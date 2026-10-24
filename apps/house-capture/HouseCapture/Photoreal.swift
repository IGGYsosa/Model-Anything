import Foundation
import Security
import simd
import SwiftUI
import WebKit

/// Where "Make photoreal" sends a room: the job server in apps/pipeline (server.py). It trains a
/// Gaussian splat from the room's photos and LiDAR and serves the web viewer for the result.
/// The address lives in UserDefaults; the bearer token in the Keychain.
enum PhotorealSettings {
  private static let urlKey = "photorealServerURL"
  private static let keychain: [String: Any] = [
    kSecClass as String: kSecClassGenericPassword,
    kSecAttrService as String: "com.iggysosa.modelanything.photoreal",
    kSecAttrAccount as String: "token",
  ]

  static var serverURL: URL? {
    get { UserDefaults.standard.string(forKey: urlKey).flatMap(validURL) }
    set { UserDefaults.standard.set(newValue?.absoluteString, forKey: urlKey) }
  }

  static func validURL(_ text: String) -> URL? {
    guard let url = URL(string: text.trimmingCharacters(in: .whitespacesAndNewlines)),
          let scheme = url.scheme?.lowercased(), scheme == "https" || scheme == "http",
          url.host?.isEmpty == false else { return nil }
    return url
  }

  static var token: String? {
    get {
      var query = keychain
      query[kSecReturnData as String] = true
      query[kSecMatchLimit as String] = kSecMatchLimitOne
      var item: CFTypeRef?
      guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess, let data = item as? Data else { return nil }
      return String(data: data, encoding: .utf8)
    }
    set {
      SecItemDelete(keychain as CFDictionary)
      guard let newValue, !newValue.isEmpty else { return }
      var item = keychain
      item[kSecValueData as String] = Data(newValue.utf8)
      item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
      SecItemAdd(item as CFDictionary, nil)
    }
  }

  static var isConfigured: Bool { serverURL != nil && !(token ?? "").isEmpty }
}

/// job.json as the server reports it (GET /jobs/{id})
struct PhotorealJob: Decodable, Equatable {
  /// a weak area (apps/pipeline/review.py): box in the scanned session's room coordinates
  struct Hint: Decodable, Equatable {
    let text: String
    let id: Int?
    let reason: String?
    let min: [Float]?
    let max: [Float]?
  }
  struct Metrics: Decodable, Equatable { let psnr: Double?; let quality: String? }
  let id: String
  let state: String
  let step: String?
  let error: String?
  let hints: [Hint]?
  let metrics: Metrics?
  /// the scanned session's ARKit world → room, column-major (for rescan boxes)
  let worldToRoom: [Float]?

  var finished: Bool { state == "done" || state == "failed" }
  var statusText: String {
    switch (state, step) {
    case ("queued", _): "Waiting for the photoreal server…"
    case (_, "train"?): "Training the photoreal scene (this takes a while)…"
    case (_, "qc"?): "Checking quality…"
    case (_, "package"?): "Packaging the scene…"
    default: "Preparing photos and LiDAR…"
    }
  }
}

/// Weak areas the room screen hands to the next capture ("Rescan weak spots"). Passed through
/// `pending` because a capture starts from several screens with the same entry point.
struct RescanRequest {
  var houseID: String
  /// the session the photoreal job was made from
  var sessionID: String
  var worldToRoom: simd_float4x4
  var areas: [(id: Int, min: SIMD3<Float>, max: SIMD3<Float>)]
  @MainActor static var pending: RescanRequest?

  init?(houseID: String, sessionID: String, job: PhotorealJob) {
    guard let m = job.worldToRoom, m.count == 16 else { return nil }
    let areas = (job.hints ?? []).compactMap { h -> (id: Int, min: SIMD3<Float>, max: SIMD3<Float>)? in
      guard let id = h.id, let lo = h.min, let hi = h.max, lo.count == 3, hi.count == 3 else { return nil }
      return (id, SIMD3(lo[0], lo[1], lo[2]), SIMD3(hi[0], hi[1], hi[2]))
    }
    guard !areas.isEmpty else { return nil }
    self.houseID = houseID
    self.sessionID = sessionID
    worldToRoom = simd_float4x4(columnMajor: m)
    self.areas = areas
  }

  /// The areas as boxes in `zone`'s frame, which is a relocalized live session's world:
  /// room → scanned session's world (inverse worldToRoom) → zone (its toZone). Nil when that
  /// session isn't placed in the zone, so the boxes can't be located.
  func boxes(in house: HouseProject, zone: String) -> [CoverageEngine.RescanBox]? {
    guard let session = house.session(sessionID), session.zone == zone, session.alignment.method != "unaligned" else { return nil }
    let zoneToRoom = worldToRoom * simd_float4x4(columnMajor: session.alignment.toZone).inverse
    return areas.map { area in
      var toCenter = matrix_identity_float4x4
      toCenter.columns.3 = SIMD4(-(area.min + area.max) / 2, 1)
      return CoverageEngine.RescanBox(id: area.id, worldToBox: toCenter * zoneToRoom, halfExtents: (area.max - area.min) / 2)
    }
  }
}

/// The latest job for each scan (house/room/session/index), so results survive relaunches.
enum PhotorealJobs {
  private static let key = "photorealJobs"
  static func id(for scan: String) -> String? {
    (UserDefaults.standard.dictionary(forKey: key) as? [String: String])?[scan]
  }
  static func set(_ id: String, for scan: String) {
    var all = (UserDefaults.standard.dictionary(forKey: key) as? [String: String]) ?? [:]
    all[scan] = id
    UserDefaults.standard.set(all, forKey: key)
  }
}

enum PhotorealError: LocalizedError {
  case notConfigured
  case server(Int, String)
  var errorDescription: String? {
    switch self {
    case .notConfigured: "Set the photoreal server address and token first."
    case .server(let code, let detail): "The photoreal server answered \(code): \(detail)"
    }
  }
}

enum PhotorealClient {
  /// Upload one room's export (ScanBundle.zippedRoom) and start a job. The multipart body is
  /// written to a temporary file and streamed, so large photo sets never sit in memory.
  static func upload(zip: URL, room: Int, name: String, preset: String = "interior",
                     progress: @escaping @Sendable (Double) -> Void) async throws -> PhotorealJob {
    guard let server = PhotorealSettings.serverURL, let token = PhotorealSettings.token, !token.isEmpty else {
      throw PhotorealError.notConfigured
    }
    let boundary = "Boundary-\(UUID().uuidString)"
    let body = FileManager.default.temporaryDirectory.appendingPathComponent("photoreal-\(UUID().uuidString).multipart")
    defer { try? FileManager.default.removeItem(at: body) }
    try writeMultipart(to: body, boundary: boundary, fields: ["room": String(room), "preset": preset, "name": name], file: zip)
    var request = URLRequest(url: server.appendingPathComponent("jobs"))
    request.httpMethod = "POST"
    request.timeoutInterval = 900
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    let (data, response) = try await URLSession.shared.upload(for: request, fromFile: body, delegate: UploadProgress(progress))
    try check(response, data)
    struct Created: Decodable { let id: String }
    return try await status(try JSONDecoder().decode(Created.self, from: data).id)
  }

  static func status(_ id: String) async throws -> PhotorealJob {
    guard let server = PhotorealSettings.serverURL else { throw PhotorealError.notConfigured }
    let (data, response) = try await URLSession.shared.data(from: server.appendingPathComponent("jobs/\(id)"))
    try check(response, data)
    return try JSONDecoder().decode(PhotorealJob.self, from: data)
  }

  /// The server's bundled web viewer opened on this job's scene
  static func viewerURL(_ id: String) -> URL? { PhotorealSettings.serverURL?.appendingPathComponent("view/\(id)") }

  private static func check(_ response: URLResponse, _ data: Data) throws {
    guard let http = response as? HTTPURLResponse else { return }
    guard (200..<300).contains(http.statusCode) else {
      struct Detail: Decodable { let detail: String }
      let detail = (try? JSONDecoder().decode(Detail.self, from: data).detail) ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
      throw PhotorealError.server(http.statusCode, detail)
    }
  }

  private static func writeMultipart(to url: URL, boundary: String, fields: [String: String], file: URL) throws {
    FileManager.default.createFile(atPath: url.path, contents: nil)
    let out = try FileHandle(forWritingTo: url)
    defer { try? out.close() }
    func write(_ text: String) throws { try out.write(contentsOf: Data(text.utf8)) }
    for (key, value) in fields.sorted(by: { $0.key < $1.key }) {
      try write("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(key)\"\r\n\r\n\(value)\r\n")
    }
    try write("--\(boundary)\r\nContent-Disposition: form-data; name=\"bundle\"; filename=\"room.zip\"\r\nContent-Type: application/zip\r\n\r\n")
    let input = try FileHandle(forReadingFrom: file)
    defer { try? input.close() }
    while let chunk = try input.read(upToCount: 1 << 20), !chunk.isEmpty {
      try out.write(contentsOf: chunk)
    }
    try write("\r\n--\(boundary)--\r\n")
  }
}

private final class UploadProgress: NSObject, URLSessionTaskDelegate {
  let report: @Sendable (Double) -> Void
  init(_ report: @escaping @Sendable (Double) -> Void) { self.report = report }
  func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                  totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
    guard totalBytesExpectedToSend > 0 else { return }
    report(Double(totalBytesSent) / Double(totalBytesExpectedToSend))
  }
}

struct PhotorealSettingsSheet: View {
  var onSave: () -> Void = {}
  @Environment(\.dismiss) private var dismiss
  @State private var address = PhotorealSettings.serverURL?.absoluteString ?? ""
  @State private var token = PhotorealSettings.token ?? ""

  var body: some View {
    NavigationStack {
      Form {
        Section {
          TextField("https://photoreal.example.com", text: $address)
            .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
          SecureField("Access token", text: $token)
        } footer: {
          Text("Make photoreal uploads the room's photos, LiDAR mesh and layout to this server, which trains a photoreal 3D scene from them. Use a server you run or trust.")
        }
      }
      .navigationTitle("Photoreal server").navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
        ToolbarItem(placement: .confirmationAction) {
          Button("Save") {
            PhotorealSettings.serverURL = PhotorealSettings.validURL(address)
            PhotorealSettings.token = token.trimmingCharacters(in: .whitespacesAndNewlines)
            onSave()
            dismiss()
          }.disabled(PhotorealSettings.validURL(address) == nil || token.trimmingCharacters(in: .whitespaces).isEmpty)
        }
      }
    }
  }
}

/// The trained scene in the shared web viewer (same build as the browser version).
struct PhotorealViewer: View {
  let url: URL
  @Environment(\.dismiss) private var dismiss
  var body: some View {
    NavigationStack {
      PhotorealWebView(url: url).ignoresSafeArea(edges: .bottom)
        .navigationTitle("Photoreal").navigationBarTitleDisplayMode(.inline)
        .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
    }
  }
}

private struct PhotorealWebView: UIViewRepresentable {
  let url: URL
  func makeUIView(context: Context) -> WKWebView {
    let view = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
    view.isOpaque = false
    view.backgroundColor = .black
    view.scrollView.isScrollEnabled = false // the 3D canvas owns drag and pinch
    view.load(URLRequest(url: url))
    return view
  }
  func updateUIView(_ view: WKWebView, context: Context) {}
}
