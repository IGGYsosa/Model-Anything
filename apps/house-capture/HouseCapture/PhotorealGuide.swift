import Foundation
import simd

/// Photoreal capture. Measuring needs one good view of each wall; a Gaussian splat needs every
/// surface from many overlapping viewpoints, or it smears between them. So in photoreal mode the
/// app shoots on movement (every ~15 cm or ~10° of turn) instead of on new coverage, and walks the
/// user through the passes that give a room that overlap: a loop at chest height, a loop held
/// low, then the ceiling.
struct PhotorealGuide: Equatable {
  /// photos for a good room (the training pipeline also adds the LiDAR depth frames)
  static let target = 200
  /// background capture eases off past this many photos in a room (memory, upload size)
  static let maxPhotos = 400
  /// shoot again after moving this far or turning this much: ~70% overlap between neighbours
  static let step: Float = 0.15
  static let turn: Float = .pi / 18
  /// headings around the room, for the loop passes
  static let sectors = 8
  static let loopSectors = 7
  static let ceilingPhotos = 8
  /// below the usual standing camera height by this much counts as the low pass
  static let lowDrop: Float = 0.35

  enum Pass: Equatable { case chest, low, ceiling, fill, done }

  private(set) var photos = 0
  private(set) var chest: Set<Int> = []
  private(set) var low: Set<Int> = []
  private(set) var ceiling = 0
  /// camera heights of level photos taken before the low pass, for "standing height"
  private var standing: [Float] = []

  static func wantsPhoto(after last: PhotoView?, at view: PhotoView) -> Bool {
    guard let last else { return true }
    return simd_distance(last.position, view.position) > step || simd_dot(last.forward, view.forward) < cos(turn)
  }

  /// which way round the room (0..<sectors) a level view faces
  static func sector(_ forward: SIMD3<Float>) -> Int {
    let a = atan2(forward.x, -forward.z) + .pi  // 0…2π
    return min(sectors - 1, Int(a / (2 * .pi) * Float(sectors)))
  }

  var standingHeight: Float? {
    guard !standing.isEmpty else { return nil }
    let sorted = standing.sorted()
    return sorted[sorted.count / 2]
  }

  mutating func add(_ view: PhotoView) {
    photos += 1
    let f = view.forward
    if f.y > 0.5 {
      ceiling += 1
      return
    }
    guard f.y > -0.6 else { return } // floor shots count toward the total only
    let s = Self.sector(f)
    if let h = standingHeight, standing.count >= 5, view.position.y < h - Self.lowDrop {
      low.insert(s)
    } else {
      chest.insert(s)
      if pass == .chest || standing.count < 5 { standing.append(view.position.y) }
    }
  }

  var pass: Pass {
    if chest.count < Self.loopSectors { return .chest }
    if low.count < Self.loopSectors - 1 { return .low }
    if ceiling < Self.ceilingPhotos { return .ceiling }
    return photos < Self.target ? .fill : .done
  }

  var message: String {
    switch pass {
    case .chest:
      return "Pass 1 of 3: walk slowly along the walls at chest height, facing into the room (\(chest.count) of \(Self.loopSectors) directions)."
    case .low:
      return "Pass 2 of 3: go around again holding the phone low, near your waist (\(low.count) of \(Self.loopSectors - 1) directions)."
    case .ceiling:
      return "Pass 3 of 3: tilt up to the ceiling and the top corners (\(ceiling) of \(Self.ceilingPhotos))."
    case .fill:
      return "Keep going to \(Self.target) photos: step closer to furniture, shelves and any red spots."
    case .done:
      return "Enough photos for photoreal. Save the room when you're ready."
    }
  }
}

/// Dim rooms push the camera to a long shutter and high ISO, so every frame blurs and turns
/// noisy, which costs a splat far more detail than it costs measuring. ARKit video exposes at
/// most ~1/60 s, so a sustained high ISO at that shutter means the room needs more light.
struct LightMeter: Equatable {
  static let darkISO: Float = 1000
  static let brightISO: Float = 800  // hysteresis, so the warning doesn't flicker
  private(set) var iso: Float?
  private(set) var tooDark = false

  mutating func add(iso sample: Float?, shutter: Double) {
    guard let sample, sample > 0 else { return }
    iso = iso.map { $0 * 0.9 + sample * 0.1 } ?? sample
    guard let iso else { return }
    let slow = shutter >= 1.0 / 70
    tooDark = slow && iso >= (tooDark ? Self.brightISO : Self.darkISO)
  }

  static let message = "Too dark for photoreal: turn on more lights. In dim light every photo blurs and the model comes out soft."
}
