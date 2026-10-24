#!/usr/bin/env python3
"""Turn one room of a 3D Model Anything session bundle (house-capture/5) into a
Gaussian-splat training dataset, in room coordinates (floor y=0, walls on X/Z,
the same frame as bundle_to_room.py and the viewer).

usage: bundle_to_dataset.py <session-bundle> [--room N] -o <out> [--max-blur 4] [--no-frames] [--text]

Writes (out must not exist):
  images/NNNN.jpg          the app's photos, sensor pixels untouched (EXIF stripped,
                           so no tool rotates them away from their intrinsics)
  images/fNNNNN.jpg        the 640 px color image of each LiDAR depth frame (up to 5 a second
                           while scanning): ten times the viewpoints of the photos, which is
                           what keeps a splat from smearing between photos
  sparse/0/*.bin           COLMAP model (--text for .txt): PINHOLE camera per photo, ARKit poses (no
                           structure-from-motion), LiDAR seed points with per-photo
                           visibility tracks (gsplat's depth loss uses these)
  points3D.ply             the seed points (with normals), for trainers that take a PLY
  transforms.json          Nerfstudio-style poses (OpenGL camera convention)
  metadata.json            room layout, per-image exposure/blur, held-out split
  test_images.txt          the held-out images (read by patch_trainer.py's split patch)

Photos with limited tracking or more than --max-blur px of predicted motion blur
(measured at 1920 px wide) are skipped, and depth frames over FRAME_MAX_BLUR. Every
8th kept photo is held out for scoring; depth frames always train. Depth frames also add
their confident LiDAR depth, back-projected, to the seed points: dense metric depth
targets for the views they came from.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.agents/skills/scan-to-threejs/scripts'))
from bundle_to_room import convert  # noqa: E402
from plyio import read_vertices, write_points  # noqa: E402

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # camera: +y up/-z forward → +y down/+z forward
HELD_OUT_EVERY = 8
FRAME_DEPTH_STEP = 8       # px between depth samples taken from a 256x192 LiDAR frame
FRAME_MAX_DEPTH = 4.0      # m: LiDAR beyond this is noisy
FRAME_MAX_BLUR = 3.0       # px of predicted motion blur at the depth frame's 640 px (about 9 at 1920)


def mat(cm):
    return np.array(cm, dtype=np.float64).reshape(4, 4).T  # column-major → row-major


def intrinsics(k):
    """(fx, fy, cx, cy) from the app's column-major 3×3"""
    return float(k[0]), float(k[4]), float(k[6]), float(k[7])


def qvec(r):
    """COLMAP quaternion (w, x, y, z) of a rotation matrix"""
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0] * 4
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q = np.array(q)
    return q if q[0] >= 0 else -q


def strip_exif(data):
    """JPEG bytes without APP1/Exif segments; the compressed image is copied as is."""
    if data[:2] != b'\xff\xd8':
        raise ValueError('not a JPEG')
    out, i = bytearray(data[:2]), 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError('corrupt JPEG marker')
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: the rest is image data
            out += data[i:]
            break
        length = struct.unpack('>H', data[i + 2:i + 4])[0]
        segment = data[i:i + 2 + length]
        if not (marker == 0xE1 and segment[4:10] == b'Exif\0\0'):
            out += segment
        i += 2 + length
    return bytes(out)


def pick_room(manifest, room):
    rooms = [r for r in manifest.get('rooms', []) if r.get('file')]
    if room is not None:
        if not any(r['index'] == room for r in rooms):
            raise ValueError(f'room {room} not in this bundle (has {[r["index"] for r in rooms]})')
        return room
    if len(rooms) != 1:
        raise ValueError('bundle has several rooms; choose one with --room: '
                         + ', '.join(f'{r["index"]} {r.get("name", "")}' for r in rooms))
    return rooms[0]['index']


def mesh_points(ply, w2r):
    v = read_vertices(ply)
    keep = v['inroom'] > 0 if 'inroom' in v.dtype.names else np.ones(len(v), bool)
    xyz = np.stack([v['x'], v['y'], v['z']], 1)[keep].astype(np.float64)
    xyz = xyz @ w2r[:3, :3].T + w2r[:3, 3]
    if {'nx', 'ny', 'nz'} <= set(v.dtype.names):
        normals = np.stack([v['nx'], v['ny'], v['nz']], 1)[keep].astype(np.float64) @ w2r[:3, :3].T
    else:
        normals = np.zeros_like(xyz)
    return xyz, normals


def seed_points(xyz, normals, voxel, max_points):
    """one point per voxel (the first one: mesh vertices come before depth samples)"""
    _, first = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0, return_index=True)
    first.sort()
    if len(first) > max_points:
        first = np.sort(np.random.default_rng(0).choice(first, max_points, replace=False))
    return xyz[first], normals[first]


def depth_frames(bundle, room):
    """the room's depth frames that have a color image"""
    path = bundle / 'depth/frames.json'
    if not path.is_file():
        return []
    frames = [f for f in json.loads(path.read_text()) if f.get('room') == room and f.get('rgb')]
    return [f for f in frames if (bundle / f['rgb']).is_file() and (bundle / f['file']).is_file()]


def frame_blur(frames, i):
    """Predicted motion blur (px, at the color image's size) of frame i: the app's photo formula,
    focal length x shutter x (turn rate + half the move rate), at the slower of its neighbours."""
    f = frames[i]
    a, rates = mat(f['transform']), []
    for g in frames[max(0, i - 1):i] + frames[i + 1:i + 2]:
        dt = abs(g['timestamp'] - f['timestamp'])
        if 0 < dt <= 0.6:
            b = mat(g['transform'])
            turn = math.acos(min(1.0, max(-1.0, (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2)))
            rates.append((turn + np.linalg.norm(a[:3, 3] - b[:3, 3]) / 2) / dt)
    shutter = (f.get('camera') or {}).get('shutter') or 1 / 60
    return intrinsics(f['rgbIntrinsics'])[0] * shutter * min(rates) if rates else 0.0


def frame_depth_points(bundle, f, w2r):
    """confident LiDAR depth of one frame on a sparse pixel grid, in room space, with normals
    from the depth image's neighbouring pixels (turned to face the camera)"""
    w, h = f['width'], f['height']
    raw = (bundle / f['file']).read_bytes()
    if f['file'].endswith('.f16z'):
        raw = zlib.decompress(raw, -15)
    depth = np.frombuffer(raw, '<f2').astype(np.float32).reshape(h, w)
    conf_path = bundle / f['confidence'] if f.get('confidence') else None
    if conf_path is not None and conf_path.is_file():
        conf = np.fromfile(conf_path, np.uint8).reshape(h, w)
    else:
        conf = np.full((h, w), 2, np.uint8)
    fx, fy, cx, cy = intrinsics(f['intrinsics'])
    gy, gx = np.mgrid[0:h, 0:w]
    with np.errstate(invalid='ignore'):
        cam = np.stack([(gx - cx) * depth / fx, (gy - cy) * depth / fy, depth], -1)  # OpenCV camera
    ys, xs = np.mgrid[FRAME_DEPTH_STEP // 2:h - 2:FRAME_DEPTH_STEP, FRAME_DEPTH_STEP // 2:w - 2:FRAME_DEPTH_STEP]
    z = depth[ys, xs]
    ok = (conf[ys, xs] >= 2) & np.isfinite(z) & (z > 0.1) & (z < FRAME_MAX_DEPTH)
    # flat here (no normals across edges or corners): neighbours close, and depth changing linearly
    for before, after in ((depth[ys, xs - 2], depth[ys, xs + 2]), (depth[ys - 2, xs], depth[ys + 2, xs])):
        with np.errstate(invalid='ignore'):
            ok &= np.isfinite(before) & np.isfinite(after) & (np.abs(after - z) < 0.05 * z)
            ok &= np.abs(before + after - 2 * z) < 0.01 * z
    xs, ys = xs[ok], ys[ok]
    p = cam[ys, xs]
    n = np.cross(cam[ys, xs + 2] - p, cam[ys + 2, xs] - p)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    n = np.where((np.einsum('ij,ij->i', n, p) > 0)[:, None], -n, n)
    c2w = w2r @ mat(f['transform']) @ ARKIT_TO_CV
    return p @ c2w[:3, :3].T + c2w[:3, 3], n @ c2w[:3, :3].T


def visible(w2c, k, w, h, pts):
    """Indices, pixel coords and depths of the points this camera sees (z-buffer over the
    points themselves, dilated so gaps between LiDAR vertices don't leak hidden surfaces)."""
    fx, fy, cx, cy = k
    pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    front = np.nonzero(z > 0.05)[0]
    u = fx * pc[front, 0] / z[front] + cx
    v = fy * pc[front, 1] / z[front] + cy
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx, u, v, z = front[inside], u[inside], v[inside], z[front][inside]
    cell = max(4, round(w / 240))
    gw, gh = math.ceil(w / cell), math.ceil(h / cell)
    flat = (v // cell).astype(np.int64) * gw + (u // cell).astype(np.int64)
    zbuf = np.full(gw * gh, np.inf)
    np.minimum.at(zbuf, flat, z)
    grid = zbuf.reshape(gh, gw)
    padded = np.pad(grid, 1, constant_values=np.inf)
    dilated = np.min([padded[1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)
    seen = z <= dilated.reshape(-1)[flat] * 1.03 + 0.02
    return idx[seen], u[seen], v[seen], z[seen]


def nums(values):
    return ' '.join(repr(float(v)) for v in values)


def write_colmap(sparse, cams, images, xyz, rgb, tracks, binary=True):
    """COLMAP model as binary (what gsplat's loader reads; on Windows it needs
    patch_pycolmap.py) or text (for other tools; that loader's text reader is broken)."""
    sparse.mkdir(parents=True)
    if not binary:
        write_colmap_text(sparse, cams, images, xyz, rgb, tracks)
        return
    with open(sparse / 'cameras.bin', 'wb') as f:
        f.write(struct.pack('<Q', len(cams)))
        for cid, (w, h, k) in enumerate(cams, 1):
            f.write(struct.pack('<iiQQ4d', cid, 1, w, h, *k))  # model 1 = PINHOLE
    with open(sparse / 'images.bin', 'wb') as f:
        f.write(struct.pack('<Q', len(images)))
        for iid, im in enumerate(images, 1):
            f.write(struct.pack('<i7di', iid, *im['q'], *im['t'], iid))
            f.write(im['name'].encode() + b'\0')
            f.write(struct.pack('<Q', len(im['points'])))
            for x, y, pid in im['points']:
                f.write(struct.pack('<ddq', x, y, pid))
    with open(sparse / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('<Q', len(xyz)))
        for pid in range(len(xyz)):
            f.write(struct.pack('<Q3d3BdQ', pid + 1, *xyz[pid], *rgb[pid], 0.0, len(tracks[pid])))
            for iid, j in tracks[pid]:
                f.write(struct.pack('<ii', iid, j))


def write_colmap_text(sparse, cams, images, xyz, rgb, tracks):
    with open(sparse / 'cameras.txt', 'w') as f:
        for cid, (w, h, k) in enumerate(cams, 1):
            f.write(f'{cid} PINHOLE {w} {h} {nums(k)}\n')
    with open(sparse / 'images.txt', 'w') as f:
        for iid, im in enumerate(images, 1):
            f.write(f'{iid} {nums(im["q"])} {nums(im["t"])} {iid} {im["name"]}\n')
            f.write(' '.join(f'{float(x)!r} {float(y)!r} {pid}' for x, y, pid in im['points']) + '\n')
    with open(sparse / 'points3D.txt', 'w') as f:
        for pid in range(len(xyz)):
            track = ' '.join(f'{iid} {j}' for iid, j in tracks[pid])
            f.write(f'{pid + 1} {nums(xyz[pid])} {" ".join(map(str, rgb[pid]))} 0 {track}\n')


def build(bundle, out, room=None, max_blur=4.0, voxel=0.01, max_points=300_000, track_cap=4000, binary=True,
          use_frames=True):
    bundle, out = Path(bundle), Path(out)
    if out.exists():
        raise ValueError('output directory already exists; choose a new one')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if not str(manifest.get('format', '')).startswith('house-capture/'):
        raise ValueError('expected a house-capture session bundle (manifest.json)')
    room = pick_room(manifest, room)
    layout = convert(str(bundle), room)
    w2r = mat(layout['transform']['worldToRoom'])

    kept, skipped = [], []
    for p in json.loads((bundle / 'photos/photos.json').read_text()):
        if p['room'] != room:
            continue
        blur = p.get('blurPixels')
        if not p.get('tracking', 'normal').startswith('normal'):
            skipped.append({'file': p['file'], 'reason': 'tracking ' + p['tracking']})
        elif blur is not None and blur > max_blur * p['width'] / 1920:
            skipped.append({'file': p['file'], 'reason': f'blur {blur:.1f}px'})
        elif not (bundle / p['file']).is_file():
            skipped.append({'file': p['file'], 'reason': 'missing image'})
        else:
            kept.append(p)
    if len(kept) < 3:
        raise ValueError(f'only {len(kept)} usable photos for room {room}')

    rec = next(r for r in manifest['rooms'] if r['index'] == room)
    ply = bundle / ((rec.get('mesh') or {}).get('file') or f'mesh/room-{room}.ply')
    if not ply.is_file():
        raise ValueError(f'room {room} has no LiDAR mesh ({ply.name})')
    xyz, normals = mesh_points(ply, w2r)

    frames_kept = []
    if use_frames:
        candidates = sorted(depth_frames(bundle, room), key=lambda f: f['timestamp'])
        for i, f in enumerate(candidates):
            f = {**f, 'blurPixels': frame_blur(candidates, i)}
            if f['blurPixels'] > FRAME_MAX_BLUR:
                skipped.append({'file': f['rgb'], 'reason': f'blur {f["blurPixels"]:.1f}px (depth frame)'})
            else:
                frames_kept.append(f)
        if candidates and len(xyz):
            # every frame's depth (motion blur barely affects LiDAR), inside the scanned room
            # (mesh bounds plus a margin), not through doorways
            lo, hi = xyz.min(0) - 0.25, xyz.max(0) + 0.25
            samples = [frame_depth_points(bundle, f, w2r) for f in candidates]
            extra, extra_n = np.concatenate([s[0] for s in samples]), np.concatenate([s[1] for s in samples])
            inside = np.all((extra >= lo) & (extra <= hi), 1)
            xyz, normals = np.concatenate([xyz, extra[inside]]), np.concatenate([normals, extra_n[inside]])
    xyz, normals = seed_points(xyz, normals, voxel, max_points)

    # one list of training views: photos (full resolution, scored) then depth frames (640 px)
    views = [{'name': f'{i:04d}.jpg', 'file': p['file'], 'width': p['width'], 'height': p['height'],
              'k': intrinsics(p['intrinsics']), 'transform': p['transform'], 'blur': p.get('blurPixels'),
              'camera': p.get('camera') or {}, 'heldOut': i % HELD_OUT_EVERY == 0, 'photo': p}
             for i, p in enumerate(kept)]
    for i, f in enumerate(frames_kept):
        size = Image.open(bundle / f['rgb']).size
        views.append({'name': f'f{i:05d}.jpg', 'file': f['rgb'], 'width': size[0], 'height': size[1],
                      'k': intrinsics(f['rgbIntrinsics']), 'transform': f['transform'], 'blur': f['blurPixels'],
                      'camera': f.get('camera') or {}, 'heldOut': False, 'photo': None})

    (out / 'images').mkdir(parents=True)
    cams, images, frames, meta = [], [], [], []
    best = np.full(len(xyz), -np.inf)
    best_img = np.full(len(xyz), -1)
    best_uv = np.zeros((len(xyz), 2))
    per_image = []
    for i, vw in enumerate(views):
        name, k, w, h = vw['name'], vw['k'], vw['width'], vw['height']
        (out / 'images' / name).write_bytes(strip_exif((bundle / vw['file']).read_bytes()))
        c2w = w2r @ mat(vw['transform'])  # room space, ARKit (OpenGL) camera axes
        w2c = np.linalg.inv(c2w @ ARKIT_TO_CV)
        idx, u, v, z = visible(w2c, k, w, h, xyz)
        cam = c2w[:3, 3]
        view = cam - xyz[idx]
        facing = np.einsum('ij,ij->i', normals[idx], view / np.linalg.norm(view, axis=1, keepdims=True))
        facing = np.where(np.linalg.norm(normals[idx], axis=1) > 0, facing, 0.5)
        # seed colors prefer close, head-on, sharp, high-resolution views
        score = np.where(facing > 0.2, facing / z / (1 + (vw['blur'] or 0)) * min(1.0, w / 1920), -np.inf)
        better = score > best[idx]
        best[idx[better]], best_img[idx[better]] = score[better], i
        best_uv[idx[better]] = np.stack([u[better], v[better]], 1)
        per_image.append((idx, u, v))
        cams.append((w, h, k))
        images.append({'name': name, 'q': qvec(w2c[:3, :3]), 't': w2c[:3, 3]})
        frames.append({'file_path': f'images/{name}', 'w': w, 'h': h,
                       'fl_x': k[0], 'fl_y': k[1], 'cx': k[2], 'cy': k[3],
                       'transform_matrix': c2w.tolist()})
        cs, p = vw['camera'], vw['photo'] or {}
        meta.append({'name': name, 'source': vw['file'], 'heldOut': vw['heldOut'], 'depthFrame': vw['photo'] is None,
                     'exposure': p.get('exposure'), 'iso': cs.get('iso'), 'shutter': cs.get('shutter'),
                     'exposureOffset': cs.get('exposureOffset'), 'blurPixels': vw['blur'],
                     'highRes': bool(p.get('highRes')), 'manual': bool(p.get('manual')), 'mask': None,
                     # room → OpenCV camera and pixels, for the QC/review pass
                     'w2c': w2c.reshape(-1).tolist(), 'k': list(k), 'size': [w, h]})

    # Seed colors from each point's best photo; points no photo sees well are dropped.
    seen = np.nonzero(best_img >= 0)[0]
    remap = np.full(len(xyz), -1)
    remap[seen] = np.arange(len(seen))
    rgb = np.zeros((len(seen), 3), np.uint8)
    for i, vw in enumerate(views):
        mine = seen[best_img[seen] == i]
        if len(mine):
            pixels = np.asarray(Image.open(out / 'images' / images[i]['name']).convert('RGB'))
            uv = np.clip(best_uv[mine].astype(int), 0, [vw['width'] - 1, vw['height'] - 1])
            rgb[remap[mine]] = pixels[uv[:, 1], uv[:, 0]]
    xyz, normals = xyz[seen], normals[seen]

    rng = np.random.default_rng(0)
    tracks = [[] for _ in range(len(xyz))]
    for i, (idx, u, v) in enumerate(per_image):
        ok = remap[idx] >= 0
        idx, u, v = remap[idx[ok]], u[ok], v[ok]
        if len(idx) > track_cap:
            pick = np.sort(rng.choice(len(idx), track_cap, replace=False))
            idx, u, v = idx[pick], u[pick], v[pick]
        images[i]['points'] = list(zip(u.tolist(), v.tolist(), (idx + 1).tolist()))
        for j, pid in enumerate(idx.tolist()):
            tracks[pid].append((i + 1, j))

    write_colmap(out / 'sparse/0', cams, images, xyz, rgb, tracks, binary)
    write_points(out / 'points3D.ply', xyz, rgb, normals)
    (out / 'transforms.json').write_text(json.dumps({'camera_model': 'OPENCV', 'ply_file_path': 'points3D.ply',
                                                     'frames': frames}, indent=1))
    metadata = {'format': 'model-anything-dataset/1', 'source': str(bundle), 'room': room,
                'roomName': rec.get('name'), 'heldOutEvery': HELD_OUT_EVERY, 'points': len(xyz),
                'worldToRoom': layout['transform']['worldToRoom'], 'layout': layout,
                'images': meta, 'skipped': skipped}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=1))
    (out / 'test_images.txt').write_text(''.join(m['name'] + '\n' for m in meta if m['heldOut']))
    return metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bundle')
    ap.add_argument('--room', type=int)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--max-blur', type=float, default=4.0, help='px of predicted blur at 1920 px wide')
    ap.add_argument('--voxel', type=float, default=0.01, help='seed point spacing (m)')
    ap.add_argument('--max-points', type=int, default=300_000)
    ap.add_argument('--text', action='store_true', help='write the COLMAP model as text instead of .bin')
    ap.add_argument('--no-frames', action='store_true', help="photos only: leave out the depth frames' images and depth")
    a = ap.parse_args()
    m = build(a.bundle, a.out, a.room, a.max_blur, a.voxel, a.max_points, binary=not a.text,
              use_frames=not a.no_frames)
    held = sum(i['heldOut'] for i in m['images'])
    frames = sum(i['depthFrame'] for i in m['images'])
    print(f'room {m["room"]} {m["roomName"] or ""}: {len(m["images"]) - frames} photos ({held} held out), '
          f'{frames} depth frames, {len(m["skipped"])} skipped, {m["points"]} seed points -> {a.out}')


if __name__ == '__main__':
    main()
