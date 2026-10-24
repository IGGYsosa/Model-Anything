#!/usr/bin/env python3
"""Turn one room of a 3D Model Anything session bundle (house-capture/5) into a
Gaussian-splat training dataset, in room coordinates (floor y=0, walls on X/Z,
the same frame as bundle_to_room.py and the viewer).

usage: bundle_to_dataset.py <session-bundle> [--room N] -o <out> [--max-blur 4] [--text]

Writes (out must not exist):
  images/NNNN.jpg          the app's photos, sensor pixels untouched (EXIF stripped,
                           so no tool rotates them away from their intrinsics)
  sparse/0/*.bin           COLMAP model: PINHOLE camera per photo, ARKit poses (no
                           structure-from-motion), LiDAR seed points with per-photo
                           visibility tracks (gsplat's depth loss uses these)
  points3D.ply             the seed points, for trainers that take a PLY
  transforms.json          Nerfstudio-style poses (OpenGL camera convention)
  metadata.json            room layout, per-photo exposure/blur, held-out split

Photos with limited tracking or more than --max-blur px of predicted motion blur
(measured at 1920 px wide) are skipped. Every 8th kept photo (index % 8 == 0) is
the held-out set, matching gsplat's --test_every 8.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.agents/skills/scan-to-threejs/scripts'))
from bundle_to_room import convert  # noqa: E402
from plyio import read_vertices, write_points  # noqa: E402

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # camera: +y up/-z forward → +y down/+z forward
HELD_OUT_EVERY = 8


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


def seed_points(ply, w2r, voxel, max_points):
    v = read_vertices(ply)
    keep = v['inroom'] > 0 if 'inroom' in v.dtype.names else np.ones(len(v), bool)
    xyz = np.stack([v['x'], v['y'], v['z']], 1)[keep].astype(np.float64)
    xyz = xyz @ w2r[:3, :3].T + w2r[:3, 3]
    if {'nx', 'ny', 'nz'} <= set(v.dtype.names):
        normals = np.stack([v['nx'], v['ny'], v['nz']], 1)[keep].astype(np.float64) @ w2r[:3, :3].T
    else:
        normals = np.zeros_like(xyz)
    _, first = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0, return_index=True)
    first.sort()
    if len(first) > max_points:
        first = np.sort(np.random.default_rng(0).choice(first, max_points, replace=False))
    return xyz[first], normals[first]


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


def write_colmap(sparse, cams, images, xyz, rgb, tracks, text):
    sparse.mkdir(parents=True)
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
    if not text:
        return
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


def build(bundle, out, room=None, max_blur=4.0, voxel=0.01, max_points=300_000, track_cap=4000, text=False):
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
    xyz, normals = seed_points(ply, w2r, voxel, max_points)

    (out / 'images').mkdir(parents=True)
    cams, images, frames, meta = [], [], [], []
    best = np.full(len(xyz), -np.inf)
    best_img = np.full(len(xyz), -1)
    best_uv = np.zeros((len(xyz), 2))
    per_image = []
    for i, p in enumerate(kept):
        name = f'{i:04d}.jpg'
        (out / 'images' / name).write_bytes(strip_exif((bundle / p['file']).read_bytes()))
        k = intrinsics(p['intrinsics'])
        c2w = w2r @ mat(p['transform'])  # room space, ARKit (OpenGL) camera axes
        w2c = np.linalg.inv(c2w @ ARKIT_TO_CV)
        idx, u, v, z = visible(w2c, k, p['width'], p['height'], xyz)
        cam = c2w[:3, 3]
        view = cam - xyz[idx]
        facing = np.einsum('ij,ij->i', normals[idx], view / np.linalg.norm(view, axis=1, keepdims=True))
        facing = np.where(np.linalg.norm(normals[idx], axis=1) > 0, facing, 0.5)
        score = np.where(facing > 0.2, facing / z / (1 + (p.get('blurPixels') or 0)), -np.inf)
        better = score > best[idx]
        best[idx[better]], best_img[idx[better]] = score[better], i
        best_uv[idx[better]] = np.stack([u[better], v[better]], 1)
        per_image.append((idx, u, v))
        cams.append((p['width'], p['height'], k))
        images.append({'name': name, 'q': qvec(w2c[:3, :3]), 't': w2c[:3, 3]})
        frames.append({'file_path': f'images/{name}', 'w': p['width'], 'h': p['height'],
                       'fl_x': k[0], 'fl_y': k[1], 'cx': k[2], 'cy': k[3],
                       'transform_matrix': c2w.tolist()})
        cs = p.get('camera') or {}
        meta.append({'name': name, 'source': p['file'], 'heldOut': i % HELD_OUT_EVERY == 0,
                     'exposure': p.get('exposure'), 'iso': cs.get('iso'), 'shutter': cs.get('shutter'),
                     'exposureOffset': cs.get('exposureOffset'), 'blurPixels': p.get('blurPixels'),
                     'highRes': bool(p.get('highRes')), 'manual': bool(p.get('manual')), 'mask': None})

    # Seed colors from each point's best photo; points no photo sees well are dropped.
    seen = np.nonzero(best_img >= 0)[0]
    remap = np.full(len(xyz), -1)
    remap[seen] = np.arange(len(seen))
    rgb = np.zeros((len(seen), 3), np.uint8)
    for i, p in enumerate(kept):
        mine = seen[best_img[seen] == i]
        if len(mine):
            pixels = np.asarray(Image.open(out / 'images' / images[i]['name']).convert('RGB'))
            uv = np.clip(best_uv[mine].astype(int), 0, [p['width'] - 1, p['height'] - 1])
            rgb[remap[mine]] = pixels[uv[:, 1], uv[:, 0]]
    xyz = xyz[seen]

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

    write_colmap(out / 'sparse/0', cams, images, xyz, rgb, tracks, text)
    write_points(out / 'points3D.ply', xyz, rgb)
    (out / 'transforms.json').write_text(json.dumps({'camera_model': 'OPENCV', 'ply_file_path': 'points3D.ply',
                                                     'frames': frames}, indent=1))
    metadata = {'format': 'model-anything-dataset/1', 'source': str(bundle), 'room': room,
                'roomName': rec.get('name'), 'heldOutEvery': HELD_OUT_EVERY, 'points': len(xyz),
                'worldToRoom': layout['transform']['worldToRoom'], 'layout': layout,
                'images': meta, 'skipped': skipped}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=1))
    return metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bundle')
    ap.add_argument('--room', type=int)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--max-blur', type=float, default=4.0, help='px of predicted blur at 1920 px wide')
    ap.add_argument('--voxel', type=float, default=0.01, help='seed point spacing (m)')
    ap.add_argument('--max-points', type=int, default=300_000)
    ap.add_argument('--text', action='store_true', help='also write the COLMAP model as text')
    a = ap.parse_args()
    m = build(a.bundle, a.out, a.room, a.max_blur, a.voxel, a.max_points, text=a.text)
    held = sum(i['heldOut'] for i in m['images'])
    print(f'room {m["room"]} {m["roomName"] or ""}: {len(m["images"])} photos ({held} held out), '
          f'{len(m["skipped"])} skipped, {m["points"]} seed points → {a.out}')


if __name__ == '__main__':
    main()
