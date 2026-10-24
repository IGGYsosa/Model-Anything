"""Find where a photoreal scene is weak and where to rescan, in room coordinates.

Two kinds of weakness are scored on the LiDAR seed points (they lie on the real surfaces):
  - few photos: fewer than `min_views` training photos see the surface reasonably head-on
  - inaccurate: held-out renders differ from the real photo there (blur, smears, missing detail)
Weak points are grouped into areas (connected 0.35 m cells). Each area gets a box, the side
it faces, a suggested viewpoint for new photos and, when a held-out photo sees it, a real-vs-
render comparison. The viewer draws these; the app turns the boxes into red rescan targets.
"""
from collections import deque
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from bundle_to_dataset import visible
from plyio import read_vertices

CELL = 0.35          # m, grid used to group weak points into areas
TILE = 1.5           # m, largest floor-plan extent of one area, so each names a patch you can walk to
SIDES = ['floor', 'ceiling', '+X wall', '-X wall', '+Z wall', '-Z wall', 'surface']
ERROR_LIMIT = 0.10   # mean absolute color difference (0-1) above which a surface counts as inaccurate
MIN_VIEWS = 3
MAX_AREAS = 12
MAX_WEAK_POINTS = 60_000
COMPARISON_EDGE = 640


def load_points(dataset):
    v = read_vertices(Path(dataset) / 'points3D.ply')
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    if 'nx' in v.dtype.names:
        normals = np.stack([v['nx'], v['ny'], v['nz']], 1).astype(np.float64)
    else:
        normals = np.zeros_like(xyz)
    return xyz, normals


def camera(image):
    w2c = np.array(image['w2c'], float).reshape(4, 4)
    return w2c, tuple(image['k']), int(image['size'][0]), int(image['size'][1]), np.linalg.inv(w2c)[:3, 3]


def facing_views(xyz, normals, image):
    """Indices and pixel coords of points this photo sees, keeping only reasonably head-on views."""
    w2c, k, w, h, eye = camera(image)
    idx, u, v, z = visible(w2c, k, w, h, xyz)
    n = normals[idx]
    view = eye - xyz[idx]
    cos = np.einsum('ij,ij->i', n, view) / np.maximum(np.linalg.norm(view, axis=1), 1e-9)
    ok = (np.linalg.norm(n, axis=1) == 0) | (cos > 0.25)
    return idx[ok], u[ok], v[ok]


def block_mean(err, size):
    """Average over size×size blocks, repeated back to full size: a cheap blur so single noisy
    pixels don't flag a surface."""
    h, w = err.shape
    hb, wb = h // size * size, w // size * size
    if not hb or not wb:
        return err
    blocks = err[:hb, :wb].reshape(hb // size, size, wb // size, size).mean((1, 3))
    out = np.repeat(np.repeat(blocks, size, 0), size, 1)
    return np.pad(out, ((0, h - hb), (0, w - wb)), mode='edge')


def save_comparison(canvas, out, stem):
    half = canvas.shape[1] // 2
    names = []
    for part, pixels in (('photo', canvas[:, :half]), ('render', canvas[:, half:half * 2])):
        im = Image.fromarray(pixels)
        im.thumbnail((COMPARISON_EDGE, COMPARISON_EDGE))
        name = f'review/{stem}-{part}.jpg'
        im.save(out / name, quality=85)
        names.append(name)
    return names


def components(keys):
    """Connected groups of occupied grid cells (26-neighborhood over the last three key columns;
    earlier columns, like the surface and tile, must match)."""
    cells = {}
    for i, key in enumerate(map(tuple, keys)):
        cells.setdefault(key, []).append(i)
    seen, groups = set(), []
    for start in cells:
        if start in seen:
            continue
        seen.add(start)
        queue, members = deque([start]), []
        while queue:
            c = queue.popleft()
            members += cells[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nb = c[:-3] + (c[-3] + dx, c[-2] + dy, c[-1] + dz)
                        if nb in cells and nb not in seen:
                            seen.add(nb)
                            queue.append(nb)
        groups.append(np.array(members))
    return groups


def viewpoint(center, normal, lo, hi):
    """Where to stand for new photos: ~1.6 m out from the surface, at eye height, inside the room."""
    n = normal.copy()
    if abs(n[1]) > 0.7:  # floor/ceiling patch: look at it from the room's middle
        middle = (lo + hi) / 2
        n = np.array([middle[0] - center[0], 0, middle[2] - center[2]])
    n[1] = 0
    n = n / (np.linalg.norm(n) or 1)
    eye = center + n * 1.6
    eye[0] = np.clip(eye[0], lo[0] + 0.3, hi[0] - 0.3)
    eye[2] = np.clip(eye[2], lo[2] + 0.3, hi[2] - 0.3)
    eye[1] = 1.5
    return eye


def sides(normals):
    """Which surface each point is on (index into SIDES), from its normal. Normals point into the
    room, so a floor faces +Y and the wall at +X faces -X."""
    a = np.abs(normals)
    axis = np.argmax(a, 1)
    code = np.where(axis == 1, np.where(normals[:, 1] > 0, 0, 1),
                    np.where(axis == 0, np.where(normals[:, 0] < 0, 2, 3), np.where(normals[:, 2] < 0, 4, 5)))
    return np.where(a.max(1) == 0, 6, code)


def comparison(views, members, reason):
    """The held-out view to show for an area: where the result differs most from the photo for
    an inaccurate area, the one that sees most of it otherwise. None if no view sees it."""
    best, best_score = None, 0.0
    for view in views:
        seen = np.isin(view['points'], members)
        if seen.sum() < 5:
            continue
        score = float(view['errors'][seen].mean()) if reason == 'inaccurate' else float(seen.sum())
        if score > best_score:
            best, best_score = view, score
    return best


def review(dataset, scene, renders=None, stage=None, layout=None):
    """Writes scene/review/* and returns the `review` block for scene.json."""
    dataset, scene = Path(dataset), Path(scene)
    meta = json.loads((dataset / 'metadata.json').read_text())
    xyz, normals = load_points(dataset)
    images = meta['images']
    (scene / 'review').mkdir(parents=True, exist_ok=True)

    coverage = np.zeros(len(xyz), int)
    for image in images:
        if not image['heldOut']:
            idx, _, _ = facing_views(xyz, normals, image)
            coverage[idx] += 1

    error = np.zeros(len(xyz))
    views = []
    held = [i for i in images if i['heldOut']]
    pngs = sorted(Path(renders).glob(f'{stage}_*.png')) if renders and stage else []
    poses = {p['file']: p for p in (layout or meta['layout'])['photos']}
    for n, (image, png) in enumerate(zip(held, pngs)):
        canvas = np.asarray(Image.open(png).convert('RGB'))
        half = canvas.shape[1] // 2
        gt, render = canvas[:, :half].astype(np.float64) / 255, canvas[:, half:half * 2].astype(np.float64) / 255
        mse = np.mean((gt - render) ** 2)
        diff = block_mean(np.abs(gt - render).mean(2), 8)
        idx, u, v = facing_views(xyz, normals, image)
        scale = gt.shape[1] / image['size'][0]  # renders are at the training resolution
        px = np.clip((u * scale).astype(int), 0, gt.shape[1] - 1)
        py = np.clip((v * scale).astype(int), 0, gt.shape[0] - 1)
        error[idx] = np.maximum(error[idx], diff[py, px])
        photo, rendered = save_comparison(canvas, scene, f'view-{n:02d}')
        pose = poses.get(image['source'], {})
        views.append({'name': image['name'], 'psnr': round(99.0 if mse == 0 else -10 * math.log10(mse), 2),
                      'photo': photo, 'render': rendered, 'points': idx, 'errors': diff[py, px],
                      **{k: pose[k] for k in ('position', 'forward', 'up', 'fovY', 'aspect') if k in pose}})

    inaccurate = error > ERROR_LIMIT
    weak = inaccurate | (coverage < MIN_VIEWS)
    lo, hi = xyz.min(0), xyz.max(0)
    areas = []
    weak_idx = np.nonzero(weak)[0]
    minimum = max(20, int(len(xyz) * 0.002))
    surface = sides(normals)
    # connected weak cells on the same surface, cut into walkable tiles
    keys = np.column_stack([surface[weak_idx], np.floor(xyz[weak_idx][:, [0, 2]] / TILE),
                            np.floor(xyz[weak_idx] / CELL)]).astype(np.int64)
    for group in components(keys):
        members = weak_idx[group]
        if len(members) < minimum:
            continue
        pts = xyz[members]
        normal = normals[members].mean(0)
        normal = normal / (np.linalg.norm(normal) or 1)
        center = pts.mean(0)
        blurry = int(inaccurate[members].sum())
        reason = 'inaccurate' if blurry >= len(members) * 0.3 else 'few-photos'
        severity = float(np.clip(error[members].mean() / ERROR_LIMIT, 0, 3) if reason == 'inaccurate'
                         else 1 - coverage[members].mean() / MIN_VIEWS)
        eye = viewpoint(center, normal, lo, hi)
        best = comparison(views, members, reason)
        where = SIDES[int(np.bincount(surface[members]).argmax())]
        text = (f'{where[0].upper()}{where[1:]} near ({center[0]:.1f}, {center[2]:.1f}) m: '
                + ('blurry or smeared in the result. Retake it slowly from about 1.5 m, holding still for each photo.'
                   if reason == 'inaccurate' else
                   'too few photos see it. Photograph it head-on from a couple of positions.'))
        areas.append({'reason': reason, 'severity': round(severity, 3), 'points': int(len(members)),
                      'center': [round(float(c), 3) for c in center],
                      'min': [round(float(c) - 0.1, 3) for c in pts.min(0)],
                      'max': [round(float(c) + 0.1, 3) for c in pts.max(0)],
                      'normal': [round(float(c), 3) for c in normal], 'side': where,
                      'viewpoint': {'position': [round(float(c), 3) for c in eye],
                                    'target': [round(float(c), 3) for c in center]},
                      'view': views.index(best) if best is not None else None, 'text': text})
    areas.sort(key=lambda a: (a['severity'] * math.sqrt(a['points'])), reverse=True)
    areas = areas[:MAX_AREAS]
    for i, area in enumerate(areas):
        area['id'] = i + 1

    # red dots for the viewer: x y z reason (1 inaccurate, 0 few photos), float32
    sample = weak_idx
    if len(sample) > MAX_WEAK_POINTS:
        sample = np.sort(np.random.default_rng(0).choice(sample, MAX_WEAK_POINTS, replace=False))
    packed = np.column_stack([xyz[sample], inaccurate[sample].astype(np.float64)]).astype('<f4')
    (scene / 'review' / 'weak.bin').write_bytes(packed.tobytes())

    for view in views:
        view.pop('points')
        view.pop('errors')
    return {'format': 'model-anything-review/1', 'weakPoints': 'review/weak.bin', 'weakPointCount': int(len(sample)),
            'weakFraction': round(float(weak.mean()), 4) if len(xyz) else 0,
            'thresholds': {'error': ERROR_LIMIT, 'minViews': MIN_VIEWS},
            'areas': areas, 'views': views}
