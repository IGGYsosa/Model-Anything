#!/usr/bin/env python3
"""Write a 3D Gaussian splat PLY without training, for testing the scene package and viewer.

usage: make_test_splat.py -o out.ply                       synthetic 4 × 3 m room (no inputs)
       make_test_splat.py --from-points points3D.ply -o out.ply   one small splat per LiDAR seed

The output uses the standard 3DGS layout (x y z, f_dc_0-2, opacity, scale_0-2, rot_0-3; SH
degree 0) in room coordinates, +Y up, so the viewer shows it without any rotation.
"""
import argparse

import numpy as np

from plyio import read_vertices

SH_C0 = 0.28209479177387814


def write_splat(path, xyz, rgb, scale):
    n = len(xyz)
    names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3']
    arr = np.zeros(n, dtype=[(k, '<f4') for k in names])
    arr['x'], arr['y'], arr['z'] = xyz.T
    dc = (rgb / 255.0 - 0.5) / SH_C0
    arr['f_dc_0'], arr['f_dc_1'], arr['f_dc_2'] = dc.T
    arr['opacity'] = 4.0  # logit: ~98% opaque
    arr['scale_0'] = arr['scale_1'] = arr['scale_2'] = np.log(scale)
    arr['rot_0'] = 1.0
    header = f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\n' + \
        ''.join(f'property float {k}\n' for k in names) + 'end_header\n'
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(arr.tobytes())


def synthetic_room(spacing=0.025):
    """Checkered floor, three tinted walls (the +Z side open for the dollhouse view), a box."""
    pts, cols = [], []
    xs, zs, ys = np.arange(-2, 2, spacing), np.arange(-1.5, 1.5, spacing), np.arange(0, 2.5, spacing)
    gx, gz = np.meshgrid(xs, zs)
    check = ((np.floor(gx / 0.5) + np.floor(gz / 0.5)) % 2)[..., None]
    pts.append(np.stack([gx, np.zeros_like(gx), gz], -1).reshape(-1, 3))
    cols.append((check * [190, 170, 140] + (1 - check) * [120, 95, 70]).reshape(-1, 3))
    for x0, color in ((-2, [200, 205, 215]), (2, [215, 200, 190])):
        gz, gy = np.meshgrid(zs, ys)
        pts.append(np.stack([np.full_like(gz, x0), gy, gz], -1).reshape(-1, 3))
        cols.append(np.tile(color, (gz.size, 1)))
    gx, gy = np.meshgrid(xs, ys)
    pts.append(np.stack([gx, gy, np.full_like(gx, -1.5)], -1).reshape(-1, 3))
    window = ((np.abs(gx) < 0.6) & (gy > 0.9) & (gy < 1.8)).reshape(-1, 1)
    cols.append(np.where(window, [80, 140, 220], [225, 225, 230]))
    b = np.arange(-0.4, 0.4, spacing)
    for axis in range(3):
        for sgn in (-1, 1):
            u, v = np.meshgrid(b, b)
            face = np.zeros(u.shape + (3,))
            face[..., axis] = sgn * 0.4
            face[..., (axis + 1) % 3], face[..., (axis + 2) % 3] = u, v
            face = face.reshape(-1, 3) + [0.6, 0.4, 0.2]
            pts.append(face)
            cols.append(np.tile([200, 70, 50], (len(face), 1)))
    xyz = np.vstack(pts).astype(np.float32)
    keep = xyz[:, 1] >= -1e-6
    return xyz[keep], np.vstack(cols)[keep].astype(np.float32), spacing * 0.7


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--from-points', help='point cloud PLY with x y z red green blue (dataset/points3D.ply)')
    ap.add_argument('--scale', type=float, default=0.012, help='splat radius in meters for --from-points')
    ap.add_argument('-o', '--out', required=True)
    a = ap.parse_args()
    if a.from_points:
        v = read_vertices(a.from_points)
        xyz = np.stack([v['x'], v['y'], v['z']], 1)
        rgb = np.stack([v['red'], v['green'], v['blue']], 1).astype(np.float32)
        scale = a.scale
    else:
        xyz, rgb, scale = synthetic_room()
    write_splat(a.out, xyz, rgb, scale)
    print(f'{len(xyz)} splats -> {a.out}')


if __name__ == '__main__':
    main()
