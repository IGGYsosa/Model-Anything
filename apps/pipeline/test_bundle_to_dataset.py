"""Checks for the bundle → splat dataset converter on a synthetic session.
Run: python -m unittest discover -s apps/pipeline -p 'test_*.py'
"""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from bundle_to_dataset import build, mat, strip_exif

ANGLE, OFFSET, FLOOR = math.radians(30), np.array([3.0, -1.2, -2.0]), -1.2  # room in the ARKit world
HALF = (2.0, 1.5)  # room half extents (x, z) before rotation
K = [100, 0, 0, 0, 100, 0, 80, 60, 1]


def world(r):
    """4×4 placing a room-local frame in the ARKit world"""
    c, s = math.cos(ANGLE), math.sin(ANGLE)
    m = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]) @ r
    m[:3, 3] += OFFSET
    return m


def yaw(phi, pos):
    c, s = math.cos(phi), math.sin(phi)
    m = np.array([[c, 0, s, pos[0]], [0, 1, 0, pos[1]], [-s, 0, c, pos[2]], [0, 0, 0, 1]], float)
    return m


def cm(m):
    return m.T.reshape(-1).tolist()  # row-major → column-major 16


def wall(center, along, length, height=2.5):
    u = np.array([along[0], 0, along[1]], float) / np.linalg.norm(along)
    r = np.eye(4)
    r[:3, 0], r[:3, 1], r[:3, 2] = u, [0, 1, 0], np.cross(u, [0, 1, 0])
    r[:3, 3] = [center[0], height / 2, center[1]]
    return {'identifier': f'w{center}', 'transform': cm(world(r)), 'dimensions': [length, height, 0],
            'confidence': {'high': {}}}


def make_bundle(bundle, size=(160, 120), focal=100, count=13):
    """Synthetic one-room session (room index 2) in a rotated, offset ARKit world; returns photos.
    The last photo belongs to another room; smoke_test.py makes bigger, rendered ones."""
    (bundle / 'rooms').mkdir(parents=True)
    (bundle / 'photos').mkdir()
    (bundle / 'mesh').mkdir()
    hx, hz = HALF
    room = {'walls': [wall((0, -hz), (1, 0), 2 * hx), wall((hx, 0), (0, 1), 2 * hz),
                      wall((0, hz), (-1, 0), 2 * hx), wall((-hx, 0), (0, -1), 2 * hz)],
            'objects': [], 'doors': [], 'windows': [], 'openings': []}
    (bundle / 'rooms/room-2.json').write_text(json.dumps(room))
    (bundle / 'manifest.json').write_text(json.dumps({
        'format': 'house-capture/5', 'rooms': [
            {'index': 2, 'name': 'Patio', 'file': 'rooms/room-2.json', 'mesh': {'file': 'mesh/room-2.ply'}}]}))

    # LiDAR mesh vertices on the floor and walls (room-local), plus an outside surface (inroom 0)
    g = np.linspace(-1, 1, 21)
    pts, nrm = [], []
    for a in g:
        for b in g:
            pts.append([a * hx, 0, b * hz]); nrm.append([0, 1, 0])
            for sgn in (-1, 1):
                pts.append([a * hx, 1.25 + b * 1.2, sgn * hz]); nrm.append([0, 0, -sgn])
                pts.append([sgn * hx, 1.25 + b * 1.2, a * hz]); nrm.append([-sgn, 0, 0])
    local = np.array(pts)
    inroom = np.ones(len(local), np.uint8)
    outside = np.array([[x, 1.0, hz + 3] for x in np.linspace(-1, 1, 10)])
    local = np.vstack([local, outside]); nrm = np.vstack([nrm, np.tile([0, 0, -1], (10, 1))])
    inroom = np.concatenate([inroom, np.zeros(10, np.uint8)])
    rot = world(np.eye(4))[:3, :3]
    w = local @ rot.T + OFFSET
    n = nrm @ rot.T
    vert = np.empty(len(w), dtype=[(k, '<f4') for k in ('x', 'y', 'z', 'nx', 'ny', 'nz', 'coverage')]
                    + [(k, 'u1') for k in ('red', 'green', 'blue', 'inroom')])
    for i, k in enumerate('xyz'):
        vert[k], vert['n' + k] = w[:, i], n[:, i]
    vert['coverage'], vert['red'], vert['green'], vert['blue'], vert['inroom'] = 1, 255, 0, 0, inroom
    header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(w)}\n'
              + ''.join(f'property float {k}\n' for k in ('x', 'y', 'z', 'nx', 'ny', 'nz', 'coverage'))
              + ''.join(f'property uchar {k}\n' for k in ('red', 'green', 'blue', 'inroom'))
              + 'element face 0\nproperty list uchar uint vertex_indices\nproperty uchar class\nend_header\n')
    (bundle / 'mesh/room-2.ply').write_bytes(header.encode() + vert.tobytes())

    # 12 photos walking around the middle of the room; one blurry, one with lost tracking,
    # one from another room. Every image is the same color, unlike the PLY's red heatmap.
    exif = Image.Exif()
    exif[0x0112] = 6
    photos = []
    w, h = size
    k = [focal, 0, 0, 0, focal, 0, w / 2, h / 2, 1]
    for i in range(count):
        phi = i * math.tau / (count - 1)  # walking a 0.5 m circle, looking out across the room
        file = f'photos/{i:04d}.jpg'
        Image.new('RGB', size, (10, 200, 30)).save(bundle / file, quality=95, exif=exif)
        photos.append({'file': file, 'room': 2 if i < count - 1 else 1, 'width': w, 'height': h,
                       'intrinsics': k, 'tracking': 'normal', 'blurPixels': 0.2, 'exposure': 0.01,
                       'camera': {'iso': 100 + i, 'shutter': 0.01},
                       'transform': cm(world(yaw(phi, (0.2 + 0.5 * math.sin(phi), 1.4, 0.1 + 0.5 * math.cos(phi)))))})
    photos[3]['blurPixels'] = 9.0
    photos[5]['tracking'] = 'limited:excessiveMotion'
    (bundle / 'photos/photos.json').write_text(json.dumps(photos))
    return photos


CEILING = 2.5
TILT_UP = np.array([[1, 0, 0, 0], [0, math.cos(0.5), -math.sin(0.5), 0], [0, math.sin(0.5), math.cos(0.5), 0], [0, 0, 0, 1]])


def room_depth(c2w_local, k, w, h):
    """OpenCV depth of the box room (floor, ceiling, four walls) seen from a room-local ARKit pose"""
    fx, fy, cx, cy = k
    v, u = np.mgrid[0:h, 0:w] + 0.5
    rays = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], -1) @ c2w_local[:3, :3].T  # ARKit axes
    o = c2w_local[:3, 3]
    t = np.full((h, w), np.inf)
    for axis, plane in ((1, 0.0), (1, CEILING), (0, -HALF[0]), (0, HALF[0]), (2, -HALF[1]), (2, HALF[1])):
        with np.errstate(divide='ignore', invalid='ignore'):
            ti = (plane - o[axis]) / rays[..., axis]
        t = np.where((ti > 0) & (ti < t), ti, t)
    return t  # the rays have camera z = -1, so t is the depth


def add_depth_frames(bundle, count=6):
    """LiDAR depth frames (room 2) with 64×48 color images, the camera turning slowly from the room's
    middle, tilted up; then one frame after a fast 90° turn, which should be skipped as blurred."""
    (bundle / 'depth').mkdir()
    dw, dh, kd = 128, 96, (80.0, 80.0, 64.0, 48.0)
    frames = []
    for i in range(count + 1):
        phi = 0.03 * i if i < count else 0.03 * count + math.radians(90)
        local = yaw(phi, (0.2, 1.4, 0.1)) @ TILT_UP  # sees the ceiling, which the mesh leaves out
        depth = room_depth(local, kd, dw, dh)
        name = f'{i:05d}'
        import zlib
        z = zlib.compressobj(wbits=-15)
        (bundle / f'depth/{name}.f16z').write_bytes(z.compress(depth.astype('<f2').tobytes()) + z.flush())
        (bundle / f'depth/{name}.conf').write_bytes(np.full((dh, dw), 2, np.uint8).tobytes())
        Image.new('RGB', (64, 48), (220, 40, 120)).save(bundle / f'depth/{name}.jpg', quality=95)
        frames.append({'file': f'depth/{name}.f16z', 'confidence': f'depth/{name}.conf', 'rgb': f'depth/{name}.jpg',
                       'room': 2, 'timestamp': 100 + 0.2 * i, 'width': dw, 'height': dh,
                       'intrinsics': [kd[0], 0, 0, 0, kd[1], 0, kd[2], kd[3], 1],
                       'rgbIntrinsics': [40, 0, 0, 0, 40, 0, 32, 24, 1],
                       'camera': {'iso': 200, 'shutter': 1 / 60}, 'transform': cm(world(local))})
    (bundle / 'depth/frames.json').write_text(json.dumps(frames))
    return frames


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.bundle = Path(self.temp.name) / 'session'
        self.photos = make_bundle(self.bundle)
        self.out = Path(self.temp.name) / 'dataset'
        self.meta = build(self.bundle, self.out, binary=False)  # text, to read back here

    def tearDown(self):
        self.temp.cleanup()

    def colmap_images(self):
        lines = [l for l in (self.out / 'sparse/0/images.txt').read_text().splitlines()]
        images = []
        for head, pts in zip(lines[::2], lines[1::2]):
            h = head.split()
            p = pts.split()
            images.append({'q': np.array(h[1:5], float), 't': np.array(h[5:8], float), 'name': h[9],
                           'points': [(float(p[j]), float(p[j + 1]), int(p[j + 2])) for j in range(0, len(p), 3)]})
        return images

    def points(self):
        rows = [l.split() for l in (self.out / 'sparse/0/points3D.txt').read_text().splitlines()]
        return {int(r[0]): (np.array(r[1:4], float), [int(c) for c in r[4:7]]) for r in rows}

    def test_filtering_split_and_metadata(self):
        self.assertEqual([s['file'] for s in self.meta['skipped']], ['photos/0003.jpg', 'photos/0005.jpg'])
        self.assertEqual(len(self.meta['images']), 10)
        self.assertEqual([i['heldOut'] for i in self.meta['images']].count(True), 2)
        self.assertTrue(self.meta['images'][0]['heldOut'] and self.meta['images'][8]['heldOut'])
        self.assertEqual(self.meta['images'][3]['source'], 'photos/0004.jpg')
        self.assertEqual(self.meta['room'], 2)
        self.assertEqual(len(json.loads((self.out / 'transforms.json').read_text())['frames']), 10)
        with self.assertRaises(ValueError):
            build(self.bundle, self.out)

    def test_room_coordinates(self):
        pts = np.array([p for p, _ in self.points().values()])
        self.assertGreater(len(pts), 500)
        # floor at 0, walls axis-aligned and centered, the outside (inroom 0) surface gone
        self.assertAlmostEqual(pts[:, 1].min(), 0, places=4)
        for axis, half in ((0, HALF[0]), (2, HALF[1])):
            lo, hi = sorted([abs(pts[:, axis].min()), abs(pts[:, axis].max())])
            self.assertAlmostEqual(lo, hi, places=3)
            self.assertIn(round(hi, 3), (HALF[0], HALF[1]))

    def test_projection_matches_capture_convention(self):
        w2r = mat(self.meta['worldToRoom'])
        r2w = np.linalg.inv(w2r)
        points = self.points()
        kept = [p for p in self.photos[:12] if p['file'] not in ('photos/0003.jpg', 'photos/0005.jpg')]
        checked = 0
        for im, photo in zip(self.colmap_images(), kept):
            w, x, y, z = im['q']
            rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            cam_from_world = np.linalg.inv(mat(photo['transform']))
            for u, v, pid in im['points'][::25]:
                p_room = points[pid][0]
                pc = rot @ p_room + im['t']  # COLMAP: room → OpenCV camera
                self.assertAlmostEqual(u, 100 * pc[0] / pc[2] + 80, places=4)
                self.assertAlmostEqual(v, 100 * pc[1] / pc[2] + 60, places=4)
                # reconstruct_bundle.py's projection from the original ARKit pose
                c = cam_from_world @ np.append(r2w[:3, :3] @ p_room + r2w[:3, 3], 1)
                self.assertAlmostEqual(u, 100 * c[0] / -c[2] + 80, places=4)
                self.assertAlmostEqual(v, 100 * -c[1] / -c[2] + 60, places=4)
                checked += 1
        self.assertGreater(checked, 50)

    def test_tracks_are_consistent(self):
        points = self.points()
        rows = {int(r.split()[0]): r.split()[8:] for r in (self.out / 'sparse/0/points3D.txt').read_text().splitlines()}
        images = self.colmap_images()
        for pid, track in list(rows.items())[::40]:
            for iid, j in zip(track[::2], track[1::2]):
                self.assertEqual(images[int(iid) - 1]['points'][int(j)][2], pid)
        self.assertTrue(all(len(im['points']) > 0 for im in images))

    def test_seed_colors_come_from_photos(self):
        colors = np.array([c for _, c in self.points().values()])
        self.assertTrue(np.all(np.abs(colors - [10, 200, 30]) < 12))

    def test_images_keep_pixels_and_lose_exif(self):
        original = Image.open(self.bundle / 'photos/0004.jpg')
        copy = Image.open(self.out / 'images/0003.jpg')
        self.assertEqual(original.getexif().get(0x0112), 6)
        self.assertIsNone(copy.getexif().get(0x0112))
        self.assertEqual(copy.size, (160, 120))
        self.assertTrue(np.array_equal(np.asarray(original), np.asarray(copy)))


class DepthFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.bundle = Path(self.temp.name) / 'session'
        make_bundle(self.bundle)
        add_depth_frames(self.bundle)
        self.out = Path(self.temp.name) / 'dataset'
        self.meta = build(self.bundle, self.out, binary=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_sharp_frames_train_and_never_score(self):
        frames = [i for i in self.meta['images'] if i['depthFrame']]
        self.assertEqual([f['name'] for f in frames], [f'f{i:05d}.jpg' for i in range(6)])
        self.assertFalse(any(f['heldOut'] for f in frames))
        self.assertEqual(frames[0]['size'], [64, 48])
        self.assertTrue((self.out / 'images/f00000.jpg').is_file())
        self.assertIn({'file': 'depth/00006.jpg', 'reason': self.meta['skipped'][-1]['reason']}, self.meta['skipped'])
        self.assertIn('depth frame', self.meta['skipped'][-1]['reason'])
        held = (self.out / 'test_images.txt').read_text().split()
        self.assertEqual(held, [i['name'] for i in self.meta['images'] if i['heldOut']])
        self.assertEqual(held, ['0000.jpg', '0008.jpg'])

    def test_depth_samples_lie_on_the_room_with_inward_normals(self):
        from plyio import read_vertices
        v = read_vertices(self.out / 'points3D.ply')
        xyz = np.stack([v['x'], v['y'], v['z']], 1)
        n = np.stack([v['nx'], v['ny'], v['nz']], 1)
        plain = build(self.bundle, Path(self.temp.name) / 'photos-only', binary=False, use_frames=False)
        self.assertGreater(len(xyz), plain['points'])  # depth adds points: the ceiling has no mesh at all
        ceiling = np.abs(xyz[:, 1] - CEILING) < 0.01
        self.assertGreater(ceiling.sum(), 20)
        self.assertTrue(np.all(n[ceiling, 1] < -0.9))  # facing down into the room, within 25° (float16 depth)
        # every point is on a surface of the box
        gap = np.min(np.stack([np.abs(xyz[:, 1]), np.abs(xyz[:, 1] - CEILING), np.abs(np.abs(xyz[:, 0]) - HALF[0]),
                               np.abs(np.abs(xyz[:, 2]) - HALF[1])]), 0)
        self.assertLess(gap.max(), 0.02)  # float16 depth: ~2 mm steps at 3 m, more at grazing angles
        self.assertFalse(any(i['depthFrame'] for i in plain['images']))
        data = (self.bundle / 'photos/0004.jpg').read_bytes()
        self.assertEqual(strip_exif(strip_exif(data)), strip_exif(data))


if __name__ == '__main__':
    unittest.main()
