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


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.bundle = Path(self.temp.name) / 'session'
        (self.bundle / 'rooms').mkdir(parents=True)
        (self.bundle / 'photos').mkdir()
        (self.bundle / 'mesh').mkdir()
        hx, hz = HALF
        room = {'walls': [wall((0, -hz), (1, 0), 2 * hx), wall((hx, 0), (0, 1), 2 * hz),
                          wall((0, hz), (-1, 0), 2 * hx), wall((-hx, 0), (0, -1), 2 * hz)],
                'objects': [], 'doors': [], 'windows': [], 'openings': []}
        (self.bundle / 'rooms/room-2.json').write_text(json.dumps(room))
        (self.bundle / 'manifest.json').write_text(json.dumps({
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
        self.local_points = local[inroom > 0]
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
        (self.bundle / 'mesh/room-2.ply').write_bytes(header.encode() + vert.tobytes())

        # 12 photos from the middle of the room turning around; one blurry, one with lost tracking,
        # one from another room. Every image is the same color, unlike the PLY's red heatmap.
        exif = Image.Exif()
        exif[0x0112] = 6
        photos = []
        for i in range(13):
            file = f'photos/{i:04d}.jpg'
            Image.new('RGB', (160, 120), (10, 200, 30)).save(self.bundle / file, quality=95, exif=exif)
            photos.append({'file': file, 'room': 2 if i < 12 else 1, 'width': 160, 'height': 120,
                           'intrinsics': K, 'tracking': 'normal', 'blurPixels': 0.2, 'exposure': 0.01,
                           'camera': {'iso': 100 + i, 'shutter': 0.01},
                           'transform': cm(world(yaw(i * math.tau / 12, (0.2, 1.4, 0.1))))})
        photos[3]['blurPixels'] = 9.0
        photos[5]['tracking'] = 'limited:excessiveMotion'
        (self.bundle / 'photos/photos.json').write_text(json.dumps(photos))
        self.photos = photos
        self.out = Path(self.temp.name) / 'dataset'
        self.meta = build(self.bundle, self.out, text=True)

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
        data = (self.bundle / 'photos/0004.jpg').read_bytes()
        self.assertEqual(strip_exif(strip_exif(data)), strip_exif(data))


if __name__ == '__main__':
    unittest.main()
