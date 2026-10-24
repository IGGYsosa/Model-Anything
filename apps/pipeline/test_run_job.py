"""Job runner and API checks with a stand-in trainer (no GPU or gsplat needed).
Run: python -m unittest discover -s apps/pipeline -p 'test_*.py'
"""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import zipfile

import numpy as np
from PIL import Image

import run_job
from test_bundle_to_dataset import make_bundle

# Imitates gsplat's simple_trainer outputs: stats/val_stepN.json, ply/point_cloud_N.ply and
# side-by-side (ground truth | render) held-out PNGs, the second one deliberately poor.
FAKE_TRAINER = r'''
import json, sys, os
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, os.environ['PIPELINE_DIR'])
from make_test_splat import synthetic_room, write_splat
argv = sys.argv[1:]
if '--help' in argv:
    print('--save-ply, --no-save-ply  --disable-viewer, --no-disable-viewer  --antialiased, --no-antialiased '
          '--depth-loss, --no-depth-loss --use-bilateral-grid, --no-use-bilateral-grid --pose-opt, --no-pose-opt '
          '--normalize-world-space, --no-normalize-world-space')
    sys.exit(0)
arg = lambda name: argv[argv.index(name) + 1]
Path(os.environ['ARGS_OUT']).write_text(json.dumps(argv))
result, steps, data = Path(arg('--result_dir')), int(arg('--max_steps')), Path(arg('--data_dir'))
for d in ('stats', 'ply', 'renders'):
    (result / d).mkdir(parents=True, exist_ok=True)
(result / 'stats' / f'val_step{steps - 1}.json').write_text(json.dumps({'psnr': 27.5, 'ssim': 0.9, 'lpips': 0.1, 'num_GS': 1000}))
xyz, rgb, scale = synthetic_room(0.2)
write_splat(result / 'ply' / f'point_cloud_{steps - 1}.ply', xyz, rgb, scale)
rng = np.random.default_rng(1)
images = sorted((data / 'images').glob('*.jpg'))
for i, path in enumerate(images[::8]):
    gt = np.full((30, 40, 3), 120, np.uint8)  # the photos are 160×120, trained at 1/4
    render = gt.copy() if i == 0 else rng.integers(0, 255, gt.shape, dtype=np.uint8)
    Image.fromarray(np.concatenate([gt, render], 1)).save(result / 'renders' / f'val_step{steps - 1}_{i:04d}.png')
'''


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        make_bundle(base / 'export' / 'Patio')
        self.zip = base / 'patio.zip'
        with zipfile.ZipFile(self.zip, 'w') as z:
            for path in (base / 'export').rglob('*'):
                z.write(path, path.relative_to(base / 'export'))
        self.trainer = base / 'fake_trainer.py'
        self.trainer.write_text(FAKE_TRAINER)
        self.args_out = base / 'trainer-args.json'
        self.env = {'MA_TRAINER': f'"{sys.executable}" "{self.trainer}"', 'ARGS_OUT': str(self.args_out),
                    'PIPELINE_DIR': str(Path(run_job.__file__).parent)}
        self.saved = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        self.base = base

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.temp.cleanup()

    def test_trains_scores_and_packages(self):
        job = self.base / 'job'
        data = run_job.run(self.zip, job, steps=100)
        self.assertEqual(data['state'], 'done')
        self.assertEqual(data['metrics']['quality'], 'good')
        self.assertEqual(data['room'], 2)
        args = json.loads(self.args_out.read_text())
        self.assertEqual(args[0], 'mcmc')
        for flag in ('--save-ply', '--antialiased', '--depth-loss', '--use-bilateral-grid', '--no-normalize-world-space'):
            self.assertIn(flag, args)
        self.assertNotIn('--pose-opt', args)
        scene = json.loads((job / 'scene/scene.json').read_text())
        self.assertEqual(scene['format'], 'model-anything-scene/1')
        self.assertEqual(scene['name'], 'Patio')
        self.assertTrue((job / 'scene' / scene['splat']).is_file())
        self.assertEqual(len(scene['layout']['walls']), 4)
        self.assertEqual(len(scene['layout']['polygon']), 4)
        self.assertEqual(set(scene['camera']), {'orbit', 'target', 'walk', 'walkLook'})
        self.assertEqual(len(scene['worldToRoom']), 16)

        review = scene['review']
        self.assertEqual([v['psnr'] for v in review['views']][0], 99.0)
        self.assertLess(review['views'][1]['psnr'], 22)
        for view in review['views']:
            self.assertTrue((job / 'scene' / view['photo']).is_file())
            self.assertTrue((job / 'scene' / view['render']).is_file())
        # the noisy second held-out view shows up as inaccurate areas, linked to that comparison
        inaccurate = [a for a in review['areas'] if a['reason'] == 'inaccurate']
        self.assertTrue(inaccurate)
        self.assertTrue(all(a['view'] == 1 for a in inaccurate))
        for area in review['areas']:
            self.assertTrue(all(lo < hi for lo, hi in zip(area['min'], area['max'])))
            self.assertTrue(all(lo <= c <= hi for lo, c, hi in zip(area['min'], area['center'], area['max'])))
            self.assertAlmostEqual(area['viewpoint']['position'][1], 1.5)
        self.assertLessEqual(len(review['areas']), 12)
        self.assertEqual([a['id'] for a in review['areas']], list(range(1, len(review['areas']) + 1)))
        weak = (job / 'scene' / review['weakPoints']).read_bytes()
        self.assertEqual(len(weak), review['weakPointCount'] * 16)
        # the job carries the areas for the phone; scene.json carries their text
        self.assertEqual([h['id'] for h in data['hints']], [a['id'] for a in review['areas']])
        self.assertEqual(data['worldToRoom'], scene['worldToRoom'])
        self.assertEqual(data['hints'][0]['min'], review['areas'][0]['min'])
        self.assertEqual(scene['hints'], [a['text'] for a in review['areas']])

    def test_exterior_preset_refines_poses(self):
        run_job.run(self.zip, self.base / 'job', preset='exterior', steps=10)
        self.assertIn('--pose-opt', json.loads(self.args_out.read_text()))

    def test_value_style_bools(self):
        self.assertEqual(run_job.bool_arg('pose_opt', False, '--pose_opt BOOL'), ['--pose_opt', 'False'])
        self.assertEqual(run_job.bool_arg('pose_opt', True, '--pose-opt, --no-pose-opt'), ['--pose-opt'])
        self.assertEqual(run_job.bool_arg('packed', False, '--packed, --no_packed'), ['--no_packed'])

    def test_package_existing_splat(self):
        splat = self.base / 'given.ply'
        from make_test_splat import synthetic_room, write_splat
        write_splat(splat, *synthetic_room(0.25))
        data = run_job.run(self.zip, self.base / 'job', splat=splat)
        self.assertEqual(data['state'], 'done')
        self.assertFalse(self.args_out.exists())
        # without held-out renders, only photo coverage can be judged
        review = json.loads((self.base / 'job/scene/scene.json').read_text())['review']
        self.assertEqual(review['views'], [])
        self.assertTrue(review['areas'])
        self.assertEqual({a['reason'] for a in review['areas']}, {'few-photos'})

    def test_rejects_unsafe_archives_and_records_failure(self):
        bad = self.base / 'bad.zip'
        with zipfile.ZipFile(bad, 'w') as z:
            z.writestr('../escape.txt', 'x')
        with self.assertRaises(ValueError):
            run_job.run(bad, self.base / 'job')
        state = json.loads((self.base / 'job/job.json').read_text())
        self.assertEqual(state['state'], 'failed')
        self.assertIn('unsafe path', state['error'])
        self.assertFalse((self.base / 'escape.txt').exists())


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        make_bundle(base / 'export' / 'Patio')
        self.zip = base / 'patio.zip'
        with zipfile.ZipFile(self.zip, 'w') as z:
            for path in (base / 'export').rglob('*'):
                z.write(path, path.relative_to(base / 'export'))
        trainer = base / 'fake_trainer.py'
        trainer.write_text(FAKE_TRAINER)
        env = {'MA_TOKEN': 'test-token', 'MA_JOBS_DIR': str(base / 'jobs'), 'MA_VIEWER_DIR': str(base / 'no-viewer'),
               'MA_TRAINER': f'"{sys.executable}" "{trainer}"', 'ARGS_OUT': str(base / 'args.json'),
               'PIPELINE_DIR': str(Path(run_job.__file__).parent)}
        self.saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        import server
        self.server = importlib.reload(server)
        from fastapi.testclient import TestClient
        self.client = TestClient(self.server.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.temp.cleanup()

    def upload(self, token='test-token'):
        with open(self.zip, 'rb') as f:
            return self.client.post('/jobs', files={'bundle': ('patio.zip', f, 'application/zip')},
                                    data={'preset': 'interior'}, headers={'Authorization': f'Bearer {token}'})

    def test_upload_train_and_fetch_scene(self):
        self.assertEqual(self.upload('wrong').status_code, 401)
        created = self.upload()
        self.assertEqual(created.status_code, 200, created.text)
        job_id = created.json()['id']
        deadline = time.time() + 120
        while time.time() < deadline:
            state = self.client.get(f'/jobs/{job_id}').json()
            if state['state'] in ('done', 'failed'):
                break
            time.sleep(0.5)
        self.assertEqual(state['state'], 'done', state.get('error'))
        self.assertNotIn('trace', state)
        scene = self.client.get(f'/jobs/{job_id}/scene/scene.json')
        self.assertEqual(scene.status_code, 200)
        splat = self.client.get(f'/jobs/{job_id}/scene/{scene.json()["splat"]}')
        self.assertEqual(splat.status_code, 200)
        self.assertEqual(self.client.get(f'/jobs/{job_id}/scene/../job.json').status_code, 404)
        self.assertEqual(self.client.get('/jobs/' + '0' * 32).status_code, 404)
        self.assertEqual(self.client.get('/jobs').status_code, 401)
        listed = self.client.get('/jobs', headers={'Authorization': 'Bearer test-token'}).json()
        self.assertEqual(listed[0]['id'], job_id)
        self.assertEqual(self.client.get(f'/view/{job_id}', follow_redirects=False).status_code, 503)


if __name__ == '__main__':
    unittest.main()
