#!/usr/bin/env python3
"""Run one photoreal job: phone export → dataset → gsplat training → QC → scene package.

usage: run_job.py <bundle.zip | session-dir> -o <job-dir> [--room N] [--preset interior|exterior]
                  [--steps 30000] [--data-factor 1] [--splat existing.ply]

The job directory gets:
  job.json       state (queued/running/done/failed), current step, metrics, hints, error
  dataset/       bundle_to_dataset.py output
  train/         gsplat result_dir (stats/, renders/, ply/)
  scene/         what the viewer loads: scene.json + scene.sog (or scene.ply)

--splat skips training and packages a splat you already have (a LiDAR preview from
make_test_splat.py, or a splat trained elsewhere). The trainer command comes from
$MA_TRAINER (default: python $GSPLAT_DIR/examples/simple_trainer.py), so the same
script runs in the Docker image and against a local gsplat checkout.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
from PIL import Image

from bundle_to_dataset import build
from review import review

# gsplat simple_trainer options per preset (strategy subcommand first). MCMC densification,
# anti-aliasing, a bilateral grid for the phone's per-photo auto exposure, and depth loss
# against the LiDAR seed tracks. Outdoors ARKit drifts over distance, so poses are refined too.
PRESETS = {
    'interior': {'strategy': 'mcmc', 'flags': {'antialiased': True, 'depth_loss': True,
                                               'use_bilateral_grid': True}},
    'exterior': {'strategy': 'mcmc', 'flags': {'antialiased': True, 'depth_loss': True,
                                               'use_bilateral_grid': True, 'pose_opt': True}},
}
QUALITY = ((26.0, 'good'), (22.0, 'fair'))
# gsplat sizes its learning rates and depth loss from how far apart the photos were taken.
MIN_SPREAD = 0.05     # m: photos all from one spot have no parallax and can't be trained
TARGET_SPREAD = 0.5   # m: below this, scale gsplat's scene size up to a room-sized value
# native (no Docker) gsplat install made by setup_native.ps1: env/ (Python 3.10) and gsplat/ (examples)
NATIVE = Path(os.environ.get('MA_NATIVE_DIR') or Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local/share')) / 'ModelAnything')  # held-out PSNR (dB) floors; below the last is poor


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def retry(action, attempts=40):
    """Windows can't replace a file another process has open (the server reading job.json while
    the job writes it, or the reverse); such conflicts last milliseconds."""
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


def read_json(path):
    return retry(lambda: json.loads(Path(path).read_text()))


class Job:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'job.json'
        self.data = read_json(self.path) if self.path.exists() else {'state': 'queued', 'createdAt': now()}

    def update(self, **fields):
        self.data.update(fields, updatedAt=now())
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=1))
        retry(lambda: os.replace(tmp, self.path))


def unpack(source, dest):
    """Session folder from a zip (safely extracted) or a directory: the folder with manifest.json."""
    source = Path(source)
    if source.is_file():
        dest = dest.resolve()
        with zipfile.ZipFile(source) as z:
            for info in z.infolist():
                target = (dest / info.filename).resolve()
                if dest not in target.parents and target != dest:
                    raise ValueError(f'unsafe path in archive: {info.filename}')
            z.extractall(dest)
        source = dest
    found = sorted(source.glob('manifest.json')) + sorted(source.glob('*/manifest.json')) + sorted(source.glob('*/*/manifest.json'))
    if not found:
        raise ValueError('no session bundle (manifest.json) in the upload')
    if len(found) > 1:
        raise ValueError('upload contains several sessions; share one room or session')
    return found[0].parent


def trainer_command():
    custom = os.environ.get('MA_TRAINER')
    if custom:
        if os.name != 'nt':
            return shlex.split(custom)
        return [part.strip('"') for part in shlex.split(custom, posix=False)]  # keep Windows backslashes
    if os.environ.get('GSPLAT_DIR'):  # the Docker image, or any checkout run with this Python
        return [sys.executable, str(Path(os.environ['GSPLAT_DIR']) / 'examples/simple_trainer.py')]
    python = NATIVE / 'env' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if python.exists() and (NATIVE / 'gsplat').is_dir():  # setup_native.ps1
        return [str(python), str(NATIVE / 'gsplat/examples/simple_trainer.py')]
    return [sys.executable, '/opt/gsplat/examples/simple_trainer.py']


def trainer_env(command):
    """The trainer's environment: UTF-8 output, and the pure-PyTorch fused_ssim stand-in
    (shims/) when the CUDA package isn't installed, as on native Windows."""
    env = {**os.environ, 'PYTHONUTF8': '1'}
    if subprocess.run([command[0], '-c', 'import fused_ssim'], capture_output=True).returncode:
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(Path(__file__).parent / 'shims'), env.get('PYTHONPATH')]))
    return env


def bool_arg(name, value, help_text):
    """tyro renders bools as flags (--x / --no-x) or as --x {True,False}, with - or _; follow --help."""
    dashed = name.replace('_', '-')
    for yes, no in ((f'--{dashed}', f'--no-{dashed}'), (f'--{name}', f'--no-{name}'), (f'--{name}', f'--no_{name}')):
        if re.search(re.escape(no) + r'\b', help_text):
            return [yes if value else no]
    return [f'--{name}', str(value)]


def option(name, help_text):
    """--data-dir or --data_dir, whichever this trainer's tyro version prints"""
    dashed = f'--{name.replace("_", "-")}'
    return dashed if dashed in help_text else f'--{name}'


def camera_spread(metadata):
    """Largest distance (m) of a photo from the photos' mean position: gsplat's scene scale."""
    eyes = np.array([np.linalg.inv(np.array(i['w2c'], float).reshape(4, 4))[:3, 3] for i in metadata['images']])
    return float(np.linalg.norm(eyes - eyes.mean(0), axis=1).max())


def train_args(preset, dataset, result, steps, factor, help_text, global_scale=1.0):
    p = PRESETS[preset]
    values = {'data_dir': dataset, 'result_dir': result, 'data_factor': factor, 'test_every': 8, 'max_steps': steps,
              'eval_steps': steps, 'save_steps': steps, 'ply_steps': steps}
    if global_scale != 1.0:
        values['global_scale'] = round(global_scale, 4)
    args = [p['strategy']]
    for name, value in values.items():
        args += [option(name, help_text), str(value)]
    # room coordinates in, room coordinates out: the splat must line up with the layout
    # no live viewer and no fly-through video (it fails on short camera paths and isn't used)
    flags = {'save_ply': True, 'disable_viewer': True, 'disable_video': True, 'normalize_world_space': False, **p['flags']}
    for name, value in flags.items():
        args += bool_arg(name, value, help_text)
    return args


def downscale(dataset, factor):
    """gsplat reads images_<factor>/ (intrinsics are scaled by the trainer)"""
    out = dataset / f'images_{factor}'
    out.mkdir(exist_ok=True)
    for src in sorted((dataset / 'images').glob('*.jpg')):
        im = Image.open(src)
        im.resize((im.width // factor, im.height // factor), Image.LANCZOS).save(out / src.name, quality=95)


def room_layout(layout):
    """bundle_to_room.py output → the viewer's RoomLayout"""
    walls = [{'a': w['a'], 'b': w['b'], 'u': w['u'], 'length': w['length'],
              'openings': [{'kind': o['kind'], 'u0': o['u0'], 'u1': o['u1']} for o in w['openings']]}
             for w in layout['walls']]
    polygon = layout.get('polygon')
    if not polygon:
        xs = [p[0] for w in walls for p in (w['a'], w['b'])]
        zs = [p[1] for w in walls for p in (w['a'], w['b'])]
        polygon = [[min(xs), max(zs)], [max(xs), max(zs)], [max(xs), min(zs)], [min(xs), min(zs)]]
    objects = [{k: o[k] for k in ('category', 'center', 'size', 'rotY')} for o in layout['objects']]
    return {'polygon': polygon, 'walls': walls, 'objects': objects}


def cameras(layout):
    poly = np.array(room_layout(layout)['polygon'], float)
    lo, hi = poly.min(0), poly.max(0)
    center = (lo + hi) / 2
    size = float(np.max(hi - lo))
    height = max([w['height'] for w in layout['walls']] or [2.5])
    target = [round(float(center[0]), 3), round(height * 0.4, 3), round(float(center[1]), 3)]
    # Splats look right only near where photos were taken, so the home view stands at the photo
    # farthest from the middle (the widest captured view); without photos, inside the far corner.
    corner = hi - np.minimum(0.45, (hi - lo) * 0.15)
    orbit = [round(float(corner[0]), 3), round(min(height - 0.25, 2.0), 3), round(float(corner[1]), 3)]
    if layout['photos']:
        far = max(layout['photos'], key=lambda p: (p['position'][0] - center[0]) ** 2 + (p['position'][2] - center[1]) ** 2)
        orbit = [round(float(v), 3) for v in far['position']]
    photo = layout['photos'][0] if layout['photos'] else None
    walk = [target[0], 1.6, target[2]] if not photo else [photo['position'][0], 1.6, photo['position'][2]]
    look = [walk[0] + 1, 1.5, walk[2]] if not photo else \
        [walk[0] + photo['forward'][0], 1.5, walk[2] + photo['forward'][2]]
    return {'orbit': orbit, 'target': target, 'walk': walk, 'walkLook': [round(v, 3) for v in look]}


def compress(ply, scene):
    """scene.sog via splat-transform when available, otherwise the PLY itself (Spark reads both)."""
    tool = shutil.which('splat-transform')
    if tool:
        out = scene / 'scene.sog'
        run = subprocess.run([tool, str(ply), str(out)], capture_output=True, text=True)
        if run.returncode == 0 and out.exists():
            return out.name
        print('splat-transform failed, shipping PLY:', run.stderr[-2000:], file=sys.stderr)
    shutil.copyfile(ply, scene / 'scene.ply')
    return 'scene.ply'


def run(source, job_dir, room=None, preset='interior', steps=30_000, factor=1, splat=None):
    job = Job(job_dir)
    job.update(state='running', step='unpack', preset=preset, error=None)
    try:
        upload = job.root / 'upload'
        if upload.exists():
            shutil.rmtree(upload)
        bundle = unpack(source, upload)

        job.update(step='dataset')
        dataset = job.root / 'dataset'
        if dataset.exists():
            shutil.rmtree(dataset)
        metadata = build(bundle, dataset, room)
        job.update(room=metadata['room'], roomName=metadata['roomName'], photos=len(metadata['images']),
                   skipped=len(metadata['skipped']), seedPoints=metadata['points'])

        metrics, renders, stage = {}, None, None
        if splat:
            ply = Path(splat)
        else:
            spread = camera_spread(metadata)
            job.update(step='train', cameraSpread=round(spread, 3))
            if spread < MIN_SPREAD:
                raise ValueError('The photos were all taken from about one spot. Walk around the room while '
                                 'scanning so the photos see it from different positions.')
            if factor > 1:
                downscale(dataset, factor)
            result = job.root / 'train'
            command = trainer_command()
            env = trainer_env(command)
            help_text = subprocess.run(command + [PRESETS[preset]['strategy'], '--help'],
                                       capture_output=True, text=True, env=env).stdout
            log = open(job.root / 'train.log', 'w')
            with log:
                trained = subprocess.run(command + train_args(preset, dataset, result, steps, factor, help_text,
                                                              max(1.0, TARGET_SPREAD / spread)),
                                         stdout=log, stderr=subprocess.STDOUT, env=env)
            if trained.returncode:
                raise RuntimeError(f'training failed (exit {trained.returncode}); see train.log')
            stage = f'val_step{steps - 1}'  # renders: val_step299_0000.png; stats: val_step0299.json
            stats_file = result / 'stats' / f'val_step{steps - 1:04d}.json'
            stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
            metrics = {k: stats[k] for k in ('psnr', 'ssim', 'lpips', 'cc_psnr', 'cc_ssim', 'cc_lpips', 'num_GS') if k in stats}
            plys = sorted((result / 'ply').glob('point_cloud_*.ply'), key=lambda p: int(p.stem.split('_')[-1]))
            if not plys:
                raise RuntimeError('training produced no PLY')
            ply = plys[-1]

            renders = result / 'renders'
            psnr = metrics.get('cc_psnr', metrics.get('psnr'))
            if psnr is not None:
                metrics['quality'] = next((label for floor, label in QUALITY if psnr >= floor), 'poor')

        job.update(step='qc')
        scene = job.root / 'scene'
        if scene.exists():
            shutil.rmtree(scene)
        scene.mkdir()
        # weak areas: inaccurate in held-out renders (when trained) or seen by too few photos
        findings = review(dataset, scene, renders, stage, metadata['layout'])
        hints = [{k: a[k] for k in ('id', 'text', 'reason', 'side', 'center', 'min', 'max')} for a in findings['areas']]

        job.update(step='package')
        package = {
            'format': 'model-anything-scene/1', 'name': metadata['roomName'] or f'Room {metadata["room"]}',
            'splat': compress(ply, scene), 'createdAt': now(), 'preset': preset,
            'layout': room_layout(metadata['layout']), 'camera': cameras(metadata['layout']),
            'height': max([w['height'] for w in metadata['layout']['walls']] or [2.5]),
            'metrics': metrics, 'hints': [h['text'] for h in hints], 'review': findings,
            # room → the capture session's ARKit world (column-major), for rescan targets on the phone
            'worldToRoom': metadata['worldToRoom'],
        }
        (scene / 'scene.json').write_text(json.dumps(package, indent=1))
        job.update(state='done', step=None, metrics=metrics, hints=hints, worldToRoom=metadata['worldToRoom'],
                   scene='scene/scene.json', finishedAt=now())
        return job.data
    except Exception as error:
        job.update(state='failed', error=f'{type(error).__name__}: {error}', trace=traceback.format_exc()[-4000:])
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('source', help='zip from the app, or an unpacked session folder')
    ap.add_argument('-o', '--out', required=True, help='job directory')
    ap.add_argument('--room', type=int)
    ap.add_argument('--preset', choices=sorted(PRESETS), default='interior')
    ap.add_argument('--steps', type=int, default=30_000)
    ap.add_argument('--data-factor', type=int, default=1, help='train on images downscaled by this factor')
    ap.add_argument('--splat', help='package this splat instead of training')
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    try:
        data = run(a.source, a.out, a.room, a.preset, a.steps, a.data_factor, a.splat)
    except Exception as error:
        sys.exit(f'job failed: {error}')
    print(json.dumps({k: data.get(k) for k in ('state', 'room', 'photos', 'metrics', 'hints', 'scene')}, indent=1))


if __name__ == '__main__':
    main()
