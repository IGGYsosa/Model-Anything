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
QUALITY = ((26.0, 'good'), (22.0, 'fair'))  # held-out PSNR (dB) floors; below the last is poor


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class Job:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'job.json'
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {'state': 'queued', 'createdAt': now()}

    def update(self, **fields):
        self.data.update(fields, updatedAt=now())
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)


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
    gsplat = Path(os.environ.get('GSPLAT_DIR', '/opt/gsplat'))
    return [sys.executable, str(gsplat / 'examples/simple_trainer.py')]


def bool_arg(name, value, help_text):
    """tyro renders bools as flags (--x / --no-x) or as --x {True,False}, with - or _; follow --help."""
    dashed = name.replace('_', '-')
    for yes, no in ((f'--{dashed}', f'--no-{dashed}'), (f'--{name}', f'--no-{name}'), (f'--{name}', f'--no_{name}')):
        if re.search(re.escape(no) + r'\b', help_text):
            return [yes if value else no]
    return [f'--{name}', str(value)]


def train_args(preset, dataset, result, steps, factor, help_text):
    p = PRESETS[preset]
    args = [p['strategy'], '--data_dir', str(dataset), '--result_dir', str(result),
            '--data_factor', str(factor), '--test_every', '8', '--max_steps', str(steps),
            '--eval_steps', str(steps), '--save_steps', str(steps), '--ply_steps', str(steps)]
    # room coordinates in, room coordinates out: the splat must line up with the layout
    flags = {'save_ply': True, 'disable_viewer': True, 'normalize_world_space': False, **p['flags']}
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
    # home view from just inside the far corner, above eye height: splats only look right from
    # where the photos were taken, so the dollhouse view starts inside rather than overhead
    corner = hi - np.minimum(0.45, (hi - lo) * 0.15)
    orbit = [round(float(corner[0]), 3), round(min(height - 0.25, 2.0), 3), round(float(corner[1]), 3)]
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
            job.update(step='train')
            if factor > 1:
                downscale(dataset, factor)
            result = job.root / 'train'
            command = trainer_command()
            help_text = subprocess.run(command + [PRESETS[preset]['strategy'], '--help'],
                                       capture_output=True, text=True).stdout
            log = open(job.root / 'train.log', 'w')
            with log:
                trained = subprocess.run(command + train_args(preset, dataset, result, steps, factor, help_text),
                                         stdout=log, stderr=subprocess.STDOUT)
            if trained.returncode:
                raise RuntimeError(f'training failed (exit {trained.returncode}); see train.log')
            stage = f'val_step{steps - 1}'
            stats_file = result / 'stats' / f'{stage}.json'
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
        job.update(state='done', step=None, metrics=metrics, hints=hints, scene='scene/scene.json', finishedAt=now())
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
