"""End-to-end GPU check of the photoreal pipeline, no phone needed.

Renders the synthetic room (make_test_splat.py) with gsplat into the photos of a synthetic app
session, then runs the real job: dataset → gsplat training → review → scene package. Fails
unless training reproduces the held-out photos well and the LiDAR depth loss was active.

usage: <trainer python> apps/pipeline/smoke_test.py [-o data/jobs/smoke] [--steps 2000]
(the native install's Python: %LOCALAPPDATA%/ModelAnything/env/Scripts/python.exe)
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
from gsplat.rendering import rasterization

from bundle_to_dataset import ARKIT_TO_CV, build, mat
from make_test_splat import synthetic_room
import run_job
from test_bundle_to_dataset import make_bundle


def render_photos(bundle):
    """Replace the bundle's flat placeholder photos with renders of the synthetic room."""
    with tempfile.TemporaryDirectory() as temp:
        w2r = mat(build(bundle, Path(temp) / 'dataset')['worldToRoom'])  # same room frame as the job
    xyz, rgb, scale = synthetic_room()
    device = 'cuda'
    means = torch.tensor(xyz, device=device)
    quats = torch.zeros(len(xyz), 4, device=device)
    quats[:, 0] = 1
    scales = torch.full((len(xyz), 3), float(scale), device=device)
    opacities = torch.full((len(xyz),), 0.98, device=device)
    colors = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device)
    photos = json.loads((bundle / 'photos/photos.json').read_text())
    for p in photos:
        w2c = np.linalg.inv(w2r @ mat(p['transform']) @ ARKIT_TO_CV)
        k = p['intrinsics']
        K = torch.tensor([[k[0], 0, k[6]], [0, k[4], k[7]], [0, 0, 1]], dtype=torch.float32, device=device)
        image, alpha, _ = rasterization(means, quats, scales, opacities, colors,
                                        torch.tensor(w2c, dtype=torch.float32, device=device)[None], K[None],
                                        p['width'], p['height'])
        image = image[0] + (1 - alpha[0]) * 0.1  # dark gray where nothing is drawn
        pixels = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(pixels).save(bundle / p['file'], quality=95)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-o', '--out', default=str(Path(__file__).resolve().parents[2] / 'data/jobs/smoke'))
    ap.add_argument('--steps', type=int, default=2000)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit('smoke test needs a CUDA GPU')
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    bundle = out / 'export' / 'Patio'
    make_bundle(bundle, size=(320, 240), focal=200, count=41)
    render_photos(bundle)
    data = run_job.run(bundle, out / 'job', steps=a.steps)
    log = (out / 'job/train.log').read_text(errors='replace')
    depth = [float(v) for v in re.findall(r'depth loss=([0-9.]+)', log)]
    psnr = data['metrics'].get('psnr', 0)
    print(json.dumps({'state': data['state'], 'metrics': data['metrics'], 'cameraSpread': data.get('cameraSpread'),
                      'depthLossLast': depth[-1] if depth else None, 'weakAreas': len(data['hints']),
                      'scene': str(out / 'job/scene/scene.json')}, indent=1))
    problems = []
    if psnr < 25:
        problems.append(f'held-out PSNR {psnr:.1f} dB < 25')
    if not depth or max(depth) == 0:
        problems.append('depth loss never active')
    if problems:
        sys.exit('smoke test failed: ' + '; '.join(problems))
    print(f'smoke test passed. View: ?scene=/jobs/{out.name}/job/scene/scene.json')


if __name__ == '__main__':
    main()
