"""Two small fixes to gsplat's example trainer and its COLMAP loader. Safe to run again.

1. pycolmap (rmbrualla fork): reads/writes COLMAP .bin counts as a C `long` ('L'), 8 bytes on
   Linux but 4 on Windows, where every .bin file fails to load. COLMAP stores them as uint64.
2. examples/lib_bilagrid.py color_correct(): asserts when its least-squares color fit has no
   solution, which happens for a held-out photo that is nearly one color (a blank wall). That
   aborts the run at evaluation after training finished; leave such an image uncorrected.
3. examples/simple_trainer.py depth loss: torch.where(depths > 0, 1 / depths, 0) still evaluates
   1/0 where a LiDAR point has nothing rendered yet; the forward pass hides it, but the backward
   pass turns 0 * inf into NaN and the run diverges within ~100 steps (seen on a real room scan).
   Clamp before dividing.

usage: python patch_trainer.py <gsplat>/examples   (with the trainer's Python)
setup_native.ps1 and the Dockerfile run it.
"""
from pathlib import Path
import sys

import pycolmap.scene_manager


def patch(path, replacements):
    source = path.read_text()
    patched = source
    for old, new in replacements:
        patched = patched.replace(old, new)
    if patched != source:
        path.write_text(patched)
        print(f'patched {path}')
    else:
        print(f'already patched: {path}')


patch(Path(pycolmap.scene_manager.__file__), [
    ("struct.unpack('L'", "struct.unpack('<Q'"), ("struct.pack('L'", "struct.pack('<Q'"), ("'IiLL'", "'<IiQQ'")])
patch(Path(sys.argv[1]) / 'lib_bilagrid.py', [
    ('            assert torch.all(torch.isfinite(w))\n',
     '            if not torch.all(torch.isfinite(w)):\n'
     '                return img  # no color fit (e.g. a nearly uniform photo): leave it uncorrected\n')])
patch(Path(sys.argv[1]) / 'simple_trainer.py', [
    ('disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))',
     'disp = torch.where(depths > 0.0, 1.0 / depths.clamp_min(1e-6), torch.zeros_like(depths))')])
