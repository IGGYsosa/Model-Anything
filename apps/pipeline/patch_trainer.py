"""Two small fixes to gsplat's example trainer and its COLMAP loader. Safe to run again.

1. pycolmap (rmbrualla fork): reads/writes COLMAP .bin counts as a C `long` ('L'), 8 bytes on
   Linux but 4 on Windows, where every .bin file fails to load. COLMAP stores them as uint64.
2. examples/lib_bilagrid.py color_correct(): asserts when its least-squares color fit has no
   solution, which happens for a held-out photo that is nearly one color (a blank wall). That
   aborts the run at evaluation after training finished; leave such an image uncorrected.

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
