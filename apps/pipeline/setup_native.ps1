# Install gsplat for photoreal training on Windows without Docker or a CUDA toolkit.
# Needs an NVIDIA GPU with a current driver, git, and uv (https://docs.astral.sh/uv/).
# Uses gsplat's prebuilt Windows wheel (Python 3.10, PyTorch 2.4, CUDA 12.4); about 5 GB.
#   powershell -ExecutionPolicy Bypass -File apps/pipeline/setup_native.ps1
# run_job.py finds this install automatically (override the location with MA_NATIVE_DIR).
$ErrorActionPreference = 'Stop'
$root = if ($env:MA_NATIVE_DIR) { $env:MA_NATIVE_DIR } else { Join-Path $env:LOCALAPPDATA 'ModelAnything' }
New-Item -ItemType Directory -Force $root | Out-Null
Set-Location $root

if (-not (Test-Path 'env')) { uv venv --python 3.10 env }
if (-not (Test-Path 'gsplat')) { git clone --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git gsplat }
$python = Join-Path $root 'env\Scripts\python.exe'

uv pip install --python $python torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python $python 'gsplat==1.5.3+pt24cu124' --index-url https://docs.gsplat.studio/whl/pt24cu124 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
# The examples' CUDA extras (fused-ssim, fused-bilagrid) have no Windows builds: the trainer uses
# its pure-PyTorch bilateral grid, and run_job.py supplies a pure-PyTorch fused_ssim (shims/).
Get-Content 'gsplat\examples\requirements.txt' |
  Where-Object { $_ -and $_ -notmatch '^#' -and $_ -notmatch 'fused-ssim|fused-bilagrid' } |
  Set-Content -Encoding utf8 'reqs-native.txt'
uv pip install --python $python -r reqs-native.txt
# COLMAP .bin loading on Windows, and evaluation of near-uniform photos (see patch_trainer.py)
& $python (Join-Path $PSScriptRoot 'patch_trainer.py') (Join-Path $root 'gsplat\examples')

& $python -c "import torch, gsplat; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('gsplat', gsplat.__version__)"
