# Photoreal pipeline

Turns a room scanned with the iPhone app into a photoreal Gaussian-splat scene that the web viewer (and the app, through a web view) can show. No hand modeling and no agent loop: the app's own posed photos are the training views and its LiDAR seeds and supervises the geometry.

```text
app export (zip) → bundle_to_dataset.py → gsplat training → QC → scene/scene.json + scene.sog
```

- **Dataset** (`bundle_to_dataset.py`): one room → COLMAP model in room coordinates (floor y=0, walls on X/Z, the frame `bundle_to_room.py` and the viewer use). Cameras come from ARKit, so there is no structure-from-motion step. LiDAR mesh vertices become seed points, colored from the photo that sees them best, with per-photo visibility tracks for depth supervision. Blurry and lost-tracking photos are skipped; every 8th photo is held out for scoring.
- **Training** (`run_job.py`): gsplat 1.5.3 `simple_trainer.py` with MCMC densification, anti-aliasing, a bilateral grid for the phone's per-photo auto exposure, and LiDAR depth loss. `--preset exterior` adds pose refinement for drift over larger areas. World normalization is off so the splat stays aligned with the RoomPlan layout.
- **QC** (`review.py`): held-out PSNR/SSIM/LPIPS from gsplat and a quality label (`good` / `fair` / `poor`). Every LiDAR seed point is also scored for photo coverage and for how far held-out renders differ from the real photos there. Weak points are grouped per surface into walkable areas, each with:
  - a box
  - the side it faces
  - where to stand for new photos
  - a real-vs-render comparison

  The viewer's Rescan panel shows these areas. The iPhone app's **Rescan weak spots** marks them red on the live mesh until they are photographed again.
- **Scene package**: `scene.json` (`model-anything-scene/1`) holds the splat file name, RoomPlan layout, camera presets and metrics. The splat is compressed to `.sog` with splat-transform when it is installed, otherwise shipped as `.ply`.

## Run a job

On a machine with gsplat installed (`GSPLAT_DIR` pointing at its checkout, or `MA_TRAINER` set to the trainer command):

```sh
python apps/pipeline/run_job.py patio.zip -o data/jobs/patio --preset interior
```

Then, with `npm run dev`, open `http://127.0.0.1:5188/?scene=/jobs/patio/scene/scene.json`. The dev server serves `data/jobs` and `data/scenes` at the job server's paths.

Without a GPU, `--splat file.ply` packages an existing splat instead of training. `make_test_splat.py` writes one (a synthetic closed room, or one splat per seed point with `--from-points dataset/points3D.ply`).

`--data-factor 2` trains on half-size photos if the GPU runs out of memory.

## Job server

`server.py` queues jobs and runs them one at a time.

| Request | Purpose |
| --- | --- |
| `POST /jobs` (bearer token; multipart `bundle`, optional `room`, `preset`, `name`) | Upload an app export, returns `{id}` |
| `GET /jobs/{id}` | State, step, metrics, capture hints |
| `GET /jobs/{id}/scene/...` | Scene package files |
| `GET /view/{id}` | The web viewer showing that scene |
| `GET /jobs` (bearer token) | Recent jobs |

Job ids are random 128-bit values. Reading a job needs only its id, so the viewer and the app's web view load scenes without the token. The iPhone app's **Make photoreal** uses this API, and so can an agent (a Hermes profile, for example) that queues jobs and reads the hints.

### Docker (GPU)

Docker Desktop with WSL2 and an NVIDIA driver, or any Linux host with the NVIDIA container toolkit:

```sh
VIEWER_BASE=/viewer/ npm run build
MA_TOKEN=<secret> docker compose -f apps/pipeline/compose.yaml up --build
```

In Git Bash on Windows, prefix the build with `MSYS_NO_PATHCONV=1`, or Git Bash rewrites `/viewer/` into a Windows path and the viewer's assets 404.

The image (`Dockerfile`) pins gsplat 1.5.3 (built for Ampere, Ada and Hopper GPUs) and splat-transform 3.10.1. Jobs are written to `data/jobs/`, which Git ignores. The same image runs on a rented cloud GPU.

The first image build and first real training run have not been verified yet; `run_job.py` reads the trainer's `--help` to spell its boolean flags.

## Tests

```sh
python -m unittest discover -s apps/pipeline -p 'test_*.py'
```

These cover the converter (projection against the app's camera convention, room coordinates, filtering, tracks, colors, EXIF), the job runner and the API, using a stand-in trainer. No GPU is needed. On Windows, set `PYTHONUTF8=1` so the existing scripts can print non-ASCII output.
