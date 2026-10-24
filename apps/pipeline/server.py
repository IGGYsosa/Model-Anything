"""Photoreal job API for the iPhone app, the web viewer, or an agent (e.g. a Hermes profile).

  POST /jobs                 multipart: bundle=<zip from the app>, room=<index>, preset=interior|exterior
                             → {"id": ...}; requires Authorization: Bearer $MA_TOKEN
  GET  /jobs                 recent jobs (token required)
  GET  /jobs/{id}            job.json: state, step, metrics, capture hints
  GET  /jobs/{id}/scene/...  the scene package (scene.json, scene.sog)
  GET  /view/{id}            redirects to the bundled web viewer showing that scene

Job ids are random 128-bit values; reading a job or its scene needs only the id, so the
viewer (and an iOS WKWebView) can load assets without the token. One worker trains at a time.

  MA_TOKEN=secret uvicorn server:app --host 0.0.0.0 --port 8787
Environment: MA_TOKEN (required unless MA_ALLOW_ANONYMOUS=1), MA_JOBS_DIR (default
data/jobs), MA_VIEWER_DIR (viewer build with VIEWER_BASE=/viewer/), MA_MAX_UPLOAD_MB (4096),
plus MA_TRAINER / GSPLAT_DIR for run_job.py.
"""
import json
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from run_job import PRESETS, Job, now, read_json

HERE = Path(__file__).resolve().parent
JOBS = Path(os.environ.get('MA_JOBS_DIR', HERE.parents[1] / 'data/jobs')).resolve()
VIEWER = Path(os.environ.get('MA_VIEWER_DIR', HERE.parents[1] / 'apps/room-viewer/dist'))
TOKEN = os.environ.get('MA_TOKEN', '')
ANONYMOUS = os.environ.get('MA_ALLOW_ANONYMOUS') == '1'
MAX_UPLOAD = int(os.environ.get('MA_MAX_UPLOAD_MB', '4096')) * 1024 * 1024
JOB_ID = re.compile(r'^[0-9a-f]{32}$')

if not TOKEN and not ANONYMOUS:
    raise RuntimeError('set MA_TOKEN (or MA_ALLOW_ANONYMOUS=1 for a private machine)')

app = FastAPI(title='3D Model Anything photoreal jobs')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET'], allow_headers=['*'])
work = queue.Queue()
if VIEWER.is_dir():
    app.mount('/viewer', StaticFiles(directory=VIEWER, html=True), name='viewer')


def authorized(authorization: str = Header(default='')):
    if ANONYMOUS:
        return
    if not secrets.compare_digest(authorization, f'Bearer {TOKEN}'):
        raise HTTPException(401, 'missing or wrong bearer token')


def job_dir(job_id):
    if not JOB_ID.match(job_id) or not (JOBS / job_id / 'job.json').exists():
        raise HTTPException(404, 'no such job')
    return JOBS / job_id


def worker():
    while True:
        job_id = work.get()
        root = JOBS / job_id
        job = Job(root)
        args = [sys.executable, str(HERE / 'run_job.py'), str(root / 'upload.zip'), '-o', str(root),
                '--preset', job.data.get('preset', 'interior')]
        if job.data.get('requestedRoom') is not None:
            args += ['--room', str(job.data['requestedRoom'])]
        with open(root / 'job.log', 'w') as log:
            subprocess.run(args, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        if Job(root).data.get('state') not in ('done', 'failed'):  # crashed before recording a result
            Job(root).update(state='failed', error='job runner exited unexpectedly; see job.log')
        work.task_done()


@app.on_event('startup')
def resume():
    JOBS.mkdir(parents=True, exist_ok=True)
    pending = []
    for path in JOBS.glob('*/job.json'):
        job = Job(path.parent)
        if job.data.get('state') == 'running':
            job.update(state='failed', error='server restarted during this job; upload again')
        elif job.data.get('state') == 'queued':
            pending.append((job.data.get('createdAt', ''), path.parent.name))
    for _, job_id in sorted(pending):
        work.put(job_id)
    threading.Thread(target=worker, daemon=True).start()


@app.post('/jobs', dependencies=[Depends(authorized)])
async def create(bundle: UploadFile = File(...), room: int | None = Form(None), preset: str = Form('interior'),
                 name: str | None = Form(None)):
    if preset not in PRESETS:
        raise HTTPException(422, f'preset must be one of {sorted(PRESETS)}')
    job_id = secrets.token_hex(16)
    root = JOBS / job_id
    root.mkdir(parents=True)
    size = 0
    try:
        with open(root / 'upload.zip', 'wb') as out:
            while chunk := await bundle.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, 'upload too large')
                out.write(chunk)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    Job(root).update(state='queued', createdAt=now(), preset=preset, requestedRoom=room,
                     name=name, uploadBytes=size, position=work.qsize() + 1)
    work.put(job_id)
    return {'id': job_id, 'state': 'queued', 'status': f'/jobs/{job_id}', 'view': f'/view/{job_id}'}


@app.get('/jobs', dependencies=[Depends(authorized)])
def recent(limit: int = 50):
    jobs = []
    for path in JOBS.glob('*/job.json'):
        data = read_json(path)
        jobs.append({'id': path.parent.name, **{k: data.get(k) for k in ('state', 'step', 'name', 'roomName', 'createdAt')}})
    return sorted(jobs, key=lambda j: j['createdAt'] or '', reverse=True)[:limit]


@app.get('/jobs/{job_id}')
def status(job_id: str):
    data = read_json(job_dir(job_id) / 'job.json')
    data.pop('trace', None)
    return {'id': job_id, **data}


@app.get('/jobs/{job_id}/scene/{path:path}')
def scene_file(job_id: str, path: str):
    scene = (job_dir(job_id) / 'scene').resolve()
    target = (scene / path).resolve()
    if scene not in target.parents or not target.is_file():
        raise HTTPException(404, 'no such file')
    return FileResponse(target, headers={'Cache-Control': 'public, max-age=31536000, immutable'})


@app.get('/view/{job_id}')
def view(job_id: str):
    job_dir(job_id)
    if not VIEWER.is_dir():
        raise HTTPException(503, 'viewer build not installed on this server')
    return RedirectResponse(f'/viewer/?scene=/jobs/{job_id}/scene/scene.json')
