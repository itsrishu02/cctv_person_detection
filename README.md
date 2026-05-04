# ZEEX AI Face Recognition v0

This is the Vercel/v0-style version of `zeex-face-recognition`.

It keeps the YOLO/OpenCV recognition pipeline in a local FastAPI backend and
replaces the Streamlit UI with a Next.js frontend that can be edited in v0 or
deployed on Vercel.

## What changed

- Empty frames are now reported as `EMPTY_ZONE`.
- The annotated frame shows `EMPTY ZONE / NO PERSON` when no face and no person
  detection exists in the observed frame.
- Empty-zone detection has a tunable `Empty Zone Frames` threshold in the UI.
- Streamlit was removed from this version.
- The UI lives in `frontend/` as a Next.js app.
- The recognition loop lives in `backend/api.py` as HTTP endpoints.

## Project layout

```text
zeex-face-recognition-v0/
  backend/
    api.py
    face_pipeline.py
    recognizer.py
    encode_faces.py
    config.yaml
    encodings.pkl
    models/
    requirements.txt
  frontend/
    app/
    package.json
    .env.example
```

## Run locally

Open one terminal for the backend:

```powershell
cd C:\Users\RishuSingh\Downloads\Mins_project\zeex-face-recognition-v0\backend
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Open another terminal for the frontend:

```powershell
cd C:\Users\RishuSingh\Downloads\Mins_project\zeex-face-recognition-v0\frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

## Choosing a source

Changing `Camera ID` in the frontend automatically switches to the matching
RTSP source and starts recognition:

```text
CAM-01 -> rtsp://65.1.214.31:8554/cam1
CAM-02 -> rtsp://65.1.214.31:8554/cam2
CAM-10 -> rtsp://65.1.214.31:8554/cam10
```

The log line below means the backend is working, but the RTSP endpoint itself
is not serving a stream at that path:

```text
method DESCRIBE failed: 404 Not Found
```

Use the `Path` tab with the configured sample video, use `Upload`, or replace
the RTSP URL with one that VLC can open successfully. Common RTSP paths look
like `/Streaming/Channels/101`, `/cam/realmonitor?channel=1&subtype=0`, or a
camera-specific path from the device settings.

## Empty-zone threshold

`Empty Zone Frames` controls how many consecutive processed frames must contain
zero detected people before the UI/API reports `EMPTY_ZONE`. Increase it if
empty-zone status flickers during brief detector misses; decrease it if you want
faster empty-zone detection.

## Backend API

- `GET /api/health`
- `GET /api/defaults`
- `GET /api/workers`
- `POST /api/videos/upload`
- `POST /api/recognition/start`
- `POST /api/recognition/stop`
- `POST /api/recognition/settings`
- `GET /api/recognition/status`
- `GET /api/recognition/frame`

## Vercel note

Deploy `frontend/` to Vercel and set:

```text
NEXT_PUBLIC_API_BASE_URL=<your backend URL>
```

The Python backend should run on a machine/server that can access the RTSP
camera, local video files, model files, and OpenCV dependencies.
