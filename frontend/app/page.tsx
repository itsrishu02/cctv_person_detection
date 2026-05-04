"use client";

import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Camera,
  FolderOpen,
  Play,
  Radio,
  Square,
  Upload,
  Video
} from "lucide-react";

type SourceMode = "link" | "upload" | "path";

type Worker = {
  worker_id: string;
  name: string;
  role: string;
  zone: string;
  shift: string;
  encodings: number;
};

type Stats = {
  fps: number;
  detections: number;
  known_count: number;
  unknown_count: number;
  person_no_face_count: number;
  total_people_count: number;
  no_person_frame_count: number;
  empty_zone_threshold: number;
  empty_zone: boolean;
  zone_status: string;
};

type StatusResponse = {
  running: boolean;
  message: string;
  stats: Stats;
};

type DefaultsResponse = {
  streamUrl: string;
  fallbackVideo: string;
  fallbackVideoExists: boolean;
  cameraId: string;
  zone: string;
  threshold: number;
  detectEvery: number;
  faceImgSize: number;
  faceConf: number;
  minFace: number;
  emptyZoneFrames: number;
  personEnabled: boolean;
  personConf: number;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";
const CAMERA_RTSP_BASE = "rtsp://65.1.214.31:8554";

const EMPTY_STATS: Stats = {
  fps: 0,
  detections: 0,
  known_count: 0,
  unknown_count: 0,
  person_no_face_count: 0,
  total_people_count: 0,
  no_person_frame_count: 0,
  empty_zone_threshold: 3,
  empty_zone: false,
  zone_status: "IDLE"
};

export default function Page() {
  const [mode, setMode] = useState<SourceMode>("path");
  const [streamUrl, setStreamUrl] = useState("");
  const [localPath, setLocalPath] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [cameraId, setCameraId] = useState("CAM-01");
  const [zone, setZone] = useState("Zone-A");
  const [threshold, setThreshold] = useState(0.3);
  const [detectEvery, setDetectEvery] = useState(2);
  const [faceImgSize, setFaceImgSize] = useState(1280);
  const [faceConf, setFaceConf] = useState(0.25);
  const [minFace, setMinFace] = useState(32);
  const [emptyZoneFrames, setEmptyZoneFrames] = useState(3);
  const [personEnabled, setPersonEnabled] = useState(true);
  const [personConf, setPersonConf] = useState(0.5);
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<StatusResponse>({
    running: false,
    message: "Idle",
    stats: EMPTY_STATS
  });
  const [frameUrl, setFrameUrl] = useState("");
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [backendError, setBackendError] = useState("");
  const [actionError, setActionError] = useState("");
  const skipNextCameraStart = useRef(true);
  const statusFailureCount = useRef(0);
  const runningRef = useRef(false);

  const zoneStatus = status.stats?.zone_status || "EMPTY_ZONE";
  const zoneIsEmpty = status.stats?.empty_zone ?? true;

  const statusClass = useMemo(() => {
    if (backendError) return "offline";
    if (zoneStatus === "CHECKING_EMPTY_ZONE") return "checking";
    return zoneIsEmpty ? "empty" : "occupied";
  }, [backendError, zoneIsEmpty, zoneStatus]);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/recognition/status`, { cache: "no-store" });
      if (!res.ok) throw new Error("Status request failed");
      const nextStatus = (await res.json()) as StatusResponse;
      statusFailureCount.current = 0;
      setBackendError("");
      setStatus(nextStatus);
      setRunning(nextStatus.running);
      runningRef.current = nextStatus.running;
    } catch (err) {
      statusFailureCount.current += 1;
      if (statusFailureCount.current >= 3) {
        setBackendError("Backend is not responding");
      }
      throw err;
    }
  }, []);

  useEffect(() => {
    let ignore = false;

    async function loadWorkers() {
      try {
        const res = await fetch(`${API_BASE}/api/workers`, { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as { workers: Worker[] };
        if (!ignore) setWorkers(data.workers);
      } catch {
        if (!ignore) setWorkers([]);
      }
    }

    async function loadDefaults() {
      try {
        const res = await fetch(`${API_BASE}/api/defaults`, { cache: "no-store" });
        if (!res.ok) return;
        const defaults = (await res.json()) as DefaultsResponse;
        if (ignore) return;
        const defaultCameraRtsp = rtspForCameraId(defaults.cameraId);
        setStreamUrl(defaultCameraRtsp || defaults.streamUrl);
        setLocalPath(defaults.fallbackVideo);
        if (defaults.cameraId !== cameraId) {
          skipNextCameraStart.current = true;
        }
        setCameraId(defaults.cameraId);
        setZone(defaults.zone);
        setThreshold(defaults.threshold);
        setDetectEvery(defaults.detectEvery);
        setFaceImgSize(defaults.faceImgSize);
        setFaceConf(defaults.faceConf);
        setMinFace(defaults.minFace);
        setEmptyZoneFrames(defaults.emptyZoneFrames);
        setPersonEnabled(defaults.personEnabled);
        setPersonConf(defaults.personConf);
        setMode("link");
      } catch {
        if (!ignore) setMode("link");
      }
    }

    loadDefaults();
    loadWorkers();
    fetchStatus().catch(() => {});
    return () => {
      ignore = true;
    };
  }, [fetchStatus]);

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      fetchStatus().catch(() => {});
      setFrameUrl(`${API_BASE}/api/recognition/frame?t=${Date.now()}`);
    }, 350);
    return () => window.clearInterval(timer);
  }, [fetchStatus, running]);

  useEffect(() => {
    if (!running) return;
    const timer = window.setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/recognition/settings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ emptyZoneFrames })
        });
        if (!res.ok) throw new Error(await readApiError(res));
        setStatus((current) => ({
          ...current,
          stats: {
            ...current.stats,
            empty_zone_threshold: emptyZoneFrames
          }
        }));
      } catch (err) {
        setActionError(err instanceof Error ? err.message : "Settings update failed");
      }
    }, 200);
    return () => window.clearTimeout(timer);
  }, [emptyZoneFrames, running]);

  useEffect(() => {
    const nextRtsp = rtspForCameraId(cameraId);
    if (!nextRtsp) return;

    setStreamUrl(nextRtsp);
    setMode("link");

    if (skipNextCameraStart.current) {
      skipNextCameraStart.current = false;
      return;
    }

    if (!runningRef.current) return;

    const timer = window.setTimeout(() => {
      start(nextRtsp, cameraId.trim().toUpperCase());
    }, 600);

    return () => window.clearTimeout(timer);
  }, [cameraId]);

  async function resolveSource() {
    if (mode === "link") return streamUrl.trim();
    if (mode === "path") return localPath.trim();

    if (!uploadFile) {
      throw new Error("Select a video file");
    }

    const body = new FormData();
    body.append("file", uploadFile);
    const uploadRes = await fetch(`${API_BASE}/api/videos/upload`, {
      method: "POST",
      body
    });
    if (!uploadRes.ok) {
      throw new Error(await uploadRes.text());
    }
    const uploaded = (await uploadRes.json()) as { source: string };
    return uploaded.source;
  }

  async function start(sourceOverride?: string, cameraIdOverride?: string) {
    setBusy(true);
    setActionError("");
    setBackendError("");
    try {
      const source = sourceOverride || (await resolveSource());
      if (!source) throw new Error("Source is required");
      const res = await fetch(`${API_BASE}/api/recognition/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source,
          cameraId: cameraIdOverride || cameraId,
          zone,
          threshold,
          detectEvery,
          faceImgSize,
          faceConf,
          minFace,
          emptyZoneFrames,
          personEnabled,
          personConf
        })
      });
      if (!res.ok) {
        throw new Error(await readApiError(res));
      }
      setRunning(true);
      setFrameUrl(`${API_BASE}/api/recognition/frame?t=${Date.now()}`);
      await fetchStatus();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Start failed");
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    setBusy(true);
    setActionError("");
    setBackendError("");
    try {
      await fetch(`${API_BASE}/api/recognition/stop`, { method: "POST" });
      setRunning(false);
      await fetchStatus();
    } catch {
      setActionError("Stop request failed");
    } finally {
      setBusy(false);
    }
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    setUploadFile(event.target.files?.[0] ?? null);
  }

  const metrics = [
    ["FPS", status.stats.fps.toFixed(1)],
    ["People", status.stats.total_people_count],
    ["Known", status.stats.known_count],
    ["Unknown", status.stats.unknown_count],
    ["No Face", status.stats.person_no_face_count],
    ["Faces", status.stats.detections],
    [
      "Empty Wait",
      `${status.stats.no_person_frame_count}/${status.stats.empty_zone_threshold}`
    ]
  ];

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Camera size={22} aria-hidden="true" />
          </div>
          <div>
            <h1 className="brand-title">ZEEX AI Face Recognition v0</h1>
            <p className="brand-subtitle">
              {cameraId} · {zone}
            </p>
          </div>
        </div>
        <div className={`status-pill ${statusClass}`}>
          <Activity size={16} aria-hidden="true" />
          {backendError ? "BACKEND_OFFLINE" : zoneStatus}
        </div>
      </header>

      <div className="layout">
        <aside className="panel controls">
          <div className="panel-header">
            <h2 className="panel-title">Source</h2>
            <span className="message">{status.message}</span>
          </div>

          <div className="form-stack">
            <div className="segmented" aria-label="Source mode">
              <button
                className={`segment ${mode === "link" ? "active" : ""}`}
                type="button"
                onClick={() => setMode("link")}
              >
                <Radio size={15} aria-hidden="true" />
                Link
              </button>
              <button
                className={`segment ${mode === "upload" ? "active" : ""}`}
                type="button"
                onClick={() => setMode("upload")}
              >
                <Upload size={15} aria-hidden="true" />
                Upload
              </button>
              <button
                className={`segment ${mode === "path" ? "active" : ""}`}
                type="button"
                onClick={() => setMode("path")}
              >
                <FolderOpen size={15} aria-hidden="true" />
                Path
              </button>
            </div>

            {mode === "link" && (
              <label className="field">
                <span className="label">Stream URL</span>
                <input
                  className="text-input"
                  value={streamUrl}
                  onChange={(event) => setStreamUrl(event.target.value)}
                  placeholder="rtsp://..."
                />
              </label>
            )}

            {mode === "upload" && (
              <label className="field">
                <span className="label">Video File</span>
                <input
                  className="file-input"
                  type="file"
                  accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
                  onChange={onFileChange}
                />
              </label>
            )}

            {mode === "path" && (
              <label className="field">
                <span className="label">Local Video Path</span>
                <input
                  className="text-input"
                  value={localPath}
                  onChange={(event) => setLocalPath(event.target.value)}
                  placeholder={"C:\\Users\\RishuSingh\\Downloads\\clip.mp4"}
                />
              </label>
            )}

            <label className="field">
              <span className="label">Camera ID</span>
              <input
                className="text-input"
                value={cameraId}
                onChange={(event) => setCameraId(event.target.value.toUpperCase())}
                placeholder="CAM-01"
              />
            </label>

            <label className="field">
              <span className="label">Zone</span>
              <input
                className="text-input"
                value={zone}
                onChange={(event) => setZone(event.target.value)}
              />
            </label>

            <RangeField
              label="Cosine Threshold"
              max={0.7}
              min={0.2}
              step={0.01}
              value={threshold}
              onChange={setThreshold}
            />
            <RangeField
              label="Detect Every"
              max={10}
              min={1}
              step={1}
              value={detectEvery}
              onChange={setDetectEvery}
            />
            <RangeField
              label="Face Image Size"
              max={1920}
              min={640}
              step={320}
              value={faceImgSize}
              onChange={setFaceImgSize}
            />
            <RangeField
              label="Face Confidence"
              max={0.7}
              min={0.1}
              step={0.01}
              value={faceConf}
              onChange={setFaceConf}
            />
            <RangeField
              label="Min Face"
              max={80}
              min={12}
              step={2}
              value={minFace}
              onChange={setMinFace}
            />
            <RangeField
              label="Empty Zone Frames"
              max={120}
              min={1}
              step={1}
              value={emptyZoneFrames}
              onChange={setEmptyZoneFrames}
            />
            <label className="field-row">
              <span className="label">Person Detector</span>
              <input
                checked={personEnabled}
                onChange={(event) => setPersonEnabled(event.target.checked)}
                type="checkbox"
              />
            </label>
            <RangeField
              label="Person Confidence"
              max={0.9}
              min={0.2}
              step={0.01}
              value={personConf}
              onChange={setPersonConf}
            />

            <div className="actions">
              <button
                className="button primary"
                disabled={busy || running}
                onClick={() => start()}
                type="button"
              >
                <Play size={17} aria-hidden="true" />
                Start
              </button>
              <button
                className="button secondary"
                disabled={busy || !running}
                onClick={stop}
                type="button"
              >
                <Square size={17} aria-hidden="true" />
                Stop
              </button>
            </div>

            {(backendError || actionError) && (
              <div className="error">{backendError || actionError}</div>
            )}
          </div>
        </aside>

        <section className="main">
          <section className="panel feed-panel">
            <div className="feed-header">
              <h2 className="panel-title">Observed Frame</h2>
              <span className="message">{running ? "Live" : "Idle"}</span>
            </div>
            <div className="feed">
              {frameUrl ? (
                <img alt="Annotated recognition frame" src={frameUrl} />
              ) : (
                <div className="feed-placeholder">
                  <Video size={42} aria-hidden="true" />
                  <span>No frame</span>
                </div>
              )}
            </div>
          </section>

          <section className="metrics" aria-label="Recognition metrics">
            {metrics.map(([label, value]) => (
              <div className="metric" key={label}>
                <div className="metric-label">{label}</div>
                <div className="metric-value">{value}</div>
              </div>
            ))}
          </section>

          <section className="panel">
            <div className="feed-header">
              <h2 className="panel-title">Registered Workers</h2>
              <span className="message">{workers.length} workers</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Worker ID</th>
                    <th>Name</th>
                    <th>Role</th>
                    <th>Zone</th>
                    <th>Shift</th>
                    <th>Encodings</th>
                  </tr>
                </thead>
                <tbody>
                  {workers.length === 0 ? (
                    <tr>
                      <td colSpan={6}>No workers loaded</td>
                    </tr>
                  ) : (
                    workers.map((worker) => (
                      <tr key={worker.worker_id}>
                        <td>{worker.worker_id}</td>
                        <td>{worker.name}</td>
                        <td>{worker.role || "-"}</td>
                        <td>{worker.zone || "-"}</td>
                        <td>{worker.shift || "-"}</td>
                        <td>{worker.encodings}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </section>
      </div>
    </main>
  );
}

async function readApiError(res: Response) {
  const text = await res.text();
  try {
    const data = JSON.parse(text) as { detail?: unknown };
    if (typeof data.detail === "string") return data.detail;
  } catch {
    return text;
  }
  return `Request failed with status ${res.status}`;
}

function rtspForCameraId(cameraId: string) {
  const match = cameraId.trim().match(/^CAM-(\d+)$/i);
  if (!match) return "";
  const cameraNumber = Number(match[1]);
  if (!Number.isInteger(cameraNumber) || cameraNumber <= 0) return "";
  return `${CAMERA_RTSP_BASE}/cam${cameraNumber}`;
}

function RangeField({
  label,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span className="field-row">
        <span className="label">{label}</span>
        <strong>{value}</strong>
      </span>
      <input
        className="range"
        max={max}
        min={min}
        onChange={(event) => onChange(Number(event.target.value))}
        step={step}
        type="range"
        value={value}
      />
    </label>
  );
}
