"""
Crack-AI local server
---------------------
เซิร์ฟเวอร์ภายในเครื่อง (localhost) สำหรับหน้าเว็บ index.html
- จัดการชุดข้อมูล (dataset) รูปแบบ YOLO
- สั่งเทรน / หยุดเทรน / ดูความคืบหน้า (Ultralytics YOLO)
- ตรวจจับความเสียหายจากภาพ (segmentation หรือ detection)

รัน:  python server.py            (ค่าเริ่มต้น http://127.0.0.1:8765)
      python server.py --port 9000 --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_VERSION = "2.0.0"
BASE = Path(__file__).resolve().parent          # โฟลเดอร์ server/
WEB = BASE.parent                                # โฟลเดอร์หน้าเว็บ (root ของ repo)
WORK = Path(os.environ.get("CRACKAI_WORKSPACE", WEB / "workspace")).resolve()
DATA = WORK / "dataset"
IMG_DIR = DATA / "images"
LBL_DIR = DATA / "labels"
RUNS = WORK / "runs"
WEIGHTS = WORK / "weights"
CLASSES_FILE = WORK / "classes.json"
ACTIVE_FILE = WORK / "active_model.json"
for d in (IMG_DIR, LBL_DIR, RUNS, WEIGHTS):
    d.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MEASURES = {"crack", "area", "rebar"}
DEFAULT_CLASSES = [
    {"name": "crack", "label": "รอยร้าว", "measure": "crack", "color": "#D9363E"},
    {"name": "spalling", "label": "คอนกรีตกะเทาะ/หลุดร่อน", "measure": "area", "color": "#E8730C"},
    {"name": "exposed_rebar", "label": "เหล็กเสริมโผล่/เป็นสนิม", "measure": "rebar", "color": "#2F8F5B"},
]
MODEL_FAMILIES = {"yolo26": "YOLO26", "yolo11": "YOLO11"}
MODEL_SIZES = ["n", "s", "m", "l", "x"]

def load_config() -> dict:
    """server/config.json: host, port และรายชื่อเว็บไซต์ (origin) ที่อนุญาตให้สั่งงานเซิร์ฟเวอร์นี้
    เช่นหน้าเว็บบน GitHub Pages ของคุณ https://<ชื่อผู้ใช้>.github.io"""
    cfg = {"host": "127.0.0.1", "port": 8765, "allowed_origins": []}
    p = BASE / "config.json"
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [คำเตือน] อ่าน config.json ไม่ได้: {e}")
    extra = os.environ.get("CRACKAI_ORIGINS", "")
    cfg["allowed_origins"] = [o.strip().rstrip("/") for o in list(cfg.get("allowed_origins") or []) + extra.split(",")
                              if o.strip() and "YOUR-GITHUB-NAME" not in o]
    return cfg


CONFIG = load_config()
app = FastAPI(title="Crack-AI", version=APP_VERSION)

# อนุญาต: หน้าเว็บที่เปิดจากไฟล์ในเครื่อง (origin = null), localhost และ origin ที่ระบุใน config.json
import inspect as _inspect

_cors_kw = {}
if "allow_private_network" in _inspect.signature(CORSMiddleware.__init__).parameters:
    _cors_kw["allow_private_network"] = True  # Starlette รุ่นใหม่รองรับในตัว
app.add_middleware(
    CORSMiddleware,
    **_cors_kw,
    allow_origins=CONFIG["allowed_origins"],
    allow_origin_regex=r"^(null|https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?)$",
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)


class PrivateNetworkAccess:
    """Chrome/Edge ต้องการ header นี้ก่อนยอมให้เว็บบนอินเทอร์เน็ต (เช่น GitHub Pages)
    เรียกเซิร์ฟเวอร์ในเครื่อง (127.0.0.1) — Private Network Access / Local Network Access"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "OPTIONS":
            return await self.app(scope, receive, send)
        want = any(k == b"access-control-request-private-network" for k, _ in scope["headers"])

        async def send2(msg):
            if want and msg["type"] == "http.response.start":
                msg.setdefault("headers", [])
                msg["headers"] = list(msg["headers"]) + [(b"access-control-allow-private-network", b"true")]
            await send(msg)

        return await self.app(scope, receive, send2)


if not _cors_kw:  # Starlette รุ่นเก่า: เติม header เอง
    app.add_middleware(PrivateNetworkAccess)




def require_client(x_crackai: Optional[str] = Header(default=None)) -> None:
    """ทุกคำสั่งที่เปลี่ยนข้อมูลต้องมี header นี้ -> บังคับให้เบราว์เซอร์ทำ CORS preflight
    เว็บไซต์ภายนอกจึงสั่งงานเซิร์ฟเวอร์นี้แทนผู้ใช้ไม่ได้"""
    if x_crackai != "1":
        raise HTTPException(403, "missing X-CrackAI header")


# ---------------------------------------------------------------- utils
def read_json(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(p: Path, data: Any) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def get_classes() -> list[dict]:
    c = read_json(CLASSES_FILE, None)
    if not c:
        write_json(CLASSES_FILE, DEFAULT_CLASSES)
        return [dict(x) for x in DEFAULT_CLASSES]
    return c


def guess_measure(name: str) -> str:
    n = name.lower()
    if "crack" in n or "ร้าว" in n:
        return "crack"
    if "rebar" in n or "steel" in n or "เหล็ก" in n:
        return "rebar"
    return "area"


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """แตก zip แบบกันไฟล์อันตราย (path traversal / zip bomb ขั้นพื้นฐาน)"""
    total = 0
    dest = dest.resolve()
    for info in zf.infolist():
        if info.is_dir():
            continue
        total += info.file_size
        if total > 20 * 1024**3:
            raise HTTPException(400, "ไฟล์ zip ใหญ่เกิน 20 GB หลังแตกไฟล์")
        target = (dest / info.filename).resolve()
        if not str(target).startswith(str(dest) + os.sep):
            continue  # ข้ามไฟล์ที่พยายามเขียนออกนอกโฟลเดอร์
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out, 1024 * 1024)


def parse_label_text(text: str) -> tuple[list[str], list[str], dict]:
    """ตรวจและทำความสะอาดไฟล์ label YOLO -> (บรรทัดที่ถูกต้อง, ปัญหา, สถิติ)"""
    good, issues = [], []
    st = {"seg": 0, "box": 0}
    for ln, raw in enumerate(text.splitlines(), 1):
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append(f"บรรทัด {ln}: ไม่ใช่ตัวเลข")
            continue
        if cls < 0 or any(not np.isfinite(v) for v in vals):
            issues.append(f"บรรทัด {ln}: ค่าไม่ถูกต้อง")
            continue
        if len(vals) == 4:
            st["box"] += 1
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            st["seg"] += 1
        else:
            issues.append(f"บรรทัด {ln}: จำนวนพิกัดไม่ถูกต้อง ({len(vals)})")
            continue
        if any(v < -0.01 or v > 1.01 for v in vals):
            issues.append(f"บรรทัด {ln}: พิกัดเกินช่วง 0-1 (ถูกตัดให้อยู่ในช่วง)")
        vals = [min(1.0, max(0.0, v)) for v in vals]
        good.append(" ".join([str(cls)] + [f"{v:.6f}" for v in vals]))
    return good, issues, st


def store_pair(img_bytes: bytes, ext: str, label_lines: Optional[list[str]], stem_hint: str) -> str:
    """เก็บภาพ+label ลง dataset ชื่อไฟล์อิงจาก hash ของภาพ (อัปโหลดซ้ำจะไม่ซ้ำซ้อน)"""
    h = hashlib.sha1(img_bytes).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", stem_hint)[:40].strip("_") or "img"
    name = f"{stem}_{h}"
    for old in IMG_DIR.glob(f"*_{h}.*"):  # ภาพเดิมซ้ำ -> แทนที่
        old.unlink(missing_ok=True)
        (LBL_DIR / (old.stem + ".txt")).unlink(missing_ok=True)
    (IMG_DIR / f"{name}{ext.lower()}").write_bytes(img_bytes)
    lbl = LBL_DIR / f"{name}.txt"
    if label_lines is not None:
        lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    return name


def dataset_stats() -> dict:
    classes = get_classes()
    imgs = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXT)
    per_cls: dict[int, int] = {}
    labeled = background = seg = box = 0
    max_cls = -1
    for p in imgs:
        lp = LBL_DIR / (p.stem + ".txt")
        lines = lp.read_text(encoding="utf-8").split("\n") if lp.exists() else []
        lines = [l for l in lines if l.strip()]
        if not lines:
            background += 1
            continue
        labeled += 1
        for l in lines:
            parts = l.split()
            c = int(parts[0])
            per_cls[c] = per_cls.get(c, 0) + 1
            max_cls = max(max_cls, c)
            if len(parts) == 5:
                box += 1
            else:
                seg += 1
    warnings = []
    if max_cls >= len(classes):
        warnings.append(
            f"ป้ายกำกับมีคลาสหมายเลข {max_cls} แต่ตั้งค่าคลาสไว้เพียง {len(classes)} คลาส (0-{len(classes)-1}) "
            "กรุณาเพิ่มคลาสในแท็บ 'คลาส' ให้ครบ"
        )
    if 0 < len(imgs) < 20:
        warnings.append("จำนวนภาพยังน้อย แนะนำอย่างน้อย 100-300 ภาพต่อคลาสเพื่อผลที่ใช้งานได้จริง")
    return {
        "images": len(imgs),
        "labeled": labeled,
        "background": background,
        "instances": {str(k): v for k, v in sorted(per_cls.items())},
        "seg_lines": seg,
        "box_lines": box,
        "max_class_id": max_cls,
        "warnings": warnings,
    }


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "อ่านไฟล์ภาพไม่ได้")
    return img


# ---------------------------------------------------------------- torch info (lazy)
_torch_info: Optional[dict] = None


def torch_info() -> dict:
    global _torch_info
    if _torch_info is not None:
        return _torch_info
    info: dict[str, Any] = {"torch": None, "cuda": False, "mps": False, "devices": [], "ultralytics": None}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            for i in range(torch.cuda.device_count()):
                pr = torch.cuda.get_device_properties(i)
                info["devices"].append(
                    {"id": str(i), "name": pr.name, "mem_gb": round(pr.total_memory / 1024**3, 1),
                     "cc": f"{pr.major}.{pr.minor}"}
                )
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception as e:  # torch ไม่ได้ติดตั้ง
        info["error"] = str(e)
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    _torch_info = info
    return info


# ---------------------------------------------------------------- training process
class TrainState:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.run_dir: Optional[Path] = None
        self.stopped_by_user = False
        self.lock = threading.Lock()


TRAIN = TrainState()


def is_training() -> bool:
    return TRAIN.proc is not None and TRAIN.proc.poll() is None


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def tail_text(p: Path, n_bytes: int = 6000) -> str:
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - n_bytes))
        raw = f.read().decode("utf-8", errors="replace")
    # tqdm เขียนด้วย \r -> เก็บเฉพาะสถานะล่าสุดของแต่ละบรรทัด
    lines = [seg.split("\r")[-1] for seg in raw.split("\n")]
    lines = [re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", l) for l in lines]
    return "\n".join(lines[-40:])


def run_status(run_dir: Path) -> dict:
    st = read_json(run_dir / "status.json", {})
    prog = read_json(run_dir / "progress.json", {})
    cfg = read_json(run_dir / "run_config.json", {})
    best = run_dir / "weights" / "best.pt"
    state = st.get("state", "unknown")
    if TRAIN.run_dir == run_dir:
        if is_training():
            state = "running" if state in ("running", "starting", "unknown") else state
        elif TRAIN.proc is not None and state in ("running", "starting", "unknown"):
            state = "stopped" if TRAIN.stopped_by_user else "failed"
    elif state in ("running", "starting"):
        state = "stopped"  # เซิร์ฟเวอร์ถูกปิดระหว่างเทรน
    return {
        "run": run_dir.name,
        "state": state,
        "message": st.get("message", ""),
        "error": st.get("error", ""),
        "epoch": prog.get("epoch", 0),
        "epochs": prog.get("epochs", cfg.get("hyp", {}).get("epochs", 0)),
        "history": prog.get("history", []),
        "final": prog.get("final"),
        "started": st.get("started"),
        "finished": st.get("finished"),
        "best": str(best) if best.exists() else None,
        "model": cfg.get("model"),
        "task": cfg.get("task"),
        "log": tail_text(run_dir / "train.log"),
    }


# ---------------------------------------------------------------- inference
_model_cache: dict[str, Any] = {}
_infer_lock = threading.Lock()


def get_active() -> Optional[dict]:
    a = read_json(ACTIVE_FILE, None)
    if a and Path(a.get("path", "")).exists():
        return a
    return None


def load_model(path: str):
    from ultralytics import YOLO

    if path not in _model_cache:
        _model_cache.clear()  # เก็บไว้ทีละโมเดล ประหยัดหน่วยความจำ GPU
        _model_cache[path] = YOLO(path)
    return _model_cache[path]


def model_meta(path: Path) -> dict:
    """ข้อมูลคลาสของโมเดล: ใช้ classes.json ที่บันทึกไว้ตอนเทรน ถ้าไม่มีให้เดาจากชื่อ"""
    run_classes = read_json(path.parent.parent / "classes.json", None) if path.parent.name == "weights" else None
    return {"classes": run_classes}


def pick_device(req: str) -> Optional[str]:
    info = torch_info()
    if req and req != "auto":
        return req
    if info.get("cuda"):
        return "0"
    if info.get("mps"):
        return "mps"
    return "cpu"


def instances_from_result(r, offset=(0, 0)) -> list[dict]:
    from ultralytics.utils import ops

    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    masks = r.masks.data.cpu().numpy() if r.masks is not None else None
    ox, oy = offset
    for i in range(len(boxes)):
        polys = []
        is_box = masks is None
        if masks is not None:
            m = (masks[i] > 0.5).astype(np.uint8)
            contours = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            for c in contours:
                if cv2.contourArea(c) < 1.0:
                    continue
                c = c.reshape(-1, 2).astype(np.float32)
                c = ops.scale_coords(m.shape[:2], c, r.orig_shape, normalize=False)
                c = np.asarray(c, dtype=np.float32)
                c[:, 0] += ox
                c[:, 1] += oy
                polys.append(c)
            if not polys:
                continue
        else:
            x1, y1, x2, y2 = boxes[i]
            polys.append(np.array([[x1 + ox, y1 + oy], [x2 + ox, y1 + oy], [x2 + ox, y2 + oy], [x1 + ox, y2 + oy]], np.float32))
        b = boxes[i]
        out.append({
            "cls": int(clss[i]), "conf": float(confs[i]), "polys": polys, "is_box": is_box,
            "box": [float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)],
        })
    return out


def merge_tiled(instances: list[dict], W: int, H: int) -> list[dict]:
    """รวมผลจากหลายช่อง (tile): วาดทุก instance ลงแผนที่ของแต่ละคลาส แล้วแยกเป็นก้อนที่ติดกัน
    ทำให้รอยร้าวที่พาดผ่านหลายช่องกลายเป็นชิ้นเดียว"""
    s = min(1.0, 4000.0 / max(W, H))
    w, h = max(1, int(round(W * s))), max(1, int(round(H * s)))
    merged = []
    by_cls: dict[int, list[dict]] = {}
    for ins in instances:
        by_cls.setdefault(ins["cls"], []).append(ins)
    for cls, items in by_cls.items():
        is_box = all(x["is_box"] for x in items)
        mask = np.zeros((h, w), np.uint8)
        confm = np.zeros((h, w), np.float32)
        for ins in items:
            tmp = np.zeros((h, w), np.uint8)
            pts = [np.round(p * s).astype(np.int32) for p in ins["polys"]]
            cv2.fillPoly(tmp, pts, 1)
            x1, y1 = int(max(0, ins["box"][0] * s - 2)), int(max(0, ins["box"][1] * s - 2))
            x2, y2 = int(min(w, ins["box"][2] * s + 3)), int(min(h, ins["box"][3] * s + 3))
            sub = tmp[y1:y2, x1:x2].astype(bool)
            mask[y1:y2, x1:x2][sub] = 1
            cm = confm[y1:y2, x1:x2]
            cm[sub] = np.maximum(cm[sub], ins["conf"])
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for k in range(1, n):
            x, y, bw, bh, area = stats[k]
            if area < 2:
                continue
            comp = (lab[y:y + bh, x:x + bw] == k).astype(np.uint8)
            contours = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            polys = []
            for c in contours:
                c = c.reshape(-1, 2).astype(np.float32)
                c[:, 0] = (c[:, 0] + x) / s
                c[:, 1] = (c[:, 1] + y) / s
                polys.append(c)
            conf = float(confm[y:y + bh, x:x + bw][comp.astype(bool)].max())
            merged.append({
                "cls": cls, "conf": conf, "polys": polys, "is_box": is_box,
                "box": [x / s, y / s, (x + bw) / s, (y + bh) / s],
            })
    return merged


def simplify(poly: np.ndarray, W: int, H: int) -> list[list[float]]:
    eps = 1.0
    c = poly.reshape(-1, 1, 2).astype(np.float32)
    s = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    while len(s) > 400:
        eps *= 1.6
        s = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    s[:, 0] = np.clip(s[:, 0], 0, W)
    s[:, 1] = np.clip(s[:, 1], 0, H)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in s]


# ================================================================ API
class ClassItem(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    label: str = Field(default="", max_length=80)
    measure: str = "area"
    color: str = "#888888"


class AnnotationIn(BaseModel):
    image: str  # data URL (jpeg/png)
    filename: str = "annotated"
    labels: list[dict]  # [{cls:int, points:[[x,y],...] normalized 0-1}]


class TrainIn(BaseModel):
    family: str = "yolo11"
    size: str = "n"
    task: str = "segment"
    val_split: float = 0.2
    seed: int = 0
    hyp: dict = {}
    base_weights: Optional[str] = None  # เทรนต่อจากโมเดลเดิม (path best.pt)


class ActivateIn(BaseModel):
    path: str


@app.get("/api/health")
def health():
    info = torch_info()
    return {
        "ok": True,
        "version": APP_VERSION,
        "platform": platform.system(),
        "python": platform.python_version(),
        **info,
        "training": is_training(),
        "active_model": get_active(),
        "workspace": str(WORK),
        "allowed_origins": CONFIG["allowed_origins"],
    }


@app.get("/api/classes")
def classes_get():
    return get_classes()


@app.put("/api/classes", dependencies=[Depends(require_client)])
def classes_put(items: list[ClassItem]):
    if not items:
        raise HTTPException(400, "ต้องมีอย่างน้อย 1 คลาส")
    names = set()
    out = []
    for it in items:
        n = re.sub(r"\s+", "_", it.name.strip())
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", n):
            raise HTTPException(400, f"ชื่อคลาส '{it.name}' ใช้ได้เฉพาะ a-z, 0-9, _ และ -")
        if n.lower() in names:
            raise HTTPException(400, f"ชื่อคลาส '{n}' ซ้ำ")
        names.add(n.lower())
        if it.measure not in MEASURES:
            raise HTTPException(400, f"ประเภทการวัด '{it.measure}' ไม่ถูกต้อง")
        color = it.color if re.fullmatch(r"#[0-9A-Fa-f]{6}", it.color) else "#888888"
        out.append({"name": n, "label": it.label.strip() or n, "measure": it.measure, "color": color})
    write_json(CLASSES_FILE, out)
    return out


@app.get("/api/dataset")
def dataset_get():
    return dataset_stats()


@app.delete("/api/dataset", dependencies=[Depends(require_client)])
def dataset_clear():
    if is_training():
        raise HTTPException(409, "กำลังเทรนอยู่ ล้างชุดข้อมูลไม่ได้")
    shutil.rmtree(DATA, ignore_errors=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    return dataset_stats()


@app.delete("/api/dataset/item/{name}", dependencies=[Depends(require_client)])
def dataset_delete_item(name: str):
    """ลบภาพหนึ่งภาพ (และป้ายกำกับ) ออกจากชุดข้อมูล ใช้เมื่อผู้ใช้ปิด 'ใช้เทรน' ในคลังงาน"""
    if is_training():
        raise HTTPException(409, "กำลังเทรนอยู่ กรุณารอให้เสร็จหรือหยุดก่อน")
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", name):
        raise HTTPException(400, "ชื่อไม่ถูกต้อง")
    n = 0
    for p in IMG_DIR.glob(name + ".*"):
        p.unlink(missing_ok=True)
        n += 1
    (LBL_DIR / f"{name}.txt").unlink(missing_ok=True)
    return {"deleted": n, "stats": dataset_stats()}


@app.post("/api/dataset/upload", dependencies=[Depends(require_client)])
async def dataset_upload(files: list[UploadFile] = File(...)):
    if is_training():
        raise HTTPException(409, "กำลังเทรนอยู่ กรุณารอให้เสร็จหรือหยุดก่อน")
    tmp = Path(tempfile.mkdtemp(prefix="crackai_up_"))
    try:
        for i, f in enumerate(files):
            fname = Path(f.filename or f"file{i}").name
            dest = tmp / f"_{i}" / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                while chunk := await f.read(4 * 1024 * 1024):
                    out.write(chunk)
            if dest.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(dest) as zf:
                        safe_extract(zf, dest.parent / (dest.stem + "_unzip"))
                except zipfile.BadZipFile:
                    raise HTTPException(400, f"ไฟล์ {fname} ไม่ใช่ zip ที่ถูกต้อง")
                dest.unlink()

        all_files = [p for p in tmp.rglob("*") if p.is_file() and "__MACOSX" not in p.parts]
        images = [p for p in all_files if p.suffix.lower() in IMG_EXT and not p.name.startswith("._")]
        txts = [p for p in all_files if p.suffix.lower() == ".txt" and p.name.lower() not in ("classes.txt", "readme.txt")]
        txt_by_stem: dict[str, list[Path]] = {}
        for t in txts:
            txt_by_stem.setdefault(t.stem, []).append(t)

        yaml_names = None
        for y in [p for p in all_files if p.suffix.lower() in (".yaml", ".yml")]:
            try:
                d = yaml.safe_load(y.read_text(encoding="utf-8"))
                n = d.get("names") if isinstance(d, dict) else None
                if isinstance(n, dict):
                    n = [n[k] for k in sorted(n, key=lambda k: int(k))]
                if isinstance(n, list) and n:
                    yaml_names = [str(x) for x in n]
                    break
            except Exception:
                continue
        if yaml_names is None:
            ct = [p for p in all_files if p.name.lower() == "classes.txt"]
            if ct:
                yaml_names = [l.strip() for l in ct[0].read_text(encoding="utf-8").splitlines() if l.strip()]

        added = with_label = no_label = 0
        issues: list[str] = []
        stats = {"seg": 0, "box": 0}
        for img in images:
            # หา label: โครงสร้าง .../images/x.jpg -> .../labels/x.txt ก่อน แล้วค่อยหาชื่อเดียวกัน
            lbl = None
            parts = list(img.parts)
            if "images" in parts:
                idx = len(parts) - 1 - parts[::-1].index("images")
                cand = Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")
                if cand.exists():
                    lbl = cand
            if lbl is None:
                same = img.with_suffix(".txt")
                if same.exists():
                    lbl = same
                elif len(txt_by_stem.get(img.stem, [])) == 1:
                    lbl = txt_by_stem[img.stem][0]
            lines = None
            if lbl is not None:
                lines, iss, st = parse_label_text(lbl.read_text(encoding="utf-8", errors="replace"))
                stats["seg"] += st["seg"]
                stats["box"] += st["box"]
                issues += [f"{img.name}: {x}" for x in iss]
                with_label += 1
            else:
                no_label += 1
            data = img.read_bytes()
            if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
                issues.append(f"{img.name}: อ่านภาพไม่ได้ (ข้าม)")
                continue
            store_pair(data, img.suffix, lines, img.stem)
            added += 1
        if not images:
            raise HTTPException(400, "ไม่พบไฟล์ภาพในสิ่งที่อัปโหลด (รองรับ jpg, png, bmp, webp, tif)")
        return {
            "added": added, "with_label": with_label, "without_label": no_label,
            "lines": stats, "issues": issues[:50], "issue_count": len(issues),
            "yaml_names": yaml_names, "stats": dataset_stats(),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/dataset/annotation", dependencies=[Depends(require_client)])
def dataset_add_annotation(a: AnnotationIn):
    if is_training():
        raise HTTPException(409, "กำลังเทรนอยู่ กรุณารอให้เสร็จหรือหยุดก่อน")
    m = re.match(r"data:image/(jpeg|jpg|png);base64,(.+)", a.image, re.S)
    if not m:
        raise HTTPException(400, "รูปแบบภาพไม่ถูกต้อง")
    data = base64.b64decode(m.group(2))
    decode_image(data)
    nc = len(get_classes())
    lines = []
    for lb in a.labels:
        cls = int(lb.get("cls", -1))
        pts = lb.get("points") or []
        if cls < 0 or cls >= nc or len(pts) < 3:
            continue
        flat = []
        for x, y in pts:
            flat += [min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))]
        lines.append(" ".join([str(cls)] + [f"{v:.6f}" for v in flat]))
    ext = ".png" if m.group(1) == "png" else ".jpg"
    name = store_pair(data, ext, lines, a.filename)
    return {"name": name, "labels": len(lines), "stats": dataset_stats()}


@app.post("/api/train", dependencies=[Depends(require_client)])
def train_start(cfg: TrainIn):
    with TRAIN.lock:
        if is_training():
            raise HTTPException(409, "มีการเทรนที่กำลังทำงานอยู่แล้ว")
        if cfg.family not in MODEL_FAMILIES or cfg.size not in MODEL_SIZES or cfg.task not in ("segment", "detect"):
            raise HTTPException(400, "รุ่นโมเดลไม่ถูกต้อง")
        classes = get_classes()
        st = dataset_stats()
        if st["images"] < 2:
            raise HTTPException(400, "ต้องมีภาพในชุดข้อมูลอย่างน้อย 2 ภาพ")
        if st["labeled"] == 0:
            raise HTTPException(400, "ยังไม่มีภาพที่มีป้ายกำกับ (label) เลย")
        if st["max_class_id"] >= len(classes):
            raise HTTPException(400, st["warnings"][0])
        if cfg.task == "segment" and st["box_lines"] > 0:
            raise HTTPException(
                400,
                f"ชุดข้อมูลมีป้ายกำกับแบบกรอบสี่เหลี่ยม {st['box_lines']} รายการ ซึ่งใช้เทรนแบบ Segmentation ไม่ได้ "
                "ให้เลือกชนิดงานเป็น 'Detection' หรือใช้ป้ายกำกับแบบ polygon ทั้งหมด",
            )

        # ---- hyperparameters (whitelist + ตรวจช่วงค่า)
        spec = {
            "epochs": (int, 1, 2000), "imgsz": (int, 160, 2560), "batch": (float, -1, 512),
            "patience": (int, 0, 2000), "lr0": (float, 1e-6, 0.5), "lrf": (float, 1e-4, 1.0),
            "workers": (int, 0, 32), "close_mosaic": (int, 0, 1000), "seed": (int, 0, 2**31 - 1),
            "fliplr": (float, 0, 1), "flipud": (float, 0, 1), "degrees": (float, 0, 180),
            "translate": (float, 0, 0.9), "scale": (float, 0, 0.95), "mosaic": (float, 0, 1),
            "mixup": (float, 0, 1), "hsv_h": (float, 0, 1), "hsv_s": (float, 0, 1), "hsv_v": (float, 0, 1),
            "weight_decay": (float, 0, 0.1), "warmup_epochs": (float, 0, 50),
        }
        hyp: dict[str, Any] = {}
        for k, (typ, lo, hi) in spec.items():
            if k in cfg.hyp and cfg.hyp[k] not in (None, ""):
                try:
                    v = typ(cfg.hyp[k])
                except (TypeError, ValueError):
                    raise HTTPException(400, f"ค่า {k} ไม่ถูกต้อง")
                if not (lo <= v <= hi):
                    raise HTTPException(400, f"ค่า {k} ต้องอยู่ระหว่าง {lo} ถึง {hi}")
                hyp[k] = v
        if "imgsz" in hyp:
            hyp["imgsz"] = max(32, int(round(hyp["imgsz"] / 32)) * 32)
        if "batch" in hyp:
            b = hyp["batch"]
            hyp["batch"] = -1 if b < 0 else (b if 0 < b < 1 else max(1, int(b)))
        if cfg.hyp.get("optimizer") in ("auto", "SGD", "Adam", "AdamW", "MuSGD"):
            hyp["optimizer"] = cfg.hyp["optimizer"]
        if cfg.hyp.get("cache") in ("ram", "disk"):
            hyp["cache"] = cfg.hyp["cache"]
        for b in ("amp", "cos_lr", "rect", "plots"):
            if b in cfg.hyp:
                hyp[b] = bool(cfg.hyp[b])
        dev = str(cfg.hyp.get("device", "auto"))
        if dev != "auto":
            if not re.fullmatch(r"cpu|mps|\d+(,\d+)*", dev):
                raise HTTPException(400, "ค่า device ไม่ถูกต้อง")
            hyp["device"] = dev
        if platform.system() == "Windows" and "," in dev:
            raise HTTPException(400, "Windows ไม่รองรับการเทรนหลาย GPU พร้อมกัน (DDP) กรุณาเลือก GPU เดียว")

        # ---- แบ่ง train / val
        imgs = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXT)
        rnd = random.Random(cfg.seed)
        rnd.shuffle(imgs)
        n_val = min(len(imgs) - 1, max(1, int(round(len(imgs) * min(0.5, max(0.05, cfg.val_split))))))
        val, trn = imgs[:n_val], imgs[n_val:]

        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        run_dir = RUNS / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "train.txt").write_text("\n".join(str(p) for p in trn) + "\n", encoding="utf-8")
        (run_dir / "val.txt").write_text("\n".join(str(p) for p in val) + "\n", encoding="utf-8")
        data_yaml = {
            "path": str(DATA),
            "train": str(run_dir / "train.txt"),
            "val": str(run_dir / "val.txt"),
            "names": {i: c["name"] for i, c in enumerate(classes)},
        }
        (run_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True), encoding="utf-8")
        write_json(run_dir / "classes.json", classes)

        if cfg.base_weights:
            bw = Path(cfg.base_weights).resolve()
            if not bw.exists() or bw.suffix != ".pt":
                raise HTTPException(400, "ไม่พบไฟล์โมเดลตั้งต้น")
            model = str(bw)
        else:
            suffix = "-seg" if cfg.task == "segment" else ""
            model = f"{cfg.family}{cfg.size}{suffix}.pt"
            if (WEIGHTS / model).exists():
                model = str(WEIGHTS / model)
        run_cfg = {
            "model": model, "task": cfg.task, "data": str(run_dir / "data.yaml"),
            "project": str(RUNS), "name": run_name, "hyp": hyp,
            "split": {"train": len(trn), "val": len(val)},
        }
        write_json(run_dir / "run_config.json", run_cfg)
        write_json(run_dir / "status.json", {"state": "starting", "started": time.time()})

        log = open(run_dir / "train.log", "w", encoding="utf-8")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        kw: dict[str, Any] = {}
        if platform.system() == "Windows":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        TRAIN.proc = subprocess.Popen(
            [sys.executable, str(BASE / "trainer.py"), str(run_dir / "run_config.json")],
            cwd=str(WEIGHTS), stdout=log, stderr=subprocess.STDOUT, env=env, **kw,
        )
        TRAIN.run_dir = run_dir
        TRAIN.stopped_by_user = False
        return {"run": run_name, "split": run_cfg["split"], "model": model, "hyp": hyp}


@app.get("/api/train/status")
def train_status(run: Optional[str] = None):
    if run:
        rd = (RUNS / run).resolve()
        if rd.parent != RUNS.resolve() or not rd.exists():
            raise HTTPException(404, "ไม่พบรอบการเทรน")
        return run_status(rd)
    if TRAIN.run_dir is None:
        return {"state": "idle"}
    return run_status(TRAIN.run_dir)


@app.post("/api/train/stop", dependencies=[Depends(require_client)])
def train_stop():
    if not is_training():
        return {"stopped": False}
    TRAIN.stopped_by_user = True
    kill_tree(TRAIN.proc)
    st = read_json(TRAIN.run_dir / "status.json", {})
    st.update({"state": "stopped", "finished": time.time(), "message": "หยุดโดยผู้ใช้"})
    write_json(TRAIN.run_dir / "status.json", st)
    return {"stopped": True}


@app.get("/api/models")
def models_list():
    items = []
    for rd in sorted(RUNS.iterdir(), reverse=True) if RUNS.exists() else []:
        best = rd / "weights" / "best.pt"
        if not rd.is_dir() or not best.exists():
            continue
        s = run_status(rd)
        bestm = s["final"] or (max(s["history"], key=lambda h: h.get("fitness") or 0) if s["history"] else {})
        items.append({
            "path": str(best), "run": rd.name, "state": s["state"], "task": s["task"],
            "model": Path(s["model"] or "").name, "epochs": s["epoch"],
            "metrics": {k: bestm.get(k) for k in ("mAP50(M)", "mAP50-95(M)", "mAP50(B)", "mAP50-95(B)") if k in bestm},
            "classes": read_json(rd / "classes.json", []),
            "mtime": best.stat().st_mtime,
        })
    for p in sorted(WEIGHTS.glob("custom_*.pt")):
        items.append({"path": str(p), "run": p.stem, "state": "uploaded", "task": None, "model": p.name,
                      "epochs": None, "metrics": {}, "classes": read_json(p.with_suffix(".classes.json"), []),
                      "mtime": p.stat().st_mtime})
    return {"active": get_active(), "models": items}


@app.post("/api/models/activate", dependencies=[Depends(require_client)])
def models_activate(a: ActivateIn):
    p = Path(a.path).resolve()
    if p.suffix != ".pt" or not p.exists() or not (str(p).startswith(str(RUNS.resolve())) or str(p).startswith(str(WEIGHTS.resolve()))):
        raise HTTPException(400, "ไฟล์โมเดลไม่ถูกต้อง")
    try:
        with _infer_lock:
            m = load_model(str(p))
            names = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
            task = m.task
    except Exception as e:
        raise HTTPException(400, f"โหลดโมเดลไม่สำเร็จ: {e}")
    meta = {"path": str(p), "names": names, "task": task, "activated": time.time(),
            "label": p.parent.parent.name if p.parent.name == "weights" else p.stem}
    write_json(ACTIVE_FILE, meta)
    return meta


@app.post("/api/models/upload", dependencies=[Depends(require_client)])
async def models_upload(file: UploadFile = File(...)):
    name = Path(file.filename or "model.pt").name
    if not name.endswith(".pt"):
        raise HTTPException(400, "รองรับเฉพาะไฟล์ .pt ของ Ultralytics")
    dest = WEIGHTS / ("custom_" + re.sub(r"[^A-Za-z0-9_\-.]", "_", name))
    with open(dest, "wb") as out:
        while chunk := await file.read(4 * 1024 * 1024):
            out.write(chunk)
    try:
        from ultralytics import YOLO

        m = YOLO(str(dest))
        names = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"ไฟล์นี้ไม่ใช่โมเดล YOLO ที่ใช้ได้: {e}")
    cls = [{"name": n, "label": n, "measure": guess_measure(n), "color": "#888888"} for n in names]
    write_json(dest.with_suffix(".classes.json"), cls)
    return models_activate(ActivateIn(path=str(dest)))


@app.post("/api/detect", dependencies=[Depends(require_client)])
async def detect(
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.5),
    imgsz: int = Form(0),
    tile: str = Form("auto"),  # auto | off | on
    device: str = Form("auto"),
    max_det: int = Form(300),
):
    active = get_active()
    if not active:
        raise HTTPException(400, "ยังไม่ได้เลือกโมเดลที่จะใช้ตรวจจับ กรุณาเทรนหรือเลือกโมเดลก่อน")
    img = decode_image(await image.read())
    H, W = img.shape[:2]
    conf = min(0.99, max(0.01, conf))
    iou = min(0.95, max(0.05, iou))
    t0 = time.time()
    with _infer_lock:
        model = load_model(active["path"])
        train_args = getattr(model, "ckpt", {}) or {}
        train_args = train_args.get("train_args", {}) if isinstance(train_args, dict) else {}
        sz = imgsz if imgsz >= 160 else int(train_args.get("imgsz", 640) or 640)
        sz = max(32, int(round(sz / 32)) * 32)
        dev = pick_device(device)
        use_tile = tile == "on" or (tile == "auto" and max(W, H) > 2 * sz)
        kw = dict(conf=conf, iou=iou, imgsz=sz, max_det=max(1, min(1000, max_det)), verbose=False, device=dev)
        if not use_tile:
            r = model.predict(img, **kw)[0]
            inst = instances_from_result(r)
            n_tiles = 1
        else:
            step = int(sz * 0.8)
            xs = list(range(0, max(1, W - sz), step)) + [max(0, W - sz)]
            ys = list(range(0, max(1, H - sz), step)) + [max(0, H - sz)]
            xs, ys = sorted(set(xs)), sorted(set(ys))
            inst = []
            for y in ys:
                for x in xs:
                    crop = img[y:y + sz, x:x + sz]
                    r = model.predict(crop, **kw)[0]
                    inst += instances_from_result(r, offset=(x, y))
            inst = merge_tiled(inst, W, H)
            n_tiles = len(xs) * len(ys)
        names = model.names
    # map ชื่อคลาส -> ประเภทการวัด
    run_classes = model_meta(Path(active["path"]))["classes"]
    if run_classes is None and Path(active["path"]).name.startswith("custom_"):
        run_classes = read_json(Path(active["path"]).with_suffix(".classes.json"), None)
    global_cls = {c["name"]: c for c in get_classes()}
    dets = []
    for ins in inst:
        name = names[ins["cls"]] if isinstance(names, (dict, list)) else str(ins["cls"])
        cinfo = None
        if run_classes and ins["cls"] < len(run_classes):
            cinfo = run_classes[ins["cls"]]
        cinfo = cinfo or global_cls.get(name) or {"measure": guess_measure(name), "label": name, "color": "#888888"}
        polys = [simplify(p, W, H) for p in ins["polys"]]
        polys = [p for p in polys if len(p) >= 3]
        if not polys:
            continue
        dets.append({
            "cls": ins["cls"], "name": name, "label": cinfo.get("label", name), "measure": cinfo.get("measure", "area"),
            "color": cinfo.get("color", "#888888"), "conf": round(ins["conf"], 4), "is_box": ins["is_box"],
            "box": [round(v, 1) for v in ins["box"]], "polygons": polys,
        })
    dets.sort(key=lambda d: -d["conf"])
    return {
        "model": active.get("label"), "task": active.get("task"), "image_size": [W, H], "imgsz": sz,
        "tiles": n_tiles, "device": dev, "time_ms": int((time.time() - t0) * 1000), "detections": dets,
    }


# ---------------------------------------------------------------- static (หน้าเว็บ)
# เสิร์ฟเฉพาะไฟล์หน้าเว็บ (ไม่เปิดให้เข้าถึงโฟลเดอร์ server/ หรือ workspace/)
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


def _static_route(fname: str):
    def handler():  # ไม่มีพารามิเตอร์ -> ผู้ใช้ระบุไฟล์อื่นผ่าน query ไม่ได้
        return FileResponse(WEB / fname)
    return handler


for _f in ("app.js", "library.js", "changelog.js", "favicon.svg"):
    app.add_api_route(f"/{_f}", _static_route(_f), methods=["GET"], include_in_schema=False)
app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")
(WEB / "assets").mkdir(exist_ok=True)
app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์: {exc}"})


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=CONFIG.get("host", "127.0.0.1"), help="ใช้ 0.0.0.0 ถ้าต้องการให้เครื่องอื่นในวงแลนเข้าถึง (ไม่แนะนำ)")
    ap.add_argument("--port", type=int, default=int(CONFIG.get("port", 8765)))
    a = ap.parse_args()
    print(f"\n  Crack-AI v{APP_VERSION} พร้อมใช้งาน: http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}")
    print(f"  workspace: {WORK}")
    print(f"  เว็บที่อนุญาตเพิ่มเติม: {', '.join(CONFIG['allowed_origins']) or '(ไม่มี: แก้ได้ที่ server/config.json)'}\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
