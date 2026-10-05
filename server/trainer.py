"""Worker ที่ server.py เรียกเป็น process แยก เพื่อเทรน YOLO และเขียนความคืบหน้าเป็นไฟล์ JSON"""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path


def write_json(p: Path, data) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def clean(v):
    try:
        v = float(v)
        return None if math.isnan(v) or math.isinf(v) else round(v, 5)
    except Exception:
        return None


def main() -> int:
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    run_dir = cfg_path.parent
    status_p, prog_p = run_dir / "status.json", run_dir / "progress.json"
    started = time.time()
    write_json(status_p, {"state": "running", "started": started, "message": "กำลังเตรียมโมเดล"})
    history = []

    try:
        from ultralytics import YOLO

        model = YOLO(cfg["model"])

        def on_fit_epoch_end(trainer):
            ep = trainer.epoch + 1
            final = ep > trainer.epochs  # ultralytics เรียกซ้ำอีกครั้งหลัง validate best.pt ตอนจบ
            row = {"epoch": min(ep, trainer.epochs), "time": round(time.time() - started, 1)}
            try:
                if trainer.tloss is not None:
                    for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
                        row[k] = clean(v)
            except Exception:
                pass
            for k, v in (trainer.metrics or {}).items():
                row[k.replace("metrics/", "")] = clean(v)
            row["fitness"] = clean(getattr(trainer, "fitness", None) or 0)
            try:
                row["lr"] = clean(list(trainer.lr.values())[0])
            except Exception:
                pass
            data = {"epoch": row["epoch"], "epochs": trainer.epochs, "history": history}
            if final:
                data["final"] = {k: v for k, v in row.items() if not k.startswith("train/")}
            else:
                if history and history[-1]["epoch"] == row["epoch"]:
                    history[-1] = row  # epoch เดิมถูก redo (เช่น NaN/OOM recovery)
                else:
                    history.append(row)
                write_json(status_p, {"state": "running", "started": started,
                                      "message": f"epoch {row['epoch']}/{trainer.epochs}"})
            write_json(prog_p, data)

        def on_train_start(trainer):
            write_json(prog_p, {"epoch": 0, "epochs": trainer.epochs, "history": []})
            write_json(status_p, {"state": "running", "started": started, "message": "เริ่มเทรนแล้ว"})

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(
            data=cfg["data"], project=cfg["project"], name=cfg["name"], exist_ok=True,
            task=cfg["task"], verbose=True, **cfg["hyp"],
        )
        best = run_dir / "weights" / "best.pt"
        write_json(status_p, {"state": "done", "started": started, "finished": time.time(),
                              "message": "เทรนเสร็จสมบูรณ์", "best": str(best) if best.exists() else None})
        return 0
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        msg = str(e)
        hint = ""
        low = msg.lower()
        if "out of memory" in low:
            hint = " (หน่วยความจำ GPU ไม่พอ: ลด batch หรือ imgsz หรือเลือกรุ่นเล็กลง)"
        elif "no kernel image" in low or "sm_120" in low:
            hint = " (PyTorch ไม่รองรับการ์ดจอรุ่นนี้: ติดตั้ง PyTorch ที่รองรับ CUDA 12.8 ขึ้นไป ดู README)"
        elif "download" in low or "connection" in low:
            hint = " (ดาวน์โหลดโมเดลตั้งต้นไม่ได้: ตรวจอินเทอร์เน็ต หรือวางไฟล์ .pt ไว้ใน workspace/weights)"
        write_json(status_p, {"state": "failed", "started": started, "finished": time.time(),
                              "message": "เทรนไม่สำเร็จ", "error": msg[-800:] + hint})
        return 1


if __name__ == "__main__":
    sys.exit(main())
