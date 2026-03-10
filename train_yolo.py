import os
import sys
from pathlib import Path
from typing import Optional


def _load_ultralytics() -> "YOLO":  # type: ignore[name-defined]
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[train_yolo] Failed to import ultralytics: {exc}", file=sys.stderr)
        print("[train_yolo] Make sure the Ultralytics package is installed in the container.", file=sys.stderr)
        raise
    return YOLO  # type: ignore[return-value]


def _resolve_model_name(model_name: str) -> str:
    """
    Resolve the requested model name.

    Currently this is a simple passthrough: configuration should provide
    either a valid Ultralytics model alias (e.g., 'yolo11n') or an absolute
    path to a .pt weights file that exists inside the container.
    """
    return model_name


def train() -> None:
    YOLO = _load_ultralytics()

    model_name = os.getenv("MODEL_NAME", "yolo9-t")
    imgsz = int(os.getenv("IMGSZ", "640"))
    batch = int(os.getenv("BATCH", "16"))
    epochs = int(os.getenv("EPOCHS", "100"))
    workers = int(os.getenv("WORKERS", "8"))

    data_yaml = os.getenv("DATA_YAML_PATH", "/workspace/data/dataset/data.yaml")
    project = os.getenv("PROJECT_DIR", "/workspace/runs")
    run_name = os.getenv("RUN_NAME", "run")

    model_name_fallback: Optional[str] = os.getenv("MODEL_NAME_FALLBACK") or None

    resolved_model_name = _resolve_model_name(model_name)

    print(f"[train_yolo] Requested model: {model_name} -> using: {resolved_model_name}")
    print(f"[train_yolo] Data config: {data_yaml}")
    print(f"[train_yolo] Project dir: {project}, Run name: {run_name}")

    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"[train_yolo] data.yaml not found at {data_path}", file=sys.stderr)
        sys.exit(1)

    def _run_training(name: str) -> None:
        print(f"[train_yolo] Starting training with model '{name}'")
        model = YOLO(name)
        model.train(
            data=str(data_path),
            imgsz=imgsz,
            batch=batch,
            epochs=epochs,
            workers=workers,
            project=project,
            name=run_name,
        )

    try:
        _run_training(resolved_model_name)
    except Exception as exc:
        if model_name_fallback:
            print(f"[train_yolo] Primary model '{model_name}' failed: {exc}", file=sys.stderr)
            print(f"[train_yolo] Attempting fallback model '{model_name_fallback}'")
            _run_training(model_name_fallback)
        else:
            print(
                "[train_yolo] Training failed and no MODEL_NAME_FALLBACK provided. "
                "Consider setting MODEL_NAME_FALLBACK to a known-good Ultralytics model alias "
                "(e.g., 'yolo11n', 'yolo11s').",
                file=sys.stderr,
            )
            raise


if __name__ == "__main__":
    train()

