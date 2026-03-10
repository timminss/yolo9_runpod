import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests
import yaml
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[orchestrator] Config file not found at {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def build_pod_request_body(config: Dict[str, Any], run_name: str) -> Dict[str, Any]:
    runpod_cfg = config.get("runpod", {})

    template_path = Path("runpod/runpod_pod_template.json")
    if template_path.exists():
        with template_path.open("r") as f:
            template = json.load(f)
    else:
        template = {}

    gpu_type = runpod_cfg.get("gpu_type", "NVIDIA GeForce RTX 3090")
    gpu_count = int(runpod_cfg.get("gpu_count", 1))
    volume_in_gb = int(runpod_cfg.get("volume_in_gb", 150))
    container_disk_in_gb = int(runpod_cfg.get("container_disk_in_gb", 100))
    image_name = runpod_cfg.get("image", template.get("imageName", "ultralytics/ultralytics:latest"))
    cloud_type = runpod_cfg.get("cloud_type", template.get("cloudType", "SECURE"))

    # dockerStartCmd: prefer config over template.
    start_cmd = runpod_cfg.get("command", template.get("command", []))

    pod_name = template.get("name", "ultralytics-yolo-training")
    volume_mount_path = template.get("volumeMountPath", "/workspace")
    support_public_ip = bool(template.get("supportPublicIp", False))

    # Environment variables passed into the pod. These drive the entrypoint and scripts.
    env_vars = {
        "ROBOFLOW_DATASET_URL": config.get("roboflow_dataset_url", ""),
        "RUN_NAME": run_name,
        "MODEL_NAME": config.get("model_name", "yolo9-t"),
        "IMGSZ": str(config.get("imgsz", 640)),
        "BATCH": str(config.get("batch", 16)),
        "EPOCHS": str(config.get("epochs", 100)),
        "WORKERS": str(config.get("workers", 8)),
        "PROJECT_DIR": "/workspace/runs",
        "DATA_ROOT": "/workspace/data",
    }

    # Filter out empty values just in case.
    env_vars = {k: v for k, v in env_vars.items() if v}

    body: Dict[str, Any] = {
        "name": pod_name,
        "cloudType": cloud_type,
        "gpuCount": gpu_count,
        "gpuTypeIds": [gpu_type],
        "volumeInGb": volume_in_gb,
        "containerDiskInGb": container_disk_in_gb,
        "volumeMountPath": volume_mount_path,
        "imageName": image_name,
        "supportPublicIp": support_public_ip,
        "env": env_vars,
    }

    if start_cmd:
        body["dockerStartCmd"] = start_cmd

    return body


def _get_api_base(runpod_cfg: Dict[str, Any]) -> str:
    """
    Build the base URL for the RunPod REST API.

    Current public docs use https://rest.runpod.io/v1 as the base for Pod
    operations, so we default to that unless overridden in config.
    """
    base = runpod_cfg.get("api_base_url", "https://rest.runpod.io")
    version = runpod_cfg.get("api_version", "v1")
    return f"{base.rstrip('/')}/{version}"


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(5))
def create_pod(api_base: str, api_key: str, body: Dict[str, Any]) -> str:
    url = f"{api_base}/pods"
    print(f"[orchestrator] Creating pod via {url}")
    resp = requests.post(url, headers=_auth_headers(api_key), json=body, timeout=30)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Create pod failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    pod_id = data.get("id") or data.get("podId") or data.get("pod", {}).get("id")
    if not pod_id:
        raise RuntimeError(f"Unexpected create pod response format: {data}")
    print(f"[orchestrator] Created pod with id {pod_id}")
    return str(pod_id)


def get_pod(api_base: str, api_key: str, pod_id: str) -> Dict[str, Any]:
    url = f"{api_base}/pods/{pod_id}"
    resp = requests.get(url, headers=_auth_headers(api_key), timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Get pod failed ({resp.status_code}): {resp.text}")
    return resp.json()


def delete_pod(api_base: str, api_key: str, pod_id: str) -> None:
    url = f"{api_base}/pods/{pod_id}"
    resp = requests.delete(url, headers=_auth_headers(api_key), timeout=30)
    if resp.status_code not in (200, 204):
        print(f"[orchestrator] WARNING: Failed to delete pod ({resp.status_code}): {resp.text}", file=sys.stderr)
    else:
        print(f"[orchestrator] Deleted pod {pod_id}")


def wait_for_pod_completion(api_base: str, api_key: str, pod_id: str, poll_interval: int = 30) -> None:
    """
    Poll pod status until it reaches an exited or terminated state.

    This relies on the pod exiting after the entrypoint script completes.
    """
    terminal_statuses = {"EXITED", "TERMINATED"}

    while True:
        pod = get_pod(api_base, api_key, pod_id)
        status = pod.get("status") or pod.get("desiredStatus")
        print(f"[orchestrator] Pod {pod_id} status: {status}")

        if status in terminal_statuses:
            print(f"[orchestrator] Pod {pod_id} reached terminal status: {status}")
            return

        time.sleep(poll_interval)


def download_artifacts_stub(output_tar_path: str, run_name: str) -> Path:
    """
    Placeholder for an automated artifact download mechanism.

    At the time of writing, the recommended approaches for transferring files
    from Pods are:
      - runpodctl (inside the pod)
      - SCP / rsync using the pod's public IP and port mappings
      - S3-compatible APIs for network volumes

    Since these rely on environment-specific setup (SSH keys, network volumes,
    S3 credentials), this helper currently just documents the expected path
    inside the pod and prepares a local directory where you can place the
    downloaded tarball.
    """
    artifacts_root = Path("artifacts") / run_name
    artifacts_root.mkdir(parents=True, exist_ok=True)

    print(
        "[orchestrator] NOTE: Automated file transfer from the pod is environment-specific.\n"
        f"  - Expected tarball in pod: {output_tar_path}\n"
        f"  - Place the downloaded file at: {artifacts_root / 'artifacts.tar.gz'}\n"
        "Once downloaded, you can extract it locally with:\n"
        f"  tar -xzf {artifacts_root / 'artifacts.tar.gz'} -C {artifacts_root}\n"
    )

    return artifacts_root


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Manage a RunPod-based Ultralytics YOLO training run.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config file.")
    parser.add_argument("--run-name", type=str, default=None, help="Override run_name from config.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = args.run_name or config.get("run_name") or "run"

    # API key resolution: environment variable wins, then config values.
    runpod_cfg = config.get("runpod", {})
    api_key = os.getenv("RUNPOD_API_KEY") or runpod_cfg.get("api_key") or config.get("runpod_api_key")
    if not api_key:
        print(
            "[orchestrator] No RunPod API key found. Set RUNPOD_API_KEY in the environment\n"
            "or add either 'runpod_api_key' at the top level or 'runpod.api_key' in your config file.",
            file=sys.stderr,
        )
        sys.exit(1)

    pod_body = build_pod_request_body(config, run_name)
    api_base = _get_api_base(runpod_cfg)

    pod_id = None
    try:
        pod_id = create_pod(api_base, api_key, pod_body)
        wait_for_pod_completion(api_base, api_key, pod_id)

        output_tar_path = runpod_cfg.get("output_tar_path", f"/workspace/output/{run_name}.tar.gz")
        download_artifacts_stub(output_tar_path, run_name)
    finally:
        if pod_id and runpod_cfg.get("auto_delete_pod", True):
            delete_pod(api_base, api_key, pod_id)
        elif pod_id:
            print(f"[orchestrator] auto_delete_pod is disabled; pod {pod_id} will remain running.")


if __name__ == "__main__":
    main()

