import argparse
import json
import os
import subprocess
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
    # Allow config to turn on public IP exposure for SSH-based artifact download.
    support_public_ip = bool(runpod_cfg.get("support_public_ip", template.get("supportPublicIp", True)))

    # Ports to expose; required for SSH/scp downloads. Default 22/tcp for scp.
    ports = runpod_cfg.get("ssh_ports", template.get("ports", ["22/tcp"]))

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

    if ports:
        body["ports"] = ports

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


def _unwrap_pod(pod: Dict[str, Any]) -> Dict[str, Any]:
    """Handle API response that may nest the pod under a 'pod' key."""
    if "pod" in pod and isinstance(pod["pod"], dict):
        return pod["pod"]
    return pod


def _pod_status(pod: Dict[str, Any]) -> str:
    """Resolve status from pod (API may use status or desiredStatus)."""
    return pod.get("status") or pod.get("desiredStatus") or ""


def _pod_has_ssh(pod: Dict[str, Any]) -> bool:
    """Return True if pod has publicIp and SSH port mapping."""
    ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings") or {}
    port = port_mappings.get("22") or port_mappings.get(22)
    return bool(ip and port)


def wait_for_pod_running(
    api_base: str,
    api_key: str,
    pod_id: str,
    poll_interval: int = 30,
    ssh_ready_timeout: int = 180,
) -> Dict[str, Any]:
    """
    Poll until the pod is RUNNING and has publicIp/portMappings (so we can SSH).
    After seeing RUNNING, keeps polling up to ssh_ready_timeout seconds for network info.
    """
    while True:
        pod = _unwrap_pod(get_pod(api_base, api_key, pod_id))
        status = _pod_status(pod)
        print(f"[orchestrator] Pod {pod_id} status: {status}")

        if status in ("EXITED", "TERMINATED"):
            print(f"[orchestrator] Pod {pod_id} ended before RUNNING: {status}", file=sys.stderr)
            return pod

        if status == "RUNNING":
            if _pod_has_ssh(pod):
                print(f"[orchestrator] Pod {pod_id} is RUNNING with SSH available")
                return pod
            # RUNNING but no SSH yet; wait for network info
            deadline = time.monotonic() + ssh_ready_timeout
            while time.monotonic() < deadline:
                print(f"[orchestrator] Pod RUNNING, waiting for public IP / SSH port (up to {ssh_ready_timeout}s)...")
                time.sleep(min(poll_interval, 15))
                pod = _unwrap_pod(get_pod(api_base, api_key, pod_id))
                if _pod_status(pod) != "RUNNING":
                    break
                if _pod_has_ssh(pod):
                    print(f"[orchestrator] Pod {pod_id} has SSH available")
                    return pod
            # Return current pod even if SSH not ready (caller may fall back to stub)
            return pod

        time.sleep(poll_interval)


def _ssh_test_file(
    public_ip: str, ssh_port: int, ssh_key_path: str, remote_path: str
) -> bool:
    """Return True if remote_path exists on the pod (via SSH)."""
    ssh_cmd = [
        "ssh",
        "-i", ssh_key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-p", str(ssh_port),
        f"root@{public_ip}",
        f"test -f {remote_path!r}",
    ]
    result = subprocess.run(ssh_cmd, capture_output=True)
    return result.returncode == 0


def wait_for_artifacts_ready(
    pod_info: Dict[str, Any],
    output_tar_path: str,
    ssh_key_path: str,
    poll_interval: int = 30,
    timeout_seconds: int = 86400,
) -> bool:
    """
    Poll the pod via SSH until the output tarball exists (entrypoint has finished).
    Return True if the file appeared, False on timeout or missing SSH info.
    """
    public_ip = pod_info.get("publicIp")
    port_mappings = pod_info.get("portMappings") or {}
    ssh_port = port_mappings.get("22") or port_mappings.get(22)
    if not public_ip or not ssh_port:
        print("[orchestrator] Cannot poll for artifacts: no public IP or SSH port", file=sys.stderr)
        return False
    if not Path(ssh_key_path).exists():
        print(f"[orchestrator] SSH key not found at {ssh_key_path}", file=sys.stderr)
        return False

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _ssh_test_file(public_ip, ssh_port, ssh_key_path, output_tar_path):
            print(f"[orchestrator] Artifacts ready at {output_tar_path}")
            return True
        print(f"[orchestrator] Waiting for {output_tar_path} (next check in {poll_interval}s)...")
        time.sleep(poll_interval)
    print("[orchestrator] Timeout waiting for artifacts tarball", file=sys.stderr)
    return False


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


def download_artifacts_via_scp(
    api_base: str,
    api_key: str,
    pod_id: str,
    output_tar_path: str,
    run_name: str,
    ssh_key_path: str,
    pod_info: Dict[str, Any] | None = None,
) -> Path:
    """
    Download the output tarball from the Pod using scp.

    This requires:
      - The Pod to expose SSH on a public IP (supportPublicIp: true, ports including 22/tcp).
      - An SSH private key accessible in the orchestrator container at ssh_key_path.

    If pod_info is provided (e.g. from wait_for_pod_completion), use it for publicIp/portMappings
    so we can scp even after the pod has exited and the API may no longer return them.
    """
    artifacts_root = Path("artifacts") / run_name
    artifacts_root.mkdir(parents=True, exist_ok=True)
    local_tar = artifacts_root / "artifacts.tar.gz"

    if pod_info:
        pod = pod_info
    else:
        pod = _unwrap_pod(get_pod(api_base, api_key, pod_id))
    public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings") or {}
    # portMappings can be {"22": 12345}; key might be string or int
    ssh_port = port_mappings.get("22") or port_mappings.get(22)

    if not public_ip or not ssh_port:
        print(
            "[orchestrator] Pod does not have a public IP or SSH port mapping. "
            "Falling back to manual download instructions.",
            file=sys.stderr,
        )
        return download_artifacts_stub(output_tar_path, run_name)

    if not Path(ssh_key_path).exists():
        print(
            f"[orchestrator] SSH key not found at {ssh_key_path}. "
            "Falling back to manual download instructions.",
            file=sys.stderr,
        )
        return download_artifacts_stub(output_tar_path, run_name)

    scp_cmd = [
        "scp",
        "-i",
        ssh_key_path,
        "-o",
        "StrictHostKeyChecking=no",
        "-P",
        str(ssh_port),
        f"root@{public_ip}:{output_tar_path}",
        str(local_tar),
    ]

    print(f"[orchestrator] Downloading artifacts via scp from {public_ip}:{ssh_port}")
    try:
        subprocess.run(scp_cmd, check=True)
        print(f"[orchestrator] Artifacts downloaded to {local_tar}")
    except subprocess.CalledProcessError as exc:
        print(
            f"[orchestrator] scp failed: {exc}. Falling back to manual instructions.",
            file=sys.stderr,
        )
        return download_artifacts_stub(output_tar_path, run_name)

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

    output_tar_path = runpod_cfg.get("output_tar_path", f"/workspace/output/{run_name}.tar.gz")
    output_tar_path = output_tar_path.replace("${RUN_NAME}", run_name)
    ssh_key_path = runpod_cfg.get("ssh_key_path", "volume/id_rsa_runpod")

    pod_id = None
    try:
        pod_id = create_pod(api_base, api_key, pod_body)
        pod_info = wait_for_pod_running(api_base, api_key, pod_id)

        if _pod_status(pod_info) == "RUNNING":
            if wait_for_artifacts_ready(pod_info, output_tar_path, ssh_key_path):
                download_artifacts_via_scp(
                    api_base, api_key, pod_id, output_tar_path, run_name, ssh_key_path, pod_info=pod_info
                )
        else:
            # Pod ended before RUNNING; try download anyway in case we have connection info
            download_artifacts_via_scp(
                api_base, api_key, pod_id, output_tar_path, run_name, ssh_key_path, pod_info=pod_info
            )
    finally:
        if pod_id and runpod_cfg.get("auto_delete_pod", True):
            delete_pod(api_base, api_key, pod_id)
        elif pod_id:
            print(f"[orchestrator] auto_delete_pod is disabled; pod {pod_id} will remain running.")


if __name__ == "__main__":
    main()

