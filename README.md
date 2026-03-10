# RunPod Ultralytics YOLO Training Helper

This project provides a minimal setup to:

- Launch a GPU pod on RunPod using the Ultralytics Docker image.
- Download a Roboflow YOLO-format dataset inside the pod.
- Train a YOLO model (configured for `yolo9-t` by default).
- Package training results into a tarball.
- Download artifacts back to your local machine.
- Shut down the pod to stop GPU billing.

> NOTE: This repo is designed as a starting point. You will still need to:
> - Create a RunPod account and API key.
> - Potentially tweak GPU type and region names to match what RunPod currently offers.

---

## 1. Prerequisites

- Python 3.9+ on your local machine.
- A **RunPod** account and **API key** with permission to manage pods.
- A **Roboflow** dataset exported in **YOLO format** (YOLOv8/9-style labels).
- A GPU type like **NVIDIA RTX 3090 (24GB)** or similar available in your chosen RunPod region.

Install local Python dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Configure your dataset and training

Copy the example config:

```bash
cp config.example.yaml config.yaml
```

Then edit `config.yaml`:

- **`roboflow_dataset_url`**: Paste your Roboflow YOLO download URL.
- **`model_name`**: The YOLO model alias, e.g. `yolo9-t`. If your Ultralytics install does not support this alias, you can switch to another model such as `yolo11n`/`yolo11s`.
- **`imgsz`, `batch`, `epochs`, `workers`**: Training hyperparameters.
- **`run_name`**: A short name for this training run; used for folder and artifact names.
- **`runpod` section**:
  - `cloud_type`, `region`
  - `gpu_type`, `gpu_count`
  - `container_disk_in_gb`, `volume_in_gb`
  - `image` (defaults to `ultralytics/ultralytics:latest`)
  - `command` (by default: runs `/workspace/entrypoint.sh` in the pod)

---

## 3. Set your RunPod API key

You have two options:

- **Simpler (baked into config):** Edit `volume/config.yaml` and set either:
  - Top-level `runpod_api_key: "YOUR_RUNPOD_API_KEY_HERE"`, or
  - Nested `runpod.api_key: "YOUR_RUNPOD_API_KEY_HERE"`.
- **More secure:** Export your RunPod API key in your shell or `.env`:

  ```bash
  export RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY"
  ```

The orchestration script will use `RUNPOD_API_KEY` if it is set; otherwise it falls back to the key in `config.yaml`.

---

## 4. What gets created on RunPod

The local script will:

1. Load `config.yaml` and `runpod/runpod_pod_template.json`.
2. Merge template fields with config overrides (GPU type, region, environment variables, etc.).
3. Call the RunPod **Create Pod** API to start a pod with the **Ultralytics** image.
4. Pass environment variables into the pod so that `/workspace/entrypoint.sh` can:
   - Download your Roboflow dataset.
   - Run YOLO training.
   - Package results into a tarball (e.g. `/workspace/output/<run_name>.tar.gz`).

You can customize the pod template further in `runpod/runpod_pod_template.json` as needed.

---

## 5. Container-side scripts

Inside the pod, we expect the following helper scripts under `/workspace`:

- `download_dataset.py` – downloads and unpacks the Roboflow dataset.
- `train_yolo.py` – runs training using the Ultralytics CLI or Python API.
- `entrypoint.sh` – orchestrates download + train + packaging.

There are a few ways to ensure these files are available in the pod:

- **Recommended**: Build your own small image based on `ultralytics/ultralytics:latest` that copies in these scripts and use that image in `runpod.runpod_pod_template.json`.
- Alternatively, you can clone this repo from a public Git host as part of the pod `command`, then invoke `entrypoint.sh` from the cloned folder.

The repo includes the script sources so you can copy them into a Dockerfile or another distribution mechanism.

---

## 6. Running a training run end-to-end

Once your config is set and your API key is provided (in env or config):

```bash
python manage_runpod_training.py --run-name my_first_run
```

This will:

1. Create a new RunPod pod with the configured GPU, region, and image.
2. Wait for the pod to become `RUNNING`.
3. Let the pod execute `entrypoint.sh` which:
   - Downloads the dataset from `ROBOFLOW_DATASET_URL`.
   - Runs YOLO training with the configured model and hyperparameters.
   - Writes artifacts under `/workspace/runs` and packages them into `/workspace/output/<run_name>.tar.gz`.
4. Download the tarball to `artifacts/<run_name>/` on your local machine.
5. Optionally delete the pod (based on `runpod.auto_delete_pod` in `config.yaml`).

You can add `--follow-logs` (if implemented) to stream pod logs while it runs.

---

## 7. Getting artifacts back locally

This template does **not** hard-code a single file transfer mechanism, because
RunPod supports several options (runpodctl, SCP/rsync, S3-compatible APIs for
network volumes) and the best choice depends on your environment.

The orchestration script will:

- Expect the pod to create a tarball at something like `/workspace/output/<run_name>.tar.gz`.
- Create a local directory `artifacts/<run_name>/`.
- Print instructions telling you where to place the downloaded tarball locally (for example, `artifacts/<run_name>/artifacts.tar.gz`) and how to extract it.

To actually transfer the file you can, for example:

- Use `runpodctl` from your machine or inside the pod.
- Use `scp` / `rsync` with the pod's public IP and SSH port.
- Configure a network volume and S3-compatible access, then download via standard S3 tooling.

Once the tarball is present locally, extract it:

```bash
tar -xzf artifacts/<run_name>/artifacts.tar.gz -C artifacts/<run_name>/
```

You will then see the same `/workspace/runs/...` structure inside `artifacts/<run_name>/`.

---

## 8. Inspecting results

After a successful run you should see:

- `artifacts/<run_name>/best.pt` (or equivalent YOLO weights).
- Training metrics and plots from Ultralytics (inside the unpacked `runs/train/...` folders).

You can then use Ultralytics locally to run inference, for example:

```bash
pip install ultralytics
yolo task=detect mode=predict model=artifacts/my_first_run/best.pt source=path/to/images
```

---

## 9. Notes and caveats

- The exact GPU type string and region name must match what RunPod currently offers; check the RunPod console or API docs if pod creation fails due to invalid compute options.
- `yolo9-t` is set as a default in the config, but model naming can change across Ultralytics versions. If you hit model-name errors, switch to a supported alias (e.g., `yolo11n` or `yolo11s`) in `config.yaml`.
- This project intentionally keeps the orchestration logic simple. You can extend it with:
  - Experiment tracking.
  - Multiple run queues.
  - Spot/cheaper GPU options once you’re comfortable with the basics.

