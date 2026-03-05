# OpenPI Policy Server Setup on RunPod

## 1. Create a RunPod Pod

- **GPU**: A100 80GB (recommended) or A40 48GB (budget)
- **Template**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Container disk**: 20GB
- **Volume disk**: 50GB (mounted at `/workspace`)
- **Expose TCP port**: 8000

## 2. SSH into the Pod

```bash
ssh root@<POD_IP> -p <SSH_PORT>
```

Or use the RunPod web terminal.

## 3. Install OpenPI

```bash
cd /workspace

git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Install with uv (recommended by openpi)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

uv sync
```

## 4. Clone This Repo

```bash
cd /workspace
git clone https://github.com/Hyperion-Net/piper_isaac_sim.git
cd piper_isaac_sim
```

## 5. Install Piper Dependencies (for data transforms)

```bash
cd /workspace/openpi
uv pip install mujoco numpy opencv-python-headless
```

## 6. Download Model Checkpoint

The first time you serve a policy, openpi auto-downloads the checkpoint.
To pre-download:

```bash
cd /workspace/openpi

# For pi0.5 (flow matching, best quality):
uv run python -c "from openpi.models import model_registry; model_registry.get_model('pi0_5')"

# Or for pi0-FAST (autoregressive, faster inference):
uv run python -c "from openpi.models import model_registry; model_registry.get_model('pi0_fast')"
```

## 7. Serve the Policy

```bash
cd /workspace/openpi

# Serve pi0.5 base model
uv run python -m openpi.serving.serve_policy \
    --config pi0_5 \
    --port 8000

# Or serve pi0-FAST for lower latency
uv run python -m openpi.serving.serve_policy \
    --config pi0_fast \
    --port 8000
```

The server will print `Serving on 0.0.0.0:8000` when ready.

## 8. Connect from Your Mac

Get the pod's public IP and exposed port from the RunPod dashboard.
RunPod TCP proxy gives you a URL like `<pod_id>-8000.proxy.runpod.net`.

### Option A: RunPod TCP Proxy (easiest)

```bash
# On your Mac
mjpython sort_blocks.py --mode policy --host <pod_id>-8000.proxy.runpod.net --port 443
```

### Option B: SSH Tunnel

```bash
# Terminal 1: tunnel
ssh -L 8000:localhost:8000 root@<POD_IP> -p <SSH_PORT>

# Terminal 2: run
mjpython sort_blocks.py --mode policy --host localhost --port 8000
```

## 9. Fine-Tuning on Sorting Demonstrations

To fine-tune pi0.5 on the color sorting task:

### Collect demonstrations

```bash
# On your Mac — run scripted sorting and record episodes
mjpython collect_demos.py --episodes 50 --output /path/to/demos
```

### Convert to LeRobot format

```bash
# On RunPod
cd /workspace/openpi
uv run python -m openpi.data.convert_lerobot \
    --input /path/to/demos \
    --output /workspace/piper_sorting_data \
    --task "sort the colored blocks into their matching zones"
```

### Compute normalization stats

```bash
uv run python -m openpi.training.compute_norm_stats \
    --config piper_sorting
```

### Train (LoRA fine-tune)

```bash
uv run python -m openpi.training.train \
    --config piper_sorting \
    --base_model pi0_5 \
    --data_path /workspace/piper_sorting_data \
    --output_dir /workspace/checkpoints/piper_sorting \
    --batch_size 4 \
    --num_steps 10000 \
    --lora_rank 32
```

### Serve the fine-tuned model

```bash
uv run python -m openpi.serving.serve_policy \
    --config pi0_5 \
    --checkpoint /workspace/checkpoints/piper_sorting \
    --port 8000
```

## Troubleshooting

- **OOM on A40**: Reduce batch size to 2 for fine-tuning, or use A100 80GB
- **Websocket timeout**: Check that port 8000 is exposed in RunPod pod settings
- **Slow inference**: pi0-FAST is ~3x faster than pi0.5 for inference
- **Model not found**: Ensure you have internet access on the pod for checkpoint download
