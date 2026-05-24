# Local Embedding Service

Production-ready local embedding API powered by [Text Embeddings Inference (TEI)](https://huggingface.co/docs/text-embeddings-inference) (`BAAI/bge-m3`) and a FastAPI proxy. All inference runs inside Docker on your GPU — no cloud APIs.

## Architecture

```
Client  →  FastAPI (:8080)  →  TEI (:80, /v1/embeddings)  →  GPU
```

- **POST /embed** — single text → embedding vector
- **POST /embed/batch** — multiple texts → list of vectors (one upstream call)
- **GET /health** — API + TEI upstream status

## Why TEI instead of vLLM?

For embedding-only workloads, [TEI](https://github.com/huggingface/text-embeddings-inference) is a better fit than vLLM:

| | TEI | vLLM |
|---|-----|------|
| Purpose | Embeddings only | Full LLM inference |
| Docker image | ~1–2 GB | ~8–9 GB |
| Boot time | Seconds | Minutes |
| API | OpenAI `/v1/embeddings` | OpenAI `/v1/embeddings` |

The FastAPI layer is unchanged — both backends speak the same OpenAI-compatible protocol.

## Prerequisites

- Linux server with NVIDIA GPU
- [NVIDIA driver](https://www.nvidia.com/Download/index.aspx)
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify GPU access:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## Quick start

### GPU (Linux server)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```bash
git clone https://github.com/EgorLozh/local_embedding.git
cd local_embedding

cp .env.example .env
# Set TEI_IMAGE_TAG for your GPU (see table below)

docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Verify GPU access before starting:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### CPU (local dev, no GPU)

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

First startup downloads **BAAI/bge-m3** (~570 MB) into the HuggingFace cache volume. TEI healthcheck allows up to 2 minutes (`start_period: 120s`).

Check status:

```bash
docker compose ps
docker compose logs -f tei
docker compose logs -f api
```

## API examples (curl)

### Health

```bash
curl http://localhost:8080/health
```

Response:

```json
{"status": "ok", "vllm": "ok"}
```

### Single embedding

```bash
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "пример текста"}'
```

Response:

```json
{"embedding": [0.012, -0.034, ...]}
```

### Batch embeddings

```bash
curl -X POST http://localhost:8080/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["текст 1", "текст 2"]}'
```

Response:

```json
{"embeddings": [[...], [...]]}
```

## Python client

```bash
pip install httpx
python clients/example_client.py
```

Or set a custom API URL:

```bash
EMBEDDING_API_URL=http://your-server:8080 python clients/example_client.py
```

## Configuration

Copy [`.env.example`](.env.example) to `.env`. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TEI_IMAGE_TAG` | TEI Docker tag for your GPU arch | `cuda-1.9` |
| `EMBEDDING_MODEL` | HuggingFace model ID | `BAAI/bge-m3` |
| `TEI_MAX_BATCH_TOKENS` | Max tokens per batch | `16384` |
| `TEI_MAX_CONCURRENT_REQUESTS` | Max concurrent requests | `512` |
| `API_PORT` | Host port for FastAPI | `8080` |
| `HTTP_TIMEOUT_SEC` | Upstream request timeout | `30` |
| `RETRY_MAX_ATTEMPTS` | Retries on transient upstream errors | `3` |
| `BATCH_MAX_TEXTS` | Max texts per batch request | `64` |
| `HF_TOKEN` | HuggingFace token (optional) | — |
| `HF_CACHE_DIR` | Host HuggingFace cache mount | `~/.cache/huggingface` |

### TEI image tags by GPU

| GPU | Tag |
|-----|-----|
| Universal (default) | `cuda-1.9` |
| RTX 40xx (Ada) | `89-1.9` |
| RTX 30xx / A10 (Ampere 8.6) | `86-1.9` |
| RTX 20xx / T4 (Turing, experimental) | `turing-1.9` |
| A100 (Ampere 8.0) | `1.9` |

See [TEI Docker images](https://github.com/huggingface/text-embeddings-inference#docker-images) for the full list.

## Error handling

| HTTP | Cause |
|------|--------|
| `422` | Empty or whitespace-only text; batch too large |
| `502` | Invalid response from TEI |
| `503` | TEI unreachable or model error |
| `504` | TEI request timeout |

## Low-latency notes

- FastAPI uses a shared `httpx.AsyncClient` with HTTP keep-alive to TEI.
- Batch requests send one upstream call with `input: ["...", "..."]`.
- Tune `TEI_MAX_CONCURRENT_REQUESTS` and `TEI_MAX_BATCH_TOKENS` for your GPU.

## Troubleshooting

**`could not select device driver "nvidia" with capabilities: [[gpu]]`**

Docker cannot access the GPU. Either install the NVIDIA Container Toolkit (for GPU mode), or use CPU mode.

On Ubuntu/Debian (GPU server):

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Then start with GPU overlay
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

On Windows (Docker Desktop): GPU passthrough requires WSL2 backend + recent NVIDIA drivers. For local testing, use CPU mode instead:

```powershell
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

**`no space left on device` during pull**

The old vLLM image is ~8.7 GB. Free disk space before pulling TEI:

```bash
# Remove failed/partial layers and unused images
docker system prune -a

# Check what's using space
docker system df
```

**TEI stays unhealthy**

- Check logs: `docker compose logs tei`
- Ensure GPU is visible inside the container.
- Pick the correct `TEI_IMAGE_TAG` for your GPU architecture.
- First model download can take several minutes.

**API returns `503` / degraded health**

- Wait until TEI finishes loading (`/health` on port 80 inside the stack).
- Confirm `VLLM_BASE_URL=http://tei:80` in `.env`.

**CUDA / driver mismatch**

- Use a pinned `TEI_IMAGE_TAG` matching your GPU compute capability (see table above).

## Development (local, without Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start TEI separately, then:
export VLLM_BASE_URL=http://localhost:8080
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## License

MIT (adjust as needed).
