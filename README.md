# Local Embedding Service

Production-ready local embedding API powered by [vLLM](https://docs.vllm.ai/) (`BAAI/bge-m3`) and a FastAPI proxy. All inference runs inside Docker on your GPU — no cloud APIs.

## Architecture

```
Client  →  FastAPI (:8080)  →  vLLM (:8000, /v1/embeddings)  →  GPU
```

- **POST /embed** — single text → embedding vector
- **POST /embed/batch** — multiple texts → list of vectors (one vLLM call)
- **GET /health** — API + vLLM upstream status

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

```bash
git clone https://github.com/EgorLozh/local_embedding.git
cd local_embedding

cp .env.example .env
# Edit .env if needed (GPU memory, HF_TOKEN, ports)

docker compose up -d --build
```

First startup downloads **BAAI/bge-m3** (~2 GB) into the HuggingFace cache volume. vLLM healthcheck allows up to 5 minutes (`start_period: 300s`).

Check status:

```bash
docker compose ps
docker compose logs -f vllm
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
| `VLLM_GPU_MEMORY_UTILIZATION` | Fraction of GPU VRAM for vLLM | `0.5` |
| `VLLM_MAX_MODEL_LEN` | Max sequence length | `8192` |
| `VLLM_MAX_NUM_SEQS` | Max concurrent sequences | `256` |
| `API_PORT` | Host port for FastAPI | `8080` |
| `HTTP_TIMEOUT_SEC` | Upstream request timeout | `30` |
| `RETRY_MAX_ATTEMPTS` | Retries on transient vLLM errors | `3` |
| `BATCH_MAX_TEXTS` | Max texts per batch request | `64` |
| `HF_TOKEN` | HuggingFace token (optional) | — |
| `HF_CACHE_DIR` | Host HuggingFace cache mount | `~/.cache/huggingface` |

## Error handling

| HTTP | Cause |
|------|--------|
| `422` | Empty or whitespace-only text; batch too large |
| `502` | Invalid response from vLLM |
| `503` | vLLM unreachable or model error |
| `504` | vLLM request timeout |

## Low-latency notes

- FastAPI uses a shared `httpx.AsyncClient` with HTTP keep-alive to vLLM.
- Batch requests send one upstream call with `input: ["...", "..."]`.
- Tune `VLLM_MAX_NUM_SEQS` and `VLLM_GPU_MEMORY_UTILIZATION` for your GPU.
- vLLM container uses `ipc: host` as recommended by vLLM docs.

## Troubleshooting

**vLLM stays unhealthy**

- Check logs: `docker compose logs vllm`
- Ensure GPU is visible inside the container.
- Lower `VLLM_GPU_MEMORY_UTILIZATION` if OOM occurs.
- First model download can take several minutes.

**API returns `503` / degraded health**

- Wait until vLLM finishes loading (`/health` on port 8000 inside the stack).
- Confirm `VLLM_BASE_URL=http://vllm:8000` in `.env`.

**CUDA / driver mismatch**

- Use a pinned `VLLM_IMAGE_TAG` compatible with your driver, or set `VLLM_ENABLE_CUDA_COMPATIBILITY=1` on the vLLM service if supported.

## Development (local, without Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start vLLM separately, then:
export VLLM_BASE_URL=http://localhost:8000
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## License

MIT (adjust as needed).
