# VectorOps — Decentralized AI Cluster Intelligence

Energy-efficient, thermal/VRAM-aware scheduling for fragmented GPU clusters.
Built for [Hackathon: AI Innovation — Track: AI for Cluster Intelligence].

## What's running

- **Backend** (`backend/`): FastAPI service with a live telemetry simulator
  (2 clusters, 12 nodes, realistic diurnal + bursty load patterns), a
  weighted-scoring scheduler with migration-cost and fairness-aging terms,
  a Random Forest failure predictor (classifier + time-to-bottleneck
  regressor), a cost optimizer, and an AI copilot bridge to a local LLM.
- **Frontend** (`frontend/dashboard.html`): single-file React dashboard
  (no build step — open directly in a browser). Animated VRAM/temp gauges
  per node, a chaos-injection button per cluster, a live scheduler decision
  log, cost report, and a copilot chat panel.

## Quick start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt

# generate synthetic training data + train the failure prediction models
python data/generate_dataset.py
python data/train_model.py

# start the API (also boots the live telemetry sim loop)
uvicorn app.main:app --reload --port 8000
```

Verify it's up: `curl http://localhost:8000/health`

### 2. Frontend

Just open `frontend/dashboard.html` directly in a browser (double-click it,
or `open frontend/dashboard.html`). It polls `http://localhost:8000` every
2 seconds — no npm, no build tooling.

### 3. (Optional) AI Copilot — local LLM

The copilot works in an offline fallback mode by default. For full natural
language explanations:

```bash
# install Ollama: https://ollama.com
ollama serve
ollama pull llama3
```

Once Ollama is running on `localhost:11434`, the copilot panel in the
dashboard will automatically start using it — no restart needed.

## API reference

| Method | Endpoint          | Description                                      |
|--------|-------------------|---------------------------------------------------|
| GET    | `/nodes`          | Live telemetry for all 12 nodes                   |
| GET    | `/nodes/{cluster}`| Live telemetry filtered by cluster (A or B)       |
| POST   | `/schedule`       | Run the scheduler for a job, returns the decision |
| GET    | `/decisions`      | Recent scheduling decision log                    |
| GET    | `/cost`           | Cost optimizer report (idle GPUs, $ wasted)       |
| POST   | `/chaos`          | Inject an artificial bottleneck (demo tool)       |
| POST   | `/chaos/clear`    | Clear active chaos on a cluster                   |
| POST   | `/copilot/ask`    | Ask the AI copilot about current cluster state    |
| GET    | `/health`         | Liveness + model-ready check                      |

## Demo script (suggested)

1. Open the dashboard — Cluster A (lab, RTX 4090s) is already running hot
   (~85-90°C, high util) while Cluster B (faculty, RTX 3080s) sits idle.
2. Click **"Schedule Job (hottest node)"** — watch the decision log explain
   *why* it evacuated/placed the job the way it did, citing real scores.
3. Click **"⚡ Inject Chaos"** on Cluster B — watch its nodes light up
   unsafe in real time as temp/util spike, then decay back down over ~30s
   as the simulator's chaos decay kicks in.
4. Ask the copilot: *"Why is A-01 flagged unsafe?"* or *"Why didn't the
   last job go to B-06?"* — it answers using the live JSON state, not a
   canned template.
5. Point to the **Cost Optimizer** panel: idle GPU count converted into
   real $/hr of equivalent cloud compute being wasted.

## Architecture notes

- **No thrashing**: the scheduler subtracts a migration-cost penalty
  (checkpoint + restart overhead, higher for cross-cluster moves) from the
  score delta before deciding to migrate — a small score improvement alone
  won't trigger a move.
- **Fairness/starvation guard**: nodes accumulate a small "aging" bonus the
  longer they wait without a job, so a busy-but-safe node isn't perpetually
  skipped in favor of an idle one.
- **Two-tier failure prediction**: a classifier for binary risk (`P(unsafe
  within 30 min)`) and a regressor for `minutes_to_bottleneck`, so the
  scheduler can act proactively, not just reactively at the 85°C/90% line.
- **Realistic telemetry patterns**: util/temp correlation, diurnal drift,
  and burst arrivals are modeled after characteristics documented in real
  production GPU cluster traces (Alibaba PAI, Microsoft Philly), since raw
  temperature isn't published in those datasets.

## Team

- **User** — System Architect / AI Integrator (this repo)
- **Nafis** — Full-Stack (FastAPI + React integration)
- **Riti** — ML (failure prediction model)
- **Komol** — Mentor
