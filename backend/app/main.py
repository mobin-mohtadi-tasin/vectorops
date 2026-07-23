"""
VectorOps - FastAPI Backend

Endpoints:
  GET  /nodes                 -> live telemetry for all nodes (with fail risk)
  GET  /nodes/{cluster}       -> live telemetry filtered by cluster (A/B)
  POST /schedule              -> run scheduler for a hypothetical job, returns decision
  GET  /decisions              -> recent scheduling decisions log
  GET  /cost                  -> cost optimizer report
  POST /chaos                 -> inject an artificial bottleneck into a cluster (demo)
  POST /chaos/clear            -> clear active chaos
  POST /copilot/ask           -> ask the AI copilot a question about cluster state
  GET  /health                -> liveness check
"""
import asyncio
import uuid
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Optional
from pydantic import BaseModel

from app.telemetry import engine
from app import scheduler as sched_mod
from app import failure_model
from app import cost_optimizer
from app.copilot import ask_copilot
from app.models import ScheduleDecision, ChaosRequest, CopilotQuery, Job

recent_decisions: List[ScheduleDecision] = []
MAX_DECISION_LOG = 50


failure_model.score_nodes(engine.snapshot())


async def background_tick_loop():
    while True:
        engine.tick()
        failure_model.score_nodes(engine.snapshot())
        await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_tick_loop())
    yield
    task.cancel()


app = FastAPI(title="VectorOps API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_index():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(current_dir, "../../frontend/dashboard.html"))
    return FileResponse(path)


@app.get("/health")
def health():
    return {"status": "ok", "models_ready": failure_model.models_ready(), "tick": engine.tick_count}


@app.get("/nodes")
def get_nodes():
    return engine.snapshot()


@app.get("/nodes/{cluster}")
def get_nodes_by_cluster(cluster: str):
    nodes = [n for n in engine.snapshot() if n.cluster.upper() == cluster.upper()]
    if not nodes:
        raise HTTPException(status_code=404, detail=f"No nodes found for cluster '{cluster}'")
    return nodes


class ScheduleRequest(BaseModel):
    job_vram_gb: float
    current_node_id: Optional[str] = None
    job_id: Optional[str] = None


@app.post("/schedule")
def schedule_job(req: ScheduleRequest):
    nodes = engine.snapshot()
    decision = sched_mod.choose_node(nodes, req.job_vram_gb, req.current_node_id)
    decision.job_id = req.job_id or f"job-{uuid.uuid4().hex[:6]}"

    recent_decisions.append(decision)
    if len(recent_decisions) > MAX_DECISION_LOG:
        recent_decisions.pop(0)

    return decision


@app.get("/decisions")
def get_decisions():
    return recent_decisions[-20:]


@app.get("/cost")
def get_cost_report():
    return cost_optimizer.cluster_cost_report(engine.snapshot())


@app.post("/chaos")
def inject_chaos(req: ChaosRequest):
    if req.cluster.upper() not in ("A", "B"):
        raise HTTPException(status_code=400, detail="cluster must be 'A' or 'B'")
    engine.inject_chaos(req.cluster.upper(), req.intensity)
    return {"status": "chaos injected", "cluster": req.cluster.upper(), "intensity": req.intensity}


@app.post("/chaos/clear")
def clear_chaos(req: ChaosRequest):
    if req.cluster.upper() not in ("A", "B"):
        raise HTTPException(status_code=400, detail="cluster must be 'A' or 'B'")
    engine.clear_chaos(req.cluster.upper())
    return {"status": "chaos cleared", "cluster": req.cluster.upper()}


@app.post("/copilot/ask")
def copilot_ask(query: CopilotQuery):
    nodes = engine.snapshot()
    result = ask_copilot(query.question, nodes, recent_decisions)
    return result
