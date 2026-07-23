"""
VectorOps - FastAPI Backend
"""
import asyncio
import uuid
import os
import datetime
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
from app.job_queue import queue_manager
from app.notifications import notif_engine
from app.agent import ask_agent
from app.feedback import feedback_store
from app.models import (
    ScheduleDecision, ChaosRequest, CopilotQuery, Job,
    QueueJobSubmit, QueueJobItem, NotificationItem, SupportQuery, SupportMessage,
    AgentQuery, FeedbackSubmission, FeedbackItem, UserAllocation
)

recent_decisions: List[ScheduleDecision] = []
MAX_DECISION_LOG = 50

support_messages: List[SupportMessage] = [
    SupportMessage(id="msg-1", sender="support", text="Hello! Welcome to VectorOps IT Support. How can we assist with your GPU allocation or workload today?", timestamp="09:00 AM"),
]

failure_model.score_nodes(engine.snapshot())


async def background_tick_loop():
    while True:
        engine.tick()
        nodes = engine.snapshot()
        failure_model.score_nodes(nodes)
        notif_engine.observe_nodes(nodes)
        await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_tick_loop())
    yield
    task.cancel()


app = FastAPI(title="VectorOps API", version="0.2.0", lifespan=lifespan)

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


# --- NEW WORK / QUEUE ENDPOINTS ---

@app.get("/queue", response_model=List[QueueJobItem])
def get_queue():
    return queue_manager.list_jobs()


@app.post("/queue/submit", response_model=QueueJobItem)
def submit_queue_job(req: QueueJobSubmit):
    return queue_manager.submit_job(req)


@app.post("/queue/cancel/{job_id}", response_model=QueueJobItem)
def cancel_queue_job(job_id: str):
    res = queue_manager.cancel_job(job_id)
    if not res:
        raise HTTPException(status_code=404, detail="Job not found")
    return res


@app.delete("/queue/{job_id}")
def delete_queue_job(job_id: str):
    success = queue_manager.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "job_id": job_id}


# --- NEW NOTIFICATIONS ENDPOINTS ---

@app.get("/notifications", response_model=List[NotificationItem])
def get_notifications():
    return notif_engine.get_notifications()


@app.post("/notifications/read")
def mark_notifications_read(notif_id: Optional[str] = None):
    notif_engine.mark_read(notif_id)
    return {"status": "success"}


# --- NEW SUPPORT IT ENDPOINTS ---

@app.get("/support/messages", response_model=List[SupportMessage])
def get_support_messages():
    return support_messages


@app.post("/support/chat", response_model=SupportMessage)
def post_support_message(q: SupportQuery):
    now_str = datetime.datetime.now().strftime("%I:%M %p")
    user_msg = SupportMessage(id=f"msg-{uuid.uuid4().hex[:4]}", sender="user", text=q.message, timestamp=now_str)
    support_messages.append(user_msg)
    
    # Auto reply simulation
    reply_text = "Thank you for reaching out to VectorOps IT Support! An engineer has been notified of your inquiry regarding GPU cluster resources."
    if "vram" in q.message.lower() or "memory" in q.message.lower():
        reply_text = "IT Support Notice: VRAM allocation requests over 24GB require approval from your lab director. We have logged your request."
    elif "reset" in q.message.lower() or "node" in q.message.lower():
        reply_text = "IT Support Notice: Node health check initiated. If a node is stuck in 'unsafe' state, emergency evacuation will be executed automatically."
    
    reply_msg = SupportMessage(id=f"msg-{uuid.uuid4().hex[:4]}", sender="support", text=reply_text, timestamp=now_str)
    support_messages.append(reply_msg)
    return reply_msg


# --- NEW USER TRAINING ALLOCATION ENDPOINTS ---

@app.get("/user/allocation", response_model=UserAllocation)
def get_user_allocation():
    nodes = engine.snapshot()
    used_vram = sum(n.vram_used_gb for n in nodes if n.cluster == "A") * 0.35
    return UserAllocation(
        username="Demo User (Lab Tier)",
        tier="Researcher / Lab Tier 1",
        allocated_vram_gb=48.0,
        used_vram_gb=round(used_vram, 1),
        max_jobs=5,
        active_jobs=2,
        allocated_nodes=["A-01", "A-02", "B-01"]
    )


# --- NEW OPENROUTER AGENT ENDPOINTS ---

@app.post("/agent/chat")
def chat_agent(req: AgentQuery):
    nodes = engine.snapshot()
    return ask_agent(req.prompt, req.api_key, nodes)


# --- NEW FEEDBACK ENDPOINTS ---

@app.get("/feedback", response_model=List[FeedbackItem])
def get_feedback():
    return feedback_store.get_all()


@app.post("/feedback", response_model=FeedbackItem)
def submit_feedback(req: FeedbackSubmission):
    return feedback_store.add_feedback(req)
