"""
VectorOps - Shared data models
"""
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import Optional, List, Literal, Dict, Any


class NodeTelemetry(BaseModel):
    node_id: str
    cluster: str                 # "A" or "B"
    gpu_model: str
    vram_total_gb: float
    vram_used_gb: float
    vram_util_pct: float         # 0-100
    gpu_core_util_pct: float     # 0-100
    temp_c: float
    power_draw_w: float
    active_jobs: int
    idle_minutes: float          # consecutive minutes under 10% util
    status: Literal["healthy", "warning", "unsafe"] = "healthy"
    fail_risk_pct: Optional[float] = None
    minutes_to_bottleneck: Optional[float] = None
    last_migrated_job: Optional[str] = None
    wait_time_s: float = 0.0     # for scheduler fairness/aging


class Job(BaseModel):
    job_id: str
    vram_required_gb: float
    est_duration_min: int
    current_node: Optional[str] = None
    priority: int = 1


class ScheduleDecision(BaseModel):
    job_id: str
    from_node: Optional[str]
    to_node: str
    reason: str
    score_breakdown: dict
    migrated: bool


class ChaosRequest(BaseModel):
    cluster: str
    intensity: float = 1.0   # multiplier on temp/util spike


class CopilotQuery(BaseModel):
    question: str


class QueueJobSubmit(BaseModel):
    name: str
    vram_gb: float
    priority: Literal["High", "Medium", "Low"] = "Medium"
    user: str = "User-01"


class QueueJobItem(BaseModel):
    job_id: str
    name: str
    user: str
    vram_gb: float
    priority: str
    status: Literal["Queued", "Running", "Completed", "Cancelled"] = "Queued"
    submitted_at: str
    assigned_node: Optional[str] = None


class NotificationItem(BaseModel):
    id: str
    timestamp: str
    title: str
    message: str
    type: Literal["info", "warning", "danger", "success"] = "info"
    read: bool = False


class SupportMessage(BaseModel):
    id: str
    sender: Literal["user", "support"]
    text: str
    timestamp: str


class SupportQuery(BaseModel):
    message: str


class AgentQuery(BaseModel):
    prompt: str
    api_key: Optional[str] = None


class FeedbackSubmission(BaseModel):
    user_name: str
    email: Optional[str] = None
    category: Literal["bug", "feature", "ui", "general"] = "general"
    rating: int = 5
    comments: str


class FeedbackItem(FeedbackSubmission):
    id: str
    created_at: str


class UserAllocation(BaseModel):
    username: str
    tier: str
    allocated_vram_gb: float
    used_vram_gb: float
    max_jobs: int
    active_jobs: int
    allocated_nodes: List[str]
