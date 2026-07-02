"""
VectorOps - Shared data models
"""
from pydantic import BaseModel
from typing import Optional, List, Literal


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
