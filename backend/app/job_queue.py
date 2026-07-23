"""
VectorOps - Job Queue Manager
"""
import uuid
import datetime
from typing import List, Optional
from app.models import QueueJobItem, QueueJobSubmit

INITIAL_JOBS: List[QueueJobItem] = [
    QueueJobItem(
        job_id="job-8912",
        name="llama3-8b-finetune",
        user="Alex (NLP Lab)",
        vram_gb=16.0,
        priority="High",
        status="Running",
        submitted_at="10:14 AM",
        assigned_node="A-01"
    ),
    QueueJobItem(
        job_id="job-7431",
        name="resnet50-eval-batch",
        user="Sarah (Vision)",
        vram_gb=8.0,
        priority="Medium",
        status="Running",
        submitted_at="10:22 AM",
        assigned_node="B-03"
    ),
    QueueJobItem(
        job_id="job-5542",
        name="whisper-transcribe-v2",
        user="Audio-Team",
        vram_gb=6.0,
        priority="Low",
        status="Queued",
        submitted_at="10:30 AM",
        assigned_node=None
    ),
    QueueJobItem(
        job_id="job-3319",
        name="diffusion-checkpoint-eval",
        user="GenAI-Lab",
        vram_gb=12.0,
        priority="High",
        status="Queued",
        submitted_at="10:35 AM",
        assigned_node=None
    ),
]


class JobQueueManager:
    def __init__ (self):
        self.jobs: List[QueueJobItem] = list(INITIAL_JOBS)

    def list_jobs(self) -> List[QueueJobItem]:
        return self.jobs

    def submit_job(self, req: QueueJobSubmit) -> QueueJobItem:
        now_str = datetime.datetime.now().strftime("%I:%M %p")
        job = QueueJobItem(
            job_id=f"job-{uuid.uuid4().hex[:4]}",
            name=req.name,
            user=req.user,
            vram_gb=req.vram_gb,
            priority=req.priority,
            status="Queued",
            submitted_at=now_str,
            assigned_node=None
        )
        self.jobs.insert(0, job)
        return job

    def cancel_job(self, job_id: str) -> Optional[QueueJobItem]:
        for job in self.jobs:
            if job.job_id == job_id:
                job.status = "Cancelled"
                return job
        return None

    def delete_job(self, job_id: str) -> bool:
        for i, job in enumerate(self.jobs):
            if job.job_id == job_id:
                self.jobs.pop(i)
                return True
        return False


queue_manager = JobQueueManager()
