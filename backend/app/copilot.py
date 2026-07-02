"""
VectorOps - AI Cluster Copilot

Calls a local LLM via Ollama (default: llama3) to explain scheduling
decisions in natural language. Instead of a canned template, the live
cluster state (node telemetry + last scheduling decisions) is serialized
and injected as grounding context, so questions like "why didn't job X
move to Cluster B?" get answered with the *actual* current scores and
thresholds rather than a generic explanation.

Requires Ollama running locally: `ollama serve` + `ollama pull llama3`.
If Ollama isn't reachable, falls back to a deterministic templated
explanation so the rest of the demo still works offline.
"""
import json
import urllib.request
import urllib.error
from typing import List, Dict, Optional
from app.models import NodeTelemetry, ScheduleDecision
from app import scheduler as sched_mod

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

SYSTEM_PROMPT = """You are VectorOps Copilot, an AI assistant embedded in a GPU cluster \
scheduler. You are given the CURRENT live telemetry for every node and the scheduler's \
recent decisions as JSON context. Answer the user's question using ONLY that data -- cite \
specific node IDs, temperatures, utilization %, and scores. Keep answers to 3-5 sentences, \
plain language, no markdown headers. If asked "why" about a decision, explain it in terms \
of the weighted scoring formula (VRAM headroom, thermal headroom, compute headroom, \
migration cost, fairness aging) and the actual numbers involved."""


def _build_context(nodes: List[NodeTelemetry], recent_decisions: List[ScheduleDecision]) -> str:
    node_summary = [
        {
            "node_id": n.node_id, "cluster": n.cluster, "status": n.status,
            "temp_c": n.temp_c, "gpu_util_pct": n.gpu_core_util_pct,
            "vram_util_pct": n.vram_util_pct, "fail_risk_pct": n.fail_risk_pct,
            "minutes_to_bottleneck": n.minutes_to_bottleneck,
            "score": sched_mod.node_score(n) if sched_mod.is_safe(n) else "FILTERED_UNSAFE",
        }
        for n in nodes
    ]
    decisions_summary = [d.dict() for d in recent_decisions[-5:]]
    return json.dumps({"nodes": node_summary, "recent_decisions": decisions_summary}, indent=None)


def _fallback_answer(question: str, nodes: List[NodeTelemetry], recent_decisions: List[ScheduleDecision]) -> str:
    """Deterministic offline fallback if Ollama isn't running."""
    if recent_decisions:
        d = recent_decisions[-1]
        return (
            f"[offline mode -- Ollama not reachable] Most recent decision: {d.reason}"
        )
    hottest = max(nodes, key=lambda n: n.temp_c)
    idlest = min(nodes, key=lambda n: n.gpu_core_util_pct)
    return (
        f"[offline mode -- Ollama not reachable] Right now {hottest.node_id} is hottest at "
        f"{hottest.temp_c}C / {hottest.gpu_core_util_pct}% util, while {idlest.node_id} is "
        f"most idle at {idlest.gpu_core_util_pct}% util. Start `ollama serve` for full "
        f"natural-language answers."
    )


def ask_copilot(
    question: str,
    nodes: List[NodeTelemetry],
    recent_decisions: Optional[List[ScheduleDecision]] = None,
    model: str = OLLAMA_MODEL,
) -> Dict:
    recent_decisions = recent_decisions or []
    context = _build_context(nodes, recent_decisions)
    prompt = f"{SYSTEM_PROMPT}\n\nCURRENT CLUSTER STATE:\n{context}\n\nQUESTION: {question}\n\nANSWER:"

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return {"answer": body.get("response", "").strip(), "source": "ollama", "model": model}
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return {"answer": _fallback_answer(question, nodes, recent_decisions), "source": "fallback", "model": None}
