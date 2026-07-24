"""
VectorOps - OpenRouter AI Agent Integration
"""
import os
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional
from app.models import NodeTelemetry

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

SYSTEM_PROMPT = """You are VectorOps Assistant, an AI agent specialized STRICTLY in GPU cluster telemetry, workload scheduling, VRAM allocation, thermal optimization, and VectorOps platform usage.

CRITICAL SECURITY & TOKEN-SAVING DIRECTIVE:
- You ONLY answer questions related to VectorOps, GPU cluster telemetry, cluster nodes, VRAM optimization, failure prediction, job scheduling, and IT support for cluster workloads.
- If the user asks ANY question NOT related to GPU clusters, VectorOps, server performance, machine learning hardware, or job scheduling (for example: general coding, recipes, sports, creative writing, history, or general trivia), you MUST decline politely with the exact following message domain notice:
"I am a specialized VectorOps Cluster Agent trained strictly to assist with GPU telemetry, VRAM scheduling, node diagnostics, and cluster operations. Please ask questions related to VectorOps cluster management to conserve token allocation."
- Keep your answers concise, professional, and directly actionable (2-4 sentences max). Cite node metrics or scoring rules when relevant."""


def _domain_guard_check(prompt: str) -> bool:
    """Pre-check prompt keywords to reject obvious off-topic prompts before calling OpenRouter to save tokens."""
    prompt_lower = prompt.lower()
    allowed_keywords = [
        "gpu", "vram", "cluster", "node", "scheduler", "schedule", "temp", "thermal",
        "power", "util", "vectorops", "job", "queue", "workload", "fail", "bottleneck",
        "cost", "chaos", "evacuate", "migrate", "allocation", "quota", "memory", "cuda",
        "oom", "support", "help", "a-0", "b-0", "c-0", "d-0", "e-0", "f-0", "g-0",
        "rtx", "4090", "3080", "a100", "h100", "l4", "t4", "ada", "slurm", "k8s"
    ]
    return any(k in prompt_lower for k in allowed_keywords)


def _offline_fallback(prompt: str, nodes: Optional[List[NodeTelemetry]] = None) -> str:
    """Intelligent domain-guarded offline fallback."""
    if not _domain_guard_check(prompt):
        return (
            "I am a specialized VectorOps Cluster Agent trained strictly to assist with "
            "GPU telemetry, VRAM scheduling, node diagnostics, and cluster operations. "
            "Please ask questions related to VectorOps cluster management to conserve token allocation."
        )
    
    prompt_lower = prompt.lower()
    if "temp" in prompt_lower or "hot" in prompt_lower or "thermal" in prompt_lower:
        return "Cluster nodes operating above 85°C are flagged as unsafe by the scheduler. Check active nodes for thermal throttling or trigger an evacuation in the Work queue."
    elif "vram" in prompt_lower or "memory" in prompt_lower or "allocat" in prompt_lower:
        return "VRAM headroom is weighted at 40% in the placement scoring algorithm. High-memory jobs (>40GB) are automatically routed to Cluster C (A100) or Cluster E (H100) nodes."
    elif "queue" in prompt_lower or "job" in prompt_lower:
        return "Jobs in the Work queue are evaluated based on required VRAM and priority. Migration costs are penalized to avoid thrashing jobs for marginal score gains."
    else:
        return "VectorOps is currently monitoring 28 nodes across 7 GPU clusters (Clusters A through G). You can inspect live telemetry on the Home dashboard or manage pending workloads in the Work tab."


def ask_agent(prompt: str, api_key: Optional[str] = None, nodes: Optional[List[NodeTelemetry]] = None) -> Dict:
    # 1. Check local pre-guard filter
    if not _domain_guard_check(prompt):
        return {
            "answer": (
                "I am a specialized VectorOps Cluster Agent trained strictly to assist with "
                "GPU telemetry, VRAM scheduling, node diagnostics, and cluster operations. "
                "Please ask questions related to VectorOps cluster management to conserve token allocation."
            ),
            "source": "domain_guard",
            "guarded": True
        }

    key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()

    # 2. If no API key configured, use fallback
    if not key:
        return {
            "answer": _offline_fallback(prompt, nodes),
            "source": "offline_fallback",
            "guarded": False
        }

    # 3. Call OpenRouter API
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://vectorops.app",
        "X-Title": "VectorOps Agent"
    }

    payload = json.dumps({
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 250
    }).encode("utf-8")

    req = urllib.request.Request(OPENROUTER_API_URL, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            answer = data["choices"][0]["message"]["content"].strip()
            return {"answer": answer, "source": "openrouter", "model": DEFAULT_MODEL, "guarded": False}
    except Exception as e:
        return {
            "answer": _offline_fallback(prompt, nodes),
            "source": f"fallback ({type(e).__name__})",
            "guarded": False
        }
