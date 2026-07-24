"""
VectorOps - Cost Optimizer

Flags GPUs idle (<10% util) for >30 min and converts wasted capacity into
real currency: local electricity cost paid for nothing, vs. what the
equivalent compute would have cost on Kaggle/Colab-style cloud pricing.
"""
from typing import List, Dict
from app.models import NodeTelemetry

ELECTRICITY_RATE_USD_PER_KWH = 0.09     # rough regional avg, adjustable
IDLE_THRESHOLD_PCT = 10.0
IDLE_THRESHOLD_MIN = 30.0

# rough $/GPU-hour equivalents for cloud alternatives, by tier
CLOUD_RATE_USD_PER_GPU_HOUR = {
    "RTX 4090": 0.79,      # ~ comparable Colab Pro+ / Lambda spot tier
    "RTX 3080": 0.45,      # ~ comparable Kaggle/Colab free-to-plus tier
    "NVIDIA A100": 1.89,   # ~ Enterprise A100 SXM 80GB tier
    "NVIDIA L4": 0.55,     # ~ Google Cloud L4 inference tier
    "NVIDIA H100": 3.49,   # ~ Hyperscale H100 SXM5 tier
    "RTX 6000 Ada": 1.25,  # ~ Workstation Ada Generation tier
    "Tesla T4": 0.35,      # ~ Standard T4 cloud tier
    "default": 0.60,
}


def evaluate_node(node: NodeTelemetry) -> Dict:
    is_idle = node.gpu_core_util_pct < IDLE_THRESHOLD_PCT and node.idle_minutes >= IDLE_THRESHOLD_MIN
    cloud_rate = CLOUD_RATE_USD_PER_GPU_HOUR.get(node.gpu_model, CLOUD_RATE_USD_PER_GPU_HOUR["default"])

    if is_idle:
        idle_hours = node.idle_minutes / 60.0
        wasted_kwh = (node.power_draw_w / 1000.0) * idle_hours
        wasted_electricity_usd = wasted_kwh * ELECTRICITY_RATE_USD_PER_KWH
        opportunity_cost_usd = cloud_rate * idle_hours
    else:
        wasted_electricity_usd = 0.0
        opportunity_cost_usd = 0.0

    return {
        "node_id": node.node_id,
        "cluster": node.cluster,
        "is_idle": is_idle,
        "idle_minutes": node.idle_minutes,
        "wasted_electricity_usd": round(wasted_electricity_usd, 4),
        "opportunity_cost_usd_per_hour": round(opportunity_cost_usd, 2) if is_idle else 0.0,
        "cloud_equivalent_rate": cloud_rate,
    }


def cluster_cost_report(nodes: List[NodeTelemetry]) -> Dict:
    rows = [evaluate_node(n) for n in nodes]
    idle_rows = [r for r in rows if r["is_idle"]]

    total_wasted_electricity = round(sum(r["wasted_electricity_usd"] for r in idle_rows), 4)
    total_opportunity_cost = round(sum(r["opportunity_cost_usd_per_hour"] for r in idle_rows), 2)

    by_cluster: Dict[str, Dict] = {}
    for r in rows:
        c = by_cluster.setdefault(r["cluster"], {"idle_nodes": 0, "total_nodes": 0, "opportunity_cost_usd": 0.0})
        c["total_nodes"] += 1
        if r["is_idle"]:
            c["idle_nodes"] += 1
            c["opportunity_cost_usd"] += r["opportunity_cost_usd_per_hour"]
    for c in by_cluster.values():
        c["opportunity_cost_usd"] = round(c["opportunity_cost_usd"], 2)

    return {
        "nodes": rows,
        "idle_node_count": len(idle_rows),
        "total_nodes": len(rows),
        "total_wasted_electricity_usd": total_wasted_electricity,
        "total_cloud_opportunity_cost_usd": total_opportunity_cost,
        "by_cluster": by_cluster,
        "headline": (
            f"{len(idle_rows)} of {len(rows)} GPUs are idle right now. Equivalent cloud "
            f"compute for that idle capacity would cost ~${total_opportunity_cost:.2f}/hr "
            f"-- while local electricity for the idle draw is only ~${total_wasted_electricity:.4f}."
        ),
    }
