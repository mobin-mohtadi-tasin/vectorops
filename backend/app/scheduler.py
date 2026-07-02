"""
VectorOps - Smart Scheduler

Pipeline:
1. FILTER  - drop nodes that are unsafe (temp > 85C, util > 90%, or predicted
             high failure risk).
2. SCORE   - weighted-matrix score across VRAM headroom, thermal headroom,
             and compute headroom.
3. PENALIZE- subtract a migration-cost term so the scheduler doesn't thrash
             jobs back and forth for marginal gains.
4. AGE     - add a small fairness bonus to nodes/jobs that have been
             waiting, so a perpetually-busy-but-safe node isn't starved.
"""
from typing import List, Optional
from app.models import NodeTelemetry, ScheduleDecision

UNSAFE_TEMP_C = 85.0
UNSAFE_UTIL_PCT = 90.0
UNSAFE_FAIL_RISK_PCT = 70.0

# weighted matrix
W_VRAM = 0.40
W_TEMP = 0.30
W_UTIL = 0.20
W_AGING = 0.10

MIGRATION_BASE_COST = 8.0     # flat overhead (checkpoint + restart), in score points
MIGRATION_GB_COST = 0.15      # extra cost per GB that needs to move
MIN_GAIN_TO_MIGRATE = 6.0     # score improvement must clear this bar to bother moving


def is_safe(node: NodeTelemetry) -> bool:
    if node.temp_c >= UNSAFE_TEMP_C:
        return False
    if node.gpu_core_util_pct >= UNSAFE_UTIL_PCT:
        return False
    if node.fail_risk_pct is not None and node.fail_risk_pct >= UNSAFE_FAIL_RISK_PCT:
        return False
    return True


def node_score(node: NodeTelemetry) -> float:
    """Higher = better placement target. Each component normalized 0-100."""
    vram_headroom = 100 - node.vram_util_pct
    temp_headroom = max(0, 100 - node.temp_c)   # cooler = better
    util_headroom = 100 - node.gpu_core_util_pct
    aging_bonus = min(20, node.wait_time_s / 10.0)  # caps so aging can't dominate

    score = (
        W_VRAM * vram_headroom
        + W_TEMP * temp_headroom
        + W_UTIL * util_headroom
        + W_AGING * aging_bonus
    )
    return round(score, 2)


def migration_cost(job_vram_gb: float, same_cluster: bool) -> float:
    cost = MIGRATION_BASE_COST + MIGRATION_GB_COST * job_vram_gb
    if not same_cluster:
        cost *= 1.4  # cross-cluster network hop costs more
    return round(cost, 2)


def choose_node(
    nodes: List[NodeTelemetry],
    job_vram_gb: float,
    current_node_id: Optional[str] = None,
) -> ScheduleDecision:
    safe_nodes = [n for n in nodes if is_safe(n) and n.vram_total_gb - n.vram_used_gb >= job_vram_gb]

    if not safe_nodes:
        # nothing safe fits -- pick least-bad option and flag it
        fallback = min(nodes, key=lambda n: (n.temp_c, n.gpu_core_util_pct))
        return ScheduleDecision(
            job_id="",
            from_node=current_node_id,
            to_node=fallback.node_id,
            reason="NO SAFE NODE AVAILABLE -- placed on least-loaded node as emergency fallback",
            score_breakdown={},
            migrated=current_node_id != fallback.node_id,
        )

    scored = {n.node_id: node_score(n) for n in safe_nodes}
    current = next((n for n in safe_nodes if n.node_id == current_node_id), None)
    current_was_unsafe = current_node_id is not None and current is None and any(
        n.node_id == current_node_id for n in nodes
    )

    best_id = max(scored, key=scored.get)
    best_score = scored[best_id]

    if current_was_unsafe:
        origin = next(n for n in nodes if n.node_id == current_node_id)
        return ScheduleDecision(
            job_id="",
            from_node=current_node_id,
            to_node=best_id,
            reason=(
                f"Evacuated {current_node_id} (unsafe: {origin.temp_c}C / "
                f"{origin.gpu_core_util_pct}% util) -> {best_id} (score {best_score}), "
                f"no migration-cost threshold applies to unsafe evacuations"
            ),
            score_breakdown=scored,
            migrated=True,
        )

    if current is None:
        # fresh placement, no migration cost to weigh
        return ScheduleDecision(
            job_id="",
            from_node=None,
            to_node=best_id,
            reason=f"New job placed on {best_id} (highest score {best_score}, no current node to compare)",
            score_breakdown=scored,
            migrated=True,
        )

    current_score = scored.get(current_node_id, node_score(current))
    same_cluster = current.cluster == next(n.cluster for n in safe_nodes if n.node_id == best_id)
    cost = migration_cost(job_vram_gb, same_cluster)
    net_gain = (best_score - current_score) - cost

    if best_id == current_node_id or net_gain < MIN_GAIN_TO_MIGRATE:
        return ScheduleDecision(
            job_id="",
            from_node=current_node_id,
            to_node=current_node_id,
            reason=(
                f"Stayed on {current_node_id}: best alternative {best_id} only nets "
                f"{net_gain:.1f} pts after migration cost ({cost:.1f}) -- below threshold "
                f"({MIN_GAIN_TO_MIGRATE})"
            ),
            score_breakdown=scored,
            migrated=False,
        )

    return ScheduleDecision(
        job_id="",
        from_node=current_node_id,
        to_node=best_id,
        reason=(
            f"Migrated {current_node_id} -> {best_id}: net gain {net_gain:.1f} pts "
            f"(score {current_score}->{best_score}, migration cost {cost:.1f})"
        ),
        score_breakdown=scored,
        migrated=True,
    )
