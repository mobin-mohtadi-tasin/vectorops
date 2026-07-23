"""
VectorOps - Telemetry Simulation Engine

Generates realistic, live-updating GPU cluster telemetry for two clusters
(A = bottlenecked lab cluster, B = idle faculty cluster). Patterns are
informed by known characteristics of real production GPU cluster traces
(Alibaba PAI / Microsoft Philly): bursty job arrivals, diurnal load cycles,
and a strong correlation between core utilization and thermal output.

This acts as the "live" data source for the FastAPI backend. It also
exposes a chaos-injection hook so the dashboard can trigger a bottleneck
on demand for the demo.
"""
import random
import time
import math
from typing import Dict
from app.models import NodeTelemetry

random.seed(42)

CLUSTER_CONFIG = {
    "A": {"n_nodes": 6, "gpu_model": "RTX 4090", "vram_total": 24, "base_util": 78, "base_temp": 68},
    "B": {"n_nodes": 6, "gpu_model": "RTX 3080", "vram_total": 10, "base_util": 22, "base_temp": 48},
}

JOB_NAME_POOL = [
    "resnet50-train", "bert-finetune", "yolov8-infer", "llama3-8b-lora",
    "stable-diffusion-batch", "whisper-transcribe", "gnn-train", "ppo-rl-agent",
]


def _temp_from_util(util: float, ambient: float = 24.0) -> float:
    """Temperature rises with util plus stochastic noise. Mirrors the
    strong util<->temp correlation seen in real GPU telemetry, since raw
    trace datasets rarely log temperature directly."""
    k = 0.55
    noise = random.gauss(0, 1.4)
    return round(ambient + 20 + k * util + noise, 1)


def _power_from_util(util: float, tdp: float = 350.0) -> float:
    idle_draw = tdp * 0.12
    return round(idle_draw + (tdp - idle_draw) * (util / 100.0), 1)


class ClusterSimEngine:
    """Holds live mutable state for every node and advances it each tick."""

    def __init__(self):
        self.nodes: Dict[str, NodeTelemetry] = {}
        self.tick_count = 0
        self.chaos: Dict[str, float] = {"A": 0.0, "B": 0.0}  # active chaos intensity per cluster
        self._init_nodes()

    def _init_nodes(self):
        for cluster, cfg in CLUSTER_CONFIG.items():
            for i in range(cfg["n_nodes"]):
                node_id = f"{cluster}-{i+1:02d}"
                util = max(0, min(99, random.gauss(cfg["base_util"], 8)))
                vram_total = cfg["vram_total"]
                vram_used = round(vram_total * (util / 100.0) * random.uniform(0.7, 1.0), 2)
                temp = _temp_from_util(util, ambient=26 if cluster == "A" else 24)
                self.nodes[node_id] = NodeTelemetry(
                    node_id=node_id,
                    cluster=cluster,
                    gpu_model=cfg["gpu_model"],
                    vram_total_gb=vram_total,
                    vram_used_gb=vram_used,
                    vram_util_pct=round(100 * vram_used / vram_total, 1),
                    gpu_core_util_pct=round(util, 1),
                    temp_c=temp,
                    power_draw_w=_power_from_util(util),
                    active_jobs=random.randint(0, 3) if util > 15 else 0,
                    idle_minutes=0.0 if util > 10 else random.uniform(0, 45),
                    status="healthy",
                )

    def inject_chaos(self, cluster: str, intensity: float = 1.0):
        c_upper = cluster.upper()
        other = "B" if c_upper == "A" else "A"
        
        # Accumulate chaos intensity on target cluster
        self.chaos[c_upper] = min(3.0, self.chaos.get(c_upper, 0.0) + intensity)
        self.chaos[other] = 0.0

        # Immediately apply +10 GB VRAM equivalent spike to target cluster nodes
        for node_id, node in self.nodes.items():
            if node.cluster == c_upper:
                # Add 10 GB VRAM consumption per click (capped near total capacity)
                boost_vram = 10.0
                node.vram_used_gb = min(round(node.vram_total_gb * 0.96, 2), round(node.vram_used_gb + boost_vram, 2))
                node.vram_util_pct = round(100 * node.vram_used_gb / node.vram_total_gb, 1)
                node.gpu_core_util_pct = min(99.5, round(node.gpu_core_util_pct + 45.0, 1))
                node.temp_c = min(96.0, round(node.temp_c + 20.0, 1))
                node.power_draw_w = _power_from_util(node.gpu_core_util_pct)
                node.active_jobs += 2
                node.status = "unsafe"
            elif node.cluster == other:
                # Instantly clear chaos and restore opposite cluster nodes to healthy state
                node.gpu_core_util_pct = max(10.0, round(node.gpu_core_util_pct * 0.3, 1))
                node.vram_used_gb = round(node.vram_total_gb * 0.25, 2)
                node.vram_util_pct = round(100 * node.vram_used_gb / node.vram_total_gb, 1)
                node.temp_c = _temp_from_util(node.gpu_core_util_pct, ambient=24.0)
                node.power_draw_w = _power_from_util(node.gpu_core_util_pct)
                node.status = "healthy"

    def clear_chaos(self, cluster: str):
        c_upper = cluster.upper()
        self.chaos[c_upper] = 0.0
        for node_id, node in self.nodes.items():
            if node.cluster == c_upper:
                node.gpu_core_util_pct = max(10.0, round(node.gpu_core_util_pct * 0.3, 1))
                node.vram_used_gb = round(node.vram_total_gb * 0.25, 2)
                node.vram_util_pct = round(100 * node.vram_used_gb / node.vram_total_gb, 1)
                node.temp_c = _temp_from_util(node.gpu_core_util_pct, ambient=24.0)
                node.power_draw_w = _power_from_util(node.gpu_core_util_pct)
                node.status = "healthy"

    def tick(self):
        """Advance simulated time by one step (~2s of wall clock ~= a few
        minutes of simulated cluster activity), with diurnal drift + bursts."""
        self.tick_count += 1
        diurnal = 6 * math.sin(self.tick_count / 40.0)

        for node_id, node in self.nodes.items():
            cfg = CLUSTER_CONFIG[node.cluster]
            chaos_boost = self.chaos.get(node.cluster, 0.0) * 32  # big deliberate spike

            drift = random.gauss(0, 4)
            burst = 18 if random.random() < 0.05 else 0  # occasional job burst
            target_util = cfg["base_util"] + diurnal + drift + burst + chaos_boost
            new_util = max(1, min(99.5, 0.6 * node.gpu_core_util_pct + 0.4 * target_util))

            vram_ratio = min(0.98, (new_util / 100.0) * random.uniform(0.85, 1.05))
            vram_used = round(node.vram_total_gb * vram_ratio, 2)
            temp = _temp_from_util(new_util, ambient=26 if node.cluster == "A" else 24)
            temp += self.chaos.get(node.cluster, 0.0) * 6  # chaos also directly raises temp (throttling)

            idle_minutes = (node.idle_minutes + 2) if new_util < 10 else 0.0
            active_jobs = node.active_jobs
            if burst and active_jobs < 4:
                active_jobs += 1
            elif new_util < 10 and active_jobs > 0 and random.random() < 0.3:
                active_jobs -= 1

            node.gpu_core_util_pct = round(new_util, 1)
            node.vram_used_gb = vram_used
            node.vram_util_pct = round(100 * vram_used / node.vram_total_gb, 1)
            node.temp_c = round(temp, 1)
            node.power_draw_w = _power_from_util(new_util)
            node.active_jobs = active_jobs
            node.idle_minutes = round(idle_minutes, 1)
            node.wait_time_s = node.wait_time_s + 2 if node.active_jobs == 0 else 0

            if node.temp_c >= 85 or node.gpu_core_util_pct >= 90 or node.vram_util_pct >= 92:
                node.status = "unsafe"
            elif node.temp_c >= 75 or node.gpu_core_util_pct >= 75 or node.vram_util_pct >= 78:
                node.status = "warning"
            else:
                node.status = "healthy"

        # chaos auto-decays gracefully
        for c in self.chaos:
            if self.chaos[c] > 0:
                self.chaos[c] = max(0.0, self.chaos[c] - 0.03)

    def snapshot(self):
        return list(self.nodes.values())


engine = ClusterSimEngine()
