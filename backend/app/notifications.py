"""
VectorOps - Notifications & Alert Engine
"""
import uuid
import datetime
from typing import List, Optional
from app.models import NotificationItem, NodeTelemetry

INITIAL_NOTIFS: List[NotificationItem] = [
    NotificationItem(
        id="notif-1",
        timestamp="10:35 AM",
        title="High Thermal Alert on Node A-01",
        message="Temperature reached 86°C during high burst utilization. Auto-migration check triggered.",
        type="danger",
        read=False
    ),
    NotificationItem(
        id="notif-2",
        timestamp="10:20 AM",
        title="Workload Evacuation Completed",
        message="Job resnet50-eval migrated from A-04 to B-02 due to thermal throttling risk.",
        type="success",
        read=False
    ),
    NotificationItem(
        id="notif-3",
        timestamp="10:05 AM",
        title="Cluster B Idle Capacity Alert",
        message="4 nodes in Cluster B have been idle for >30 minutes. Cost Optimizer suggests consolidating jobs.",
        type="warning",
        read=True
    ),
]


class NotificationEngine:
    def __init__(self):
        self.notifications: List[NotificationItem] = list(INITIAL_NOTIFS)
        self._alerted_nodes = set()

    def get_notifications(self) -> List[NotificationItem]:
        return self.notifications

    def mark_read(self, notif_id: Optional[str] = None):
        if notif_id:
            for n in self.notifications:
                if n.id == notif_id:
                    n.read = True
        else:
            for n in self.notifications:
                n.read = True

    def observe_nodes(self, nodes: List[NodeTelemetry]):
        now_str = datetime.datetime.now().strftime("%I:%M %p")
        for n in nodes:
            if n.status == "unsafe" and n.node_id not in self._alerted_nodes:
                self._alerted_nodes.add(n.node_id)
                self.notifications.insert(0, NotificationItem(
                    id=f"notif-{uuid.uuid4().hex[:4]}",
                    timestamp=now_str,
                    title=f"CRITICAL: Node {n.node_id} Unsafe",
                    message=f"Node {n.node_id} in Cluster {n.cluster} reached unsafe conditions: {n.temp_c}°C / {n.gpu_core_util_pct}% util.",
                    type="danger",
                    read=False
                ))
            elif n.status == "healthy" and n.node_id in self._alerted_nodes:
                self._alerted_nodes.remove(n.node_id)


notif_engine = NotificationEngine()
