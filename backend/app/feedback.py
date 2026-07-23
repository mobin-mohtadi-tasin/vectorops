"""
VectorOps - Feedback Store
"""
import uuid
import datetime
from typing import List
from app.models import FeedbackItem, FeedbackSubmission

INITIAL_FEEDBACK: List[FeedbackItem] = [
    FeedbackItem(
        id="fb-101",
        user_name="Dr. Marcus Vance",
        email="marcus@lab.edu",
        category="feature",
        rating=5,
        comments="The thermal prediction model saved our fine-tuning run from crashing yesterday! Great job on the automated evacuations.",
        created_at="July 22, 2026"
    ),
    FeedbackItem(
        id="fb-102",
        user_name="Elena Rostova",
        email="elena@vision.io",
        category="ui",
        rating=4,
        comments="Love the clean light interface. Would be great to see job execution history chart in the Work tab.",
        created_at="July 23, 2026"
    ),
]


class FeedbackStore:
    def __init__(self):
        self.feedback_list: List[FeedbackItem] = list(INITIAL_FEEDBACK)

    def get_all(self) -> List[FeedbackItem]:
        return self.feedback_list

    def add_feedback(self, submission: FeedbackSubmission) -> FeedbackItem:
        now_str = datetime.datetime.now().strftime("%B %d, %Y")
        item = FeedbackItem(
            id=f"fb-{uuid.uuid4().hex[:4]}",
            user_name=submission.user_name,
            email=submission.email,
            category=submission.category,
            rating=submission.rating,
            comments=submission.comments,
            created_at=now_str
        )
        self.feedback_list.insert(0, item)
        return item


feedback_store = FeedbackStore()
