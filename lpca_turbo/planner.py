"""
Multi-Hop Adaptive Research Planner.
Deconstructs topics into orthogonal search vectors for overview, technical, verification, and counter-perspectives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SubQuery:
    query: str
    dimension: str  # 'overview', 'technical', 'verification', 'counter_perspective'
    priority: int = 1
    description: str = ""


class ResearchPlanner:
    """Generates clean, high-signal multi-hop search queries across 4 cognitive dimensions."""

    DIMENSION_TEMPLATES = {
        "overview": [
            "{topic} overview architecture features",
            "what is {topic} core concepts",
        ],
        "technical": [
            "{topic} implementation benchmark specs",
            "{topic} API github technical architecture",
        ],
        "verification": [
            "{topic} documentation primary source 2025 2026",
            "{topic} performance evaluation accuracy",
        ],
        "counter_perspective": [
            "{topic} limitations drawbacks bottlenecks",
            "{topic} alternatives comparison issues",
        ],
    }

    def __init__(self, max_queries_per_dimension: int = 1):
        self.max_queries = max(1, max_queries_per_dimension)

    def sanitize_topic(self, topic: str) -> str:
        clean = re.sub(r"[^\w\s\-\.\/]", " ", topic)
        return " ".join(clean.split()).strip()

    def generate_plan(self, topic: str) -> List[SubQuery]:
        clean_topic = self.sanitize_topic(topic)
        sub_queries: List[SubQuery] = []
        for dimension, templates in self.DIMENSION_TEMPLATES.items():
            for idx, template in enumerate(templates[: self.max_queries]):
                sub_queries.append(
                    SubQuery(
                        query=template.format(topic=clean_topic),
                        dimension=dimension,
                        priority=idx + 1,
                        description=f"{dimension.capitalize()} vector for {clean_topic}",
                    )
                )
        return sub_queries

    def plan_to_dict(self, sub_queries: List[SubQuery]) -> List[Dict[str, Any]]:
        return [
            {
                "query": sq.query,
                "dimension": sq.dimension,
                "priority": sq.priority,
                "description": sq.description,
            }
            for sq in sub_queries
        ]
