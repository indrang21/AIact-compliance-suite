"""Rubric definitions."""
from dataclasses import dataclass
from typing import List


@dataclass
class RubricCriterion:
    name: str
    description: str
    weight: float = 1.0


@dataclass
class Rubric:
    article_number: str
    criteria: List[RubricCriterion]
