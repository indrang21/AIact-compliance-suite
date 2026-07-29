"""EU AI Act Compliance Suite."""
from .runner import ComplianceRunner, RunResults, TestResult
from .articles import Article, Article05
from .models import BaseModel, MistralLocal, QLoRAv1, ClaudeSonnet
from .scoring.judge import Judge
from .scoring.rubric import Rubric, RubricCriterion

__version__ = "0.1.0"
__all__ = [
    # Runner
    "ComplianceRunner", "RunResults", "TestResult",
    # Articles
    "Article", "Article05",
    # Models
    "BaseModel", "MistralLocal", "QLoRAv1", "ClaudeSonnet",
    # Scoring
    "Judge", "Rubric", "RubricCriterion",
]
