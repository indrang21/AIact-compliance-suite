"""Core evaluation orchestration."""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from tqdm import tqdm

from .articles.base import Article
from .models.base import BaseModel
from .scoring.judge import Judge


@dataclass
class TestResult:
    """Single test result — one prompt tested against one model."""
    prompt_id: str
    article_number: str
    category: str
    prompt: str
    model_name: str
    model_response: str
    judge_scores: Dict[str, Any] = field(default_factory=dict)
    overall_score: float = 0.0
    verdict: str = "unknown"


@dataclass
class RunResults:
    """Full results of a compliance run."""
    article_number: str
    article_title: str
    language: str
    model_name: str
    judge_name: str
    timestamp: str
    test_results: List[TestResult] = field(default_factory=list)

    @property
    def num_tests(self) -> int:
        return len(self.test_results)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.test_results if r.verdict == "pass")

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.test_results if r.verdict == "fail")

    @property
    def pass_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return self.num_passed / self.num_tests

    @property
    def avg_score(self) -> float:
        if not self.test_results:
            return 0.0
        return sum(r.overall_score for r in self.test_results) / self.num_tests

    def summary(self) -> Dict[str, Any]:
        """Aggregated summary stats."""
        # Group by category
        by_category = {}
        for r in self.test_results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0, "avg_score": 0.0}
            by_category[cat]["total"] += 1
            if r.verdict == "pass":
                by_category[cat]["passed"] += 1
            by_category[cat]["avg_score"] += r.overall_score

        for cat, stats in by_category.items():
            stats["avg_score"] = stats["avg_score"] / stats["total"]
            stats["pass_rate"] = stats["passed"] / stats["total"]

        return {
            "article": self.article_number,
            "article_title": self.article_title,
            "language": self.language,
            "model": self.model_name,
            "judge": self.judge_name,
            "timestamp": self.timestamp,
            "num_tests": self.num_tests,
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "by_category": by_category,
        }


class ComplianceRunner:
    """Runs AI Act compliance tests against LLM models."""

    def __init__(
        self,
        model: BaseModel,
        article: Article,
        judge: Judge,
        language: str = "en",
    ):
        """
        Args:
            model: model under test.
            article: article to test (e.g., Article05()).
            judge: LLM-as-judge instance.
            language: 'en' or 'fi'.
        """
        self.model = model
        self.article = article
        self.judge = judge
        self.language = language

    def run(self, verbose: bool = True) -> RunResults:
        """Execute compliance tests and return results."""
        # Load test prompts + rubric
        prompts = self.article.get_test_prompts(language=self.language)
        rubric = self.article.get_rubric()

        # Article metadata for judge
        article_data = {
            "article_number": self.article.article_number,
            "article_title": self.article.article_title,
            "article_text": self.article.article_text,
        }

        results = RunResults(
            article_number=self.article.article_number,
            article_title=self.article.article_title,
            language=self.language,
            model_name=self.model.name,
            judge_name=self.judge.judge_model.name,
            timestamp=datetime.now().isoformat(),
        )

        # Run each test prompt
        iterator = tqdm(prompts, desc=f"Testing {self.model.name}") if verbose else prompts

        for prompt_data in iterator:
            # Step 1: Get model response
            try:
                model_response = self.model.generate(prompt_data["prompt"])
            except Exception as e:
                model_response = f"[ERROR: {e}]"

            # Step 2: Judge scores
            try:
                judge_scores = self.judge.score(
                    article_data=article_data,
                    test_prompt=prompt_data,
                    model_response=model_response,
                    rubric=rubric,
                )
            except Exception as e:
                judge_scores = {
                    "overall_score": 0.0,
                    "verdict": "error",
                    "summary": f"Judge error: {e}",
                    "criterion_scores": {},
                }

            # Step 3: Record result
            result = TestResult(
                prompt_id=prompt_data["id"],
                article_number=self.article.article_number,
                category=prompt_data.get("category", "unknown"),
                prompt=prompt_data["prompt"],
                model_name=self.model.name,
                model_response=model_response,
                judge_scores=judge_scores,
                overall_score=judge_scores.get("overall_score", 0.0),
                verdict=judge_scores.get("verdict", "error"),
            )
            results.test_results.append(result)

        return results
