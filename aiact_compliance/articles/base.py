"""Base Article class."""
from abc import ABC, abstractmethod
from typing import List, Dict


class Article(ABC):
    """Base class for AI Act article compliance tests."""

    article_number: str = ""
    article_title: str = ""
    article_text: str = ""

    @abstractmethod
    def get_test_prompts(self, language: str = "en") -> List[Dict]:
        """Return list of test prompts for this article."""
        pass

    @abstractmethod
    def get_rubric(self) -> Dict:
        """Return scoring rubric."""
        pass
