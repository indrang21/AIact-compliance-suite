"""Core evaluation orchestration."""


class ComplianceRunner:
    """Runs compliance tests against LLM models."""

    def __init__(self, model, articles, language="en"):
        self.model = model
        self.articles = articles
        self.language = language

    def run(self):
        raise NotImplementedError
