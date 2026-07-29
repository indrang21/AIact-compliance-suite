"""Compliance report generator."""


class ReportGenerator:
    def __init__(self, results):
        self.results = results

    def to_html(self):
        raise NotImplementedError

    def to_markdown(self):
        raise NotImplementedError
