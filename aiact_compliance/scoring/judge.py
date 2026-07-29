"""LLM-as-judge scoring."""


class Judge:
    def __init__(self, judge_model):
        self.judge_model = judge_model

    def score(self, response, rubric):
        raise NotImplementedError
