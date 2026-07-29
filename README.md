# EU AI Act Compliance Suite

Article-level compliance evaluation suite for LLMs, aligned with the EU AI Act (Regulation 2024/1689).

Maps existing eval benchmarks and custom test suites to specific AI Act articles, providing structured compliance reports across multiple models and languages (EN/FI).

## Features

- **Article-level testing**: Tests LLM behavior against specific AI Act articles (Art. 5, 9, 10, 13, 14, 15, 50, 51-56)
- **Multilingual**: English and Finnish test suites
- **Multi-model**: Supports local models (Mistral 7B, Llama), fine-tuned adapters, and API models (DeepSeek V3)
- **Rubric + LLM-as-judge scoring**: Structured evaluation with expert rubrics
- **Multiple output formats**: CLI, Python library, Streamlit dashboard

## Installation

```bash
pip install aiact-compliance-suite
```

Or from source:

```bash
git clone https://github.com/indrang21/aiact-compliance-suite.git
cd aiact-compliance-suite
pip install -e .[dev]
```

## Quick Start

### CLI

```bash
aiact-eval run --model mistral-7b --article art05 --language en
```

### Python

```python
from aiact_compliance import ComplianceRunner
from aiact_compliance.models import MistralLocal
from aiact_compliance.articles import Article05

runner = ComplianceRunner(
    model=MistralLocal(),
    articles=[Article05()],
    language="en",
)
results = runner.run()
report = results.to_markdown()
```

### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

## Articles Covered

| Article | Title | Status |
|---------|-------|--------|
| Art. 5 | Prohibited AI Practices | ✅ |
| Art. 9 | Risk Management System | 🚧 |
| Art. 10 | Data and Data Governance | 🚧 |
| Art. 13 | Transparency and Provision of Information | 🚧 |
| Art. 14 | Human Oversight | 🚧 |
| Art. 15 | Accuracy, Robustness, and Cybersecurity | 🚧 |
| Art. 50 | Transparency Obligations for GPAI | 🚧 |
| Art. 51-56 | GPAI Models with Systemic Risk | 🚧 |

## Methodology

See [docs/methodology.md](docs/methodology.md) for detailed methodology.

## License

MIT — see [LICENSE](LICENSE).

## Related Projects

- [Suomi-RAG](https://github.com/indrang21/Suomi-RAG-bilingual-multi-regulation-compliance-RAG): Bilingual EU AI Act + GDPR retrieval system used for article text lookup.
- [Mistral 7B FI-EN QLoRA v1](https://huggingface.co/indrani191919/mistral-7b-fi-en-qlora-v1): Fine-tuned bilingual model included as one of the tested models.
