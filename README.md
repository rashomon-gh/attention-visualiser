# attention-visualiser

| Status | Coverage |
|--------|----------|
| [![tests](https://github.com/ShawonAshraf/attention-visualiser/actions/workflows/tests.yml/badge.svg)](https://github.com/ShawonAshraf/attention-visualiser/actions/workflows/tests.yml) | [![codecov](https://codecov.io/github/ShawonAshraf/attention-visualiser/graph/badge.svg?token=UqcZYGp3Rj)](https://codecov.io/github/ShawonAshraf/attention-visualiser) |


a module to visualise attention layer activations from transformer based models from huggingface

## installation

```bash
pip install git+https://github.com/ShawonAshraf/attention-visualiser
```

## usage

```python
from visualiser import AttentionVisualiser
from transformers import AutoModel, AutoTokenizer

# visualising activations from gpt
model_name = "openai-community/openai-gpt"

model = AutoModel.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Look on my Works, ye Mighty, and despair!"
encoded_inputs = tokenizer.encode_plus(text, truncation=True, return_tensors="pt")

visualiser = AttentionVisualiser(model, tokenizer)

# visualise from the first attn layer
visualiser.visualise_attn_layer(0, encoded_inputs)

```


## local dev

```bash
# env setup

uv sync
source .venv/bin/activate

# tests
uv run pytest

# tests with coverage
uv run pytest --cov --cov-report=xml
```
