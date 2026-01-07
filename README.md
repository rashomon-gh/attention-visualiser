# attention-visualiser

a module to visualise attention layer activations from transformer based models from huggingface

## installation

```bash
pip install git+https://codeberg.org/rashomon/attention-visualiser
```

## usage

```python
from attention_visualiser import AttentionVisualiser
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
