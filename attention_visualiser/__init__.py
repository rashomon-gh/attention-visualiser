"""Attention visualiser for transformer-based models.

This package provides tools to visualize attention layer activations from
Hugging Face transformer models. It supports PyTorch-based models and allows
for easy visualization of attention weights across different layers.

Example:
    >>> from transformers import AutoModel, AutoTokenizer
    >>> from attention_visualiser import AttentionVisualiser
    >>> model = AutoModel.from_pretrained("bert-base-uncased")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> visualiser = AttentionVisualiser(model, tokenizer)
    >>> encoded = tokenizer.encode_plus("Hello world", return_tensors="pt")
    >>> visualiser.visualise_attn_layer(0, encoded)
"""

from .pt import AttentionVisualiserPytorch as AttentionVisualiser

__all__ = ["AttentionVisualiser"]
