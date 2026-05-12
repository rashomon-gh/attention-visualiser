import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding
from attention_visualiser.base import BaseAttentionVisualiser
import numpy as np
from typing import Optional


class AttentionVisualiserPytorch(BaseAttentionVisualiser):
    """Attention visualizer for PyTorch-based transformer models.

    This class implements the abstract methods from BaseAttentionVisualiser
    specifically for models implemented in PyTorch. It handles the extraction
    and processing of attention weights from PyTorch transformer models.

    Attributes:
        model: A PyTorch-based transformer model from Hugging Face
        tokenizer: A tokenizer matching the model
        config: Dictionary containing visualization configuration parameters
    """

    def __init__(
        self, model: AutoModel, tokenizer: AutoTokenizer, config: Optional[dict] = None
    ) -> None:
        """Initialize the PyTorch-specific attention visualizer.

        Args:
            model: A PyTorch-based transformer model from Hugging Face
            tokenizer: A tokenizer matching the model
            config: Optional dictionary with visualization parameters
        """
        super().__init__(model, tokenizer, config)

    def compute_attentions(self, encoded_input: BatchEncoding) -> tuple:
        """Compute attention weights for the given input using a PyTorch model.

        Runs the PyTorch model in inference mode with output_attentions flag set to True
        and extracts the attention weights from the model output.

        Args:
            encoded_input: The encoded input from the tokenizer

        Returns:
            A tuple containing attention weights from all layers of the model
        """
        if encoded_input == self.current_input:
            # return from cache
            return self.cache

        # else recompute
        with torch.no_grad():
            output = self.model(**encoded_input, output_attentions=True)  # type: ignore

        attentions = output.attentions

        # update cache and current input
        self.current_input = encoded_input
        self.cache = attentions

        return attentions

    def get_attention_vector_mean(
        self, attention: torch.Tensor, axis: int = 0
    ) -> np.ndarray:
        """Calculate mean of PyTorch attention vectors along specified axis.

        Computes the mean of the attention tensor and converts it to a NumPy array.

        Args:
            attention: PyTorch tensor containing attention weights
            axis: Axis along which to compute the mean (default: 0)

        Returns:
            NumPy array of mean attention values
        """
        return torch.mean(attention, dim=axis).detach().cpu().numpy()
