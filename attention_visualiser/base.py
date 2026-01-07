import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, FlaxAutoModel
from transformers import BatchEncoding
from loguru import logger
from einops import rearrange
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BaseAttentionVisualiser(ABC):
    """Base abstract class for visualizing attention weights in transformer models.

    This class provides the foundation for visualizing attention weights from
    different transformer model implementations. Concrete subclasses must implement
    methods for computing attention values and processing attention vectors.

    Attributes:
        model: A transformer model from the Hugging Face library
        tokenizer: A tokenizer matching the model
        config: Dictionary containing visualization configuration parameters
    """

    def __init__(
        self,
        model: AutoModel | FlaxAutoModel,
        tokenizer: AutoTokenizer,
        config: Optional[dict] = None,
    ) -> None:
        """Initialize the attention visualizer with a model and tokenizer.

        Args:
            model: A transformer model from Hugging Face (PyTorch or Flax)
            tokenizer: A tokenizer matching the model
            config: Optional dictionary with visualization parameters
                   Default parameters include:
                   - figsize: Tuple specifying figure dimensions
                   - cmap: Colormap for the heatmap
                   - annot: Whether to annotate heatmap cells with values
                   - xlabel: Label for x-axis
                   - ylabel: Label for y-axis
        """
        self.model = model
        self.tokenizer = tokenizer

        logger.info(f"Model config: {self.model.config}")  # type: ignore

        if not config:
            self.config = {
                "figsize": (15, 15),
                "cmap": "viridis",
                "annot": True,
                "xlabel": "",
                "ylabel": "",
            }
            logger.info(f"Setting default visualiser config: {self.config}")
        else:
            logger.info(f"Visualiser config: {config}")
            self.config = config

        # a cache for storing already computed attention vectors
        # these need to be updated by the `compute_attentions`
        # method
        self.current_input = None
        self.cache = None

    def id_to_tokens(self, encoded_input: BatchEncoding) -> list[str]:
        """Convert token IDs to readable token strings.

        Args:
            encoded_input: The encoded input from the tokenizer

        Returns:
            List of token strings corresponding to the input IDs
        """
        tokens = self.tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])  # type: ignore
        return tokens

    @abstractmethod
    def compute_attentions(self, encoded_input: BatchEncoding) -> tuple:
        """Compute attention weights for the given input.

        This method must be implemented by concrete subclasses to compute
        attention weights specific to the model implementation.

        Args:
            encoded_input: The encoded input from the tokenizer

        Returns:
            A tuple containing attention weights
        """
        pass

    @abstractmethod
    def get_attention_vector_mean(
        self, attention: torch.Tensor, axis: int = 0
    ) -> np.ndarray:
        """Calculate mean of attention vectors along specified axis.

        This method must be implemented by concrete subclasses to handle
        either PyTorch or JAX tensors appropriately.

        Args:
            attention: Attention tensor from the model
            axis: Axis along which to compute the mean (default: 0)

        Returns:
            NumPy array of mean attention values
        """
        pass

    def visualise_attn_layer(self, idx: int, encoded_input: BatchEncoding) -> None:
        """Visualize attention weights for a specific layer.

        Creates a heatmap visualization of the attention weights for the specified
        layer index.

        Args:
            idx: Index of the attention layer to visualize.
                 Negative indices count from the end (-1 is the last layer).
            encoded_input: The encoded input from the tokenizer

        Raises:
            AssertionError: If idx is outside the range of available attention layers
        """
        tokens = self.id_to_tokens(encoded_input)

        attentions = self.compute_attentions(encoded_input)
        n_attns = len(attentions)

        # idx must no exceed attn_heads
        assert idx < n_attns, (
            f"index must be less than the number of attention outputs in the model, which is: {n_attns}"
        )

        # setting idx = -1 will get the last attention layer activations but
        # the plot title will also show -1
        if idx < 0:
            idx = n_attns + idx

        # get rid of the additional dimension since single input
        attention = rearrange(attentions[idx], "1 a b c -> a b c")
        # take mean over dim 0
        attention = self.get_attention_vector_mean(attention)

        plt.figure(figsize=self.config.get("figsize"))
        sns.heatmap(
            attention,
            cmap=self.config.get("cmap"),
            annot=self.config.get("annot"),
            xticklabels=tokens,
            yticklabels=tokens,
        )

        plt.title(f"Attention Weights for Layer idx: {idx}")
        plt.xlabel(self.config.get("xlabel"))  # type: ignore
        plt.ylabel(self.config.get("ylabel"))  # type: ignore
        plt.show()
