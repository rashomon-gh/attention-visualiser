import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from attention_visualiser.visualiser import AttentionVisualiser


class TestAttentionVisualiser:
    @pytest.fixture
    def mock_model(self):
        class MockConfig:
            num_attention_heads = 12
            num_hidden_layers = 12

        class MockModel:
            def __init__(self):
                self.config = MockConfig()
                self.call_count = 0

            def __call__(self, **kwargs):
                self.call_count += 1
                self.last_kwargs = kwargs
                return self.output  # type: ignore

        model = MockModel()
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def __init__(self):
                self.call_count = 0

            def convert_ids_to_tokens(self, ids):
                self.call_count += 1
                return ["[CLS]", "Hello", "world", "[SEP]"]

        return MockTokenizer()

    @pytest.fixture
    def visualiser(self, mock_model, mock_tokenizer):
        return AttentionVisualiser(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

    @pytest.fixture
    def mock_encoded_input(self):
        class MockEncodedInput:
            def __init__(self):
                self.data = {"input_ids": torch.tensor([[101, 7592, 2088, 102]])}

            def __getitem__(self, key):
                return self.data.get(key)

            def __eq__(self, other):
                return id(self) == id(other)

            # Add these methods to support ** unpacking
            def keys(self):
                return self.data.keys()

            def __iter__(self):
                return iter(self.data)

            def get(self, key, default=None):
                return self.data.get(key, default)

        return MockEncodedInput()

    @pytest.fixture
    def mock_attention_data(self):
        # Create mock attention data: (batch_size, num_heads, seq_len, seq_len)
        # Shape: (1, 12, 4, 4) - 1 batch, 12 attention heads, sequence length 4
        return torch.ones((1, 12, 4, 4)) * 0.25

    def test_init(self, mock_model, mock_tokenizer):
        """Test initialization of AttentionVisualiser."""
        visualiser = AttentionVisualiser(mock_model, mock_tokenizer)

        assert visualiser.model == mock_model
        assert visualiser.tokenizer == mock_tokenizer
        assert visualiser.config is not None
        assert visualiser.current_input is None
        assert visualiser.cache is None

    def test_id_to_tokens(self, visualiser, mock_encoded_input):
        """Test the id_to_tokens method."""
        tokens = visualiser.id_to_tokens(mock_encoded_input)

        assert visualiser.tokenizer.call_count == 1
        assert tokens == ["[CLS]", "Hello", "world", "[SEP]"]

    def test_compute_attentions_new_input(
        self, visualiser, mock_encoded_input, mock_attention_data
    ):
        """Test compute_attentions with new input."""

        # Setup mock return value for model call
        class MockOutput:
            def __init__(self, attn_data):
                self.attentions = attn_data

        # Add the output attribute to the model
        output = MockOutput(mock_attention_data)
        visualiser.model.output = output

        # Initialize call_count if not present
        if not hasattr(visualiser.model, "call_count"):
            visualiser.model.call_count = 0
        if not hasattr(visualiser.model, "last_kwargs"):
            visualiser.model.last_kwargs = {}

        # Call the method
        attentions = visualiser.compute_attentions(mock_encoded_input)

        # Verify the model was called
        assert visualiser.model.call_count > 0
        assert visualiser.model.last_kwargs.get("output_attentions")

        # Verify the return value and cache update
        assert attentions is mock_attention_data
        assert visualiser.current_input == mock_encoded_input
        assert visualiser.cache is mock_attention_data

    def test_compute_attentions_cached_input(
        self, visualiser, mock_encoded_input, mock_attention_data
    ):
        """Test compute_attentions with cached input."""
        # Setup cache
        visualiser.current_input = mock_encoded_input
        visualiser.cache = mock_attention_data
        visualiser.model.call_count = 0

        # Call the method
        attentions = visualiser.compute_attentions(mock_encoded_input)

        # Verify model wasn't called and cached data was returned
        assert visualiser.model.call_count == 0
        assert attentions is mock_attention_data

    def test_get_attention_vector_mean(self, visualiser):
        """Test get_attention_vector_mean method."""
        # Create test data
        test_data = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])

        # Expected result: mean along axis 0
        expected = np.array([[0.3, 0.4], [0.5, 0.6]])

        # Call the method
        result = visualiser.get_attention_vector_mean(test_data)

        # Verify
        np.testing.assert_allclose(result, expected)

    def test_get_attention_vector_mean_different_axis(self, visualiser):
        """Test get_attention_vector_mean with a different axis."""
        # Create test data
        test_data = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])

        # Expected result: mean along axis 1
        expected = np.array([[0.2, 0.3], [0.6, 0.7]])

        # Call the method
        result = visualiser.get_attention_vector_mean(test_data, axis=1)

        # Verify
        np.testing.assert_allclose(result, expected)

    def test_custom_config(self, mock_model, mock_tokenizer):
        """Test initialization with custom configuration."""
        custom_config = {
            "figsize": (10, 10),
            "cmap": "coolwarm",
            "annot": False,
            "xlabel": "Tokens",
            "ylabel": "Tokens",
        }

        visualiser = AttentionVisualiser(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=custom_config,
        )

        assert visualiser.config == custom_config

    def test_visualise_attn_layer_valid_index(
        self, visualiser, mock_encoded_input, mock_attention_data
    ):
        """Test visualise_attn_layer with a valid layer index."""
        class MockOutput:
            def __init__(self, attn_data):
                self.attentions = [attn_data] * 12  # 12 layers

        output = MockOutput(mock_attention_data)
        visualiser.model.output = output

        with patch("attention_visualiser.base.plt.figure"), \
             patch("attention_visualiser.base.plt.show"), \
             patch("attention_visualiser.base.sns.heatmap") as mock_heatmap:

            visualiser.visualise_attn_layer(0, mock_encoded_input)

            # Verify heatmap was called with correct parameters
            mock_heatmap.assert_called_once()
            call_args = mock_heatmap.call_args
            assert call_args[1]["cmap"] == visualiser.config["cmap"]
            assert call_args[1]["annot"] == visualiser.config["annot"]

    def test_visualise_attn_layer_negative_index(
        self, visualiser, mock_encoded_input, mock_attention_data
    ):
        """Test visualise_attn_layer with negative index (last layer)."""
        class MockOutput:
            def __init__(self, attn_data):
                self.attentions = [attn_data] * 12

        output = MockOutput(mock_attention_data)
        visualiser.model.output = output

        with patch("attention_visualiser.base.plt.figure"), \
             patch("attention_visualiser.base.plt.show"), \
             patch("attention_visualiser.base.plt.title") as mock_title, \
             patch("attention_visualiser.base.sns.heatmap"):

            # -1 should get the last layer (index 11)
            visualiser.visualise_attn_layer(-1, mock_encoded_input)

            # Title should show the resolved positive index
            mock_title.assert_called_once()
            title_arg = mock_title.call_args[0][0]
            assert "Layer idx: 11" in title_arg

    def test_visualise_attn_layer_index_out_of_bounds(
        self, visualiser, mock_encoded_input, mock_attention_data
    ):
        """Test visualise_attn_layer with index exceeding available layers."""
        class MockOutput:
            def __init__(self, attn_data):
                self.attentions = [attn_data] * 12

        output = MockOutput(mock_attention_data)
        visualiser.model.output = output

        with pytest.raises(AssertionError) as exc_info:
            visualiser.visualise_attn_layer(15, mock_encoded_input)

        assert "index must be less than" in str(exc_info.value)

    def test_visualise_attn_layer_custom_config(
        self, mock_model, mock_tokenizer, mock_encoded_input, mock_attention_data
    ):
        """Test visualise_attn_layer uses custom config parameters."""
        custom_config = {
            "figsize": (8, 8),
            "cmap": "plasma",
            "annot": False,
            "xlabel": "Source",
            "ylabel": "Target",
        }

        visualiser = AttentionVisualiser(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=custom_config,
        )

        class MockOutput:
            def __init__(self, attn_data):
                self.attentions = [attn_data] * 12

        output = MockOutput(mock_attention_data)
        visualiser.model.output = output

        with patch("attention_visualiser.base.plt.show"), \
             patch("attention_visualiser.base.sns.heatmap") as mock_heatmap:

            visualiser.visualise_attn_layer(0, mock_encoded_input)

            # Verify heatmap received custom config
            heatmap_kwargs = mock_heatmap.call_args[1]
            assert heatmap_kwargs["cmap"] == "plasma"
            assert heatmap_kwargs["annot"] is False

    def test_cache_invalidation_on_new_input(
        self, visualiser, mock_encoded_input, mock_attention_data
    ):
        """Test that cache is properly invalidated when input changes."""
        class MockOutput:
            def __init__(self, attn_data):
                self.attentions = [attn_data] * 12

        output = MockOutput(mock_attention_data)
        visualiser.model.output = output

        # First call - should compute
        attentions1 = visualiser.compute_attentions(mock_encoded_input)
        assert visualiser.model.call_count == 1

        # Second call with same input - should use cache
        attentions2 = visualiser.compute_attentions(mock_encoded_input)
        assert visualiser.model.call_count == 1  # No increment
        assert attentions1 is attentions2

        # Create a different input
        class DifferentEncodedInput:
            def __init__(self):
                self.data = {"input_ids": torch.tensor([[101, 7592, 102]])}

            def __getitem__(self, key):
                return self.data.get(key)

            def __eq__(self, other):
                return id(self) == id(other)

            def keys(self):
                return self.data.keys()

            def __iter__(self):
                return iter(self.data)

            def get(self, key, default=None):
                return self.data.get(key, default)

        different_input = DifferentEncodedInput()

        # Third call with different input - should recompute
        attentions3 = visualiser.compute_attentions(different_input)
        assert visualiser.model.call_count == 2

    def test_id_to_tokens_with_multiple_tokens(self, mock_model, mock_tokenizer):
        """Test id_to_tokens with various token sequences."""
        class MockTokenizerWithMultiple:
            def __init__(self):
                self.call_count = 0

            def convert_ids_to_tokens(self, ids):
                self.call_count += 1
                return ["[CLS]", "Hello", "world", "!", "[SEP]"]

        tokenizer = MockTokenizerWithMultiple()
        visualiser = AttentionVisualiser(model=mock_model, tokenizer=tokenizer)

        encoded_input = {"input_ids": torch.tensor([[101, 7592, 2088, 999, 102]])}
        tokens = visualiser.id_to_tokens(encoded_input)

        assert tokens == ["[CLS]", "Hello", "world", "!", "[SEP]"]

    def test_get_attention_vector_mean_edge_cases(self, visualiser):
        """Test get_attention_vector_mean with edge cases."""
        # Single value tensor
        single_value = torch.tensor([[[5.0]]])
        result = visualiser.get_attention_vector_mean(single_value)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 5.0)

        # Larger tensor
        large_tensor = torch.randn(2, 4, 8, 8)
        result = visualiser.get_attention_vector_mean(large_tensor)
        assert result.shape == (4, 8, 8)

    def test_compute_attentions_with_torch_no_grad(
        self, visualiser, mock_encoded_input
    ):
        """Test that compute_attentions uses torch.no_grad context."""
        class MockOutput:
            def __init__(self):
                self.attentions = [torch.ones((1, 12, 4, 4)) * 0.25] * 12

        output = MockOutput()
        visualiser.model.output = output

        # Track if no_grad was used
        original_no_grad = torch.no_grad
        with patch("torch.no_grad", wraps=original_no_grad) as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock(return_value=None)
            mock_no_grad.return_value.__exit__ = Mock(return_value=None)

            visualiser.compute_attentions(mock_encoded_input)

            # Verify no_grad was called
            mock_no_grad.assert_called_once()

    def test_attention_return_type(self, visualiser, mock_encoded_input):
        """Test that compute_attentions returns correct tuple type."""
        class MockOutput:
            def __init__(self):
                self.attentions = (torch.ones((1, 12, 4, 4)) * 0.25,) * 12

        output = MockOutput()
        visualiser.model.output = output

        result = visualiser.compute_attentions(mock_encoded_input)

        assert isinstance(result, tuple)
        assert len(result) == 12
        assert all(isinstance(attn, torch.Tensor) for attn in result)
