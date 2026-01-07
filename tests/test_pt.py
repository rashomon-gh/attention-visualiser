import pytest
import numpy as np
import torch
from attention_visualiser import AttentionVisualiser


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
