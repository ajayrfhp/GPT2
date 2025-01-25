import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, config):
        """Initializes the MultiHeadAttention module

        Args:
            d_in (_type_): Input dimension
            d_out (_type_): Output dimension
            heads (_type_): Number of heads
        """

        super().__init__()
        d_in = config["emb_dim"]
        d_out = config["emb_dim"]
        heads = config["heads"]

        assert d_out % heads == 0, "d_out must be divisible by heads"

        self.heads = heads
        self.d_out = d_out

        self.head_dim = int(d_out / heads)  # dimension of one head

        # Define nn.Linear layers for Q, K, and V
        self.W_Q = nn.Linear(d_in, d_out)  # (d_in, d_out)
        self.W_K = nn.Linear(d_in, d_out)  # (d_in, d_out)
        self.W_V = nn.Linear(d_in, d_out)  # (d_in, d_out)

        self.out_project = nn.Linear(d_out, d_out)

        self.self_attention_matrix = None

    def forward(self, x):
        # shape of input is (BS, T, d_in)

        token_length = x.shape[1]

        # Calculate Q, K, and V using nn.Linear layers
        Q = self.W_Q(x)  # (BS, T, d_out)
        Q = Q.view(
            x.shape[0], token_length, self.heads, self.head_dim
        )  # (BS, T, heads, d_out)
        Q = Q.permute(0, 2, 1, 3)  # (BS, heads, T, d_out)

        K = self.W_K(x)  # (BS, T, d_out)
        K = K.view(
            x.shape[0], token_length, self.heads, self.head_dim
        )  # (BS, T, heads, d_out)
        K = K.permute(0, 2, 1, 3)  # (BS, heads, T, d_out)

        V = self.W_V(x)  # (BS, T, d_out)
        V = V.view(
            x.shape[0], token_length, self.heads, self.head_dim
        )  # (BS, T, heads, d_out)
        V = V.permute(0, 2, 1, 3)  # (BS, heads, T, d_out)

        # Calculate attention scores
        self_attention_scores = torch.matmul(
            Q, K.transpose(-1, -2)
        )  # (BS, heads, T, T)
        self_attention_scores = self_attention_scores / torch.sqrt(
            torch.tensor(self.head_dim)
        )  # (BS, heads, T, T)

        # Apply causal attention mask
        mask = torch.triu(
            torch.ones_like(self_attention_scores, dtype=bool), diagonal=1
        )  # (BS, heads, T, T)
        self_attention_scores = torch.where(
            mask, -float("inf"), self_attention_scores
        )  # (BS, heads, T, T)

        self_attention_matrix = torch.softmax(
            self_attention_scores, dim=-1
        )  # (BS, heads, T, T)

        # Save attention matrix for testing
        self.self_attention_matrix = self_attention_matrix

        XV = (
            torch.matmul(self_attention_matrix, V)
            .transpose(1, 2)
            .reshape((x.shape[0], token_length, -1))
        )  # (BS, T, heads, d_out)

        o = self.out_project(XV)  # (BS, T, d_out)

        return o


class LayerNorm(nn.Module):
    """Layer Normalization module"""

    def __init__(self, config):
        """Initializes the LayerNorm module

        Args:
            config (_type_): config
        """
        super().__init__()
        self.config = config
        self.scale = nn.Parameter(torch.ones(config["emb_dim"]))
        self.bias = nn.Parameter(torch.zeros(config["emb_dim"]))

    def forward(self, x):
        """Forward pass

        Args:
            x (_type_): input tensor

        Returns:
            _type_: normalized output tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) * self.scale / std + self.bias


class FeedForward(nn.Module):
    """Feed Forward module

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config):
        """Initializes the FeedForward module
        Expands to 4 times the input dimension, applies GELU activation, and then reduces back to the original dimension

        Args:
            config (_type_): config
        """
        super().__init__()
        self.config = config
        self.feed_forward = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, x):
        """Forward pass

        Args:
            x (_type_): input tensor

        Returns:
            _type_: output tensor
        """
        return self.feed_forward(x)


class TransformerBlock(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config):
        """_summary_

        Args:
            config (_type_): config
        """
        super().__init__()
        self.config = config

        self.layer_norm1 = LayerNorm(config)
        self.layer_norm2 = LayerNorm(config)
        self.self_attention_block = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): input tensor

        Returns:
            _type_: output tensor
        """
        x_skip = x
        x = self.layer_norm1(x)
        x = self.self_attention_block(x)
        x = nn.Dropout(self.config["drop_out"])(x)
        x = x + x_skip

        x_skip = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = nn.Dropout(self.config["drop_out"])(x)

        x = x + x_skip

        return x
