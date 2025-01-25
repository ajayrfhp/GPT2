import torch
import torch.nn as nn
from models import TransformerBlock


class GPT2(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config):
        """_summary_

        Args:
            config (_type_): _description_
        """
        super().__init__()
        self.config = config

        # token embedding
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])

        # position embedding
        self.position_embedding = nn.Embedding(
            config["context_length"], config["emb_dim"]
        )

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["layers"])]
        )

        self.output_projection = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=False
        )

    def forward(self, x):
        """Forward pass

        Args:
            x (_type_): Input tensor

        Returns:
            _type_: Output tensor
        """
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(torch.arange(x.shape[1]))
        x = token_embeddings + position_embeddings

        x = self.transformer_blocks(x)

        x = self.output_projection(x)

        return x
