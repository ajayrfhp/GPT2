# Initialize GPT2 model
# Train GPT2 model
# Test GPT2 model
# Generate text using GPT2 model
import torch
import tiktoken
from gpt2 import GPT2
from text_processing import generate_text_greedy, text_to_idx, idx_to_text


def train_gpt2(gpt2, config):
    pass


def test_gpt2(gpt2):
    pass


if __name__ == "__main__":
    # Define configuration
    config = {
        "emb_dim": 768,
        "heads": 12,
        "layers": 12,
        "vocab_size": 50257,
        "context_length": 1024,
        "device": torch.device("cpu"),
        "drop_out": 0.1,
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    # Initialize GPT2 model
    gpt2 = GPT2(config)
    gpt2.to(config["device"])

    # Train GPT2 model
    train_gpt2(gpt2, config)

    # Test GPT2 model
    test_gpt2(gpt2)

    starting_context = "The cat"

    # Generate text using GPT2 model
    # generate_text(gpt2, config)

    generated_text = generate_text_greedy(
        gpt2,
        text_to_idx(starting_context, tokenizer),
        max_new_tokens=100,
        context_size=100,
    )

    print(idx_to_text(generated_text, tokenizer))
