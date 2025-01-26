import torch
import tiktoken
from text_processing import generate_text_greedy, text_to_idx, idx_to_text


def test_generate_text_greedy():
    vocab_size = 5
    embedding_dim = 10
    batch_size = 1
    max_new_tokens = 4
    context_size = 4
    input_token_size = 20

    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab_size, embedding_dim),
        torch.nn.Linear(embedding_dim, embedding_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(embedding_dim, vocab_size),
    )

    # batch size of 1 and 20 tokens. Vocabulary size is 5
    # context size is 4

    idx = torch.randint(0, vocab_size, (batch_size, input_token_size))
    output = generate_text_greedy(model, idx, max_new_tokens, context_size)
    assert output.shape == (batch_size, input_token_size + max_new_tokens)


def test_e2e_encoding():
    text = "hello world"
    tokenizer = tiktoken.get_encoding("gpt2")
    idx = text_to_idx(text, tokenizer)
    assert idx_to_text(idx, tokenizer) == text
