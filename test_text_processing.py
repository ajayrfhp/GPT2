import torch
import tiktoken
from text_processing import (
    generate_text_greedy,
    text_to_idx,
    idx_to_text,
    GPTDatasetV1,
)


def test_generate_text_greedy():
    """test text generation"""
    vocab_size = 5
    embedding_dim = 10
    batch_size = 1
    max_new_tokens = 4
    context_size = 6
    input_token_size = 20

    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab_size, embedding_dim),
        torch.nn.Linear(embedding_dim, embedding_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(embedding_dim, vocab_size),
    )

    # batch size of 1 and 20 tokens. Vocabulary size is 5
    # context size is 6 and we generate 4 new tokens
    # so the output should be of size 1 x 24

    idx = torch.randint(0, vocab_size, (batch_size, input_token_size))
    output = generate_text_greedy(model, idx, max_new_tokens, context_size)
    assert output.shape == (batch_size, input_token_size + max_new_tokens)


def test_e2e_encoding():
    """test end to end encoding and decoding"""
    sample_texts = [
        "This is a test",
        "This is another test",
        "This is a test with a long sentence that has multiple words",
        "Hello there !!,",
    ]
    for text in sample_texts:
        tokenizer = tiktoken.get_encoding("gpt2")
        idx = text_to_idx(text, tokenizer)
        assert idx_to_text(idx, tokenizer) == text


def test_gptdataset():
    """test dataset generation"""
    sample_texts = "This is a test"
    tokenizer = tiktoken.get_encoding("gpt2")
    context = 3
    stride = 1
    dataset = GPTDatasetV1(
        sample_texts, tokenizer, context=context, stride=stride, log=True
    )

    assert (
        len(dataset) == 1
    ), "This is a test with context 3 and stride 1 has only one example"
    assert dataset.input_ids[0].shape == (
        context,
    ), "input_ids should have shape (context,)"
    assert dataset.target_ids[0].shape == (
        context,
    ), "target_ids should have shape (context,)"

    context = 2
    stride = 1
    dataset = GPTDatasetV1(
        sample_texts, tokenizer, context=context, stride=stride, log=True
    )
    assert (
        len(dataset) == 2
    ), "This is a test with context 2 and stride 1 has only one example"
    assert dataset.input_ids[0].shape == (
        context,
    ), "input_ids should have shape (context,)"
    assert dataset.target_ids[0].shape == (
        context,
    ), "target_ids should have shape (context,)"
