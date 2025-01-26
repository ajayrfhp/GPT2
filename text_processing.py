import torch


def generate_text_greedy(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        logits = model(
            idx[:, -context_size:]
        )  # consider only last set of context size tokens

        next_token_logit = logits[:, -1, :]

        probs = torch.softmax(next_token_logit, dim=-1)

        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_idx(text, tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def idx_to_text(idx, tokenizer):
    return tokenizer.decode(idx[0].tolist())
