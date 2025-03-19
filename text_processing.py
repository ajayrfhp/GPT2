import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import urllib


class GPTDatasetV1(Dataset):
    """GPT Dataset V1"""

    def __init__(self, text, tokenizer, context, stride, log=False):
        self.input_ids = []
        self.target_ids = []
        self.stride = stride
        self.log = log

        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        if self.log:
            print("First 20 tokens are", tokens[0:20])
            print("First 20 decoded tokens are", tokenizer.decode(tokens[0:20]))

        for i in range(0, len(tokens) - context, stride):
            input_chunk = tokens[i : i + context]
            target_chunk = tokens[i + 1 : i + 1 + context]

            if len(input_chunk) != context or len(target_chunk) != context:
                continue

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

            if self.log and i <= 4:
                print(f"input ids", input_chunk)
                print(f"target ids", target_chunk)
                print(f"Input: {tokenizer.decode(input_chunk)}")
                print(f"Target: {tokenizer.decode(target_chunk)}")
                print(stride)
                print("\n\n")

        print(f"Number of training examples: {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    text,
    batch_size=4,
    context=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    log=False,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, context, stride, log=log)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    if log:
        print("First batch is", next(iter(dataloader)))

    return dataloader


def read_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def get_download_data(
    url="https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
    file_path="./data/verdict.txt",
):
    urllib.request.urlretrieve(url, file_path)
    return read_data(file_path)


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
