# Initialize GPT2 model
# Train GPT2 model
# Test GPT2 model
# Generate text using GPT2 model
import torch
import tiktoken
from models import GPT2
from text_processing import (
    generate_text_greedy,
    text_to_idx,
    idx_to_text,
    create_dataloader_v1,
    get_download_data,
)


def train(model, train_data, test_data, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(config["num_epochs"]):
        model.train()
        batch_idx = 0
        batch_loss = 0
        for inpt, target in train_data:
            batch_idx += 1
            optimizer.zero_grad()
            inpt = inpt.to(config["device"])
            target = target.to(config["device"])
            predictions = model(inpt)
            predictions = predictions.flatten(0, 1)
            target = target.flatten(0, 1)
            loss = torch.nn.CrossEntropyLoss()(predictions, target)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            if batch_idx % 100 == 0:
                avg_batch_loss = batch_loss / batch_idx
                print(f"Average batch loss: {avg_batch_loss}")
                batch_loss = 0
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {loss.item()}")

        test_loss = 0
        for inpt, target in test_data:
            inpt = inpt.to(config["device"])
            target = target.to(config["device"])
            predictions = model(inpt)
            predictions = predictions.flatten(0, 1)
            target = target.flatten(0, 1)
            loss = criterion(predictions, target).item()
            test_loss += loss
        print(f"Test loss: {test_loss}")


if __name__ == "__main__":
    # Define configuration
    config = {
        "emb_dim": 768,
        "heads": 12,
        "layers": 12,
        "vocab_size": 50257,
        "context_length": 128,
        "device": torch.device("cuda"),
        "drop_out": 0.1,
        "train_test_split": 0.8,
        "num_epochs": 10,
        "model_path": "./models/gpt2.pth",
    }

    raw_text = get_download_data()
    print(len(raw_text))
    split_idx = int(len(raw_text) * config["train_test_split"])
    train_text = raw_text[:split_idx]
    test_text = raw_text[split_idx:]

    print(len(train_text))
    print(len(test_text))

    train_data = create_dataloader_v1(
        train_text,
        batch_size=4,
        context=config["context_length"],
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        log=False,
    )

    test_data = create_dataloader_v1(
        test_text,
        batch_size=4,
        context=config["context_length"],
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        log=False,
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    # Initialize GPT2 model
    gpt2 = GPT2(config)
    gpt2.to(config["device"])

    # Train GPT2 model
    train(gpt2, train_data, test_data, config)

    starting_context = "The cat"

    generated_text = generate_text_greedy(
        gpt2,
        text_to_idx(starting_context, tokenizer).to(config["device"]),
        max_new_tokens=100,
        context_size=100,
        config=config,
    )

    print(idx_to_text(generated_text, tokenizer))

    # Save the model
    torch.save(gpt2.state_dict(), config["model_path"])

    # Load the model
    loaded_gpt2 = GPT2(config)
    loaded_gpt2.load_state_dict(torch.load(config["model_path"]))
    loaded_gpt2.to(config["device"])

    # Generate text using GPT2 model
    # generate_text(gpt2, config)

    generated_text = generate_text_greedy(
        loaded_gpt2,
        text_to_idx(starting_context, tokenizer).to(config["device"]),
        max_new_tokens=100,
        context_size=100,
        config=config,
    )

    print(idx_to_text(generated_text, tokenizer))
