import torch
import tokenizers
import transformers
import tiktoken
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import tempfile
from tokenizers import AddedToken
from transformers import PreTrainedTokenizerFast
import os


def train(model, train_data, test_data, config, use_fp_16=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler() if use_fp_16 else None

    for epoch in range(config["num_epochs"]):
        model.train()
        batch_idx = 0
        batch_loss = 0
        for batch in train_data:
            if type(batch) == dict:
                inpt = batch["input_ids"]
                target = batch["output_ids"]
                attention_mask = batch["attention_mask"]
            else:
                inpt, target, attention_mask = batch
            batch_idx += 1
            optimizer.zero_grad()
            inpt = inpt.to(config["device"])
            target = target.to(config["device"])
            attention_mask = attention_mask.to(config["device"])
            device_type = inpt.device.type
            with torch.amp.autocast(enabled=use_fp_16, device_type=device_type):
                predictions = model(inpt)
                predictions = predictions.flatten(0, 1)
                target = target.flatten(0, 1)
                attention_mask = attention_mask.flatten(0, 1)

                masked_indices = attention_mask == 0
                target[masked_indices] = 0
                predictions[masked_indices] = 0

                loss = criterion(predictions, target)

            if use_fp_16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_loss += loss.item()
            if batch_idx % 100 == 0 or batch_idx == 1:
                avg_batch_loss = batch_loss / batch_idx
                perplexity = torch.exp(torch.tensor(avg_batch_loss)).item()
                print(
                    f"At epoch {epoch+1} batch {batch_idx} of num_batches {config['num_train_batches']} Average batch loss: {avg_batch_loss} Perplexity: {perplexity}"
                )

        with torch.no_grad():
            test_loss_total = 0
            test_loss_running = 0
            model.eval()
            test_batch_idx = 0
            for batch in test_data:
                if type(batch) == dict:
                    inpt = batch["input_ids"]
                    target = batch["output_ids"]
                    attention_mask = batch["attention_mask"]
                else:
                    inpt, target, attention_mask = batch
                test_batch_idx += 1
                inpt = inpt.to(config["device"])
                target = target.to(config["device"])
                attention_mask = attention_mask.to(config["device"])
                device_type = inpt.device.type

                with torch.amp.autocast(enabled=use_fp_16, device_type=device_type):
                    predictions = model(inpt)
                    predictions = predictions.flatten(0, 1)
                    attention_mask = attention_mask.flatten(0, 1)

                    target = target.flatten(0, 1)

                    masked_indices = attention_mask == 0
                    target[masked_indices] = 0
                    predictions[masked_indices] = 0
                    loss = criterion(predictions, target).item()

                test_loss_total += loss
                test_loss_running += loss
                if test_batch_idx % 100 == 0 or test_batch_idx == 1:
                    avg_test_loss = test_loss_running / test_batch_idx
                    print(
                        f"At epoch {epoch+1} batch {test_batch_idx} of num_batches {config['num_test_batches']} Average test loss: {avg_test_loss}"
                    )
            test_loss_total /= len(test_data)
            test_perplexity = torch.exp(torch.tensor(test_loss_total))
            print(
                f"Test loss without mask: at epoch {epoch} {test_loss_total} Test perplexity without mask: {test_perplexity}"
            )

        model_path = f"{config['model_path']}_epoch_{epoch+1}.pt"
        # save model and optimizer state
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": test_loss_total,
            },
            model_path,
        )


def slide_window(text_batch, wrapped_tokenizer, max_length=128):
    """ """

    # --- Step 1 & 2: Tokenize, Add EOS, Create Shifted Inputs/Outputs (Raw) ---

    all_tokens = []
    input_ids_raw = []
    output_ids_raw = []

    eos_token_id = wrapped_tokenizer.convert_tokens_to_ids(wrapped_tokenizer.eos_token)

    for text in text_batch["text"]:

        raw_tokens = wrapped_tokenizer.tokenize(text, truncation=True)
        # iterate over tokens in chunks of size max_length
        for i in range(0, len(raw_tokens), max_length - 1):
            tokens = raw_tokens[i : i + max_length - 1]

            assert len(tokens) <= max_length - 1, "Token length exceeds max_length"
            token_ids = wrapped_tokenizer.convert_tokens_to_ids(tokens)
            token_ids.append(eos_token_id)  # Add EOS ID

            # Create input/output pairs (before padding/truncation)
            current_input_ids = token_ids[:-1]
            current_output_ids = token_ids[1:]

            input_ids_raw.append(current_input_ids)
            output_ids_raw.append(current_output_ids)
            all_tokens.append(tokens)

    padded_inputs = wrapped_tokenizer.pad(
        {"input_ids": input_ids_raw},
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )

    padded_outputs = wrapped_tokenizer.pad(
        {"input_ids": output_ids_raw},
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": padded_inputs["input_ids"],
        "attention_mask": padded_inputs["attention_mask"],
        "output_ids": padded_outputs["input_ids"],
    }


def get_train_tokenizer(train_dataset, vocab_size=10000):
    # re-train tokenizer on train_dataset

    # Initialize a new tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Define special tokens
    special_tokens = ["<|endoftext|>"]
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)

    # Create a temporary file with raw text for training
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        for text in train_dataset["text"]:
            f.write(text + "\n")
        temp_file_path = f.name

    # Train the tokenizer
    tokenizer.train(files=[temp_file_path], trainer=trainer)

    # Convert to Hugging Face tokenizer

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        padding_side="left",
    )

    # Clean up the temporary file
    os.unlink(temp_file_path)

    print(
        f"Tokenizer trained on custom dataset with vocabulary size: {wrapped_tokenizer.vocab_size}"
    )
    return wrapped_tokenizer


def get_sum_parameters_of_model(model, millions=True):
    """Get number of parameters of model

    Args:
        model (_type_): _description_
        millions (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    scale = 1e6 if millions else 1
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / scale


def get_memory_footprint_of_model(model, millions=True):
    """_summary_

    Args:
        model (_type_): _description_
        millions (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    scale = 1e6 if millions else 1
    return (
        sum(p.element_size() * p.numel() for p in model.parameters() if p.requires_grad)
        / scale
    )
