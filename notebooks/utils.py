import torch


def train(model, train_data, test_data, config, use_fp_16=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler() if use_fp_16 else None

    for epoch in range(config["num_epochs"]):
        model.train()
        batch_idx = 0
        batch_loss = 0
        for inpt, target, attention_mask in train_data:
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
            for inpt, target, attention_mask in test_data:
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
