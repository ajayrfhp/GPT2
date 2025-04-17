import torch


def train(model, train_data, test_data, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
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

            # apply attention mask to target and penalize only over non-masked tokens

            predictions = model(inpt)
            predictions = predictions.flatten(0, 1)
            target = target.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)

            masked_indices = attention_mask == 0
            target[masked_indices] = 0
            predictions[masked_indices] = 0

            loss = torch.nn.CrossEntropyLoss()(predictions, target)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            if batch_idx % 10 == 0 or batch_idx == 1:
                avg_batch_loss = batch_loss / batch_idx
                print(
                    f"At epoch {epoch+1} batch {batch_idx} of num_batches {config['num_train_batches']}Average batch loss: {avg_batch_loss}"
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
                if test_batch_idx % 10 == 0 or test_batch_idx == 1:
                    avg_test_loss = test_loss_running / test_batch_idx
                    print(
                        f"At epoch {epoch+1} batch {test_batch_idx} of num_batches {config['num_test_batches']}Average test loss: {avg_test_loss}"
                    )
            test_loss_total /= len(test_data)
            test_perplexity = torch.exp(torch.tensor(test_loss_total))
            print(
                f"Test loss without mask: at epoch {epoch} {test_loss_total} Test perplexity without mask: {test_perplexity}"
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
