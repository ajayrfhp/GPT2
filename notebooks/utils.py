import torch


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
                print(
                    f"At epoch{epoch+1} batch{batch_idx}Average batch loss: {avg_batch_loss}"
                )
                batch_loss = 0
        perplexity = torch.exp(loss)
        print(
            f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {loss.item()} Perplexity: {perplexity.item()}"
        )

        test_loss = 0
        for inpt, target in test_data:
            inpt = inpt.to(config["device"])
            target = target.to(config["device"])
            predictions = model(inpt)
            predictions = predictions.flatten(0, 1)
            target = target.flatten(0, 1)
            loss = criterion(predictions, target).item()
            test_loss += loss
        test_loss /= len(test_data)
        test_perplexity = torch.exp(test_loss)
        print(f"Test loss: {test_loss} Test perplexity: {test_perplexity}")


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
