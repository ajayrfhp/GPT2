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


def get_memory_footprint_of_model(model):
    """Get memory footprint of model

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    return sum(
        p.element_size() * p.numel() for p in model.parameters() if p.requires_grad
    )
