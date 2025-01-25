def get_sum_parameters_of_model(model, millions=True):
    scale = 1e6 if millions else 1
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / scale

def get_memory_footprint_of_model(model):
    return sum(p.element_size() * p.numel() for p in model.parameters() if p.requires_grad)