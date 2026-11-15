"""Small runtime helpers shared by quantized and unquantized runners."""


def describe_model_device_map(model):
    """Return an auditable device map across Transformers model classes."""
    accelerate_map = getattr(model, 'hf_device_map', None)
    if accelerate_map:
        return {str(key): str(value) for key, value in accelerate_map.items()}

    device = getattr(model, 'device', None)
    if device is None:
        embeddings = model.get_input_embeddings()
        device = getattr(getattr(embeddings, 'weight', None), 'device', None)
    if device is None:
        raise RuntimeError('Loaded model exposes neither hf_device_map nor a parameter device')
    return {'': str(device)}
