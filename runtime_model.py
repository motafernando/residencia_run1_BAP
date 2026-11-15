"""Small runtime helpers shared by quantized and unquantized runners."""
import copy
from pathlib import Path


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


def ensure_chat_template(tokenizer, token_snapshot):
    """Load standalone chat_template.jinja when tokenizer_config omits it."""
    if getattr(tokenizer, 'chat_template', None):
        return tokenizer.chat_template
    template_path = Path(token_snapshot) / 'chat_template.jinja'
    if template_path.exists():
        tokenizer.chat_template = template_path.read_text(encoding='utf-8')
    return getattr(tokenizer, 'chat_template', None)


def prepare_generation_config(base_generation_config, settings, tokenizer):
    """Build deterministic generation config without leftover sampling flags."""
    generation = copy.deepcopy(base_generation_config)
    updated = dict(settings)
    updated['pad_token_id'] = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    generation.update(**updated)
    if generation.eos_token_id is None:
        generation.eos_token_id = tokenizer.eos_token_id
    if not getattr(generation, 'do_sample', False):
        for flag in ('temperature', 'top_p', 'top_k'):
            if hasattr(generation, flag):
                setattr(generation, flag, None)
    return generation
