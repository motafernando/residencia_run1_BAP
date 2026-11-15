"""Strict, explicit configuration for the two quantization study conditions."""

def validate_quantization(value):
    if not isinstance(value, dict) or value.get('method') != 'bitsandbytes':
        raise ValueError('Expected a bitsandbytes quantization configuration')
    int8 = {'method': 'bitsandbytes', 'load_in_8bit': True}
    nf4 = {'method': 'bitsandbytes', 'load_in_4bit': True,
           'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': 'bfloat16',
           'bnb_4bit_use_double_quant': False}
    if value == int8:
        return 'int8'
    if value == nf4:
        return 'nf4'
    raise ValueError('Unsupported quantization settings: require reviewed int8 or NF4 profile')


def build_quantization(value, torch, BitsAndBytesConfig):
    mode = validate_quantization(value)
    if mode == 'int8':
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=False)


def summarize_quantized_model(model, bnb, requested):
    mode = validate_quantization(requested)
    cls = bnb.nn.Linear8bitLt if mode == 'int8' else bnb.nn.Linear4bit
    names = [name for name, module in model.named_modules() if isinstance(module, cls)]
    loaded = getattr(model, 'is_loaded_in_8bit' if mode == 'int8' else 'is_loaded_in_4bit', False)
    if not loaded or not names:
        raise RuntimeError('Model did not load quantized modules in requested precision')
    devices = {str(key): str(value) for key, value in model.hf_device_map.items()}
    if any(device not in ('0', 'cuda:0') for device in devices.values()):
        raise RuntimeError('Model was not loaded entirely on the requested GPU')
    return {'mode': mode, 'module_class': cls.__name__, 'module_count': len(names),
            'module_names': names, 'loaded_flag': bool(loaded), 'requested': requested,
            'effective_transformers_config': model.config.quantization_config.to_dict()
                if hasattr(model.config.quantization_config, 'to_dict') else model.config.quantization_config}
