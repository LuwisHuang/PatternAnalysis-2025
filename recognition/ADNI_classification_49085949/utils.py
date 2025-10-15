"""
ConvNeXt model utility functions
Provides constructors for different ConvNeXt variants with optional pretrained weights
"""

import torch
from modules import ConvNeXt


MODEL_URLS = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def _load_pretrained_weights(model, url, strict=False):
    """Load pretrained weights from a URL"""
    try:
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=strict)
    except Exception:
        pass
    return model


def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = MODEL_URLS['convnext_tiny_22k'] if in_22k else MODEL_URLS['convnext_tiny_1k']
        strict = kwargs.get('in_chans', 3) == 3 and kwargs.get('num_classes', 1000) == 1000
        model = _load_pretrained_weights(model, url, strict=strict)
    return model


def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = MODEL_URLS['convnext_small_22k'] if in_22k else MODEL_URLS['convnext_small_1k']
        strict = kwargs.get('in_chans', 3) == 3 and kwargs.get('num_classes', 1000) == 1000
        model = _load_pretrained_weights(model, url, strict=strict)
    return model


def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = MODEL_URLS['convnext_base_22k'] if in_22k else MODEL_URLS['convnext_base_1k']
        strict = kwargs.get('in_chans', 3) == 3 and kwargs.get('num_classes', 1000) == 1000
        model = _load_pretrained_weights(model, url, strict=strict)
    return model


def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = MODEL_URLS['convnext_large_22k'] if in_22k else MODEL_URLS['convnext_large_1k']
        strict = kwargs.get('in_chans', 3) == 3 and kwargs.get('num_classes', 1000) == 1000
        model = _load_pretrained_weights(model, url, strict=strict)
    return model


def convnext_xlarge(pretrained=False, in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        url = MODEL_URLS['convnext_xlarge_22k']
        strict = kwargs.get('in_chans', 3) == 3 and kwargs.get('num_classes', 1000) == 1000
        model = _load_pretrained_weights(model, url, strict=strict)
    return model
