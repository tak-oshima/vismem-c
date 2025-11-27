from .base import ImageGenerator
from .openai_backend import OpenAIBackend
from .openai_gptimage1 import OpenAIGPTImage1Backend
from .sd_backend import StableDiffusionBackend
from .google_imagen import GoogleImagenBackend

__all__ = ['ImageGenerator', 'OpenAIBackend', 'StableDiffusionBackend','GoogleImagenBackend','OpenAIGPTImage1Backend']

def get_backend(backend_name: str, config: dict = None):
    """Factory function to get the appropriate backend."""
    backends = {
        'openai': OpenAIBackend,
        'sd': StableDiffusionBackend,
        'stable_diffusion': StableDiffusionBackend,
        'gptimage1': OpenAIGPTImage1Backend,
        'imagen': GoogleImagenBackend
              }
    
    if backend_name not in backends:
        raise ValueError(f"Unknown backend: {backend_name}. Available: {list(backends.keys())}")
    
    return backends[backend_name](config)
