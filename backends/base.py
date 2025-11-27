from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ImageGenerator(ABC):
    """Abstract base class for image generation backends."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    def generate(self, prompt: str, count: int = 1, **kwargs) -> List[str]:
        """
        Generate images from a text prompt.
        
        Args:
            prompt: Text description for image generation
            count: Number of images to generate
            **kwargs: Additional backend-specific parameters
            
        Returns:
            List of file paths to generated images
        """
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of the backend."""
        pass
