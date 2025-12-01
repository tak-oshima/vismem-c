import os
import requests
from typing import List, Dict, Any
from pathlib import Path
import time
from .base import ImageGenerator

try:
    import openai
except ImportError:
    raise ImportError("OpenAI library not installed. Install with: pip install openai")

class OpenAIBackend(ImageGenerator):
    """Image generation backend using OpenAI's DALL-E API."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.size = config.get('size', '1024x1024')
        self.quality = config.get('quality', 'standard')
        self.model = config.get('model', 'dall-e-3')
        
    def generate(self, prompt: str, count: int = 1, output_dir: str = 'outputs/images', **kwargs) -> List[str]:
        """
        Generate images using OpenAI DALL-E API.
        
        Args:
            prompt: Text description for image generation
            count: Number of images to generate (DALL-E 3 supports 1 image per call)
            output_dir: Directory to save images
            **kwargs: Additional parameters
            
        Returns:
            List of file paths to generated images
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        generated_files = []
        
        # DALL-E 3 only supports n=1, so we loop for multiple images
        for i in range(count):
            try:
                start_time = time.time()
                
                response = openai.images.generate(
                    model=self.model,
                    prompt=prompt,
                    size=self.size,
                    quality=self.quality,
                    n=1
                )
                
                generation_time = time.time() - start_time
                
                # Get image URL
                image_url = response.data[0].url
                
                # Download image
                img_data = requests.get(image_url).content
                
                # Generate filename
                timestamp = int(time.time() * 1000)
                filename = f"openai_{timestamp}_{i}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                
                generated_files.append({
                    'filepath': filepath,
                    'generation_time': generation_time,
                    'backend': self.get_backend_name()
                })
                
                print(f"Generated image {i+1}/{count}: {filepath} (took {generation_time:.2f}s)")
                
            except Exception as e:
                print(f"Error generating image {i+1}: {str(e)}")
                continue
        
        return generated_files
    
    def get_backend_name(self) -> str:
        return "openai"
