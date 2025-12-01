import os
from typing import List, Dict, Any
from pathlib import Path
import time
import torch
from .base import ImageGenerator

try:
    from diffusers import DiffusionPipeline
except ImportError:
    raise ImportError("Diffusers library not installed. Install with: pip install diffusers")

class StableDiffusionBackend(ImageGenerator):
    """Image generation backend using Stable Diffusion."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Use a model that's publicly available and doesn't require authentication
        self.model_id = config.get('model_id', 'runwayml/stable-diffusion-v1-5')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Stable Diffusion model: {self.model_id}")
        print(f"Using device: {self.device}")
        print(f"Note: First run will download ~4-5 GB model. This may take several minutes...")
        
        try:
            # Load pipeline WITHOUT requiring authentication
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            self.pipeline = self.pipeline.to(self.device)
            
            # Optional: Enable memory efficient attention
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            # Enable safety checker
            if hasattr(self.pipeline, 'safety_checker'):
                self.pipeline.safety_checker = None  # Disable for faster generation
                
            print("✓ Model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            print("Make sure you have internet connection for first-time download.")
            raise
    
    def generate(self, prompt: str, count: int = 1, output_dir: str = 'outputs/images', **kwargs) -> List[str]:
        """
        Generate images using Stable Diffusion.
        
        Args:
            prompt: Text description for image generation
            count: Number of images to generate
            output_dir: Directory to save images
            **kwargs: Additional parameters (num_inference_steps, guidance_scale, etc.)
            
        Returns:
            List of file paths to generated images
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        generated_files = []
        
        num_inference_steps = kwargs.get('num_inference_steps', 30)  # Reduced for speed
        guidance_scale = kwargs.get('guidance_scale', 7.5)
        
        for i in range(count):
            try:
                start_time = time.time()
                
                print(f"  Generating image {i+1}/{count}...")
                
                # Generate image
                if self.device == 'cuda':
                    with torch.autocast('cuda'):
                        output = self.pipeline(
                            prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            height=512,
                            width=512
                        )
                        image = output.images[0]
                else:
                    output = self.pipeline(
                        prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=512,
                        width=512
                    )
                    image = output.images[0]
                
                generation_time = time.time() - start_time
                
                # Generate filename
                timestamp = int(time.time() * 1000)
                filename = f"sd_{timestamp}_{i}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save image
                image.save(filepath)
                
                generated_files.append({
                    'filepath': filepath,
                    'generation_time': generation_time,
                    'backend': self.get_backend_name()
                })
                
                print(f"  ✓ Saved: {filepath} ({generation_time:.2f}s)")
                
            except Exception as e:
                print(f"  ✗ Error generating image {i+1}: {str(e)}")
                continue
        
        return generated_files
    
    def get_backend_name(self) -> str:
        return "stable_diffusion"
