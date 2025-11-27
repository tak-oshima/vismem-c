"""
Google Vertex AI Imagen 4 backend for image generation.
Uses the correct ImageGenerationModel API from vertexai.preview.vision_models.
"""

import os
import time
from typing import Dict, Any, List
from pathlib import Path

try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    print("[warn] Vertex AI SDK not installed. Run: pip install --upgrade google-cloud-aiplatform>=1.114.0")

from .base import ImageGenerator


class GoogleImagenBackend(ImageGenerator):
    """Backend using Google Vertex AI Imagen 4.0 model."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Google Imagen backend."""
        super().__init__(config)
        
        if not VERTEX_AI_AVAILABLE:
            raise ImportError("Vertex AI SDK not installed. Run: pip install --upgrade google-cloud-aiplatform>=1.114.0")
        
        # Set credentials path
        credential_path = "google-cloud-key.json"
        if os.path.exists(credential_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
        else:
            print("[warn] google-cloud-key.json not found. Ensure GOOGLE_APPLICATION_CREDENTIALS is set.")
        
        # Initialize Vertex AI
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or "local-cogency-479508-i6"
        location = "us-central1"
        
        try:
            vertexai.init(project=project_id, location=location)
            print(f"[Imagen] Initialized Vertex AI with project: {project_id}, location: {location}")
        except Exception as e:
            print(f"[Imagen] Error initializing Vertex AI: {e}")
            raise
        
        self.project_id = project_id
        self.location = location
        # Use GA model name (not preview)
        self.model_name = "imagen-4.0-generate-001"  # GA version
    
    def generate(self, prompt: str, count: int = 1, output_dir: str = 'outputs/images', **kwargs):
        """Generate images using Vertex AI Imagen 4.0."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files: List[Dict[str, Any]] = []
        
        try:
            # Load the model using ImageGenerationModel (NOT GenerativeModel)
            print(f"[Imagen] Loading model: {self.model_name}")
            model = ImageGenerationModel.from_pretrained(self.model_name)
            
            for i in range(count):
                try:
                    start_time = time.time()
                    
                    print(f"[Imagen] Generating image {i+1}/{count}...")
                    
                    # Generate image using the correct method
                    # Returns ImageGenerationResponse object (NOT a list)
                    response = model.generate_images(
                        prompt=prompt,
                        number_of_images=1,
                        language="en",
                        aspect_ratio="1:1",
                        safety_filter_level="block_some",
                        add_watermark=False  # No watermark for cleaner images
                    )
                    
                    generation_time = time.time() - start_time
                    
                    # Access the images list from the response object
                    # response.images is the actual list of GeneratedImage objects
                    if response.images and len(response.images) > 0:
                        timestamp = int(time.time() * 1000)
                        filename = f"imagen_{timestamp}_{i}.png"
                        filepath = output_dir / filename
                        
                        # Save using the built-in save method
                        response.images[0].save(location=str(filepath), include_generation_parameters=False)
                        
                        generated_files.append({
                            'filepath': str(filepath),
                            'generation_time': generation_time,
                            'backend': self.get_backend_name(),
                            'model': self.model_name,
                        })
                        
                        print(f"✓ Generated image {i + 1}/{count}: {filepath} (took {generation_time:.2f}s)")
                    else:
                        print(f"✗ No image data returned for image {i + 1}")
                    
                except Exception as e:
                    print(f"✗ Error generating image {i + 1}: {type(e).__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        except Exception as e:
            print(f"✗ Fatal error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        return generated_files
    
    def get_backend_name(self) -> str:
        return "imagen"
