"""Google Vertex AI Imagen image generation backend."""

import os
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    from google.api_core import exceptions as google_exceptions
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

from backends.base import ImageGenerator


class GoogleImagenBackend(ImageGenerator):
    """Google Vertex AI Imagen backend for image generation."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Google Imagen backend."""
        super().__init__(config)

        if not VERTEX_AI_AVAILABLE:
            raise ImportError(
                "Vertex AI SDK not available. Install: pip install google-cloud-aiplatform"
            )

        # Optional local service account key
        credential_path = "google-cloud-key.json"
        if os.path.exists(credential_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

        project_id = (
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT_ID")
            or "your-default-project-id"
        )
        location = "us-central1"

        try:
            vertexai.init(project=project_id, location=location)
            logger.info(
                f"Vertex AI initialized: project={project_id}, location={location}"
            )
        except Exception as e:
            logger.error(f"Vertex AI init failed: {e}")
            raise

        self.model_name = "imagegeneration@006"
        try:
            self.model = ImageGenerationModel.from_pretrained(self.model_name)
            logger.info("Imagen model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Imagen model: {e}")
            raise

        self.enhance_prompt = config.get("enhance_prompt", True) if config else True

    def simplify_prompt(self, prompt: str) -> str:
        """Create a simplified version of the prompt for safety filter bypass."""
        # Remove potentially problematic keywords
        problematic_words = [
            "real",
            "realistic",
            "photorealistic",
            "hyper-realistic",
            "celebrity",
            "person",
            "face",
            "people",
            "human",
            "violence",
            "graphic",
            "blood",
            "gore",
        ]

        simplified = prompt
        for word in problematic_words:
            simplified = simplified.lower().replace(word.lower(), "")

        # Clean up multiple spaces
        simplified = " ".join(simplified.split())

        if not simplified or len(simplified) < 5:
            simplified = "A beautiful scene"

        return simplified

    def enhance_prompt_text(self, prompt: str) -> str:
        """Enhance prompt for better image generation."""
        if not self.enhance_prompt or not prompt:
            return prompt

        enhancements = {
            "receipt": "professional product photograph, clear text, high resolution, perfect lighting",
            "cafe": "architectural photograph, warm ambiance, cozy interior, golden hour lighting",
            "food": "food photograph, appetizing, professional lighting, detailed textures, high resolution",
            "document": "document photograph, high contrast, clear text, sharp details, commercial quality",
            "outdoor": "outdoor photograph, natural lighting, scenic, high resolution, professional composition",
            "portrait": "portrait photograph, studio lighting, sharp focus, beautiful lighting",
            "architecture": "architectural photograph, professional lighting, geometric precision, detailed",
            "still life": "still life photograph, studio lighting, professional composition, detailed textures",
        }

        lower_prompt = prompt.lower()
        for keyword, enhancement in enhancements.items():
            if keyword in lower_prompt:
                return f"{prompt}, {enhancement}"

        return f"{prompt}, high quality photograph, sharp focus, detailed, well-lit"

    def _map_safety_level(self, logical_level: str) -> str:
        """
        Map logical safety levels to Imagen's string-based safety_filter_level.
        
        Supported levels:
          - "block_most"
          - "block_some" 
          - "block_few" (least restrictive)
        
        We avoid "block_none" to prevent allowlist-only errors.
        """
        if logical_level == "block_some":
            return "block_some"
        if logical_level == "block_none":
            # Map to least restrictive public option (avoids allowlist error)
            return "block_few"
        # Default
        return "block_some"

    def generate(
        self,
        prompt: str,
        count: int = 1,
        output_dir: str = "outputs/images",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate images using Vertex AI Imagen with retry logic."""
        enhance = kwargs.get("enhance_prompt", self.enhance_prompt)
        original_prompt = prompt

        if enhance:
            prompt = self.enhance_prompt_text(prompt)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Keep your original retry logic and safety_level naming
        prompt_variants = [
            (prompt, "block_some"),  # Original prompt, moderate filter
            (self.simplify_prompt(prompt), "block_some"),  # Simplified, same level
            (self.simplify_prompt(original_prompt), "block_none"),  # Simplified original, mapped to block_few
        ]

        results: List[Dict[str, Any]] = []
        start_time = time.time()

        for attempt, (prompt_to_try, safety_level) in enumerate(prompt_variants):
            try:
                api_level = self._map_safety_level(safety_level)
                logger.info(
                    f"[Attempt {attempt+1}] Generating with prompt: "
                    f"{prompt_to_try[:80]}..."
                )
                logger.info(
                    f"  Safety level (logical): {safety_level}, "
                    f"API safety_filter_level: {api_level}"
                )

                response = self.model.generate_images(
                    prompt=prompt_to_try,
                    number_of_images=count,
                    safety_filter_level=api_level,
                    add_watermark=False,
                )

                # Check if we got images
                if (
                    response
                    and hasattr(response, "images")
                    and response.images
                    and len(response.images) > 0
                ):
                    logger.info(
                        f"✓ SUCCESS on attempt {attempt+1}: "
                        f"Generated {len(response.images)} image(s)"
                    )

                    for idx, image in enumerate(response.images):
                        timestamp = int(time.time() * 1000)
                        filename = f"imagen_{timestamp}_{idx}.png"
                        filepath = output_path / filename

                        image.save(
                            location=str(filepath),
                            include_generation_parameters=False,
                        )

                        results.append(
                            {
                                "filepath": str(filepath),
                                "filename": filename,
                                "generation_time": time.time() - start_time,
                                "backend": self.get_backend_name(),
                                "prompt": prompt_to_try,
                                "attempt": attempt + 1,
                            }
                        )

                    return results
                else:
                    logger.warning(
                        f"Attempt {attempt+1} returned empty images: {response}"
                    )
                    continue

            except google_exceptions.FailedPrecondition as e:
                logger.error(
                    "Vertex AI FailedPrecondition during image generation: %s", e
                )
                raise
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt == len(prompt_variants) - 1:
                    logger.error(
                        f"All generation attempts failed for prompt: {original_prompt}"
                    )
                    raise
                continue

        logger.error("Image generation failed: all attempts returned empty responses")
        return []

    def get_backend_name(self) -> str:
        """Return backend name."""
        return "Google Imagen (Vertex AI)"
