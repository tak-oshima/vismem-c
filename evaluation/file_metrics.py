import os
from typing import Dict, Any
from PIL import Image

def compute_file_metrics(filepath: str) -> Dict[str, Any]:
    """
    Compute file-level metrics for an image.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'exists': os.path.exists(filepath)
    }
    
    if not metrics['exists']:
        return metrics
    
    # File size
    metrics['file_size_bytes'] = os.path.getsize(filepath)
    metrics['file_size_mb'] = round(metrics['file_size_bytes'] / (1024 * 1024), 2)
    
    # Image dimensions
    try:
        with Image.open(filepath) as img:
            metrics['width'], metrics['height'] = img.size
            metrics['dimensions'] = f"{metrics['width']}x{metrics['height']}"
            metrics['mode'] = img.mode
            metrics['format'] = img.format
    except Exception as e:
        metrics['error'] = f"Failed to read image: {str(e)}"
    
    return metrics
