import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

def preprocess_image_for_ocr(image_path: str) -> np.ndarray:
    """
    Preprocess image for better OCR results.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text_from_image(image_path: str, preprocess: bool = True) -> str:
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image_path: Path to the image
        preprocess: Whether to preprocess the image
        
    Returns:
        Extracted text as string
    """
    try:
        if preprocess:
            img = preprocess_image_for_ocr(image_path)
            text = pytesseract.image_to_string(img)
        else:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
        
        return text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def check_text_in_image(image_path: str, expected_texts: List[str], 
                        case_sensitive: bool = False) -> Dict[str, Any]:
    """
    Check if expected text appears in the image.
    
    Args:
        image_path: Path to the image
        expected_texts: List of text strings expected in the image
        case_sensitive: Whether to perform case-sensitive matching
        
    Returns:
        Dictionary with OCR results
    """
    ocr_text = extract_text_from_image(image_path)
    
    if ocr_text.startswith("ERROR:"):
        return {
            'success': False,
            'error': ocr_text,
            'ocr_text': '',
            'expected_texts': expected_texts,
            'found_texts': [],
            'missing_texts': expected_texts
        }
    
    # Normalize text for comparison
    search_text = ocr_text if case_sensitive else ocr_text.lower()
    
    found_texts = []
    missing_texts = []
    
    for expected in expected_texts:
        search_term = expected if case_sensitive else expected.lower()
        if search_term in search_text:
            found_texts.append(expected)
        else:
            missing_texts.append(expected)
    
    return {
        'success': len(missing_texts) == 0,
        'ocr_text': ocr_text,
        'expected_texts': expected_texts,
        'found_texts': found_texts,
        'missing_texts': missing_texts,
        'match_rate': len(found_texts) / len(expected_texts) if expected_texts else 1.0
    }

def evaluate_ocr_batch(image_metadata_pairs: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Evaluate OCR for a batch of images.
    
    Args:
        image_metadata_pairs: List of (image_path, metadata) tuples
        
    Returns:
        List of evaluation results
    """
    results = []
    
    for image_path, metadata in image_metadata_pairs:
        expected_texts = metadata.get('expected_text', [])
        ocr_result = check_text_in_image(image_path, expected_texts)
        
        result = {
            'image_path': image_path,
            'category': metadata.get('category', 'Unknown'),
            **ocr_result
        }
        
        results.append(result)
    
    return results
