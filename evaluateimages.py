#!/usr/bin/env python3
"""
Evaluation script for generated images.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from unittest import result
from evaluation import compute_file_metrics, check_text_in_image

def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def evaluate_images(input_dir: str = 'outputs/images',
                   metadata_dir: str = 'outputs/metadata',
                   output_file: str = None,
                   run_ocr: bool = True,
                   run_metrics: bool = True):
    """
    Evaluate generated images.
    
    Args:
        input_dir: Directory containing images
        metadata_dir: Directory containing metadata files
        output_file: Path to save evaluation report
        run_ocr: Whether to run OCR evaluation
        run_metrics: Whether to compute file metrics
    """
    print(f"=== Visual Memory Image Evaluation ===")
    print(f"Input directory: {input_dir}")
    print(f"Metadata directory: {metadata_dir}")
    print(f"OCR evaluation: {run_ocr}")
    print(f"File metrics: {run_metrics}")
    print()
    
    # Find all images
    image_files = list(Path(input_dir).glob('*.png')) + list(Path(input_dir).glob('*.jpg'))
    print(f"Found {len(image_files)} images")
    print()
    
    results = []
    
    for image_path in image_files:
        print(f"Evaluating: {image_path.name}")
        
        result = {
            'filename': image_path.name,
            'filepath': str(image_path)
        }
        
        # Load metadata
        metadata_path = Path(metadata_dir) / f"{image_path.stem}.json"
        if metadata_path.exists():
            metadata = load_metadata(str(metadata_path))
            result['category'] = metadata.get('category', 'Unknown')
            result['prompt'] = metadata.get('prompt', '')
            result['generation_time'] = metadata.get('generation_time', 0)
            result['backend'] = metadata.get('backend', 'Unknown')
        else:
            print(f"  ⚠ Metadata not found: {metadata_path}")
            metadata = {}
        
        # Compute file metrics
        if run_metrics:
            metrics = compute_file_metrics(str(image_path))
            result['dimensions'] = metrics.get('dimensions', 'Unknown')
            result['file_size_mb'] = metrics.get('file_size_mb', 0)
            result['format'] = metrics.get('format', 'Unknown')
            print(f"  Dimensions: {result['dimensions']}")
            print(f"  File size: {result['file_size_mb']} MB")
        
        # Run OCR evaluation
        if run_ocr and 'expected_text' in metadata:
            expected_texts = metadata['expected_text']
            ocr_result = check_text_in_image(str(image_path), expected_texts)
            
            result['ocr_success'] = ocr_result.get('success', False)
            result['expected_texts'] = ocr_result.get('expected_texts', [])
            result['found_texts'] = ocr_result.get('found_texts', [])
            result['missing_texts'] = ocr_result.get('missing_texts', [])
            result['match_rate'] = ocr_result.get('match_rate', 0.0)

            
            status = "✓ PASS" if ocr_result['success'] else "✗ FAIL"
            print(f"  OCR Check: {status}")
            print(f"  Expected: {expected_texts}")
            print(f"  Found: {ocr_result['found_texts']}")
            if ocr_result['missing_texts']:
                print(f"  Missing: {ocr_result['missing_texts']}")
        
        results.append(result)
        print()
    
    # Generate summary statistics
    total_images = len(results)
    ocr_passes = sum(1 for r in results if r.get('ocr_success', False))
    ocr_fails = total_images - ocr_passes
    avg_generation_time = sum(r.get('generation_time', 0) for r in results) / total_images if total_images > 0 else 0
    avg_file_size = sum(r.get('file_size_mb', 0) for r in results) / total_images if total_images > 0 else 0
    
    summary = {
        'total_images': total_images,
        'ocr_passes': ocr_passes,
        'ocr_fails': ocr_fails,
        'ocr_pass_rate': ocr_passes / total_images if total_images > 0 else 0,
        'avg_generation_time': avg_generation_time,
        'avg_file_size_mb': avg_file_size,
        'results': results
    }
    
    # Print summary
    print("=== Evaluation Summary ===")
    print(f"Total images: {total_images}")
    if run_ocr:
        print(f"OCR passes: {ocr_passes}")
        print(f"OCR fails: {ocr_fails}")
        print(f"OCR pass rate: {summary['ocr_pass_rate']*100:.1f}%")
    if run_metrics:
        print(f"Avg generation time: {avg_generation_time:.2f}s")
        print(f"Avg file size: {avg_file_size:.2f} MB")
    
    # Save report
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport saved to: {output_file}")
    else:
        # Default output location
        output_file = 'outputs/evaluation/evaluation_report.json'
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate generated images')
    parser.add_argument('--input', type=str, default='outputs/images',
                       help='Directory containing images')
    parser.add_argument('--metadata', type=str, default='outputs/metadata',
                       help='Directory containing metadata files')
    parser.add_argument('--output', type=str,
                       help='Path to save evaluation report')
    parser.add_argument('--ocr', action='store_true', default=True,
                       help='Run OCR evaluation')
    parser.add_argument('--metrics', action='store_true', default=True,
                       help='Compute file metrics')
    parser.add_argument('--no-ocr', dest='ocr', action='store_false',
                       help='Skip OCR evaluation')
    parser.add_argument('--no-metrics', dest='metrics', action='store_false',
                       help='Skip file metrics')
    
    args = parser.parse_args()
    
    evaluate_images(
        input_dir=args.input,
        metadata_dir=args.metadata,
        output_file=args.output,
        run_ocr=args.ocr,
        run_metrics=args.metrics
    )

if __name__ == '__main__':
    main()
