#!/usr/bin/env python3
"""
Image generation script 
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from promptagent import PromptAgent
from backends import get_backend

os.environ['OPENAI_API_KEY'] = "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"
os.environ['GOOGLE_API_KEY'] = "AQ.Ab8RN6LfgnVF7-c3cTlqkjrCDwAaLiXwvxIsePDCL3TAwTIpjA"

def save_metadata(metadata: Dict[str, Any], output_dir: str = 'outputs/metadata'):
    #Save metadata to JSON file.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filename = os.path.basename(metadata['filepath']).replace('.png', '.json')
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filepath

def generate_images(backend_name: str = 'openai',
                   count: int = 10, 
                   categories: List[str] = None,
                   output_dir: str = 'outputs/images',
                   backend_config: Dict[str, Any] = None):
    """
    Main function to generate images.
    
    Args:
        backend_name: Name of the image generation backend
        count: Number of images to generate
        categories: List of categories to sample from
        output_dir: Directory to save images
        backend_config: Configuration for the backend
    """
    print(f"=== Visual Memory Image Generation ===")
    print(f"Backend: {backend_name}")
    print(f"Count: {count}")
    print(f"Categories: {categories or 'All'}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize prompt agent
    prompt_agent = PromptAgent()
    
    # Initialize backend
    backend = get_backend(backend_name, backend_config)
    
    # Generate prompts
    print(f"Generating {count} prompts...")
    prompt_batch = prompt_agent.generate_batch(count, categories)
    print(f"Generated {len(prompt_batch)} prompts")
    print()
    
    # Generate images
    all_metadata = []
    
    for i, prompt_data in enumerate(prompt_batch, 1):
        prompt_text = prompt_data['prompt']
        metadata = prompt_data['metadata']
        
        print(f"[{i}/{count}] Generating image for prompt:")
        print(f"  Category: {metadata['category']}")
        print(f"  Prompt: {prompt_text[:80]}...")
        
        
        try:
            generated_files = backend.generate(prompt=prompt_text,count=1,output_dir=output_dir            )
            
            if generated_files:
                file_info = generated_files[0]
                
                # Combine metadata
                full_metadata = {
                    **metadata,
                    'prompt': prompt_text,
                    'filepath': file_info['filepath'],
                    'generation_time': file_info['generation_time'],
                    'backend': file_info['backend']
                }
                
                # Save metadata
                metadata_path = save_metadata(full_metadata)
                print(f" Image saved: {file_info['filepath']}")
                print(f" Metadata saved: {metadata_path}")
                print(f" Generation time: {file_info['generation_time']:.2f}s")
                
                all_metadata.append(full_metadata)
            else:
                print(f" Failed to generate image")
                
        except Exception as e:
            print(f" Error: {str(e)}")
        
        print()
    
    # Save summary
    summary = {
        'total_generated': len(all_metadata),
        'backend': backend_name,
        'categories': categories or 'all',
        'images': all_metadata
    }
    
    summary_path = 'outputs/generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"=== Generation Complete ===")
    print(f"Total images generated: {len(all_metadata)}")
    print(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate visual memory images')
    parser.add_argument('--backend', type=str, default='openai',
                   choices=['openai', 'gptimage1', 'sd', 'stable_diffusion', 'imagen'],
                   help='Backend to use for image generation')
    parser.add_argument('--count', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--categories', type=str, nargs='+', help='Categories to generate (space-separated)')
    parser.add_argument('--output-dir', type=str, default='outputs/images', help='Output directory for images')
    parser.add_argument('--api-key', type=str, help='API key for OpenAI (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Prepare backend config
    backend_config = {}
    if args.api_key:
        backend_config['api_key'] = args.api_key
    
    # Generate images
    generate_images(backend_name=args.backend, count=args.count, categories=args.categories, output_dir=args.output_dir, backend_config=backend_config )

if __name__ == '__main__':
    main()
