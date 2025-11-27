import random
import json
from typing import Dict, Any, List
from .templates import PromptTemplates

class PromptAgent:
    """Agent for generating visual memory scenario prompts."""
    
    def __init__(self, examples_file: str = 'data/examples.json', templates_file: str = 'data/prompt_templates.json'):
        self.examples_file = examples_file
        self.templates = PromptTemplates(templates_file)
        
        # Load examples
        with open(examples_file, 'r') as f:
            data = json.load(f)
            self.examples = data.get('examples', [])
    
    def generate_prompt(self, category: str = None) -> Dict[str, Any]:
        """
        Generate a prompt and metadata for a given category.
        
        Args:
            category: Specific category to generate for, or None for random
            
        Returns:
            Dictionary with 'prompt' and 'metadata' keys
        """
        categories = ['Traveling', 'Food', 'StructuredText', 'FridgePantry', 'Hobby']
        
        if category is None:
            category = random.choice(categories)
        
        # Normalize category name
        category = category.title()
        
        if category == 'Traveling':
            return self.templates.generate_traveling_prompt()
        elif category == 'Food':
            return self.templates.generate_food_prompt()
        elif category == 'Structuredtext' or category == 'Receipt':
            return self.templates.generate_receipt_prompt()
        elif category == 'Fridgepantry':
            return self.templates.generate_fridge_prompt()
        elif category == 'Hobby':
            return self.templates.generate_hobby_prompt()
        else:
            # Fallback to random example
            example = random.choice(self.examples)
            return {
                'prompt': example['prompt'],
                'metadata': example['metadata']
            }
    
    def generate_batch(self, count: int, categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate a batch of prompts.
        
        Args:
            count: Number of prompts to generate
            categories: List of categories to sample from, or None for all
            
        Returns:
            List of prompt/metadata dictionaries
        """
        results = []
        
        for _ in range(count):
            category = random.choice(categories) if categories else None
            result = self.generate_prompt(category)
            results.append(result)
        
        return results
