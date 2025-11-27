import random
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta


class PromptTemplates:
    """Template-based prompt generation for different memory categories."""
    
    def __init__(self, templates_file: str = 'data/prompt_templates.json'):
        with open(templates_file, 'r') as f:
            self.templates = json.load(f)
    
    def generate_traveling_prompt(self) -> Dict[str, Any]:
        """Generate a traveling scenario prompt."""
        template_data = self.templates['Traveling']
        template = random.choice(template_data['templates'])
        
        # Get the required fields for this template
        required_fields = template['fields']
        
        # Generate data based on what the template needs
        location_type = random.choice(template_data['location_types'])
        location = random.choice(template_data['locations'])
        date = self._random_date()
        
        # Build kwargs dict based on required fields
        format_kwargs = {}
        metadata = {
            'category': 'Traveling',
            'date': date.strftime('%Y-%m-%d'),
            'expected_text': []
        }
        
        # Populate format_kwargs based on what this template needs
        if 'location_type' in required_fields:
            format_kwargs['location_type'] = location_type
            metadata['location_type'] = location_type
        
        if 'location_name' in required_fields:
            format_kwargs['location_name'] = location
            metadata['location_name'] = location
            metadata['expected_text'].append(location)
        
        if 'date' in required_fields:
            format_kwargs['date'] = date.strftime('%B %d, %Y')
        
        if 'city' in required_fields:
            cities = ['Phoenix', 'New York', 'San Francisco', 'Seattle', 'Boston']
            city = random.choice(cities)
            format_kwargs['city'] = city
            metadata['city'] = city
        
        if 'state' in required_fields:
            states = ['Arizona', 'New York', 'California', 'Washington', 'Massachusetts']
            state = random.choice(states)
            format_kwargs['state'] = state
            metadata['state'] = state
        
        if 'country' in required_fields:
            countries = ['Japan', 'France', 'Italy', 'Germany', 'Spain', 'United Kingdom']
            country = random.choice(countries)
            format_kwargs['country'] = country
            metadata['country'] = country
            metadata['expected_text'].append(country)
        
        if 'passport_number' in required_fields:
            passport = f"US{random.randint(10000000, 99999999)}"
            format_kwargs['passport_number'] = passport
            metadata['passport_number'] = passport
        
        # Format the prompt with available kwargs
        prompt = template['prompt'].format(**format_kwargs)
        
        # Enhance prompt with specific details for better image generation
        if 'airport' in location_type.lower():
            enhanced_prompt = (
                "Photorealistic, high-resolution color photograph inside a modern airport terminal. "
                f"{prompt} "
                f"A large overhead sign clearly shows the text \"{location}\" in sharp, readable letters. "
                "Bright, even lighting, high contrast text on the sign, no motion blur. "
                "Natural colors, professional travel photography, no illustration, no cartoon, no CGI."
            )
        elif 'train' in location_type.lower():
            enhanced_prompt = (
                "Photorealistic color photograph of a busy train station platform. "
                f"{prompt} "
                f"A station sign with the name \"{location}\" is clearly visible and easy to read. "
                "Daytime lighting, sharp focus on the sign and surrounding details, natural colors, no illustration or cartoon."
            )
        elif 'landmark' in location_type.lower() or 'border' in location_type.lower():
            enhanced_prompt = (
                "Photorealistic daytime travel photograph at a well-known landmark or border crossing. "
                f"{prompt} "
                f"A physical sign, plaque, or monument in the scene clearly displays the location name \"{location}\" in legible text. "
                "Bright natural daylight, sharp focus, rich but realistic colors, professional travel photography, no illustration or cartoon."
            )
        elif 'sign' in location_type.lower():
            enhanced_prompt = (
                "Photorealistic outdoor photograph of a large roadside or street sign. "
                f"{prompt} "
                f"The sign prominently displays the text \"{location}\" in large, readable letters. "
                "High contrast between text and background, bright daylight, no blur, no illustration or cartoon."
            )
        else:
            enhanced_prompt = (
                "Photorealistic, high-resolution travel photograph. "
                f"{prompt} "
                f"In the scene, a screen, sign, or printed document clearly shows the location name \"{location}\" in readable text. "
                "Sharp focus, bright but natural lighting, realistic colors, no illustration, no cartoon, no CGI."
            )
        
        return {'prompt': enhanced_prompt, 'metadata': metadata}
    
    def generate_food_prompt(self) -> Dict[str, Any]:
        """Generate a food scenario prompt."""
        restaurants = ['Marugame Udon', 'Chipotle', 'In-N-Out Burger', 'Olive Garden', 'Panda Express']
        dishes = ['Nikutama', 'Burrito Bowl', 'Double-Double', 'Pasta Alfredo', 'Orange Chicken']
        cities = ['Phoenix', 'Los Angeles', 'San Francisco', 'New York', 'Chicago']
        
        restaurant = random.choice(restaurants)
        dish = random.choice(dishes)
        city = random.choice(cities)
        date = self._random_date()
        
        prompt = f"Photo of {dish} with visible {restaurant} sign or menu in background"
        
        # Enhanced prompt with specific visual details
        enhanced_prompt = (
            f"Photorealistic restaurant food photograph of {dish} on a plate in the foreground. "
            f"In the background, the restaurant name \"{restaurant}\" appears on a sign, menu, or wall and is clearly readable. "
            "Sharp focus on the dish, with the restaurant branding also in focus and easy to read. "
            "Natural, bright but realistic lighting (indoor restaurant or daylight from windows), high resolution, crisp details, vibrant but true-to-life colors. "
            "Professional food photography, no illustration, no cartoon, no CGI."
        )
        
        metadata = {
            'category': 'Food',
            'restaurant': restaurant,
            'dish': dish,
            'city': city,
            'date': date.strftime('%Y-%m-%d'),
            'expected_text': [restaurant]
        }
        
        return {'prompt': enhanced_prompt, 'metadata': metadata}
    
    def generate_receipt_prompt(self) -> Dict[str, Any]:
        """Generate a receipt scenario prompt."""
        stores = ['Target', 'Walmart', 'CVS', 'Walgreens', 'Safeway']
        store = random.choice(stores)
        total = round(random.uniform(10.0, 150.0), 2)
        date = self._random_date()
        
        prompt = f"Receipt from {store} showing total ${total:.2f} dated {date.strftime('%B %d, %Y')}"
        date_str = date.strftime('%m/%d/%Y')
        
        # Enhanced prompt emphasizing text clarity
        enhanced_prompt = (
            f"Photorealistic macro photograph of a paper retail receipt from \"{store}\" lying flat on a neutral surface. "
            f"The store name \"{store}\" is printed clearly at the top of the receipt. "
            f"The total amount \"${total:.2f}\" and the date \"{date_str}\" are large, sharp, and easy to read. "
            "High contrast black text on white receipt paper, edge-to-edge sharp focus with no blur. "
            "Bright, even lighting with no shadows or glare on the text. "
            "All printed lines and numbers are crisp and legible, professional document photography, no illustration or cartoon."
        )
        
        metadata = {
            'category': 'StructuredText',
            'type': 'receipt',
            'store': store,
            'total': f"{total:.2f}",
            'date': date.strftime('%Y-%m-%d'),
            'expected_text': [store, f"{total:.2f}", date_str]
        }
        
        return {'prompt': enhanced_prompt, 'metadata': metadata}
    
    def generate_fridge_prompt(self) -> Dict[str, Any]:
        """Generate a fridge/pantry item prompt."""
        items = [
            {'item': 'milk carton', 'brand': 'Organic Valley'},
            {'item': 'yogurt container', 'brand': 'Chobani'},
            {'item': 'orange juice bottle', 'brand': 'Tropicana'},
            {'item': 'egg carton', 'brand': 'Happy Egg Co'}
        ]
        
        selected = random.choice(items)
        exp_date = datetime.now() + timedelta(days=random.randint(3, 30))
        exp_date_str = exp_date.strftime('%m/%d/%Y')
        
        prompt = f"Close-up photo of {selected['item']} with expiration date '{exp_date.strftime('EXP %m/%d/%Y')}' visible on label"
        
        # Enhanced prompt emphasizing date label clarity
        enhanced_prompt = (
            f"Photorealistic close-up product photograph of a {selected['item']} from the brand \"{selected['brand']}\". "
            f"The product label clearly shows the expiration date text \"EXP {exp_date_str}\" in the main area of the label. "
            "The expiration date text is in sharp focus and easy to read, with high contrast between text and background. "
            "Macro lens look with shallow depth of field, but the date text and brand logo are fully sharp and legible. "
            "Soft, even lighting with no glare or reflections on the label, neutral or simple fridge/interior background. "
            "Professional product photography, no illustration, no cartoon, no CGI."
        )
        
        metadata = {
            'category': 'FridgePantry',
            'item': selected['item'],
            'brand': selected['brand'],
            'date': exp_date.strftime('%Y-%m-%d'),
            'expected_text': [exp_date_str, 'EXP']
        }
        
        return {'prompt': enhanced_prompt, 'metadata': metadata}
    
    def generate_hobby_prompt(self) -> Dict[str, Any]:
        """Generate a hobby scenario prompt."""
        hobbies = [
            {
                'type': 'fishing',
                'item': '8 lb bass',
                'location': 'Lake Tahoe',
                'details': 'A fisherman proudly displaying a caught fish, clear scenic mountain lake background'
            },
            {
                'type': 'hiking',
                'item': 'trail marker',
                'location': 'Grand Canyon',
                'details': 'A hiker at a clearly visible trail marker sign in a canyon landscape'
            },
            {
                'type': 'photography',
                'item': 'camera',
                'location': 'Yosemite',
                'details': 'A photographer with camera at a scenic overlook with mountains visible'
            },
        ]
        
        hobby = random.choice(hobbies)
        date = self._random_date()
        
        prompt = f"Photo of a person holding {hobby['item']} at {hobby['location']}"
        
        # Enhanced prompt with environmental context
        enhanced_prompt = (
            "Photorealistic outdoor adventure photograph. "
            f"{hobby['details']}. "
            f"A sign, marker, or engraved plaque in the scene clearly shows the location name \"{hobby['location']}\" in readable text. "
            "Bright natural daylight, sharp focus on both the person and the location marker. "
            "Detailed scenic background landscape with realistic colors, professional travel/adventure photography, no illustration or cartoon."
        )
        
        metadata = {
            'category': 'Hobby',
            'hobby_type': hobby['type'],
            'item': hobby['item'],
            'location': hobby['location'],
            'date': date.strftime('%Y-%m-%d'),
            'expected_text': [hobby['location']]
        }
        
        return {'prompt': enhanced_prompt, 'metadata': metadata}
    
    def _random_date(self, days_back: int = 365) -> datetime:
        """Generate a random date within the last N days."""
        days_ago = random.randint(1, days_back)
        return datetime.now() - timedelta(days=days_ago)
