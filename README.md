# Modular Visual Memory Image Generation Prototype

A modular system for generating synthetic visual memory images for episodic memory datasets, designed for integration with LoCoMo-style builders.

## Features

- **Multi-category prompts**: Traveling, Food, Structured Text, Fridge/Pantry, Hobby
- **Pluggable backends**: OpenAI DALL-E 3 and Stable Diffusion support
- **Automated evaluation**: OCR validation and file metrics
- **Clean architecture**: Modular, extensible, well-documented

## Installation

1. Clone the repository
2. Install dependencies:pip install -r requirements.txt

3. Install Tesseract OCR:
   - **Mac**: `brew install tesseract`
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

4. Set up API keys (for OpenAI backend):export OPENAI_API_KEY='your-api-key-here'
## Usage

### Generate Images




