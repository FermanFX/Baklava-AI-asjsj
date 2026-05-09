'''Konfigurasiya modulu.
Butun model, kalori ve analiz parametrleri burada saxlanilir.
Bu deyerler layihenin davranisini idare edir.
'''

from pathlib import Path

MODEL_CONFIG = {
    "model_id": "microsoft/Florence-2-large",
    "model_cache_dir": str(Path(__file__).resolve().parent.parent / "models"),
    "device": "cuda",
    "dtype": "float16",
    "max_new_tokens": 1024,
    "num_beams": 3,
}

CALORIE_CONFIG = {
    "default_portion_g": 150,
    "confidence_threshold": 0.4,
    "include_portion_estimate": True,
}

ANALYSIS_CONFIG = {
    "detailed_caption_task": "<MORE_DETAILED_CAPTION>",
    "caption_task": "<CAPTION>",
    "detection_task": "<OD>",
    "extraction_prompt": (
        "List every identifiable food item, ingredient, and drink visible in this image. "
        "Be specific about quantities and preparations."
    ),
}

API_CONFIG = {
    "enabled": True,
    "api_key_env": "GEMINI_API_KEY",
    "base_url": "https://generativelanguage.googleapis.com/v1beta",
    "model": "gemini-2.5-flash",
    "max_tokens": 1024,
    "caption_prompt": (
        "Identify all food items in this image and estimate the total calories. "
        "Return ONLY a JSON object with this exact structure (no markdown, no code blocks): "
        '{"foods": [{"name": "food name", "calories": 123}], '
        '"total_calories": 456, "description": "brief description of the meal"}'
    ),
    "detection_prompt": "List all identifiable food objects in this image.",
    "retry_max_attempts": 5,
    "retry_base_delay": 2.0,
}
