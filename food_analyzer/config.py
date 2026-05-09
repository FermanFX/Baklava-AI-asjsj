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
    "api_key_env": "OPENAI_API_KEY",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "max_tokens": 1024,
    "caption_prompt": (
        "Describe this image in great detail. Focus on all visible food items, "
        "ingredients, drinks, and their preparation methods. Be specific and thorough."
    ),
    "detection_prompt": "List all identifiable objects in this image with their locations.",
}
