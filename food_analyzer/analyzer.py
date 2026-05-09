'''Analiz modulu.
FoodAnalyzer sinifi ile sekil uzerinde yemek analizi aparilir.
VLM caption-dan yemekler cixarilir, NutritionDB ile kalori hesablanir.
'''

import re
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from .config import ANALYSIS_CONFIG
from .model import VisionLanguageModel
from .nutrition_db import NutritionDB, FOOD_DATABASE


@dataclass
class DetectedFood:
    '''Detect olunmus yemek melumati.

    Attribute-ler:
        name: Yemeyin adi
        confidence: Detection inam deyeri (0.0 - 1.0)
        estimated_calories: Tehmini kalori
        estimated_portion_g: Tehmini porsiya (qram)
        calories_per_100g: 100q-da kalori
        protein: Zulal (q)
        carbs: Karbohidrat (q)
        fat: Yag (q)
    '''
    name: str
    confidence: float = 1.0
    estimated_calories: Optional[float] = None
    estimated_portion_g: Optional[float] = None
    calories_per_100g: Optional[float] = None
    protein: Optional[float] = None
    carbs: Optional[float] = None
    fat: Optional[float] = None


@dataclass
class AnalysisResult:
    '''Analiz neticesi.

    Attribute-ler:
        foods: Detect olunmus yemeklerin siyahisi
        total_calories: Umumi kalori
        total_protein: Umumi zulal
        total_carbs: Umumi karbohidrat
        total_fat: Umumi yag
        raw_caption: AI captioni (xam)
        confidence_score: Analiz inam deyeri
    '''
    foods: list[DetectedFood] = field(default_factory=list)
    total_calories: float = 0.0
    total_protein: float = 0.0
    total_carbs: float = 0.0
    total_fat: float = 0.0
    raw_caption: str = ""
    confidence_score: float = 0.0


class FoodAnalyzer:
    '''Esas analiz sinifi.
    VLM ve NutritionDB istifade ederek yemek analizi aparir.
    '''

    def __init__(self, model: VisionLanguageModel, nutrition_db: Optional[NutritionDB] = None):
        '''FoodAnalyzer sinifini yaradir.

        Parameterler:
            model: VisionLanguageModel instance-i
            nutrition_db: NutritionDB instance-i (default: yeni baz)
        '''
        self.model = model
        self.nutrition_db = nutrition_db or NutritionDB()

    def analyze(self, image: Image.Image) -> AnalysisResult:
        '''Sekli analiz edir: caption, ingredient cixarma, kalori hesablama.

        Isleme axini:
        1. Florence-2 modeline sekil gonderilir -> caption alinir
        2. Caption-dan yemek adlari cixarilir
        3. Her yemek ucun NutritionDB-de kalori axtarilir
        4. Umumi kalori, protein, carb, fat hesablanir

        Parameterler:
            image: PIL.Image formatinda sekil

        Qaytaran deyer:
            AnalysisResult: Analiz neticeleri
        '''
        result = AnalysisResult()

        caption = self.model.generate_caption(
            image, task=ANALYSIS_CONFIG["detailed_caption_task"]
        )
        result.raw_caption = caption

        detected_names = self._extract_food_items(caption)
        seen = set()
        for food_name in detected_names:
            normalized = food_name.lower().strip()
            if normalized in seen:
                continue
            seen.add(normalized)

            db_result = self.nutrition_db.estimate_calories(normalized)
            if db_result:
                food = DetectedFood(
                    name=db_result["name"],
                    estimated_calories=db_result["estimated_calories"],
                    estimated_portion_g=db_result["estimated_portion_g"],
                    calories_per_100g=db_result["calories_per_100g"],
                    protein=db_result.get("protein"),
                    carbs=db_result.get("carbs"),
                    fat=db_result.get("fat"),
                    confidence=0.7,
                )
            else:
                food = DetectedFood(name=normalized, confidence=0.3)

            result.foods.append(food)

        for food in result.foods:
            if food.estimated_calories is not None:
                result.total_calories += food.estimated_calories
            portion = food.estimated_portion_g or 0
            if food.protein is not None:
                result.total_protein += food.protein * portion / 100
            if food.carbs is not None:
                result.total_carbs += food.carbs * portion / 100
            if food.fat is not None:
                result.total_fat += food.fat * portion / 100

        result.total_calories = round(result.total_calories, 1)
        result.total_protein = round(result.total_protein, 1)
        result.total_carbs = round(result.total_carbs, 1)
        result.total_fat = round(result.total_fat, 1)
        result.confidence_score = self._compute_confidence(result)

        return result

    def _extract_food_items(self, caption: str) -> list[str]:
        '''AI caption-indan yemek adlarini cixarir.

        FOOD_DATABASE-deki adlari caption icinde axtarir,
        aliases-i yoxlayir ve tekrarlari silir.
        Hec ne tapilmazsa regex fallback istifade olunur.

        Parameterler:
            caption: AI captioni

        Qaytaran deyer:
            list[str]: Tapilan yemek adlari
        '''
        caption_lower = caption.lower()
        detected = []

        for food_entry in FOOD_DATABASE:
            name = food_entry["name"]
            if name in caption_lower:
                detected.append(name)
                continue

            words = name.split()
            if len(words) >= 2 and all(word in caption_lower for word in words):
                detected.append(name)

        from .nutrition_db import FOOD_ALIASES
        for alias, canonical in FOOD_ALIASES.items():
            if alias in caption_lower and canonical not in detected:
                detected.append(canonical)

        detected = self._deduplicate_foods(detected)
        return detected if detected else [self._fallback_extract(caption)]

    def _deduplicate_foods(self, foods: list[str]) -> list[str]:
        '''Tekrarlanan yemek adlarini silir.

        Mes: "chicken chicken breast" varsa, sadece "chicken breast" saxlanilir.

        Parameterler:
            foods: Yemek adlari siyahisi

        Qaytaran deyer:
            list[str]: Tekrarsiz siyahi
        '''
        result = []
        for f in foods:
            is_subset = any(f != g and f in g for g in foods)
            if not is_subset:
                result.append(f)
        return result

    def _fallback_extract(self, caption: str) -> str:
        '''Regex ile caption-dan yemek adi cixarma (esas metod islemezse).

        Parameterler:
            caption: AI captioni

        Qaytaran deyer:
            str: Tapilan ilk yemek adi ve ya "mixed meal"
        '''
        food_indicators = [
            r"(?:a\s+)?(?:plate\s+of|bowl\s+of|serving\s+of|some\s+)?(\w+(?:\s+\w+)?\s*(?:soup|salad|stew|curry|pasta|rice|bread|meat|fish|cake|pie|egg|roll|wrap|kebab|pizza|burger|sandwich|stir-fry|roast|steak))",
            r"(?:contains?\s+)?(\w+(?:\s+\w+)?)\s+(?:and|with|on|in)",
            r"(\w+(?:\s+\w+)?)\s+is\s+(?:a\s+)?(?:popular|typical|traditional|delicious)",
        ]
        for pattern in food_indicators:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "mixed meal"

    def _compute_confidence(self, result: AnalysisResult) -> float:
        '''Analiz inam deyerini hesablayir.

        Baza ile eslesen yemeklerin nisbeti esasinda.

        Parameterler:
            result: Analiz neticesi

        Qaytaran deyer:
            float: Inam deyeri (0.0 - 1.0)
        '''
        if not result.foods:
            return 0.0
        matched = sum(1 for f in result.foods if f.estimated_calories is not None)
        return round(matched / len(result.foods), 2)

    def analyze_with_details(self, image: Image.Image) -> dict:
        '''Sekli analiz edir ve neticeni dict formatinda qaytarir.

        JSON-a cevrilmek ucun elverislidir.

        Parameterler:
            image: PIL.Image formatinda sekil

        Qaytaran deyer:
            dict: {
                "foods": [...],
                "total_nutrition": {"calories": ..., "protein_g": ..., ...},
                "confidence": ...,
                "raw_caption": "..."
            }
        '''
        result = self.analyze(image)
        summary = {
            "foods": [],
            "total_nutrition": {
                "calories": result.total_calories,
                "protein_g": result.total_protein,
                "carbs_g": result.total_carbs,
                "fat_g": result.total_fat,
            },
            "confidence": result.confidence_score,
            "raw_caption": result.raw_caption,
        }
        for food in result.foods:
            fd = {"name": food.name.title(), "confidence": food.confidence}
            if food.estimated_calories is not None:
                fd["nutrition"] = {
                    "calories": food.estimated_calories,
                    "portion_g": food.estimated_portion_g,
                    "calories_per_100g": food.calories_per_100g,
                    "protein": food.protein,
                    "carbs": food.carbs,
                    "fat": food.fat,
                }
            summary["foods"].append(fd)
        return summary

    def analyze_with_caption(self, image: Image.Image, custom_prompt: str = "") -> AnalysisResult:
        '''Xususi prompt ile analiz.

        Custom prompt istifade edildikde, AI captioni birbasha
        yemek adi kimi qebul olunur.

        Parameterler:
            image: PIL.Image formatinda sekil
            custom_prompt: AI-a verilecek xususi prompt

        Qaytaran deyer:
            AnalysisResult: Analiz neticeleri
        '''
        result = AnalysisResult()
        task = custom_prompt if custom_prompt else ANALYSIS_CONFIG["detailed_caption_task"]

        caption = self.model.generate_caption(image, task=task)
        result.raw_caption = caption

        if custom_prompt:
            db_result = self.nutrition_db.estimate_calories(caption)
            if db_result:
                food = DetectedFood(
                    name=db_result["name"],
                    estimated_calories=db_result["estimated_calories"],
                    estimated_portion_g=db_result["estimated_portion_g"],
                    calories_per_100g=db_result["calories_per_100g"],
                    protein=db_result.get("protein"),
                    carbs=db_result.get("carbs"),
                    fat=db_result.get("fat"),
                    confidence=0.8,
                )
                result.foods.append(food)
                result.total_calories = food.estimated_calories or 0
                portion = food.estimated_portion_g or 0
                if food.protein is not None:
                    result.total_protein += food.protein * portion / 100
                if food.carbs is not None:
                    result.total_carbs += food.carbs * portion / 100
                if food.fat is not None:
                    result.total_fat += food.fat * portion / 100

        result.confidence_score = self._compute_confidence(result)
        return result