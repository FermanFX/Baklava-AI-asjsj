'''Food Analyzer paketinin giris noktasi.
Xarici istifade ucun asagidaki sinif ve funksiyalar export olunur:
  - NutritionDB: Qida bazasi
  - model_exists_locally: Modelin lokal olub-olmadigini yoxlayir
  - FoodAnalyzer: Esas analiz sinifi (lazy import)
  - get_vision_model: Vision-Language Model yaradici (lazy import)
'''

from .nutrition_db import NutritionDB
from .model import model_exists_locally

__all__ = ["FoodAnalyzer", "get_vision_model", "model_exists_locally", "NutritionDB"]
