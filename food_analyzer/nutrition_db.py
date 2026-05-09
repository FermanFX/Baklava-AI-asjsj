'''Qida bazasi modulu.
150+ yemek mehsulu ucun kalori, protein, karbohidrat ve yag deyerleri.
Azərbaycan ve Turk mutfeyine aid yemekler de daxildir.
'''

import re
from difflib import SequenceMatcher
from typing import Optional

FoodInfo = dict[str, float | str]

FOOD_DATABASE: list[FoodInfo] = [
    # Protein / Meat
    {"name": "chicken breast", "calories_per_100g": 165, "protein": 31, "carbs": 0, "fat": 3.6, "serving_g": 150},
    {"name": "chicken thigh", "calories_per_100g": 209, "protein": 26, "carbs": 0, "fat": 11, "serving_g": 130},
    {"name": "chicken wing", "calories_per_100g": 203, "protein": 18, "carbs": 0, "fat": 14, "serving_g": 100},
    {"name": "beef steak", "calories_per_100g": 271, "protein": 26, "carbs": 0, "fat": 19, "serving_g": 200},
    {"name": "ground beef", "calories_per_100g": 250, "protein": 24, "carbs": 0, "fat": 17, "serving_g": 150},
    {"name": "pork chop", "calories_per_100g": 231, "protein": 25, "carbs": 0, "fat": 14, "serving_g": 170},
    {"name": "lamb chop", "calories_per_100g": 294, "protein": 25, "carbs": 0, "fat": 21, "serving_g": 150},
    {"name": "fish fillet", "calories_per_100g": 120, "protein": 22, "carbs": 0, "fat": 3, "serving_g": 150},
    {"name": "salmon", "calories_per_100g": 208, "protein": 20, "carbs": 0, "fat": 13, "serving_g": 150},
    {"name": "tuna", "calories_per_100g": 132, "protein": 28, "carbs": 0, "fat": 1.5, "serving_g": 100},
    {"name": "shrimp", "calories_per_100g": 99, "protein": 24, "carbs": 0.2, "fat": 0.3, "serving_g": 100},
    {"name": "egg", "calories_per_100g": 155, "protein": 13, "carbs": 1.1, "fat": 11, "serving_g": 50},
    {"name": "egg white", "calories_per_100g": 52, "protein": 11, "carbs": 0.7, "fat": 0.2, "serving_g": 100},
    {"name": "bacon", "calories_per_100g": 541, "protein": 37, "carbs": 1.4, "fat": 42, "serving_g": 30},
    {"name": "sausage", "calories_per_100g": 325, "protein": 14, "carbs": 3, "fat": 29, "serving_g": 80},
    {"name": "ham", "calories_per_100g": 145, "protein": 20, "carbs": 1.5, "fat": 6.5, "serving_g": 80},

    # Grains / Carbs
    {"name": "rice", "calories_per_100g": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "serving_g": 200},
    {"name": "white rice", "calories_per_100g": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "serving_g": 200},
    {"name": "brown rice", "calories_per_100g": 111, "protein": 2.6, "carbs": 23, "fat": 0.9, "serving_g": 200},
    {"name": "pasta", "calories_per_100g": 131, "protein": 5, "carbs": 25, "fat": 1.1, "serving_g": 200},
    {"name": "spaghetti", "calories_per_100g": 131, "protein": 5, "carbs": 25, "fat": 1.1, "serving_g": 200},
    {"name": "bread", "calories_per_100g": 265, "protein": 9, "carbs": 49, "fat": 3.2, "serving_g": 50},
    {"name": "white bread", "calories_per_100g": 265, "protein": 9, "carbs": 49, "fat": 3.2, "serving_g": 50},
    {"name": "whole wheat bread", "calories_per_100g": 247, "protein": 13, "carbs": 41, "fat": 3.4, "serving_g": 50},
    {"name": "noodle", "calories_per_100g": 138, "protein": 4.5, "carbs": 25, "fat": 2.1, "serving_g": 200},
    {"name": "ramen", "calories_per_100g": 138, "protein": 4.5, "carbs": 25, "fat": 2.1, "serving_g": 200},
    {"name": "couscous", "calories_per_100g": 112, "protein": 3.8, "carbs": 23, "fat": 0.2, "serving_g": 180},
    {"name": "quinoa", "calories_per_100g": 120, "protein": 4.4, "carbs": 21, "fat": 1.9, "serving_g": 180},
    {"name": "oatmeal", "calories_per_100g": 71, "protein": 2.5, "carbs": 12, "fat": 1.5, "serving_g": 250},
    {"name": "cereal", "calories_per_100g": 379, "protein": 7, "carbs": 80, "fat": 2.5, "serving_g": 40},
    {"name": "potato", "calories_per_100g": 77, "protein": 2, "carbs": 17, "fat": 0.1, "serving_g": 200},
    {"name": "french fries", "calories_per_100g": 312, "protein": 3.4, "carbs": 38, "fat": 17, "serving_g": 150},
    {"name": "mashed potato", "calories_per_100g": 86, "protein": 2, "carbs": 17, "fat": 1.5, "serving_g": 200},

    # Vegetables
    {"name": "broccoli", "calories_per_100g": 34, "protein": 2.8, "carbs": 7, "fat": 0.4, "serving_g": 100},
    {"name": "carrot", "calories_per_100g": 41, "protein": 0.9, "carbs": 10, "fat": 0.2, "serving_g": 80},
    {"name": "tomato", "calories_per_100g": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2, "serving_g": 100},
    {"name": "cucumber", "calories_per_100g": 15, "protein": 0.7, "carbs": 3.6, "fat": 0.1, "serving_g": 100},
    {"name": "lettuce", "calories_per_100g": 15, "protein": 1.4, "carbs": 2.9, "fat": 0.2, "serving_g": 50},
    {"name": "spinach", "calories_per_100g": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4, "serving_g": 80},
    {"name": "onion", "calories_per_100g": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1, "serving_g": 80},
    {"name": "garlic", "calories_per_100g": 149, "protein": 6.4, "carbs": 33, "fat": 0.5, "serving_g": 10},
    {"name": "bell pepper", "calories_per_100g": 26, "protein": 1, "carbs": 6, "fat": 0.3, "serving_g": 100},
    {"name": "mushroom", "calories_per_100g": 22, "protein": 3.1, "carbs": 3.3, "fat": 0.3, "serving_g": 80},
    {"name": "corn", "calories_per_100g": 86, "protein": 3.3, "carbs": 19, "fat": 1.2, "serving_g": 100},
    {"name": "green beans", "calories_per_100g": 31, "protein": 1.8, "carbs": 7, "fat": 0.2, "serving_g": 100},
    {"name": "peas", "calories_per_100g": 81, "protein": 5.4, "carbs": 14, "fat": 0.4, "serving_g": 100},
    {"name": "avocado", "calories_per_100g": 160, "protein": 2, "carbs": 8.5, "fat": 15, "serving_g": 75},
    {"name": "zucchini", "calories_per_100g": 17, "protein": 1.2, "carbs": 3.1, "fat": 0.3, "serving_g": 100},
    {"name": "cabbage", "calories_per_100g": 25, "protein": 1.3, "carbs": 5.8, "fat": 0.1, "serving_g": 100},
    {"name": "cauliflower", "calories_per_100g": 25, "protein": 1.9, "carbs": 5, "fat": 0.3, "serving_g": 100},
    {"name": "eggplant", "calories_per_100g": 25, "protein": 1, "carbs": 6, "fat": 0.2, "serving_g": 100},

    # Fruits
    {"name": "apple", "calories_per_100g": 52, "protein": 0.3, "carbs": 14, "fat": 0.2, "serving_g": 150},
    {"name": "banana", "calories_per_100g": 89, "protein": 1.1, "carbs": 23, "fat": 0.3, "serving_g": 120},
    {"name": "orange", "calories_per_100g": 47, "protein": 0.9, "carbs": 12, "fat": 0.1, "serving_g": 150},
    {"name": "strawberry", "calories_per_100g": 32, "protein": 0.7, "carbs": 8, "fat": 0.3, "serving_g": 100},
    {"name": "blueberry", "calories_per_100g": 57, "protein": 0.7, "carbs": 14, "fat": 0.3, "serving_g": 80},
    {"name": "grape", "calories_per_100g": 69, "protein": 0.7, "carbs": 18, "fat": 0.2, "serving_g": 100},
    {"name": "watermelon", "calories_per_100g": 30, "protein": 0.6, "carbs": 7.6, "fat": 0.2, "serving_g": 200},
    {"name": "pineapple", "calories_per_100g": 50, "protein": 0.5, "carbs": 13, "fat": 0.1, "serving_g": 100},
    {"name": "mango", "calories_per_100g": 60, "protein": 0.8, "carbs": 15, "fat": 0.4, "serving_g": 150},
    {"name": "kiwi", "calories_per_100g": 61, "protein": 1.1, "carbs": 15, "fat": 0.5, "serving_g": 75},
    {"name": "lemon", "calories_per_100g": 29, "protein": 1.1, "carbs": 9.3, "fat": 0.3, "serving_g": 50},

    # Dairy
    {"name": "milk", "calories_per_100g": 42, "protein": 3.4, "carbs": 5, "fat": 1, "serving_g": 250},
    {"name": "cheese", "calories_per_100g": 350, "protein": 25, "carbs": 1.3, "fat": 28, "serving_g": 40},
    {"name": "cheddar cheese", "calories_per_100g": 404, "protein": 25, "carbs": 1.3, "fat": 33, "serving_g": 40},
    {"name": "mozzarella", "calories_per_100g": 280, "protein": 28, "carbs": 3.1, "fat": 17, "serving_g": 50},
    {"name": "parmesan", "calories_per_100g": 431, "protein": 38, "carbs": 4.1, "fat": 29, "serving_g": 20},
    {"name": "yogurt", "calories_per_100g": 59, "protein": 10, "carbs": 3.6, "fat": 0.7, "serving_g": 150},
    {"name": "butter", "calories_per_100g": 717, "protein": 0.9, "carbs": 0.1, "fat": 81, "serving_g": 10},
    {"name": "cream", "calories_per_100g": 340, "protein": 2.8, "carbs": 3, "fat": 36, "serving_g": 30},
    {"name": "ice cream", "calories_per_100g": 207, "protein": 3.5, "carbs": 24, "fat": 11, "serving_g": 100},

    # Legumes
    {"name": "lentil", "calories_per_100g": 116, "protein": 9, "carbs": 20, "fat": 0.4, "serving_g": 150},
    {"name": "chickpea", "calories_per_100g": 139, "protein": 7.6, "carbs": 23, "fat": 2.6, "serving_g": 150},
    {"name": "beans", "calories_per_100g": 132, "protein": 8.7, "carbs": 24, "fat": 0.5, "serving_g": 150},
    {"name": "tofu", "calories_per_100g": 76, "protein": 8, "carbs": 1.9, "fat": 4.8, "serving_g": 100},

    # Soups & Stews
    {"name": "tomato soup", "calories_per_100g": 33, "protein": 0.7, "carbs": 7.5, "fat": 0.3, "serving_g": 300},
    {"name": "chicken soup", "calories_per_100g": 36, "protein": 2.5, "carbs": 3.5, "fat": 1.5, "serving_g": 300},
    {"name": "vegetable soup", "calories_per_100g": 28, "protein": 1.2, "carbs": 5, "fat": 0.4, "serving_g": 300},

    # Sauces & Oils
    {"name": "olive oil", "calories_per_100g": 884, "protein": 0, "carbs": 0, "fat": 100, "serving_g": 15},
    {"name": "vegetable oil", "calories_per_100g": 884, "protein": 0, "carbs": 0, "fat": 100, "serving_g": 15},
    {"name": "ketchup", "calories_per_100g": 101, "protein": 1, "carbs": 27, "fat": 0.1, "serving_g": 15},
    {"name": "mayonnaise", "calories_per_100g": 700, "protein": 1, "carbs": 0.6, "fat": 78, "serving_g": 15},
    {"name": "mustard", "calories_per_100g": 66, "protein": 3.7, "carbs": 6, "fat": 3.3, "serving_g": 10},
    {"name": "soy sauce", "calories_per_100g": 53, "protein": 8, "carbs": 4.7, "fat": 0.1, "serving_g": 15},

    # Snacks & Fast Food
    {"name": "pizza", "calories_per_100g": 266, "protein": 11, "carbs": 33, "fat": 10, "serving_g": 200},
    {"name": "hamburger", "calories_per_100g": 250, "protein": 14, "carbs": 30, "fat": 9, "serving_g": 200},
    {"name": "hot dog", "calories_per_100g": 290, "protein": 11, "carbs": 18, "fat": 19, "serving_g": 100},
    {"name": "sandwich", "calories_per_100g": 220, "protein": 12, "carbs": 28, "fat": 7, "serving_g": 180},
    {"name": "chips", "calories_per_100g": 536, "protein": 7, "carbs": 49, "fat": 34, "serving_g": 50},
    {"name": "cookie", "calories_per_100g": 488, "protein": 5.5, "carbs": 64, "fat": 23, "serving_g": 30},
    {"name": "cake", "calories_per_100g": 350, "protein": 4, "carbs": 55, "fat": 13, "serving_g": 100},
    {"name": "chocolate", "calories_per_100g": 546, "protein": 4.9, "carbs": 61, "fat": 31, "serving_g": 40},
    {"name": "donut", "calories_per_100g": 421, "protein": 5.6, "carbs": 49, "fat": 23, "serving_g": 60},
    {"name": "croissant", "calories_per_100g": 406, "protein": 8.2, "carbs": 46, "fat": 21, "serving_g": 60},

    # Turkish / Azerbaijani cuisine
    {"name": "dolma", "calories_per_100g": 120, "protein": 6, "carbs": 10, "fat": 6, "serving_g": 200},
    {"name": "kebab", "calories_per_100g": 180, "protein": 22, "carbs": 2, "fat": 9, "serving_g": 200},
    {"name": "kofte", "calories_per_100g": 210, "protein": 18, "carbs": 8, "fat": 12, "serving_g": 150},
    {"name": "lahmacun", "calories_per_100g": 220, "protein": 12, "carbs": 28, "fat": 7, "serving_g": 150},
    {"name": "pide", "calories_per_100g": 260, "protein": 12, "carbs": 32, "fat": 9, "serving_g": 150},
    {"name": "borek", "calories_per_100g": 300, "protein": 10, "carbs": 28, "fat": 17, "serving_g": 120},
    {"name": "plov", "calories_per_100g": 180, "protein": 7, "carbs": 24, "fat": 6, "serving_g": 250},
    {"name": "dovga", "calories_per_100g": 35, "protein": 2.5, "carbs": 3, "fat": 1.5, "serving_g": 300},
    {"name": "dushbara", "calories_per_100g": 85, "protein": 5, "carbs": 10, "fat": 3, "serving_g": 300},
    {"name": "qutab", "calories_per_100g": 180, "protein": 7, "carbs": 22, "fat": 7, "serving_g": 100},
    {"name": "saj", "calories_per_100g": 160, "protein": 15, "carbs": 5, "fat": 9, "serving_g": 250},
    {"name": "shashlik", "calories_per_100g": 190, "protein": 24, "carbs": 1, "fat": 10, "serving_g": 200},
    {"name": "yogurtlu", "calories_per_100g": 60, "protein": 4, "carbs": 5, "fat": 2.5, "serving_g": 150},

    # Beverages
    {"name": "water", "calories_per_100g": 0, "protein": 0, "carbs": 0, "fat": 0, "serving_g": 250},
    {"name": "cola", "calories_per_100g": 42, "protein": 0, "carbs": 10.6, "fat": 0, "serving_g": 330},
    {"name": "orange juice", "calories_per_100g": 45, "protein": 0.7, "carbs": 10.4, "fat": 0.2, "serving_g": 250},
    {"name": "apple juice", "calories_per_100g": 46, "protein": 0.1, "carbs": 11.3, "fat": 0.1, "serving_g": 250},
    {"name": "coffee", "calories_per_100g": 1, "protein": 0.1, "carbs": 0, "fat": 0, "serving_g": 240},
    {"name": "tea", "calories_per_100g": 1, "protein": 0, "carbs": 0.3, "fat": 0, "serving_g": 240},
    {"name": "ayran", "calories_per_100g": 24, "protein": 1.5, "carbs": 1.6, "fat": 1.3, "serving_g": 250},
    {"name": "beer", "calories_per_100g": 43, "protein": 0.5, "carbs": 3.6, "fat": 0, "serving_g": 330},
    {"name": "wine", "calories_per_100g": 82, "protein": 0.1, "carbs": 2.6, "fat": 0, "serving_g": 150},
]

FOOD_ALIASES = {
    "fries": "french fries",
    "fry": "french fries",
    "steak": "beef steak",
    "tomato sauce": "tomato soup",
    "egg": "egg",
    "boiled egg": "egg",
    "fried egg": "egg",
    "scrambled egg": "egg",
    "chicken": "chicken breast",
    "mashed potatoes": "mashed potato",
    "french fry": "french fries",
    "pasta": "pasta",
    "spaghetti bolognese": "pasta",
    "mac and cheese": "pasta",
    "cheese": "cheese",
    "cheddar": "cheddar cheese",
    "bell peppers": "bell pepper",
    "red pepper": "bell pepper",
    "green pepper": "bell pepper",
    "shrimp": "shrimp",
    "prawn": "shrimp",
    "coke": "cola",
    "soda": "cola",
    "popcorn": "chips",
    "french bread": "bread",
    "baguette": "bread",
}


def normalize_food_name(name: str) -> str:
    '''Yemek adini normallashdirir: kicik herf, xususi karakterleri silir.

    Parameterler:
        name: Normallashdirilacaq yemek adi

    Qaytaran deyer:
        str: Temizlenmis yemek adi
    '''
    name = name.lower().strip()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def find_best_match(query: str, threshold: float = 0.45) -> Optional[FoodInfo]:
    '''Sorguya en yaxin yemeyi bazada tapir.

    Evvel aliases yoxlanir, sonra tam eslesme, daha sonra
    SequenceMatcher ve soz eslesmesi ile en yaxsi netice qaytarilir.

    Parameterler:
        query: Yemek adi sorgusu
        threshold: Eslesme threshold-u (0.0 - 1.0)

    Qaytaran deyer:
        FoodInfo | None: Tapilan yemek melumatlari ve ya None
    '''
    query = normalize_food_name(query)

    if query in FOOD_ALIASES:
        query = FOOD_ALIASES[query]

    for food in FOOD_DATABASE:
        if query == food["name"]:
            return food

    best_score = 0.0
    best_match = None
    for food in FOOD_DATABASE:
        db_name = food["name"]
        score = SequenceMatcher(None, query, db_name).ratio()
        if score > best_score:
            best_score = score
            best_match = food

    words = query.split()
    for food in FOOD_DATABASE:
        db_words = food["name"].split()
        common = set(words) & set(db_words)
        if common:
            word_score = len(common) / max(len(words), len(db_words))
            if word_score > best_score:
                best_score = word_score
                best_match = food

    if best_score >= threshold:
        return best_match
    return None


def search_foods(query: str) -> list[FoodInfo]:
    '''Yemek adina gore bazada axtaris edir.

    Parameterler:
        query: Axtaris sorgusu

    Qaytaran deyer:
        list[FoodInfo]: Uygun yemeklerin siyahisi
    '''
    query = normalize_food_name(query)
    results = []
    for food in FOOD_DATABASE:
        if query in food["name"] or food["name"] in query:
            results.append(food)
    return results


class NutritionDB:
    '''Qida bazasi sinifi.
    Yemek melumatlarini axtarmaq, kalori hesablamaq ucun istifade olunur.
    '''

    @staticmethod
    def lookup(food_name: str) -> Optional[FoodInfo]:
        '''Yemek adina gore bazada axtaris edir.

        Parameterler:
            food_name: Yemek adi

        Qaytaran deyer:
            FoodInfo | None: Yemek melumatlari
        '''
        return find_best_match(food_name)

    @staticmethod
    def estimate_calories(food_name: str, portion_g: Optional[float] = None) -> Optional[dict]:
        '''Yemeyin kalorisini porsiya olcusune gore hesablayir.

        Porsiya verilmeyibse, default deyer istifade olunur.
        Qaytardigi dict: name, calories_per_100g, protein, carbs, fat,
        estimated_portion_g, estimated_calories

        Parameterler:
            food_name: Yemek adi
            portion_g: Porsiya olcusu (qram)

        Qaytaran deyer:
            dict | None: Kalori melumatlari
        '''
        match = find_best_match(food_name)
        if match is None:
            return None

        serving = portion_g or match["serving_g"]
        cal_per_100 = match["calories_per_100g"]
        total_cal = cal_per_100 * (serving / 100)

        result = dict(match)
        result["estimated_portion_g"] = serving
        result["estimated_calories"] = round(total_cal, 1)
        return result

    @staticmethod
    def search(query: str) -> list[FoodInfo]:
        '''Yemek adina gere axtaris.

        Parameterler:
            query: Axtaris sorgusu

        Qaytaran deyer:
            list[FoodInfo]: Uygun yemekler
        '''
        return search_foods(query)