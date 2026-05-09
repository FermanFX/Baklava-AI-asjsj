'''Komekci funksiyalar modulu.
Sekil yukleme ve kalori formatlama kimi sade emeliyyatlar.
'''

from pathlib import Path

import requests
from PIL import Image


def load_image(image_source: str) -> Image.Image:
    '''Sekil faylini lokal diskden ve ya URL-den yukleyir.

    Parameterler:
        image_source: Fayl yolu (mes: "yemek.jpg") ve ya URL (mes: "https://...")

    Qaytaran deyer:
        PIL.Image: RGB formatinda sekil

    Istisnalar:
        FileNotFoundError: Fayl tapilmadigda
        requests.RequestException: URL yuklenmediyinde
    '''
    source = image_source.strip()

    if source.startswith(("http://", "https://")):
        response = requests.get(source, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {source}")
    return Image.open(path).convert("RGB")


def format_calories(cal: float) -> str:
    '''Kalori deyerini insan ucun oxunaqli formata cevirir.

    100 kcal-den yuxari deyerler tam eded, asagi deyerler 1 desimetrle gosterilir.

    Parameterler:
        cal: Kalori deyeri

    Qaytaran deyer:
        str: Formatlanmis kalori (mes: "250 kcal" ve ya "45.5 kcal")
    '''
    if cal >= 100:
        return f"{cal:.0f} kcal"
    return f"{cal:.1f} kcal"
