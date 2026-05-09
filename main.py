'''AI Food Analyzer - CLI giris noktasi.
Terminaldan istifade ucun hazirlanmishdir.
Sekil fayli ve ya URL qebul edir, yemek analizi neticelerini gosterir.
'''

import argparse
import sys
from pathlib import Path

from food_analyzer.analyzer import FoodAnalyzer
from food_analyzer.model import get_vision_model, release_vision_model
from food_analyzer.utils import load_image, format_calories


def print_banner():
    '''Terminalda giris basligini gosterir.'''
    print("=" * 58)
    print("  AI Food Analyzer v1.0")
    print("  Analyze meal photos · Identify ingredients · Estimate calories")
    print("=" * 58)


def print_result(result: dict, show_caption: bool = False):
    '''Analiz neticelerini terminalda formatli gosterir.

    Her bir yemek, porsiya, kalori ve makro deyerleri cedvel seklinde.
    Umumi kalori ve qidali deyerler de gosterilir.

    Parameterler:
        result: analyze_with_details() qaytardigi dict
        show_caption: Xam AI caption gosterilsin?
    '''
    print()
    print("─" * 58)
    print("  ANALYSIS RESULTS")
    print("─" * 58)

    if result["foods"]:
        print(f"\n  Detected Foods ({len(result['foods'])}):")
        print(f"  {'─' * 40}")
        for food in result["foods"]:
            name = food["name"]
            conf = food["confidence"]
            if "nutrition" in food:
                n = food["nutrition"]
                cal = format_calories(n["calories"])
                portion = f"{n['portion_g']}g"
                print(f"  {name:<20} {portion:>8}  {cal:>8}  (p:{n.get('protein', 0):.0f}g "
                      f"c:{n.get('carbs', 0):.0f}g f:{n.get('fat', 0):.0f}g)")
            else:
                print(f"  {name:<20} {'?':>8}  {'?':>8}  (low confidence)")
    else:
        print("\n  No specific foods detected.")

    print()
    print("─" * 40)
    tn = result["total_nutrition"]
    print(f"  TOTAL CALORIES:  {format_calories(tn['calories']):>10}")
    print(f"  PROTEIN:         {tn['protein_g']:>5.1f}g")
    print(f"  CARBS:           {tn['carbs_g']:>5.1f}g")
    print(f"  FAT:             {tn['fat_g']:>5.1f}g")
    print(f"  CONFIDENCE:      {result['confidence'] * 100:.0f}%")
    print("─" * 40)

    if show_caption and result.get("raw_caption"):
        print(f"\n  Raw AI Caption:")
        print(f"  {result['raw_caption']}")
        print("─" * 40)

    print()


def analyze_single(args):
    '''Tek bir seklin analizini aparir ve neticeni gosterir.

    Isleme axini:
    1. Sekil yuklenir
    2. Modelin lokal olub-olmadigi yoxlanilir
    3. Model yuklenir (lazim varsa endirilir)
    4. Analiz aparilir
    5. Neticeler gosterilir

    Parameterler:
        args: argparse-den gelen argumentler
    '''
    print_banner()
    print(f"\n  Loading image: {args.image}")

    try:
        image = load_image(args.image)
    except Exception as e:
        print(f"  Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Image size: {image.size[0]}x{image.size[1]}")

    print(f"  Analyzing with AI vision API...")

    try:
        model = get_vision_model(device=args.device)
        analyzer = FoodAnalyzer(model)
        result = analyzer.analyze_with_details(image)
        print_result(result, show_caption=args.verbose)
    except Exception as e:
        print(f"  Analysis error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if args.release:
            release_vision_model()


def main():
    '''CLI-nin esas giris noktasi.

    Argumentleri tehlil edir ve uygun emeliyyati yerine yetirir:
    - Normal rejim: analyze_single() cagirir
    - JSON rejimi: Neticeni JSON formatinda cixarir
    '''
    parser = argparse.ArgumentParser(
        description="AI Food Analyzer - Analyze meal photos to identify ingredients and estimate calories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py pizza.jpg
  python main.py meal.png --verbose
  python main.py https://example.com/food.jpg --device cpu
  python main.py batch ./food_photos/
        """,
    )

    parser.add_argument("image", help="Path to image file or URL")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Override device (auto-detected if not specified)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show raw AI caption")
    parser.add_argument("--release", action="store_true",
                        help="Release model from memory after analysis")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()

    if args.json:
        import json
        try:
            image = load_image(args.image)
            model = get_vision_model(device=args.device)
            analyzer = FoodAnalyzer(model)
            result = analyzer.analyze_with_details(image)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)
        finally:
            if args.release:
                release_vision_model()
    else:
        analyze_single(args)


if __name__ == "__main__":
    main()