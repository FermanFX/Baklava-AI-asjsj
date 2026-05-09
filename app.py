'''AI Food Analyzer - Gradio Web Interface
Veb brauzerde istifade ucun grafik interfeys.
Sekil yuklenir, analiz aparilir, neticeler Markdown formatinda gosterilir.
'''

import io
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

from food_analyzer.analyzer import FoodAnalyzer
from food_analyzer.model import get_vision_model
from food_analyzer.utils import format_calories


def create_app():
    '''Gradio web interfeysini yaradir ve qaytarir.

    Interfeysde:
    - Sekil yukleme
    - Analiz duymesi
    - Neticelerin Markdown cedveli

    Qaytaran deyer:
        gr.Blocks: Gradio app instance-i
    '''
    model = get_vision_model()
    analyzer = FoodAnalyzer(model)

    def analyze(image: Image.Image) -> str:
        '''Sekli analiz edir ve Markdown formatinda neticeni qaytarir.

        Parameterler:
            image: PIL.Image formatinda sekil

        Qaytaran deyer:
            str: Markdown formatinda netice metni
        '''
        if image is None:
            return "Please upload an image."

        result = analyzer.analyze_with_details(image)
        lines = []
        lines.append("# AI Food Analyzer - Results\n")
        lines.append(f"## Total Nutrition\n")
        tn = result["total_nutrition"]
        lines.append(f"- **Calories**: {format_calories(tn['calories'])}")
        lines.append(f"- **Protein**: {tn['protein_g']:.1f}g")
        lines.append(f"- **Carbs**: {tn['carbs_g']:.1f}g")
        lines.append(f"- **Fat**: {tn['fat_g']:.1f}g")
        lines.append(f"- **Confidence**: {result['confidence'] * 100:.0f}%\n")

        if result["foods"]:
            lines.append("## Detected Foods\n")
            lines.append("| Food | Portion | Calories | Protein | Carbs | Fat |")
            lines.append("|------|---------|----------|---------|-------|-----|")
            for food in result["foods"]:
                if "nutrition" in food:
                    n = food["nutrition"]
                    lines.append(
                        f"| {food['name']} | {n['portion_g']}g | "
                        f"{format_calories(n['calories'])} | {n.get('protein', 0):.0f}g | "
                        f"{n.get('carbs', 0):.0f}g | {n.get('fat', 0):.0f}g |"
                    )
                else:
                    lines.append(f"| {food['name']} | ? | ? | ? | ? | ? |")

        lines.append(f"\n---\n*Raw AI analysis: {result.get('raw_caption', 'N/A')}*")
        return "\n".join(lines)

    with gr.Blocks(
        title="AI Food Analyzer",
        theme=gr.themes.Soft(),
        css="""
        .app-title { text-align: center; font-size: 1.8em; margin-bottom: 0.5em; }
        .app-desc { text-align: center; color: #666; margin-bottom: 1.5em; }
        """,
    ) as demo:
        '''Gradio interfeys qurulusu:
        1. Basliq ve izahat
        2. Sekil yukleme + analiz duymesi
        3. Neticeler Markdown
        '''
        gr.Markdown(
            '<div class="app-title">AI Food Analyzer</div>',
        )
        gr.Markdown(
            '<div class="app-desc">Upload a photo of your meal to identify ingredients and estimate calories</div>',
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Meal Photo",
                    height=400,
                )
                analyze_btn = gr.Button("Analyze Meal", variant="primary", size="lg")

            with gr.Column(scale=1):
                output = gr.Markdown(
                    label="Analysis Results",
                    value="Upload an image and click 'Analyze Meal'",
                )

        analyze_btn.click(
            fn=analyze,
            inputs=image_input,
            outputs=output,
        )

        gr.Markdown(
            "---\n"
            "**How it works**: Your photo is analyzed by a Vision-Language AI model "
            "(Florence-2) that identifies food items. Calories are estimated using "
            "a built-in nutrition database of 150+ foods.\n\n"
            "**Note**: First run will download the AI model (~2GB). "
            "Results are estimates and should not be used for medical purposes."
        )

    return demo


if __name__ == "__main__":
    '''App-i ise salir: Gradio serveri basladir.'''
    demo = create_app()
    demo.launch(share=False)