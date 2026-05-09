import json
import re

import gradio as gr
from PIL import Image

from food_analyzer.model import get_vision_model

model = get_vision_model()


def analyze(image: Image.Image) -> str:
    if image is None:
        return "Please upload an image."

    try:
        caption = model.generate_caption(image)
    except Exception as e:
        return f"Error: {e}"

    try:
        json_match = re.search(r'\{.*\}', caption, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            total = data.get("total_calories", 0)
            foods = data.get("foods", [])
            desc = data.get("description", "")

            lines = []
            lines.append(f"## Total Calories: **{total} kcal**")
            if desc:
                lines.append(f"\n{desc}\n")
            if foods:
                lines.append("### Foods Detected\n")
                for f in foods:
                    name = f.get("name", "?")
                    cal = f.get("calories", "?")
                    lines.append(f"- **{name}**: {cal} kcal")
            return "\n".join(lines)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return f"**AI Response:**\n\n{caption}"


with gr.Blocks(title="AI Food Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Food Analyzer")
    gr.Markdown("Upload a food photo to get instant calorie estimation")

    image_input = gr.Image(type="pil", label="Upload Photo", height=400)
    output = gr.Markdown(value="Upload an image to analyze")

    image_input.change(fn=analyze, inputs=image_input, outputs=output)

if __name__ == "__main__":
    demo.launch(share=False)
