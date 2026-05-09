# AI Food Analyzer

Analyze meal photos to identify ingredients and estimate calories using an AI Vision API.

## How It Works

1. **AI Vision API** (e.g. OpenAI GPT-4o) generates a detailed caption of the food image
2. **Ingredient Extraction** parses the caption to identify specific food items
3. **Nutrition Database** (150+ foods) matches detected items to calorie/nutrition values
4. **Calorie Estimation** calculates total calories based on portion estimates

## Installation

```bash
pip install -r requirements.txt
```

> **Note**: Open `api_settings.py` in the project root and fill in your API key.
> Or set the `OPENAI_API_KEY` environment variable.

## Usage

### CLI

```bash
# Analyze a local image
python main.py meal.jpg

# Analyze from URL
python main.py https://example.com/food.jpg

# Show raw AI caption
python main.py meal.jpg --verbose

# Use CPU only
python main.py meal.jpg --device cpu

# JSON output
python main.py meal.jpg --json
```

### Web Interface

```bash
python app.py
```

Then open the URL shown in terminal (usually `http://127.0.0.1:7860`).

## Project Structure

```
food_analyzer/
  ├── __init__.py        # Package exports
  ├── config.py          # Model, API & analysis configuration
  ├── model.py           # Vision-Language Model wrapper (Florence-2 / API)
  ├── nutrition_db.py    # Food nutrition database (150+ foods)
  ├── analyzer.py        # Food analysis pipeline
  └── utils.py           # Image loading utilities
main.py                  # CLI entry point
app.py                   # Gradio web interface
api_settings.py          # Your API key & endpoint settings (git-ignored)
requirements.txt         # Python dependencies
```

## Configuration

Edit `food_analyzer/config.py` to change model behavior:

| Config | Key | Default | Description |
|--------|-----|---------|-------------|
| `API_CONFIG` | `enabled` | `True` | Use API (False = local Florence-2) |
| `API_CONFIG` | `api_key_env` | `OPENAI_API_KEY` | Env variable name for API key |
| `API_CONFIG` | `base_url` | `https://api.openai.com/v1` | API endpoint |
| `API_CONFIG` | `model` | `gpt-4o` | Vision model name |
| `API_CONFIG` | `max_tokens` | `1024` | Max response length |

Set `API_CONFIG["enabled"] = False` to use the local Florence-2 model instead (requires ~2GB download). Don't do that as false. Because model file note found.

## Limitations

- Calorie estimates are approximate
- Portion sizes are estimated from defaults (not measured)
- The AI may not detect every ingredient in complex dishes
- An API key is required (set in `api_settings.py` or `OPENAI_API_KEY` env variable)
- Results should not be used for medical purposes

## License

MIT
