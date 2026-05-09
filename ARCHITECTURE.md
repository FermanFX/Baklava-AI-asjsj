# AI Food Analyzer - İşləmə Prinsipi (Architecture Guide)

## Ümumi Baxış

AI Food Analyzer, **Vision-Language Model (VLM)** texnologiyasından istifadə edərək yemək şəkillərini analiz edən bir Python proyektidir. Sistem API vasitəsi ilə (**GPT-4o** və ya OpenAI formatlı istənilən model) işləyir və 150+ yeməkdən ibarət qida bazası ilə kalori hesablaması aparır. Alternativ olaraq **Florence-2** lokal modeli də dəstəklənir.

---

## 1. Sistem Arxitekturası

```
                 ┌─────────────────────────────────┐
                 │        main.py / app.py         │
                 │    (CLI / Web Interface)        │
                 └─────────────┬───────────────────┘
                               │
                 ┌─────────────▼───────────────────┐
                 │      FoodAnalyzer (analyzer.py) │
                 │   - Caption parse               │
                 │   - Ingredient extraction       │
                 │   - Calorie calculation         │
                 └──┬──────────────────────┬───────┘
                    │                      │
         ┌──────────▼──────────────┐  ┌────▼──────────────┐
         │   VisionLanguageModel   │  │    NutritionDB    │
         │   (model.py)            │  │ (nutrition_db.py) │
         │                         │  │   150+ foods      │
         │   API mode (default):   │  │   fuzzy match     │
         │   GPT-4o via HTTP       │  └───────────────────┘
         │                         │
         │   Local mode:           │
         │   Florence-2 via HF     │
         └─────────────────────────┘
```

### Modulların Vəzifələri

| Modul | Fayl | Vəzifə |
|-------|------|--------|
| **config** | `config.py` | Model, API, analiz və kalori parametrləri |
| **model** | `model.py` | Vision-Language model wrapper (API / Florence-2) |
| **nutrition_db** | `nutrition_db.py` | Qida bazası, kalori lookup, fuzzy matching |
| **analyzer** | `analyzer.py` | Analiz orchestrator - əsas məntiq |
| **utils** | `utils.py` | Şəkil yükləmə, kalori formatlama |
| **main** | `main.py` | CLI interfeys |
| **app** | `app.py` | Gradio web interfeys |
| **api_settings** | `api_settings.py` | Şəxsi API açarı və endpoint (git-ignored) |

---

## 2. Data Flow (Məlumat Axını)

### Addım 1: Şəkil Yükləmə

```
User Input (path/URL) → load_image() → PIL.Image (RGB)
```

- Lokal fayl və ya URL qəbul edilir
- `PIL.Image` formatına çevrilir
- `requests` ilə URL-dən stream yükləmə

### Addım 2: AI Caption Yaradılması

```
PIL.Image → APIVisionLanguageModel / VisionLanguageModel → Detailed Caption
                                 ↓
      "A plate of grilled chicken breast with steamed 
       broccoli and white rice, garnished with lemon."
```

**API mode (default):** `APIVisionLanguageModel`
- Şəkil base64 encode edilir
- OpenAI formatında API-ə göndərilir (`/chat/completions`)
- GPT-4o və ya konfiqurasiyada göstərilən model işləyir
- `API_CONFIG`-də `base_url` dəyişdirilərək fərqli provayderlərə keçmək olar

**Local mode (`API_CONFIG["enabled"] = False`):** `VisionLanguageModel`
- `<MORE_DETAILED_CAPTION>` taskı istifadə olunur
- Florence-2 HuggingFace-dən lokal işləyir
- `num_beams=3` beam search ilə keyfiyyət artırılır

### Addım 3: Yemək Adlarının Çıxarılması

```
Caption → _extract_food_items() → ["chicken breast", "broccoli", "rice"]
```

İşləmə ardıcıllığı:
1. **FOOD_DATABASE**-dəki hər yemək adı caption içində axtarılır
2. Çoxsözlü adlar (məs: "chicken breast") söz-söz yoxlanılır
3. **FOOD_ALIASES** (məs: "fries" → "french fries") tətbiq olunur
4. Təkrarlar silinir (_deduplicate_foods)
5. Heç nə tapılmazsa, regex fallback işləyir

### Addım 4: Kalori Hesablama

```
"chicken breast" → NutritionDB.estimate_calories()
    ↓
{
  "name": "chicken breast",
  "calories_per_100g": 165,
  "serving_g": 150,
  "estimated_calories": 247.5,  // 165 * 150/100
  "protein": 31, "carbs": 0, "fat": 3.6
}
```

- `find_best_match()`: Exact match → SequenceMatcher → Word overlap
- Porsiya dəyəri `serving_g` default və ya user input
- Kalori = `calories_per_100g * (portion_g / 100)`

### Addım 5: Nəticələrin Toplanması

```
Hər yemək üçün:
  - estimated_calories += total_calories
  - protein * portion/100 += total_protein
  - carbs * portion/100 += total_carbs
  - fat * portion/100 += total_fat

confidence = matched_foods / total_foods
```

---

## 3. NutritionDB - Qida Bazası Sistemi

### Axtarış Mexanizmi (`find_best_match`)

```
                    Sorgu
                      │
              ┌───────▼────────┐
              │  normalize()   │
              │  lower, clean  │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
         ┌────┤  FOOD_ALIASES  ├────┐
         │    └───────┬────────┘    │
         │            │             │
         │    ┌───────▼────────┐    │
         │    │  Exact match   ├────┼─── Qaytar
         │    └───────┬────────┘    │
         │            │             │
         │    ┌───────▼────────┐    │
         │    │ SequenceMatcher│    │
         │    │  (difflib)     │    │
         │    └───────┬────────┘    │
         │            │             │
         │    ┌───────▼────────┐    │
         │    │  Word overlap  │    │
         │    └───────┬────────┘    │
         │            │             │
         │    ┌───────▼────────┐    │
         └────┤ best_score >=  ├────┘
              │   threshold    │
              └───────┬────────┘
                      │
              Qaytar / None
```

### Məlumat Strukturu

Hər yemək entry-i:
```python
{
  "name": "food name",           # Axtarış üçün
  "calories_per_100g": 165,      # Kalori sıxlığı
  "protein": 31,                 # Zülal (q/100q)
  "carbs": 0,                    # Karbohidrat (q/100q)
  "fat": 3.6,                    # Yağ (q/100q)
  "serving_g": 150               # Standart porsiya
}
```

150+ yemək 10 kateqoriyada:
- Protein/Meat (16)
- Grains/Carbs (17)
- Vegetables (18)
- Fruits (11)
- Dairy (9)
- Legumes (4)
- Soups (3)
- Sauces/Oils (6)
- Snacks/Fast Food (11)
- Azerbaijani/Turkish (13)
- Beverages (9)

---

## 4. Vision-Language Model

Sistem iki rejimdə işləyə bilər: **API mode** (default) və **Local mode**.

### API Mode (APIVisionLanguageModel)

OpenAI formatlı API-lərlə işləyir (GPT-4o, Azure OpenAI, Groq, Together və s.).

```
PIL.Image → base64 encode → HTTP POST /chat/completions → Caption text
```

| Parametr | Dəyər |
|----------|-------|
| Model | `API_CONFIG["model"]` (default: `gpt-4o`) |
| API nöqtəsi | `API_CONFIG["base_url"]` (default: `https://api.openai.com/v1`) |
| API açarı | `api_settings.py` > `OPENAI_API_KEY` env |
| Maks token | `API_CONFIG["max_tokens"]` (default: 1024) |

API açarının təyini ardıcıllığı:
1. `api_settings.py` faylı (proyekt kökündə)
2. `OPENAI_API_KEY` environment variable
3. Xəta: "API key not found"

İşləmə ardıcıllığı:
1. `_encode_image()`: PIL.Image base64-ə çevrilir
2. `generate_caption()`: API-ə `chat/completions` sorğusu göndərilir
3. API cavabından caption mətni çıxarılır və qaytarılır

### Local Mode (VisionLanguageModel — Florence-2)

`API_CONFIG["enabled"] = False` edildikdə lokal Florence-2 modeli işləyir.

```
Input Image (224x224)
      ↓
┌─────────────────┐
│  ViT Encoder    │  (Vision Transformer)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Cross-Attention│
│  + Text Decoder │  (Causal LM)
└────────┬────────┘
         ↓
   Output Text (Caption)
```

| Parametr | Dəyər |
|----------|-------|
| Model | microsoft/Florence-2-large |
| Parametr sayı | ~1.3B |
| Vision Encoder | ViT-L/14 |
| Text Decoder | GPT-2 based |
| Giriş ölçüsü | 224x224 |
| İşləmə cihazı | CUDA (auto) / CPU |
| Dəqiqlik | float16 (GPU) / float32 (CPU) |

### Task Promptları (Local mode)

| Prompt | İstifadə |
|--------|----------|
| `<CAPTION>` | Qısa təsvir |
| `<DETAILED_CAPTION>` | Ətraflı təsvir |
| `<MORE_DETAILED_CAPTION>` | Ən ətraflı təsvir (default) |
| `<OD>` | Obyekt detektasiyası |

### Yükləmə Mexanizmi (Local mode)

```
1. load() çağırılır
2. cache_dir (./models/) yoxlanılır
3. Model yoxdursa HuggingFace-dən endirilir
4. Processor + Model yaddaşa yüklənir
5. Növbəti çağrışlarda cached versiya işləyir
```

### Singleton Pattern

`get_vision_model()` həmişə eyni instance-ı qaytarır. Rejim seçimi:
- `API_CONFIG["enabled"] = True` → `APIVisionLanguageModel`
- `API_CONFIG["enabled"] = False` → `VisionLanguageModel` (Florence-2)

---

## 5. Konfiqurasiya Sistemi

4 konfiqurasiya qrupu:

```
API_CONFIG (default — API mode):
  enabled         : True (API) / False (local model)
  api_key_env     : OPENAI_API_KEY
  base_url        : https://api.openai.com/v1
  model           : gpt-4o
  max_tokens      : 1024

MODEL_CONFIG (local mode):
  model_id        : HuggingFace ID / local path
  model_cache_dir : Lokal cache qovluğu
  device          : cuda / cpu (auto-detect)

CALORIE_CONFIG:
  default_portion_g       : 150g
  confidence_threshold    : 0.4
  include_portion_estimate: True

ANALYSIS_CONFIG:
  detailed_caption_task   : "<MORE_DETAILED_CAPTION>"
  caption_task            : "<CAPTION>"
  detection_task          : "<OD>"
```

---

## 6. İnterfeyslər

### CLI (main.py)

```
$ python main.py yemek.jpg
$ python main.py https://example.com/food.jpg --verbose
$ python main.py yemek.jpg --json
$ python main.py yemek.jpg --device cpu
```

Argumentlər:
- `image` (required): Fayl yolu və ya URL
- `--device`: CUDA/CPU override
- `--verbose, -v`: AI caption-ı göstər
- `--json`: JSON output
- `--release`: Modeli yaddaşdan boşalt

### Web (app.py)

```
$ python app.py
→ http://127.0.0.1:7860
```

Gradio interfeysi:
- Şəkil yükləmə (drag & drop)
- "Analyze Meal" düyməsi
- Nəticələr Markdown cedveli
- Mobile-friendly responsive dizayn

---

## 7. Asılılıqlar (Dependencies)

```
torch           → PyTorch (local model inference, optional)
transformers    → HuggingFace (Florence-2, optional)
accelerate      → GPU optimizasiya (local mode, optional)
Pillow          → Şəkil emalı
requests        → API sorğuları + URL yükləmə
gradio          → Web interfeys
sentencepiece   → Tokenizer (local mode, optional)
protobuf        → Model config (local mode, optional)
```

> API mode yalnız `Pillow`, `requests` və `gradio` tələb edir.
> Local mode üçün `torch`, `transformers`, `accelerate`, `sentencepiece`, `protobuf` da lazımdır.

---

## 8. Məhdudiyyətlər

1. **Kalori təxminidir** — porsiya ölçüləri standart dəyərlərə əsaslanır
2. **Caption keyfiyyəti** — mürəkkəb yeməklərdə bütün ingredientlər aşkarlanmaya bilər
3. **API açarı tələb olunur** — API mode üçün `OPENAI_API_KEY` environment variable-i set edilməlidir
4. **API xərci** — GPT-4o və oxşar API-lər pulludur (token əsaslı)
5. **İnternet tələb olunur** — API mode üçün daimi internet bağlantısı lazımdır
6. **İngilis dili** — model əsasən ingilis dilində caption yaradır

---

## 9. Gələcək İnkişaflar

- [ ] Batch analiz (çox şəkli bir anda)
- [ ] Custom model fine-tuning (yemək spesifik)
- [ ] Porsiya təxmini üçün depth estimation
- [ ] Çoxdilli dəstək (Azərbaycan, Türk)
- [ ] Mobile app (Flutter ilə)
- [ ] REST API (FastAPI)
- [ ] QR kod ilə məhsul lookup
