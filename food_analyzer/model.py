'''Vision-Language Model modulu.
Florence-2 modelini idare edir: yukleme, caption yaratma, obyekt detektasiyasi.
Model HuggingFace-den lokal cache qovluguna endirilir ve ya API ile ishlenir.
'''

import base64
import io
import os
from pathlib import Path
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .config import API_CONFIG, MODEL_CONFIG


class VisionLanguageModel:
    '''Florence-2 Vision-Language Model sinifi.

    Modeli idare edir: yukleyir, caption yaradir, obyekt detektasiyasi edir.
    Singleton patterni ile isleyir - bir defe yuklenir, tekrar istifade olunur.

    Parameterler:
        model_id: HuggingFace model adi ve ya lokal yol
        device: "cuda" ve ya "cpu"
        cache_dir: Modelin ke$lendiyi lokal qovluq
    '''

    def __init__(self, model_id: str = None, device: str = None, cache_dir: str = None):
        '''VisionLanguageModel sinifini yaradir.

        Konfigurasiyani hazirlayir ve cihazi teyin edir.
        Model bu anda yuklenmir - load() cagirilana qeder gozleyir.
        '''
        self.config = MODEL_CONFIG.copy()
        if model_id:
            self.config["model_id"] = model_id
        if device:
            self.config["device"] = device
        if cache_dir:
            self.config["model_cache_dir"] = cache_dir

        self.device = self._resolve_device()
        self.processor = None
        self.model = None
        self._loaded = False

    def _resolve_device(self) -> str:
        '''Istifade olunacaq cihazi teyin edir.

        CUDA movcuddursa GPU istifade olunur, yoxsa CPU.
        '''
        if self.config["device"] == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if self.config["device"] == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_model_local_path(self, model_id: str, cache_dir: str) -> str:
        '''Modelin lokal cache-deki tam yolunu qaytarir.

        HuggingFace cache strukturuna uygun yol yaradir.
        '''
        model_name = model_id.replace("/", "--")
        return str(Path(cache_dir) / f"models--{model_name}")

    def load(self):
        '''Modeli yaddasa yukleyir (eger yuklenmeyibse).

        Processor ve model HuggingFace-den göturulur ve cache_dir-e
        kayd edilir. Tekrar cagrildiqda artiq yuklenmis model istifade olunur.
        '''
        if self._loaded:
            return

        model_id = self.config["model_id"]
        cache_dir = self.config["model_cache_dir"]
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        os.makedirs(cache_dir, exist_ok=True)

        is_local = os.path.isdir(model_id)
        kwargs = {
            "trust_remote_code": True,
            "cache_dir": cache_dir,
        }

        self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **kwargs,
        ).to(self.device)

        self.model.eval()
        self._loaded = True
        if not is_local:
            print(f"  Model cached at: {self._get_model_local_path(model_id, cache_dir)}")

    def unload(self):
        '''Modeli yaddasdan bosaldir.

        CUDA istifade olunursa cache-i temizleyir.
        '''
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_caption(self, image: Image.Image, task: str = "<MORE_DETAILED_CAPTION>") -> str:
        '''Sekil ucun AI caption yaradir.

        Florence-2 modeline sekil gonderir ve metin tesviri alir.
        "<MORE_DETAILED_CAPTION>" taski en detalli caption verir.

        Parameterler:
            image: PIL.Image formatinda sekil
            task: Florence-2 task promptu (caption, detailed_caption, OD)

        Qaytaran deyer:
            str: AI terefinden yaradilmis metin tesviri
        '''
        self.load()

        inputs = self.processor(text=task, images=image, return_tensors="pt")

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        generated_ids = self.model.generate(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=self.config["max_new_tokens"],
            num_beams=self.config["num_beams"],
            early_stopping=False,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

        return parsed.get(task, generated_text)

    @torch.no_grad()
    def detect_objects(self, image: Image.Image) -> dict:
        '''Sekildeki obyektleri detect edir.

        Florence-2-nin "<OD>" (Object Detection) taskindan istifade edir.
        Qaytardigi dict-de label, bounding box koordinatlari ve confidence deyerleri var.

        Parameterler:
            image: PIL.Image formatinda sekil

        Qaytaran deyer:
            dict: Detect olunmus obyektlerin siyahisi (label, bbox, score)
        '''
        self.load()
        task = "<OD>"

        inputs = self.processor(text=task, images=image, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        generated_ids = self.model.generate(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=self.config["max_new_tokens"],
            num_beams=self.config["num_beams"],
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        return self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )


class APIVisionLanguageModel:
    '''API-based Vision-Language Model sinifi.

    Lokal model yuklemek yerine API vasitesi ile caption yaradir.
    OpenAI formatinda olan istənilən API ile ishleyir.

    API ayarlari (key, base_url, model) ucun prioritet siyahisi:

    1. Birbasha parametr (model_name, base_url, max_tokens)
    2. api_settings.py fayli (kok qovluqda)
    3. Environment variable (OPENAI_API_KEY)
    '''

    def __init__(self, model_name: str = None, base_url: str = None, max_tokens: int = None):
        self.config = API_CONFIG.copy()

        self.api_key = self._resolve_api_key()
        if not self.api_key:
            raise ValueError(
                f"API key not found.\n"
                f"  1. Fill in api_settings.py in the project root folder\n"
                f"  2. Or set {self.config['api_key_env']} environment variable\n"
                f"  3. Or disable API mode in config.py (API_CONFIG['enabled'] = False)"
            )

        if model_name:
            self.config["model"] = model_name
        elif self._settings_model:
            self.config["model"] = self._settings_model

        if base_url:
            self.config["base_url"] = base_url
        elif self._settings_base_url:
            self.config["base_url"] = self._settings_base_url

        if max_tokens:
            self.config["max_tokens"] = max_tokens
        elif self._settings_max_tokens:
            self.config["max_tokens"] = self._settings_max_tokens

        self._loaded = True

    def _resolve_api_key(self) -> str:
        self._settings_base_url = None
        self._settings_model = None
        self._settings_max_tokens = None
        try:
            from api_settings import API_KEY, BASE_URL, MODEL, MAX_TOKENS
            if API_KEY:
                self._settings_base_url = BASE_URL
                self._settings_model = MODEL
                self._settings_max_tokens = MAX_TOKENS
                return API_KEY
        except (ImportError, ModuleNotFoundError):
            pass
        return os.environ.get(self.config["api_key_env"], "")

    def _encode_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_caption(self, image: Image.Image, task: str = "<MORE_DETAILED_CAPTION>") -> str:
        prompt = self.config["caption_prompt"]
        base64_image = self._encode_image(image)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.config["max_tokens"],
        }

        response = requests.post(
            f"{self.config['base_url'].rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def detect_objects(self, image: Image.Image) -> dict:
        prompt = self.config["detection_prompt"]
        base64_image = self._encode_image(image)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.config["max_tokens"],
        }

        response = requests.post(
            f"{self.config['base_url'].rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return {"bboxes": [], "labels": [text]}

    def load(self):
        pass

    def unload(self):
        pass


_MODEL_INSTANCE: Optional[VisionLanguageModel | APIVisionLanguageModel] = None


def get_vision_model(
    model_id: str = None, device: str = None, cache_dir: str = None, use_api: bool = None
) -> VisionLanguageModel | APIVisionLanguageModel:
    '''Singleton model yaradicisi.

    API_CONFIG["enabled"] = True ise API model, eks halda lokal model qaytarir.
    use_api parametri ile forced edile biler.

    Parameterler:
        model_id: HuggingFace model adi (lokal mod ucun)
        device: "cuda" ve ya "cpu" (lokal mod ucun)
        cache_dir: Cache qovlugu (lokal mod ucun)
        use_api: True = API, False = lokal, None = config deyeri

    Qaytaran deyer:
        VisionLanguageModel | APIVisionLanguageModel: Model instance-i
    '''
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    if use_api is None:
        use_api = API_CONFIG["enabled"]

    if use_api:
        _MODEL_INSTANCE = APIVisionLanguageModel()
    else:
        _MODEL_INSTANCE = VisionLanguageModel(
            model_id=model_id, device=device, cache_dir=cache_dir
        )

    return _MODEL_INSTANCE


def release_vision_model():
    '''Modeli yaddasdan bosaldir ve singleton-i sifirlayir.'''
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        _MODEL_INSTANCE.unload()
        _MODEL_INSTANCE = None


def model_exists_locally(model_id: str = None, cache_dir: str = None) -> bool:
    '''Model faylinin lokal cache-de olub-olmadigini yoxlayir.

    Parameterler:
        model_id: HuggingFace model adi
        cache_dir: Cache qovlugu

    Qaytaran deyer:
        bool: True eger model lokal varsa
    '''
    cfg = MODEL_CONFIG.copy()
    mid = model_id or cfg["model_id"]
    cd = cache_dir or cfg["model_cache_dir"]
    path = Path(cd) / f"models--{mid.replace('/', '--')}"
    return path.exists() and any(path.iterdir())