# classify.py

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from google.cloud import vision
from google.cloud import secretmanager
import os
from dotenv import load_dotenv
import json
from google.oauth2 import service_account
import logging
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional, Dict
import requests
import httpx
import asyncio  
from collections import Counter 
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends, status 
from auth import get_current_user
import google.generativeai as genai
import requests
import os
import logging
import re
# ...

# Schemas and Models
from schemas import BreedIdentificationResult, PetCreate, AnnouncementCreate, SearchRequest, AnnouncementResponse
from models import Pet as PetModel, Announcement as AnnouncementModel, User as UserModel

# Database and Auth dependencies
from database import get_db
from auth import get_current_user_id

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
router = APIRouter(tags=["classify"])

# API-ключ для Mistral (оставляем, если он используется где-то еще)
MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN")

# --- НАСТРОЙКА GEMINI API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")
gemini_generative_model = None

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY не указан. Функции проверки текста на токсичность и получения рекомендаций через Gemini будут недоступны.")
else:
    try:
        # Настройка прокси через переменные окружения
        if HTTPS_PROXY:
            os.environ['HTTP_PROXY'] = HTTPS_PROXY
            os.environ['HTTPS_PROXY'] = HTTPS_PROXY
            logger.info(f"Установлен прокси для Gemini API: {HTTPS_PROXY}")
        else:
            logger.warning("HTTPS_PROXY не указан в .env файле.")

        # Дополнительное логирование текущих переменных окружения
        logger.debug(f"Текущие переменные окружения для прокси: HTTP_PROXY={os.getenv('HTTP_PROXY')}, HTTPS_PROXY={os.getenv('HTTPS_PROXY')}")

        genai.configure(api_key=GEMINI_API_KEY)
        gemini_generative_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"Google Gemini Client инициализирован с моделью {GEMINI_MODEL_NAME}.")
    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать Google Gemini Client: {e}", exc_info=True)
        gemini_generative_model = None
        
try:
    # Assumes breeds_map.json is in the same directory or accessible from where the app runs
    with open("breeds_map.json", "r", encoding="utf-8") as f:
        BREEDS_MAP = json.load(f)
    logger.info("Файл breeds_map.json успешно загружен.")
except FileNotFoundError:
    logger.error("Критическая ошибка: Файл breeds_map.json не найден. Анализ и перевод пород невозможны.")
    BREEDS_MAP = {"types": {}, "dog": {}, "cat": {}} # Default empty map
except json.JSONDecodeError as e:
    logger.error(f"Критическая ошибка: Ошибка декодирования JSON в файле breeds_map.json: {e}. Анализ и перевод пород невозможны.")
    BREEDS_MAP = {"types": {}, "dog": {}, "cat": {}} # Default empty map
except Exception as e:
    logger.error(f"Критическая ошибка: Неизвестная ошибка при загрузке breeds_map.json: {e}. Анализ и перевод пород невозможны.")
    BREEDS_MAP = {"types": {}, "dog": {}, "cat": {}} # Default empty map


# --- Google Cloud Credentials ---
def get_credentials():
    """
    Получает учетные данные Google Cloud из Secret Manager (если RENDER=true)
    или возвращает None для использования ADC/переменной окружения.
    """
    try:
        if os.getenv("RENDER"):
            logger.info("Обнаружена среда Render. Получение ключей из Secret Manager...")
            client = secretmanager.SecretManagerServiceClient()
            secret_name = "projects/pet-recognizer-bot/secrets/pet-recognizer-bot-credentials/versions/latest" 
            response = client.access_secret_version(request={"name": secret_name})
            payload = response.payload.data.decode("UTF-8")
            info = json.loads(payload)
            logger.info("Учетные данные успешно получены из Secret Manager.")
            return service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        else:
            logger.info("Среда Render не обнаружена. Используются стандартные методы поиска учетных данных (ADC, GOOGLE_APPLICATION_CREDENTIALS).")
            return None
    except ImportError:
         logger.error("Ошибка импорта google.cloud.secretmanager. Установите: `pip install google-cloud-secret-manager`")
         return None
    except Exception as e:
        logger.error(f"Ошибка при получении учетных данных Google: {e}", exc_info=True)
        return None

# --- Google Vision Client Initialization ---
vision_client = None
try:
    logger.info("Попытка инициализации Google Vision Client...")
    credentials = get_credentials()
    if credentials:
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        logger.info("Google Vision Client инициализирован с учетными данными из get_credentials().")
    else:
        vision_client = vision.ImageAnnotatorClient()
        logger.info("Google Vision Client инициализирован без явных учетных данных (используется стандартный поиск).")
    # Можно добавить простой пинг API здесь для проверки при старте, если необходимо
except Exception as e:
    logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать Google Vision Client при старте: {e}", exc_info=True)
    # vision_client останется None, функции должны это проверять

# --- Core Logic Functions ---

def detect_breed(image_data: bytes) -> Dict[str, Optional[str]]:
    """
    Анализирует изображение через Google Vision Web Detection, определяет тип и породу.
    Использует ГЛОБАЛЬНЫЙ vision_client и BREEDS_MAP.
    Возвращает: {"type": "Dog" | "Cat" | None, "breed": "ключ_порода" | None}
    """
    if not vision_client:
         logger.error("Vision API client не инициализирован. Анализ породы невозможен.")
         return {"type": None, "breed": None}

    if not BREEDS_MAP or not BREEDS_MAP.get("types"): 
        logger.error("Словарь пород BREEDS_MAP пуст или некорректен. Анализ породы невозможен.")
        return {"type": None, "breed": None}

    try:
        image = vision.Image(content=image_data)
        logger.info("Отправка запроса Web Detection в Vision API...")
        response = vision_client.web_detection(image=image)
        logger.info("Ответ от Web Detection API получен.")

        if response.error.message:
            logger.error(f"Vision API Web Detection Error: {response.error.message}")
            return {"type": None, "breed": None}

        web_detection = response.web_detection
        if not web_detection.web_entities:
            logger.info("Web Detection не вернул веб-сущностей.")
            return {"type": None, "breed": None}

        logger.debug("--- Web Detection Entities Received (detect_breed): ---")
        for entity in web_detection.web_entities:
             logger.debug(f"  Entity: desc='{entity.description}', score={entity.score:.4f}, id='{entity.entity_id}'")
        logger.debug("--- End of Web Detection Entities ---")

        detected_type = None
        plausible_breeds = []
        best_dog_term_score = 0.0
        best_cat_term_score = 0.0
        GENERAL_TERM_THRESHOLD = 0.5

        # 1. Анализ сущностей
        for entity in web_detection.web_entities:
            desc_lower_key = entity.description.lower().replace(" ", "_")
            desc_lower_text = entity.description.lower()
            score = entity.score

            is_dog_breed = desc_lower_key in BREEDS_MAP.get("dog", {})
            is_cat_breed = desc_lower_key in BREEDS_MAP.get("cat", {})

            if is_dog_breed:
                plausible_breeds.append((score, desc_lower_key, "Dog"))
            elif is_cat_breed:
                plausible_breeds.append((score, desc_lower_key, "Cat"))

            # Проверка общих терминов
            if any(term in desc_lower_text for term in ["dog", "puppy", "canine", "собака", "щенок", "пес"]):
                best_dog_term_score = max(best_dog_term_score, score)
            if any(term in desc_lower_text for term in ["cat", "kitten", "feline", "кошка", "котенок", "кот"]):
                best_cat_term_score = max(best_cat_term_score, score)

        # 2. Выбор лучшей породы (если есть)
        best_breed_key = None
        best_breed_type = None
        if plausible_breeds:
            plausible_breeds.sort(key=lambda item: item[0], reverse=True)
            best_breed_score, best_breed_key, best_breed_type = plausible_breeds[0]
            logger.info(f"Лучшая порода определена как (ключ): {best_breed_key} (score={best_breed_score:.2f}), Тип: {best_breed_type}")
        else:
            logger.info("Конкретная порода из словаря не найдена.")

        # 3. Определение финального типа
        if best_breed_type:
             detected_type = best_breed_type
        else:
             # Определяем тип по общим терминам, если порода не найдена
             if best_dog_term_score > best_cat_term_score and best_dog_term_score >= GENERAL_TERM_THRESHOLD:
                 detected_type = "Dog"
             elif best_cat_term_score > best_dog_term_score and best_cat_term_score >= GENERAL_TERM_THRESHOLD:
                 detected_type = "Cat"
             else:
                 logger.warning(f"Не удалось уверенно определить тип животного ни по породам, ни по общим терминам (dog_score={best_dog_term_score:.2f}, cat_score={best_cat_term_score:.2f}).")
                 detected_type = None

        logger.info(f"Финальный результат detect_breed: type={detected_type}, breed_key={best_breed_key}")
        return {"type": detected_type, "breed": best_breed_key}

    except Exception as e:
        logger.error(f"Ошибка при анализе изображения в detect_breed: {e}", exc_info=True)
        return {"type": None, "breed": None}


def check_image_for_animals(image_data: bytes) -> bool:
    if not vision_client:
        logger.error("Vision API client не инициализирован. Проверка изображения (Object Localization) невозможна.")
        return False
    try:
        image = vision.Image(content=image_data)
        logger.info("Отправка запроса Object Localization в Vision API...")
        objects = vision_client.object_localization(image=image).localized_object_annotations
        logger.info(f"Получено {len(objects)} объектов от Object Localization.")
        animal_count = 0
        detected_object_names = [obj.name for obj in objects]
        logger.debug(f"Обнаруженные объекты: {detected_object_names}")
        for name in detected_object_names:
            if name.lower() in ["dog", "cat", "собака", "кошка", "пес", "кот"]:
                animal_count += 1
                logger.info(f"Обнаружено животное: {name} (подсчет: {animal_count})")
        logger.info(f"Итоговое количество животных (кошка/собака): {animal_count}")
        if animal_count == 1:
            logger.info("Проверка изображения (check_image_for_animals) пройдена: найдена ровно 1 кошка или собака.")
            return True
        else:
            logger.warning(f"Проверка изображения не пройдена: найдено {animal_count} животных (требуется ровно 1). Объекты: {detected_object_names}")
            return False
    except Exception as e:
        logger.error(f"Ошибка при проверке изображения в check_image_for_animals: {e}", exc_info=True)
        return False


def check_text_for_toxicity(text: str) -> bool:
    if not text or not text.strip():
        logger.info("Empty text, considered non-toxic")
        return False
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.warning("GEMINI_API_KEY not found")
        return False
    proxy_url = os.getenv('HTTPS_PROXY')
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    # Step 1: Check for threats to life (targeting cats or dogs)
    threat_pattern = r'\b(убью|убить|убийство|зарежу|прикончу|атакую|убей|убивай)\b.*\b(кошка|кот|собака|пёс|пес)\b'
    if re.search(threat_pattern, text, flags=re.IGNORECASE):
        logger.info(f"Text contains threats to life targeting a cat or dog, marked as toxic: {text[:50]}...")
        return True
    # Step 2: Check for threats of sexual violence (targeting cats or dogs)
    sexual_threat_pattern = r'\b(изнасилую|трахну|трах|секс|насилую|похоть|изнасилование)\b.*\b(кошка|кот|собака|пёс|пес)\b'
    if re.search(sexual_threat_pattern, text, flags=re.IGNORECASE):
        logger.info(f"Text contains threats of sexual violence targeting a cat or dog, marked as toxic: {text[:50]}...")
        return True
    # Step 3: Check for Russian swearing
    swearing_pattern = r'\b(хуй|пиздец|блять|ебать|сука|пизда|долбоёб|еблан|нахуй|блядь|хуесос|пидор|гандон)\b'
    if re.search(swearing_pattern, text, flags=re.IGNORECASE):
        logger.info(f"Text contains Russian swearing, marked as toxic: {text[:50]}...")
        return True
    # Step 4: Use Gemini API for deeper context analysis
    prompt = (
        f"Is the following Russian text offensive, toxic, or inappropriate when describing a cat or dog? "
        f"Evaluate it in the context of pet-related content (e.g., adoption announcements). "
        f"Check for: "
        f"1. Threats to life (e.g., 'убью', 'убить' specifically targeting a cat or dog), "
        f"2. Threats of sexual violence (e.g., 'изнасилую', 'трахну' targeting a cat or dog), "
        f"3. Russian swearing (e.g., 'пиздец', 'блять'), "
        f"4. Insults, hate speech, or sexual content. "
        f"Consider phrases like 'убью собаку' or 'трахну кота' as toxic, but 'убью муху' as non-toxic unless context suggests otherwise. "
        f"Text: '{text}'\n"
        f"Answer ONLY with 'toxic' or 'not toxic'. Be strict and assume ambiguity as 'toxic' if it could relate to a cat or dog."
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        logger.info(f"Toxicity check for '{text[:50]}...'")
        response = requests.post(url, json=payload, headers=headers, proxies=proxies, timeout=10)
        response.raise_for_status()
        content = response.json()['candidates'][0]['content']['parts'][0]['text'].strip().lower()
        logger.info(f"Toxicity result: '{content}'")
        return content == "toxic"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            logger.warning(f"HTTP 400 error, assuming toxic: {e.response.text}")
            return True
        logger.error(f"HTTP error: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"General error: {e}", exc_info=True)
        return False

async def get_recommendations_gemini(subject: str) -> str:
    if not subject:
        logger.info("No subject provided, returning None")
        return None
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.warning("GEMINI_API_KEY not found")
        return None
    proxy_url = os.getenv('HTTPS_PROXY')
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    prompt = (
        f"You are a veterinary assistant. Provide 2-3 short recommendations for care and feeding of '{subject}'. "
        f"Answer only with a numbered or bulleted list in Russian, no introductions or conclusions."
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        logger.info(f"Recommendations request for '{subject}'")
        response = requests.post(url, json=payload, headers=headers, proxies=proxies, timeout=10)
        response.raise_for_status()
        text = response.json()['candidates'][0]['content']['parts'][0]['text']
        logger.info(f"Recommendations received: {text}")
        return text
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}", exc_info=True)
        return None


async def get_recommendations_mistral(subject: str) -> Optional[str]:
    """Получает рекомендации по уходу от Mistral API (asynchronous)."""
    if not subject: return None
    if not MISTRAL_TOKEN:
        logger.warning("Пропускаем запрос рекомендаций, MISTRAL_TOKEN не указан.")
        return None

    mistral_url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_TOKEN}",
        "Content-Type": "application/json"
    }
    prompt = (
        f"Ты - ассистент-ветеринар. Дай 2-3 основные краткие рекомендации по уходу и кормлению для '{subject}'. "
        f"Ответ дай только нумерованным или маркированным списком рекомендаций на русском языке, без вступлений и заключений."
    )
    payload = {
        "model": "mistral-large-latest", # Или mistral-small-latest
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 200
    }

    logger.info(f"Запрос рекомендаций к Mistral для: '{subject}'")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(mistral_url, headers=headers, json=payload, timeout=30) # Увеличим таймаут
        response.raise_for_status()
        result = response.json()
        logger.info(f"Ответ от Mistral (рекомендации): {result}")

        recommendations_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if recommendations_text and "error" not in recommendations_text.lower() and "не могу" not in recommendations_text.lower():
             logger.info(f"Получены рекомендации для '{subject}'.")
             return recommendations_text
        else:
             logger.warning(f"Mistral вернул пустой или некорректный ответ для рекомендаций по '{subject}'.")
             return None

    except httpx.RequestError as e:
         logger.error(f"Сетевая ошибка при запросе рекомендаций к Mistral для '{subject}': {e}")
         return None
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций от Mistral для '{subject}': {e}", exc_info=True)
        return None

# --- Form Data Parsers (Dependencies) ---

def pet_form(
    animal_type: str = Form(...),
    name: Optional[str] = Form(None),
    gender: str = Form(...),
    age: Optional[int] = Form(None),
    breed: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    # Добавим остальные поля для полноты, хотя они могут быть не в PetCreate
    isNeuteredStr: Optional[str] = Form(None, alias='isNeutered'),
    isVaccinatedStr: Optional[str] = Form(None, alias='isVaccinated'),
    size: Optional[str] = Form(None),
) -> dict:
    """Собирает данные о питомце из формы в словарь."""
    # Собираем все поля, которые есть в PetModel
    return {
        "animal_type": animal_type, "name": name, "gender": gender, "age": age,
        "breed": breed, "color": color, "isNeuteredStr": isNeuteredStr,
        "isVaccinatedStr": isVaccinatedStr, "size": size
    }

def announcement_form(
    keywords: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
) -> dict:
    """Собирает данные об объявлении из формы в словарь."""
    return {"keywords": keywords, "description": description, "city": city}

def search_form(
    animal_type: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    breed: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None, description="Ключевые слова через запятую")
) -> SearchRequest:
    """Собирает параметры поиска из формы в Pydantic модель SearchRequest."""
    keywords_list = [kw.strip() for kw in keywords.split(",") if kw and kw.strip()] if keywords else None
    return SearchRequest(
        animal_type=animal_type, gender=gender, age=age, breed=breed,
        color=color, keywords=keywords_list
    )

# --- API Endpoints ---

@router.post("/create_announcement", response_model=AnnouncementResponse)
async def create_announcement(
    pet_data: dict = Depends(pet_form),
    announcement_data: dict = Depends(announcement_form),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id_str: str = Depends(get_current_user_id)
):
    """
    Создает новое объявление после сохранения изображения и прохождения модерации.
    """
    logger.info(f"Запрос на создание объявления от пользователя ID: {user_id_str}")

    # --- User Check ---
    try:
        user_id_int = int(user_id_str)
        user = db.query(UserModel).filter(UserModel.id == user_id_int).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User not found.")
    except ValueError:
         raise HTTPException(status_code=400, detail="Invalid user ID format.")
    except Exception as e:
        logger.error(f"Ошибка БД при поиске пользователя {user_id_str}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error checking user.")

    # --- Boolean Conversion ---
    isNeutered_bool = None
    if pet_data['isNeuteredStr'] == 'Да': isNeutered_bool = True
    elif pet_data['isNeuteredStr'] == 'Нет': isNeutered_bool = False

    isVaccinated_bool = None
    if pet_data['isVaccinatedStr'] == 'Да': isVaccinated_bool = True
    elif pet_data['isVaccinatedStr'] == 'Нет': isVaccinated_bool = False

    # --- Image Handling ---
    file_path_os_absolute = ""
    file_path_url = ""
    image_content_for_check = None

    try:
        script_dir = os.path.dirname(__file__)
        images_dir_absolute = os.path.abspath(os.path.join(script_dir, "../images"))
        os.makedirs(images_dir_absolute, exist_ok=True)

        timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in (image.filename or "image") if c.isalnum() or c in ['.', '_', '-']).strip() or "image"
        unique_filename = f"{timestamp_now}_{user_id_int}_{safe_filename}"

        file_path_os_absolute = os.path.join(images_dir_absolute, unique_filename)
        file_path_url = f"images/{unique_filename}" # Используем / для URL пути

        image_content_for_check = await image.read()
        with open(file_path_os_absolute, "wb") as buffer:
            buffer.write(image_content_for_check)
        logger.info(f"Изображение сохранено: {file_path_os_absolute}")

    except Exception as e:
         logger.error(f"Ошибка сохранения файла изображения: {e}", exc_info=True)
         # Не удаляем файл здесь, так как он мог не создаться
         raise HTTPException(status_code=500, detail="Could not save image file.")
    finally:
        # Закрывать файл не нужно после await image.read()
        pass

    # --- Moderation ---
    moderation_passed = True
    moderation_error_detail = ""

    # 1. Image Moderation
    if image_content_for_check:
        logger.info("Проверка изображения (check_image_for_animals)...")
        if not check_image_for_animals(image_content_for_check):
            logger.warning(f"Модерация изображения не пройдена для {image.filename}.")
            moderation_passed = False
            moderation_error_detail = "Модерация изображения не пройдена: на фото должна быть одна кошка или одна собака."
        else:
            logger.info("Проверка изображения пройдена.")
    else:
        # Это странная ситуация, файл должен был быть прочитан
        logger.error("Не удалось получить контент изображения для модерации.")
        moderation_passed = False
        moderation_error_detail = "Не удалось прочитать содержимое изображения для модерации."

    # 2. Text Moderation (only if image moderation passed)
    if moderation_passed:
        logger.info("Проверка текста объявления на токсичность...")
        # Собираем весь текст из форм
        full_text_to_check = (
            f"{pet_data.get('name') or ''} {pet_data.get('breed') or ''} "
            f"{announcement_data.get('description') or ''} {announcement_data.get('keywords') or ''} "
            f"{announcement_data.get('city') or ''}"
        ).strip()

        if full_text_to_check:
            if check_text_for_toxicity(full_text_to_check):
                logger.warning(f"Модерация текста не пройдена для объявления пользователя {user_id_int}.")
                moderation_passed = False
                moderation_error_detail = "Модерация текста не пройдена: обнаружено недопустимое содержимое."
            else:
                logger.info("Проверка текста на токсичность пройдена.")
        else:
            logger.info("Нет текста для проверки на токсичность.")

    # --- Action based on Moderation ---
    if not moderation_passed:
        # Удаляем сохраненный файл, если модерация не пройдена
        if file_path_os_absolute and os.path.exists(file_path_os_absolute):
            try:
                os.remove(file_path_os_absolute)
                logger.info(f"Файл удален из-за ошибки модерации: {file_path_os_absolute}")
            except Exception as remove_exc:
                 logger.error(f"Ошибка при удалении файла {file_path_os_absolute} после ошибки модерации: {remove_exc}")
        raise HTTPException(status_code=400, detail=moderation_error_detail)

    # --- Database Operations (only if moderation passed) ---
    try:
         new_pet = PetModel(
             animal_type=pet_data['animal_type'],
             name=pet_data['name'],
             gender=pet_data['gender'],
             age=pet_data['age'],
             breed=pet_data['breed'],
             color=pet_data['color'],
             isNeutered=isNeutered_bool,
             isVaccinated=isVaccinated_bool,
             size=pet_data['size']
         )
         db.add(new_pet)
         db.flush() # Get pet ID

         new_announcement = AnnouncementModel(
             user_id=user_id_int,
             pet_id=new_pet.id,
             keywords=announcement_data['keywords'],
             description=announcement_data['description'],
             status="опубликовано",
             timestamp=datetime.now(), # Use datetime object
             image_path=file_path_url, # Store URL path
             city=announcement_data['city']
         )
         db.add(new_announcement)
         db.commit()
         db.refresh(new_pet)
         db.refresh(new_announcement)
         # Eager load for response model
         db.refresh(user)
         new_announcement.user = user
         new_announcement.pet = new_pet

         logger.info(f"Объявление ID {new_announcement.id} для питомца ID {new_pet.id} успешно создано. Image path (URL): {new_announcement.image_path}")
         return new_announcement

    except Exception as db_exc:
          db.rollback()
          logger.error(f"Ошибка базы данных при создании объявления: {db_exc}", exc_info=True)
          # Cleanup saved file if DB fails
          if file_path_os_absolute and os.path.exists(file_path_os_absolute):
              try:
                  os.remove(file_path_os_absolute)
                  logger.info(f"Удален файл изображения {file_path_os_absolute} из-за ошибки БД.")
              except Exception as remove_exc:
                  logger.error(f"Ошибка при удалении файла изображения {file_path_os_absolute} после ошибки БД: {remove_exc}")
          raise HTTPException(status_code=500, detail="Database error creating announcement.")


@router.post("/search_announcements")
async def search_announcements(
    search_params: SearchRequest = Depends(search_form),
    image: Optional[UploadFile] = File(None), # Reference image
    user_id_str: str = Depends(get_current_user_id), # For logging/auth
    db: Session = Depends(get_db)
):
    """
    Ищет объявления по параметрам и/или референсному изображению.
    Автозаполняет параметры поиска, если они не указаны и изображение предоставлено.
    """
    logger.info(f"Поиск объявлений для пользователя ID {user_id_str} с параметрами: {search_params}")
    if image: logger.info(f"Загружено референсное изображение: {image.filename}")

    # --- User Check (optional, for context/logging) ---
    try:
        user_id_int = int(user_id_str)
        user = db.query(UserModel).filter(UserModel.id == user_id_int).first()
        if not user: logger.warning(f"Пользователь {user_id_int} не найден, но поиск продолжается.")
    except ValueError:
        logger.warning(f"Некорректный user_id: {user_id_str}")
    except Exception as e:
        logger.error(f"Ошибка БД при проверке пользователя {user_id_str} для поиска: {e}")
        # Continue search even if user check fails

    suggested_animal_type = None
    suggested_breed_display = None

    # --- Reference Image Processing ---
    if image and image.filename:
        image_data = None
        try:
            image_data = await image.read()
            logger.info(f"Определение типа/породы по референсному изображению: {image.filename}")
            detection_result = detect_breed(image_data=image_data)

            if detection_result and detection_result.get("type"):
                raw_type = detection_result["type"] # Dog/Cat
                raw_breed_key = detection_result.get("breed") # english_cocker_spaniel/None

                suggested_animal_type = BREEDS_MAP.get("types", {}).get(raw_type, raw_type) # Собака/Кошка

                if raw_breed_key:
                    type_map_key = raw_type.lower() # dog/cat
                    suggested_breed_display = BREEDS_MAP.get(type_map_key, {}).get(raw_breed_key) # Русский перевод
                    if not suggested_breed_display:
                        suggested_breed_display = raw_breed_key.replace("_", " ").capitalize() # Форматируем ключ
                    logger.info(f"Определено по изображению: Тип={suggested_animal_type}, Порода={suggested_breed_display}")
                else:
                    suggested_breed_display = "Беспородный(ая)"
                    logger.info(f"Определен по изображению: Тип={suggested_animal_type}, Порода не определена.")

                # Auto-fill search parameters if not provided by user
                if suggested_animal_type and not search_params.animal_type:
                    search_params.animal_type = suggested_animal_type
                    logger.info(f"Автозаполнение типа: {suggested_animal_type}")
                if suggested_breed_display and suggested_breed_display != "Беспородный(ая)" and not search_params.breed:
                     search_params.breed = suggested_breed_display # Используем русское имя
                     logger.info(f"Автозаполнение породы: {suggested_breed_display}")
            else:
                logger.warning(f"Не удалось определить тип/породу по референсному изображению {image.filename}.")
        except Exception as e:
            logger.error(f"Ошибка при обработке референсного изображения {image.filename}: {e}", exc_info=True)
            # Continue search without suggestions
        finally:
             pass

    # --- Database Search ---
    try:
        query = db.query(AnnouncementModel).join(PetModel).filter(AnnouncementModel.status == "опубликовано")

        # Apply filters from (potentially auto-filled) search_params
        if search_params.animal_type: query = query.filter(PetModel.animal_type.ilike(f"%{search_params.animal_type}%"))
        if search_params.gender: query = query.filter(PetModel.gender.ilike(search_params.gender))
        if search_params.age is not None: query = query.filter(PetModel.age == search_params.age)
        if search_params.breed: query = query.filter(PetModel.breed.ilike(f"%{search_params.breed}%"))
        if search_params.color: query = query.filter(PetModel.color.ilike(f"%{search_params.color}%"))
        if search_params.keywords:
            for keyword in search_params.keywords:
                 kw_ilike = f"%{keyword}%"
                 query = query.filter((AnnouncementModel.keywords.ilike(kw_ilike)) | (AnnouncementModel.description.ilike(kw_ilike)))

        # Execute query
        announcements = query.order_by(AnnouncementModel.timestamp.desc()).limit(50).all() # Limit results
        logger.info(f"Найдено {len(announcements)} объявлений после фильтрации.")

        result_list = [AnnouncementResponse.from_orm(ann) for ann in announcements]

        return {
            "announcements": result_list,
            "suggested_animal_type": suggested_animal_type,
            "suggested_breed": suggested_breed_display
        }

    except Exception as e:
        logger.error(f"Ошибка при выполнении поиска объявлений в БД: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error searching announcements.")


@router.post("/identify_breed", response_model=BreedIdentificationResult)
async def identify_breed_endpoint(
    images: List[UploadFile] = File(..., description="Один или несколько файлов изображений")
):
    """
    Определяет тип и породу по первому изображению.
    Возвращает результат и рекомендации по уходу от Mistral.
    """
    if not images:
        raise HTTPException(status_code=400, detail="No image files provided.")

    logger.info(f"Запрос на определение породы. Файлов: {len(images)}. Обрабатывается первый.")
    image = images[0] # Process only the first image
    image_data = None

    try:
        image_data = await image.read()
        logger.info(f"Определение породы для файла: {image.filename}")

        # 1. Detect Breed using Vision API
        detection_result = detect_breed(image_data=image_data)

        if detection_result is None or detection_result.get("type") is None:
             logger.error(f"Не удалось определить тип животного для {image.filename}. Result: {detection_result}")
             raise HTTPException(status_code=422, detail="Could not determine animal type from the image.")

        raw_type = detection_result["type"] # Dog/Cat
        raw_breed_key = detection_result.get("breed") # english_cocker_spaniel/None

        # 2. Translate results using BREEDS_MAP
        translated_type = BREEDS_MAP.get("types", {}).get(raw_type, raw_type) # Собака/Кошка
        recommendation_subject = translated_type # Default subject for recommendations
        translated_breed = "Беспородный(ая)" # Default

        if raw_breed_key:
            type_key_for_map = raw_type.lower() # dog/cat
            breed_translation = BREEDS_MAP.get(type_key_for_map, {}).get(raw_breed_key)
            if breed_translation:
                translated_breed = breed_translation
                recommendation_subject = translated_breed # Use specific breed for recommendations
                logger.info(f"Порода найдена в словаре: {raw_breed_key} -> {translated_breed}")
            else:
                # Format key if not in map
                translated_breed = raw_breed_key.replace("_", " ").capitalize()
                recommendation_subject = translated_breed
                logger.warning(f"Порода '{raw_breed_key}' не найдена в BREEDS_MAP. Используется: '{translated_breed}'")
        else:
            logger.info(f"Порода не определена Vision API. Тип: {translated_type}")

        # 3. Get Recommendations from Mistral
        recommendations = None
        if recommendation_subject:
            recommendations = await get_recommendations_gemini(recommendation_subject)

        # 4. Format and return response
        response_data = BreedIdentificationResult(
            animal_type=translated_type,
            breed=translated_breed,
            recommendations=recommendations
        )
        logger.info(f"Финальный результат для {image.filename}: {response_data}")
        return response_data

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обработке /identify_breed для {image.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing image.")
    finally:
        pass


@router.post("/identify_breed_multi", response_model=BreedIdentificationResult)
async def identify_breed_multi_endpoint(
    # Ожидаем три файла
    image_front: UploadFile = File(..., description="Изображение: Вид спереди"),
    image_side: UploadFile = File(..., description="Изображение: Вид сбоку"),
    image_top: UploadFile = File(..., description="Изображение: Вид сверху")
):
    """
    Определяет тип и породу по трем изображениям с разных ракурсов.
    Требует аутентификации.
    Возвращает результат и рекомендации по уходу от Mistral.
    """
    logger.info(f"Запрос на определение породы (мульти-фото) от пользователя. Файлы: front='{image_front.filename}', side='{image_side.filename}', top='{image_top.filename}'")

    files = {"front": image_front, "side": image_side, "top": image_top}
    image_data = {}
    results = {}

    try:
        # --- 1. Чтение контента всех файлов ---
        async def read_file(angle, file):
            try:
                content = await file.read()
                logger.info(f"Прочитан файл для ракурса '{angle}': {file.filename}")
                return angle, content
            except Exception as e:
                logger.error(f"Ошибка чтения файла для ракурса '{angle}': {e}")
                return angle, None # Возвращаем None в случае ошибки чтения

        read_tasks = [read_file(angle, file) for angle, file in files.items()]
        read_results = await asyncio.gather(*read_tasks)

        for angle, content in read_results:
            if content:
                image_data[angle] = content
            else:
                # Если хотя бы один файл не прочитался, прерываем
                raise HTTPException(status_code=400, detail=f"Не удалось прочитать файл для ракурса '{angle}'")

        # Проверяем, прочитались ли все файлы (на всякий случай)
        if len(image_data) != 3:
             raise HTTPException(status_code=400, detail="Не удалось прочитать все три файла изображений.")

        # --- 2. Последовательный анализ всех трех изображений ---
        logger.info("Начало последовательного анализа трех изображений...")
        analysis_results_list = []
        # Сохраняем порядок ключей, чтобы потом правильно сопоставить результаты
        angles_order = list(image_data.keys())  # ['front', 'side', 'top'] или в каком порядке они пришли
        for angle in angles_order:
            img_bytes = image_data[angle]
            logger.info(f"Анализ ракурса: {angle}")
            # Обычный, синхронный вызов функции detect_breed
            result = detect_breed(image_data=img_bytes)
            analysis_results_list.append(result)

        logger.info("Последовательный анализ трех изображений завершен.")

        # Сопоставляем результаты с ракурсами, используя сохраненный порядок
        results = dict(zip(angles_order, analysis_results_list))
        logger.info(f"Результаты анализа по ракурсам: {results}")

        # --- 3. Комбинирование результатов ---
        # Этот блок остается без изменений, он работает с results
        detected_types = [res.get("type") for res in results.values() if res and res.get("type")]
        detected_breed_keys = [res.get("breed") for res in results.values() if res and res.get("breed")]

        # 3.1 Определение итогового типа (голосование)
        final_type_raw = None
        if detected_types:
            type_counts = Counter(detected_types)
            most_common_type, type_count = type_counts.most_common(1)[0]
            if type_count >= 2: # Если хотя бы 2 из 3 сошлись
                final_type_raw = most_common_type # "Dog" или "Cat"
                logger.info(f"Итоговый тип определен как '{final_type_raw}' (голосов: {type_count})")
            elif len(detected_types) == 1: # Если только одно изображение дало тип
                final_type_raw = detected_types[0]
                logger.info(f"Итоговый тип определен по единственному результату: '{final_type_raw}'")
            else: # Неоднозначно (например, Dog, Cat, None или три разных типа)
                logger.warning(f"Не удалось однозначно определить тип (результаты: {detected_types}).")
                # Можно выбрать тип с наибольшей уверенностью, если бы detect_breed возвращал score
        else:
            logger.error("Ни одно изображение не позволило определить тип животного.")
            raise HTTPException(status_code=422, detail="Не удалось определить тип животного ни по одному из изображений.")

        # 3.2 Определение итоговой породы (проверка на единство)
        final_breed_key = None
        if detected_breed_keys:
            unique_breeds = set(detected_breed_keys)
            if len(unique_breeds) == 1: # Все сошлись на одной породе
                final_breed_key = detected_breed_keys[0]
                logger.info(f"Итоговая порода определена как '{final_breed_key}' (все ракурсы сошлись).")
            else:
                # Породы разные или только одна определилась - неоднозначно
                logger.warning(f"Обнаружены разные породы ({unique_breeds}) или порода определена только по одному ракурсу. Результат неоднозначен.")
                # Можно выбрать самую частую, или по front view, или вернуть None
                # Пока возвращаем None (т.е. будет "Беспородный(ая)")
                final_breed_key = None
        else:
            logger.info("Ни на одном изображении не была определена конкретная порода.")
            final_breed_key = None # Явно

        # --- 4. Перевод и получение рекомендаций ---
        translated_type = BREEDS_MAP.get("types", {}).get(final_type_raw, final_type_raw)
        recommendation_subject = translated_type # По умолчанию для рекомендаций
        translated_breed = "Беспородный(ая)"

        if final_breed_key:
            type_key_for_map = final_type_raw.lower()
            breed_translation = BREEDS_MAP.get(type_key_for_map, {}).get(final_breed_key)
            if breed_translation:
                translated_breed = breed_translation
                recommendation_subject = translated_breed
                logger.info(f"Итоговая порода найдена в словаре: {final_breed_key} -> {translated_breed}")
            else:
                translated_breed = final_breed_key.replace("_", " ").capitalize()
                recommendation_subject = translated_breed
                logger.warning(f"Итоговая порода '{final_breed_key}' не найдена в BREEDS_MAP. Используется: '{translated_breed}'")
        else:
             logger.info(f"Итоговая порода не определена или неоднозначна. Тип: {translated_type}")

        recommendations = None
        if recommendation_subject:
             recommendations = await get_recommendations_gemini(recommendation_subject)

        # --- 5. Формирование ответа ---
        response_data = BreedIdentificationResult(
            animal_type=translated_type,
            breed=translated_breed,
            recommendations=recommendations
        )
        logger.info(f"Финальный результат (мульти-фото): {response_data}")
        return response_data

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обработке /identify_breed_multi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing multi-angle image.")
    finally:
        # Закрываем все файлы
        for file_to_close in files.values():
            if file_to_close:
                try:
                    await file_to_close.close()
                except Exception as close_exc:
                    logger.warning(f"Не критичная ошибка при закрытии файла {file_to_close.filename}: {close_exc}")

