# announcements.py

import logging
import requests
from fastapi import APIRouter, Depends, Query, HTTPException, File, UploadFile, Form, status
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc, asc
from typing import List, Optional
import os
import shutil
from datetime import datetime # <-- Убедимся, что datetime импортирован
from database import get_db
# Импортируем модели с псевдонимами
from models import Announcement as AnnouncementModel, Pet as PetModel, User as UserModel
# Импортируем схему ответа
from schemas import AnnouncementResponse
# Импорт для поиска пары (если используется) - убедитесь, что он есть, если эндпоинт find_mate активен
# from schemas import PetMateInput # Похоже, не используется напрямую в параметрах find_mate
from auth import get_current_user # Используем реальную аутентификацию
# Убедимся, что UserModel импортирован (дублирование не страшно)
from models import User as UserModel

# --- Начало определения роутера ---
router = APIRouter(
    prefix="/announcements",
    tags=["announcements"]
)
# Настройка логгера
logger = logging.getLogger(__name__)

# --- Вспомогательная функция для Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

def send_telegram_notification(chat_id: int, message: str):
    """Отправляет текстовое сообщение пользователю через Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не настроен. Уведомление не отправлено.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("ok"):
            logger.info(f"Уведомление успешно отправлено пользователю {chat_id}")
            return True
        else:
            logger.error(f"Ошибка отправки уведомления пользователю {chat_id}: {result.get('description')}")
            return False
    except requests.exceptions.Timeout:
        logger.error(f"Таймаут при отправке уведомления пользователю {chat_id}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Сетевая ошибка при отправке уведомления пользователю {chat_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Неизвестная ошибка при отправке уведомления пользователю {chat_id}: {e}")
        return False

# --- Эндпоинт для получения списка объявлений ---
@router.get("", response_model=List[AnnouncementResponse])
def get_announcements(
    animal_type: Optional[str] = Query(None, description="Тип животного (Кошка или Собака)"),
    gender: Optional[str] = Query(None, description="Пол (М или Ж)"),
    age: Optional[int] = Query(None, description="Точный возраст (полных лет)"),
    breed: Optional[str] = Query(None, description="Порода (можно часть названия)"),
    color: Optional[str] = Query(None, description="Окрас (можно часть названия)"),
    city: Optional[str] = Query(None, description="Город (можно часть названия)"),
    isNeutered: Optional[bool] = Query(None, description="Статус стерилизации/кастрации (true/false)"),
    isVaccinated: Optional[bool] = Query(None, description="Статус прививок (true/false)"),
    keywords: Optional[List[str]] = Query(None, description="Список ключевых слов (характеристик)"),
    size: Optional[str] = Query(None, description="Размер питомца (маленький, средний, большой)"),
    sort_by: Optional[str] = Query("timestamp_desc", description="Сортировка: timestamp_desc (сначала новые), timestamp_asc (сначала старые)"),
    skip: int = Query(0, ge=0, description="Сколько записей пропустить"),
    limit: int = Query(20, ge=1, le=100, description="Максимальное количество записей для возврата (макс 100)"),
    db: Session = Depends(get_db)
):
    """
    Получает список опубликованных объявлений с возможностью фильтрации,
    сортировки и пагинации.
    """
    logger.info(f"Запрос списка объявлений с параметрами: animal_type={animal_type}, gender={gender}, age={age}, city={city}, isNeutered={isNeutered}, isVaccinated={isVaccinated}, keywords={keywords}, size={size}, sort_by={sort_by}, skip={skip}, limit={limit}")
    try:
        query = db.query(AnnouncementModel).options(
            joinedload(AnnouncementModel.user),
            joinedload(AnnouncementModel.pet)
        ).filter(AnnouncementModel.status == "опубликовано")

        joined_pet = False
        pet_filters = [animal_type, gender, age is not None, breed, color, isNeutered is not None, isVaccinated is not None, size is not None]
        if any(pet_filters):
            if not joined_pet:
                query = query.join(AnnouncementModel.pet)
                joined_pet = True

            if animal_type: query = query.filter(PetModel.animal_type == animal_type)
            if gender: query = query.filter(PetModel.gender == gender)
            if age is not None: query = query.filter(PetModel.age == age)
            if breed: query = query.filter(PetModel.breed.ilike(f"%{breed}%"))
            if color: query = query.filter(PetModel.color.ilike(f"%{color}%"))
            if isNeutered is not None: query = query.filter(PetModel.isNeutered == isNeutered)
            if isVaccinated is not None: query = query.filter(PetModel.isVaccinated == isVaccinated)
            if size: query = query.filter(PetModel.size == size)

        if city:
            query = query.filter(AnnouncementModel.city.ilike(f"%{city}%"))

        if keywords:
            for kw in keywords:
                clean_kw = kw.strip()
                if clean_kw:
                     query = query.filter(AnnouncementModel.keywords.ilike(f"%{clean_kw}%"))

        if sort_by == "timestamp_asc":
            query = query.order_by(asc(AnnouncementModel.timestamp))
        else:
            query = query.order_by(desc(AnnouncementModel.timestamp))

        results = query.offset(skip).limit(limit).all()
        logger.info(f"Найдено {len(results)} объявлений.")
        return results

    except Exception as e:
        logger.exception(f"Ошибка при получении объявлений: {e}")
        raise HTTPException(status_code=500, detail="Не удалось получить список объявлений")

# --- Эндпоинт Запроса на контакт ---
@router.post("/{announcement_id}/request_contact", status_code=status.HTTP_200_OK)
async def request_contact(
    announcement_id: int,
    db: Session = Depends(get_db),
    requester: UserModel = Depends(get_current_user)
):
    """ Отправляет владельцу объявления уведомление в Telegram о том, что им интересуются. """
    # Проверка наличия username у запрашивающего (остается)
    if not requester.username:
         raise HTTPException(status_code=400, detail=f"У вашего пользователя (ID: {requester.id}) не указан Telegram username, владелец не сможет с вами связаться.")

    # Получаем объявление вместе с владельцем и питомцем (остается)
    announcement = db.query(AnnouncementModel).options(
        joinedload(AnnouncementModel.user),
        joinedload(AnnouncementModel.pet)
    ).filter(AnnouncementModel.id == announcement_id).first()

    # Проверка, найдено ли объявление (остается)
    if not announcement:
        raise HTTPException(status_code=404, detail="Объявление не найдено")

    owner = announcement.user
    pet = announcement.pet

    # Проверка наличия владельца и питомца (остается)
    if not owner or not pet:
        logger.error(f"Не найден владелец или питомец для объявления {announcement_id}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка: не найден владелец или питомец")

    # Проверка, что пользователь не запрашивает контакт по своему объявлению (остается)
    if owner.id == requester.id:
         raise HTTPException(status_code=400, detail="Вы не можете запросить контакт по своему объявлению.")

    # --- НАЧАЛО УДАЛЕНИЯ/КОММЕНТИРОВАНИЯ ---
    # Эта проверка больше не нужна, так как согласие подразумевается фактом существования объявления
    # if not owner.allow_contact_requests:
    #      raise HTTPException(status_code=403, detail="Владелец не разрешил запросы на связь.")
    # --- КОНЕЦ УДАЛЕНИЯ/КОММЕНТИРОВАНИЯ ---

    # Проверка, есть ли у владельца Telegram ID для отправки уведомления (остается)
    if not owner.telegram_id:
        raise HTTPException(status_code=400, detail="Невозможно отправить запрос владельцу (отсутствует Telegram ID).")

    # Формирование сообщения (остается)
    message_text = (
        f"Здравствуйте! Пользователь @{requester.username} интересуется вашим питомцем "
        f"'{pet.name or 'Без имени'}' ({pet.breed or 'порода не указана'}).\n\n"
        f"Вы можете написать ему первым, если заинтересованы."
    )

    # Отправка уведомления (остается)
    # Убедитесь, что функция send_telegram_notification существует и работает
    success = send_telegram_notification(owner.telegram_id, message_text)

    # Возврат результата (остается)
    if success:
        return {"message": "Запрос успешно отправлен владельцу."}
    else:
        # Здесь может быть ошибка самой функции отправки в Telegram
        raise HTTPException(status_code=500, detail="Не удалось отправить уведомление владельцу.")

# --- Эндпоинт для получения одного объявления ---
@router.get("/{announcement_id}", response_model=AnnouncementResponse)
def get_single_announcement(
    announcement_id: int,
    db: Session = Depends(get_db)
):
    """ Получает данные одного объявления по его ID. """
    logger.info(f"Запрос данных для объявления ID: {announcement_id}")
    try:
        announcement = db.query(AnnouncementModel).options(
            joinedload(AnnouncementModel.user),
            joinedload(AnnouncementModel.pet)
        ).filter(
            AnnouncementModel.id == announcement_id
        ).first()

        if not announcement:
            logger.warning(f"Объявление с ID {announcement_id} не найдено")
            raise HTTPException(status_code=404, detail="Объявление не найдено")

        # Дополнительные проверки на случай ошибок целостности данных
        if not announcement.user:
             logger.error(f"Не найден связанный пользователь для объявления {announcement_id}")
        if not announcement.pet:
             logger.error(f"Не найден связанный питомец для объявления {announcement_id}")

        pet_name = announcement.pet.name if announcement.pet else 'Без имени'
        logger.info(f"Найдено объявление ID {announcement_id}: {pet_name}")
        return announcement

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.exception(f"Ошибка при получении объявления ID {announcement_id}: {e}")
        raise HTTPException(status_code=500, detail="Не удалось получить данные объявления")

# --- Эндпоинт для обновления объявления (с исправленной логикой фото) ---
@router.put("/{announcement_id}", response_model=AnnouncementResponse)
async def update_announcement(
    announcement_id: int,
    # --- Параметры Form() ---
    animal_type: str = Form(...),
    name: Optional[str] = Form(None),
    gender: str = Form(...),
    age: Optional[int] = Form(None),
    breed: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    isNeuteredStr: Optional[str] = Form(None, alias='isNeutered'), # Принимаем как строку
    isVaccinatedStr: Optional[str] = Form(None, alias='isVaccinated'), # Принимаем как строку
    size: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None), # Необязательное новое изображение
    # --- Зависимости ---
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user) # Используем реального пользователя
):
    ''' Обновляет существующее объявление. '''
    logger.info(f"Попытка обновить объявление ID: {announcement_id} пользователем ID: {current_user.id}")

    try:
        # Находим существующее объявление и связанного питомца
        announcement = db.query(AnnouncementModel).options(
            joinedload(AnnouncementModel.pet) # Сразу грузим питомца
        ).filter(
            AnnouncementModel.id == announcement_id
        ).first()

        if not announcement:
            logger.warning(f"Объявление {announcement_id} для обновления не найдено.")
            raise HTTPException(status_code=404, detail="Объявление для обновления не найдено")

        # --- Проверка Авторизации (владелец ли?) ---
        if announcement.user_id != current_user.id:
             logger.warning(f"Попытка пользователя {current_user.id} обновить чужое объявление {announcement_id}")
             raise HTTPException(status_code=403, detail="Нет прав на редактирование этого объявления")

        pet_to_update = announcement.pet
        if not pet_to_update:
             logger.error(f"Связанный питомец для объявления {announcement_id} не найден!")
             raise HTTPException(status_code=500, detail="Внутренняя ошибка: Связанный питомец не найден")

        # --- ИСПРАВЛЕННЫЙ БЛОК ОБРАБОТКИ НОВОГО ИЗОБРАЖЕНИЯ ---
        old_image_url_path = announcement.image_path # Путь из БД (напр., /images/old.jpg)
        new_image_url_path = old_image_url_path # По умолчанию оставляем старый

        if image and image.filename:
            logger.info(f"Получен новый файл изображения для обновления: {image.filename}")

            # 1. Удаляем старый файл (если он есть)
            if old_image_url_path:
                try:
                    # Определяем АБСОЛЮТНЫЙ путь к ПАПКЕ images
                    script_dir_ann = os.path.dirname(__file__) # Папка, где лежит announcements.py (Backend)
                    # Важно: Путь ../images строится относительно script_dir_ann
                    images_dir_absolute_ann = os.path.abspath(os.path.join(script_dir_ann, "../images")) # Пример: E:\PetPreed\PetPreed\images

                    # Получаем имя старого файла из URL-пути
                    old_filename = os.path.basename(old_image_url_path)
                    # Собираем полный путь к старому файлу
                    full_old_os_path = os.path.join(images_dir_absolute_ann, old_filename)

                    if os.path.exists(full_old_os_path):
                        os.remove(full_old_os_path)
                        logger.info(f"Старый файл изображения удален: {full_old_os_path}")
                    else:
                        logger.warning(f"Старый файл не найден для удаления по пути: {full_old_os_path} (ориг. URL путь: {old_image_url_path})")
                except Exception as e:
                    logger.error(f"Не удалось удалить старый файл {full_old_os_path}: {e}")
                    # Не прерываем процесс, если старый файл не удалился

            # 2. Сохраняем новый файл
            try:
                # Абсолютный путь к папке images уже есть в images_dir_absolute_ann
                os.makedirs(images_dir_absolute_ann, exist_ok=True) # Убедимся, что папка существует

                # Генерируем имя файла
                timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_filename = "".join(c for c in image.filename if c.isalnum() or c in ['.', '_', '-']).strip() or "updated_image"
                # Добавляем ID пользователя для уникальности и отслеживания
                unique_filename = f"{timestamp_now}_{current_user.id}_{safe_filename}"

                # ПОЛНЫЙ путь для сохранения файла в ОС
                new_image_os_path_absolute = os.path.join(images_dir_absolute_ann, unique_filename) # => E:\...\PetPreed\images\new_file.jpg
                # ОТНОСИТЕЛЬНЫЙ URL путь для сохранения в БД
                new_image_url_path = f"images/{unique_filename}"     # => /images/new_file.jpg

                # Читаем и сохраняем по АБСОЛЮТНОМУ пути
                # Используем shutil.copyfileobj для эффективности с большими файлами
                with open(new_image_os_path_absolute, "wb") as buffer:
                     shutil.copyfileobj(image.file, buffer)
                # Или если файлы небольшие:
                # file_content = await image.read()
                # with open(new_image_os_path_absolute, "wb") as buffer:
                #     buffer.write(file_content)

                logger.info(f"Новое изображение сохранено: {new_image_os_path_absolute}")
                # Обновляем путь в объекте объявления на НОВЫЙ URL путь
                announcement.image_path = new_image_url_path

            except Exception as e:
                 logger.error(f"Ошибка сохранения нового изображения при обновлении: {e}")
                 # НЕ МЕНЯЕМ путь, если сохранение не удалось
                 announcement.image_path = old_image_url_path # Оставляем старый путь
                 # Возможно, стоит вернуть ошибку пользователю?
                 # raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла изображения: {e}")
            finally:
                # Обязательно закрываем файл, даже если была ошибка
                await image.close()
        else:
            logger.info("Новое изображение не было загружено при обновлении.")
            announcement.image_path = old_image_url_path # Убедимся, что используется старый путь

        # --- Конвертация boolean ---
        isNeutered_bool = None
        if isNeuteredStr is not None:
             isNeutered_bool = isNeuteredStr.lower() in ['true', 'yes', 'да', '1'] # Более гибкая проверка
        isVaccinated_bool = None
        if isVaccinatedStr is not None:
             isVaccinated_bool = isVaccinatedStr.lower() in ['true', 'yes', 'да', '1'] # Более гибкая проверка
        logger.debug(f"Конвертация boolean: isNeutered='{isNeuteredStr}'->{isNeutered_bool}, isVaccinated='{isVaccinatedStr}'->{isVaccinated_bool}")

        # --- Обновление данных питомца ---
        logger.debug(f"Обновление Pet ID: {pet_to_update.id}. Старые данные: type={pet_to_update.animal_type}, name={pet_to_update.name}, ...")
        pet_to_update.animal_type = animal_type
        pet_to_update.name = name
        pet_to_update.gender = gender
        pet_to_update.age = age
        pet_to_update.breed = breed
        pet_to_update.color = color
        pet_to_update.isNeutered = isNeutered_bool
        pet_to_update.isVaccinated = isVaccinated_bool
        pet_to_update.size = size
        logger.info(f"Данные питомца ID {pet_to_update.id} подготовлены к обновлению.")

        # --- Обновление данных объявления ---
        logger.debug(f"Обновление Ann ID: {announcement.id}. Старые данные: keywords={announcement.keywords}, desc={announcement.description}, city={announcement.city}, image={old_image_url_path}")
        announcement.keywords = keywords
        announcement.description = description
        announcement.city = city
        # announcement.timestamp = datetime.now() # Раскомментируй, если нужно обновлять время при редактировании
        # announcement.image_path УЖЕ ОБНОВЛЕН выше, если было новое фото
        logger.info(f"Данные объявления ID {announcement.id} подготовлены к обновлению. Image path: {announcement.image_path}")

        # --- Логика обновления флага согласия (если нужна) ---
        # consent_given: bool = Form(False) # Добавить в параметры Form() выше
        # if consent_given is not None and consent_given != current_user.allow_contact_requests:
        #     logger.info(f"Обновляем allow_contact_requests на {consent_given} для пользователя {current_user.id}")
        #     current_user.allow_contact_requests = consent_given
        #     db.add(current_user) # Добавляем пользователя в сессию для обновления

        # --- Сохранение изменений в БД ---
        try:
            db.add(announcement) # Добавляем обновленный объект объявления (и питомца через связь)
            # Если обновляли current_user, он тоже будет добавлен
            db.commit()
            db.refresh(announcement)
            # Явно обновим связанные объекты, если они нужны для ответа
            if announcement.pet: db.refresh(announcement.pet)
            # Обновляем current_user из БД на случай, если меняли его флаг
            db.refresh(current_user)

            logger.info(f"Объявление ID {announcement_id} успешно обновлено в БД. Новый image_path (URL): {announcement.image_path}")
            return announcement # Возвращаем обновленное объявление
        except Exception as db_exc:
             db.rollback()
             logger.exception(f"Ошибка БД при обновлении объявления ID {announcement_id}: {db_exc}")
             raise HTTPException(status_code=500, detail="Ошибка базы данных при обновлении объявления.")

    except HTTPException as http_exc:
        # Просто передаем HTTP исключения (404, 403) дальше
        raise http_exc
    except Exception as e:
        # Ловим любые другие непредвиденные ошибки
        logger.exception(f"Непредвиденная ошибка при обновлении объявления ID {announcement_id}: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при обновлении объявления.")

# --- ЭНДПОИНТ: Добавить в избранное ---
@router.post("/{announcement_id}/favorite", status_code=status.HTTP_200_OK)
def add_favorite(
    announcement_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """ Добавляет объявление в список избранного текущего пользователя. """
    # Загружаем пользователя вместе с его текущим избранным (оптимизация)
    user = db.query(UserModel).options(
        joinedload(UserModel.favorite_announcements)
    ).filter(UserModel.id == current_user.id).first()
    # Проверка не нужна, если get_current_user гарантирует наличие

    announcement = db.query(AnnouncementModel).filter(AnnouncementModel.id == announcement_id).first()
    if not announcement:
        raise HTTPException(status_code=404, detail="Объявление не найдено")

    if announcement in user.favorite_announcements:
        logger.info(f"Объявление {announcement_id} уже в избранном у пользователя {user.id}")
        # Можно вернуть 200 или 304 Not Modified, но 200 с сообщением тоже нормально
        return {"detail": "Объявление уже в избранном"}

    user.favorite_announcements.append(announcement)
    try:
        db.commit()
        logger.info(f"Объявление {announcement_id} добавлено в избранное пользователя {user.id}")
        return {"detail": "Объявление добавлено в избранное"}
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка добавления в избранное для user {user.id}, ann {announcement_id}: {e}")
        raise HTTPException(status_code=500, detail="Не удалось добавить в избранное")

# --- ЭНДПОИНТ: Удалить из избранного ---
# @router.delete("/{announcement_id}/favorite", status_code=status.HTTP_204_NO_CONTENT)
# def remove_favorite(
#     announcement_id: int,
#     db: Session = Depends(get_db),
#     current_user: UserModel = Depends(get_current_user)
# ):
#     """ Удаляет объявление из списка избранного текущего пользователя. """
#     user = db.query(UserModel).options(
#         joinedload(UserModel.favorite_announcements)
#     ).filter(UserModel.id == current_user.id).first()
#
#     announcement = db.query(AnnouncementModel).filter(AnnouncementModel.id == announcement_id).first()
#     if not announcement:
#         # Идемпотентность: если объявления нет, считаем "удаление" успешным
#         logger.warning(f"Попытка удалить из избранного несуществующее объявление {announcement_id} (user {user.id})")
#         return None # 204 No Content
#
#     if announcement not in user.favorite_announcements:
#         # Идемпотентность: если уже не в избранном, считаем "удаление" успешным
#         logger.info(f"Объявления {announcement_id} нет в избранном у пользователя {user.id}, удаление не требуется.")
#         return None # 204 No Content
#
#     user.favorite_announcements.remove(announcement)
#     try:
#         db.commit()
#         logger.info(f"Объявление {announcement_id} удалено из избранного пользователя {user.id}")
#         return None # Успешное удаление, возвращаем None для статуса 204
#     except Exception as e:
#         db.rollback()
#         logger.error(f"Ошибка удаления из избранного для user {user.id}, ann {announcement_id}: {e}")
#         raise HTTPException(status_code=500, detail="Не удалось удалить из избранного")

@router.delete("/{announcement_id}/favorite", status_code=status.HTTP_204_NO_CONTENT)
def remove_favorite(
    announcement_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Удаляет объявление из списка избранного текущего пользователя.
    (Версия с явным поиском объекта и логированием).
    """
    logger.info(f"Получен запрос на удаление из избранного ann_id={announcement_id} от user_id={current_user.id}")

    # Загружаем пользователя и его избранное
    # joinedload здесь важен, чтобы коллекция favorite_announcements была загружена
    user = db.query(UserModel).options(
        joinedload(UserModel.favorite_announcements)
    ).filter(UserModel.id == current_user.id).first()

    if not user:
        # Этого не должно происходить, если get_current_user работает
        logger.error(f"Критическая ошибка: не найден пользователь {current_user.id} в remove_favorite")
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    # Находим объект Announcement (просто для проверки его существования в целом)
    announcement_to_check = db.query(AnnouncementModel).filter(AnnouncementModel.id == announcement_id).first()
    if not announcement_to_check:
        logger.warning(f"Попытка удалить из избранного несуществующее объявление {announcement_id} (user {user.id})")
        return None # 204 No Content

    # --- Явный поиск объекта в коллекции ---
    found_in_favorites = None
    for fav_ann in user.favorite_announcements:
        if fav_ann.id == announcement_id: # Сравниваем по ID
            found_in_favorites = fav_ann
            break # Нашли, выходим из цикла

    if found_in_favorites:
        # Удаляем конкретный экземпляр, найденный в коллекции
        try:
            user.favorite_announcements.remove(found_in_favorites)
            logger.info(f"Объект Ann ID {announcement_id} удален из КОЛЛЕКЦИИ избранного User ID {user.id}.")
            # Добавляем логирование ДО коммита
            # Преобразуем в список ID для читаемого лога
            current_fav_ids = [fav.id for fav in user.favorite_announcements]
            logger.info(f"ДО КОММИТА (remove): User {user.id} favorites collection IDs: {current_fav_ids}")

            db.commit() # Пытаемся сохранить изменения в БД

            logger.info(f"КОММИТ УСПЕШЕН: Объявление ID {announcement_id} удалено из избранного пользователя {user.id}")
            return None # Возвращаем None для статуса 204

        except Exception as e:
            db.rollback() # Откат в случае ошибки коммита
            logger.error(f"ОШИБКА КОММИТА при удалении из избранного для user {user.id}, ann {announcement_id}: {e}", exc_info=True)
            # Восстанавливать объект в коллекции не обязательно, т.к. сессия откатится
            raise HTTPException(status_code=500, detail="Не удалось удалить из избранного (ошибка БД при сохранении)")
    else:
        logger.info(f"Объявления {announcement_id} УЖЕ НЕТ в коллекции избранного у пользователя {user.id}, удаление не требуется.")
        return None # Возвращаем None для статуса 204


# --- ЭНДПОИНТ: Удаление объявления ---
@router.delete("/{announcement_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_announcement_endpoint(
    announcement_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """ Удаляет объявление по его ID (только владелец). """
    logger.info(f"Попытка удаления объявления ID: {announcement_id} пользователем ID: {current_user.id}")

    # Находим объявление в БД
    announcement = db.query(AnnouncementModel).filter(AnnouncementModel.id == announcement_id).first()

    if not announcement:
         # Идемпотентность: если не найдено, считаем удаленным
         logger.warning(f"Попытка удалить несуществующее объявление ID: {announcement_id} (запросил user {current_user.id})")
         return None # 204 No Content

    # Проверка Авторизации (владелец ли?)
    if announcement.user_id != current_user.id:
         logger.warning(f"Попытка пользователя {current_user.id} удалить чужое объявление {announcement_id}")
         raise HTTPException(status_code=403, detail="Нет прав на удаление этого объявления")

    try:
        # 1. Удаляем файл изображения (если он есть) - Используем ту же логику пути, что и при обновлении
        image_url_path_to_delete = announcement.image_path
        if image_url_path_to_delete:
             try:
                 script_dir_del = os.path.dirname(__file__)
                 images_dir_absolute_del = os.path.abspath(os.path.join(script_dir_del, "../images"))
                 filename_to_delete = os.path.basename(image_url_path_to_delete)
                 full_os_path_to_delete = os.path.join(images_dir_absolute_del, filename_to_delete)

                 if os.path.exists(full_os_path_to_delete):
                     os.remove(full_os_path_to_delete)
                     logger.info(f"Файл изображения удален при удалении объявления: {full_os_path_to_delete}")
                 else:
                      logger.warning(f"Файл для удаления не найден по пути: {full_os_path_to_delete} (URL был: {image_url_path_to_delete})")
             except Exception as e:
                 # Логируем ошибку удаления файла, но НЕ прерываем удаление из БД
                 logger.error(f"Не удалось удалить файл {full_os_path_to_delete} при удалении объявления {announcement_id}: {e}")

        # 2. Удаляем запись из БД
        # Если в модели Announcement настроено cascade="all, delete-orphan" для связи 'pet',
        # SQLAlchemy должен удалить и связанного питомца.
        # Если нет, возможно, питомца нужно удалять вручную перед объявлением:
        # pet_to_delete = announcement.pet
        # if pet_to_delete: db.delete(pet_to_delete)
        db.delete(announcement)
        db.commit()
        logger.info(f"Объявление ID {announcement_id} (и связанный питомец, если настроено cascade) успешно удалено из БД.")

        return None # Успешное удаление, статус 204

    except Exception as db_exc:
         db.rollback()
         logger.exception(f"Ошибка БД при удалении объявления ID {announcement_id}: {db_exc}")
         raise HTTPException(status_code=500, detail="Ошибка базы данных при удалении объявления.")

# --- ЭНДПОИНТ: Найти пару ---
@router.post("/pets/find_mate", response_model=List[AnnouncementResponse])
async def find_pet_mate(
    # --- Параметры Form() ---
    animal_type: str = Form(...),
    gender: str = Form(...), # Пол СВОЕГО питомца
    age: int = Form(...),
    breed: str = Form(...),
    name: Optional[str] = Form(None), # Имя своего питомца (не используется в поиске)
    size: Optional[str] = Form(None), # Размер своего питомца (не используется в поиске)
    city: Optional[str] = Form(None), # Город для фильтрации поиска
    image: Optional[UploadFile] = File(None), # Фото своего питомца (пока не используется)
    # --- Зависимости ---
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """ Ищет подходящих питомцев для вязки по параметрам. """
    logger.info(f"Поиск пары: тип={animal_type}, пол_своего={gender}, возраст_своего={age}, порода={breed}, город={city} (Запросил пользователь ID: {current_user.id})")

    # Определяем пол ИСКОМОГО питомца (противоположный)
    target_gender = 'Ж' if gender == 'М' else 'М'
    logger.debug(f"Ищем питомцев с полом: {target_gender}")

    try:
        query = db.query(AnnouncementModel).join(AnnouncementModel.pet).options(
            joinedload(AnnouncementModel.user), # Загружаем юзера объявления
            joinedload(AnnouncementModel.pet)   # Загружаем питомца объявления
        ).filter(
            AnnouncementModel.status == "опубликовано",     # Только активные объявления
            AnnouncementModel.user_id != current_user.id, # Исключаем свои объявления
            PetModel.animal_type == animal_type,       # Тот же вид животного
            PetModel.gender == target_gender,          # Противоположный пол
            PetModel.age >= 1,                         # Возраст питомца для вязки >= 1 года (пример)
            # Дополнительные условия (опционально):
            # PetModel.isNeutered == False, # Ищем только не стерилизованных/кастрированных
            # PetModel.breed.ilike(f"%{breed}%"), # Строгое совпадение породы (убрано ниже, т.к. есть нестрогий фильтр)
        )

        # Фильтр по породе (нестрогий)
        if breed:
            query = query.filter(PetModel.breed.ilike(f"%{breed}%"))

        # Фильтр по городу (нестрогий, если указан)
        if city:
            query = query.filter(AnnouncementModel.city.ilike(f"%{city}%"))

        # TODO: Добавить более сложную логику сопоставления? Ранжирование по совпадениям?

        # Сортировка (например, сначала новые) и ограничение
        results = query.order_by(desc(AnnouncementModel.timestamp)).limit(30).all()

        logger.info(f"Найдено {len(results)} потенциальных пар.")

        # Закрываем файл изображения своего питомца, если он был загружен
        if image:
            await image.close()

        return results

    except Exception as e:
        # Закрываем файл изображения, если он был открыт и произошла ошибка
        if image and not image.is_closed:
             await image.close()
        logger.exception(f"Ошибка при поиске пары: {e}")
        raise HTTPException(status_code=500, detail="Не удалось выполнить поиск пары")


# --- ЭНДПОИНТ: Загрузка/Обновление фото для объявления (Альтернативный/Дополнительный) ---
# Этот эндпоинт может быть полезен, если нужно загрузить фото отдельно от создания/обновления
# всего объявления. Если основной эндпоинт обновления справляется, этот может быть лишним.
@router.post("/{announcement_id}/photo", status_code=status.HTTP_201_CREATED) # Используем 201 Created для нового ресурса (фото)
async def upload_announcement_photo( # Переименовано для ясности
    announcement_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user) # Добавим проверку прав
):
    """ Загружает или заменяет фотографию для указанного объявления. """
    logger.info(f"Попытка загрузить фото для объявления {announcement_id} пользователем {current_user.id}")

    announcement = db.query(AnnouncementModel).filter(AnnouncementModel.id == announcement_id).first()
    if not announcement:
        raise HTTPException(status_code=404, detail="Объявление не найдено")

    # Проверка прав: только владелец может менять фото
    if announcement.user_id != current_user.id:
        logger.warning(f"Пользователь {current_user.id} пытался изменить фото чужого объявления {announcement_id}")
        raise HTTPException(status_code=403, detail="Нет прав на изменение фото этого объявления")

    # Используем ту же логику сохранения/удаления, что и в update_announcement
    old_image_url_path = announcement.image_path
    new_image_url_path = old_image_url_path # По умолчанию

    try:
        # 1. Удаляем старый файл (если он был)
        if old_image_url_path:
            try:
                script_dir_photo = os.path.dirname(__file__)
                images_dir_absolute_photo = os.path.abspath(os.path.join(script_dir_photo, "../images"))
                old_filename_photo = os.path.basename(old_image_url_path)
                full_old_os_path_photo = os.path.join(images_dir_absolute_photo, old_filename_photo)
                if os.path.exists(full_old_os_path_photo):
                    os.remove(full_old_os_path_photo)
                    logger.info(f"Старый файл изображения удален (через /photo): {full_old_os_path_photo}")
                # else: Не страшно, если старого файла нет
            except Exception as e:
                 logger.error(f"Не удалось удалить старый файл {full_old_os_path_photo} при загрузке нового через /photo: {e}")

        # 2. Сохраняем новый файл
        script_dir_photo = os.path.dirname(__file__)
        images_dir_absolute_photo = os.path.abspath(os.path.join(script_dir_photo, "../images"))
        os.makedirs(images_dir_absolute_photo, exist_ok=True)

        timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in ['.', '_', '-']).strip() or "uploaded_image"
        # Используем ID объявления и пользователя в имени
        unique_filename = f"ann_{announcement_id}_{current_user.id}_{timestamp_now}_{safe_filename}"

        new_image_os_path_absolute = os.path.join(images_dir_absolute_photo, unique_filename)
        new_image_url_path = f"images/{unique_filename}" # Путь для БД

        with open(new_image_os_path_absolute, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Новое изображение сохранено (через /photo): {new_image_os_path_absolute}")

        # 3. Обновляем путь в БД
        announcement.image_path = new_image_url_path
        db.add(announcement)
        db.commit()
        db.refresh(announcement)

        return {
            "message": "Фотография успешно загружена/обновлена",
            "image_url": new_image_url_path, # Возвращаем URL путь
            "announcement_id": announcement_id
        }

    except Exception as e:
        db.rollback() # Откатываем изменения в БД, если сохранение файла или коммит не удались
        logger.exception(f"Ошибка при загрузке фото для объявления {announcement_id}: {e}")
        # Пытаемся удалить частично сохраненный файл, если он есть
        if 'new_image_os_path_absolute' in locals() and os.path.exists(new_image_os_path_absolute):
            try: os.remove(new_image_os_path_absolute)
            except: pass # Игнорируем ошибки при удалении временного файла
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла изображения: {e}")
    finally:
         await file.close() # Всегда закрываем файл