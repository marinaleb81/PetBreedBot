# auth_telegram.py
import os
import hmac
import hashlib
import json
from urllib.parse import unquote, parse_qs
from typing import Dict, Optional
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Body 
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import get_db
from models import User as UserModel
from schemas import User as UserSchema, Token, TelegramInitData 
from auth import create_access_token 

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    print("ПРЕДУПРЕЖДЕНИЕ: TELEGRAM_BOT_TOKEN не найден в .env файле! Валидация initData не будет работать.")

def validate_init_data(init_data: str, bot_token: str) -> Optional[Dict]:
    """Проверяет хэш initData и возвращает данные пользователя, если валидно."""
    try:
        parsed_data = parse_qs(init_data)
        received_hash = parsed_data.pop('hash', [None])[0]

        if not received_hash:
            print("Ошибка валидации: hash отсутствует в initData")
            return None

        # Формируем строку для проверки
        data_check_string_parts = []
        for key in sorted(parsed_data.keys()):
            # Значение - это список, берем первый элемент
            data_check_string_parts.append(f"{key}={parsed_data[key][0]}")
        data_check_string = "\n".join(data_check_string_parts)

        # Считаем хеш
        secret_key = hmac.new(key=b"WebAppData", msg=bot_token.encode(), digestmod=hashlib.sha256).digest()
        calculated_hash = hmac.new(key=secret_key, msg=data_check_string.encode(), digestmod=hashlib.sha256).hexdigest()

        # Сравниваем хеши
        if calculated_hash == received_hash:
            print("initData валиден.")
            user_data_str = parsed_data.get('user', [None])[0]
            if user_data_str:
                 # Декодируем URL-кодированную строку JSON пользователя
                 user_data_json_str = unquote(user_data_str)
                 return json.loads(user_data_json_str)
            else:
                 print("Ошибка валидации: данные пользователя отсутствуют в initData")
                 return None
        else:
            print("Ошибка валидации: хеши не совпадают!")
            print(f"Received Hash: {received_hash}")
            print(f"Calculated Hash: {calculated_hash}")
            print(f"Data Check String:\n{data_check_string}")
            return None
    except Exception as e:
        print(f"Исключение при валидации initData: {e}")
        return None

@router.post("/telegram", response_model=Token)
async def authenticate_telegram(
    payload: TelegramInitData = Body(...),
    db: Session = Depends(get_db)
):
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="Telegram Bot Token не настроен на сервере")

    # Валидируем initData
    user_data = validate_init_data(payload.init_data, TELEGRAM_BOT_TOKEN)

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидные данные Telegram (initData)",
        )

    telegram_id = user_data.get('id')
    if not telegram_id:
         raise HTTPException(status_code=400, detail="Не найден ID пользователя в данных Telegram")

    # Ищем пользователя в БД
    db_user = db.query(UserModel).filter(UserModel.telegram_id == telegram_id).first()

    if not db_user:
        print(f"Создание нового пользователя с Telegram ID: {telegram_id}")
        user_status = "Усыновитель"

        new_user_data = {
            "telegram_id": telegram_id,
            "username": user_data.get('username') or f"user_{telegram_id}",
            "status": user_status
            # email и password оставляем null
        }
        db_user = UserModel(**new_user_data)
        db.add(db_user)
        try:
            db.commit()
            db.refresh(db_user)
            print(f"Новый пользователь создан: ID={db_user.id}, TG_ID={telegram_id}")
        except Exception as e:
            db.rollback()
            print(f"Ошибка при создании пользователя: {e}")
            db_user = db.query(UserModel).filter(UserModel.telegram_id == telegram_id).first()
            if not db_user: # Если все еще не найден, то реальная ошибка
                 raise HTTPException(status_code=500, detail="Не удалось создать пользователя")

    access_token = create_access_token(data={"sub": str(db_user.id)})
    print(f"Выдан JWT для пользователя: ID={db_user.id}, TG_ID={telegram_id}")

    logger.info(f"!!! ГЕНЕРАЦИЯ ТОКЕНА для DB User ID: {db_user.id} !!!")
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/generate_test_token/{user_id}", include_in_schema=False)
async def generate_test_token_for_user(user_id: int):
    print(f"--- ВНИМАНИЕ: Генерация тестового токена для пользователя ID: {user_id} ---")
    access_token = create_access_token(data={"sub": str(user_id)})
    return {"user_id": user_id, "test_access_token": access_token}