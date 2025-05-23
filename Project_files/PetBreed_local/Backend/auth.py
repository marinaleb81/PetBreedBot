# auth.py (Обновленный)

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv
from database import get_db
from models import User
from datetime import datetime, timedelta # Добавлено
from typing import Optional
import logging

load_dotenv()
logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
# Время жизни токена - можно вынести в .env
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Оставляем oauth2_scheme, если планируем использовать /users/token для чего-то еще
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/token")

# === ПЕРЕНЕСЕННАЯ ФУНКЦИЯ ===
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Используем значение по умолчанию
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
# ==========================

def get_current_user_id(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub") # Это ID из нашей БД
        if user_id is None:
            raise credentials_exception
    except JWTError as e:
        print(f"JWT Error: {e}") # Логируем ошибку
        raise credentials_exception
    return user_id # Возвращаем ID из БД (как строку)
    logger.info(f"!!! ТОКЕН ДЕКОДИРОВАН. Ищем пользователя с DB ID: {user_id} !!!")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    user_id_str = get_current_user_id(token)
    try:
        user_id = int(user_id_str) # Преобразуем в int для запроса к БД
    except ValueError:
        print(f"Invalid user ID format in token: {user_id_str}")
        raise HTTPException(status_code=401, detail="Invalid user ID format")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        # Может быть полезно проверить, не удалили ли юзера после выдачи токена
        print(f"User with DB ID {user_id} not found in DB, but token was valid.")
        raise HTTPException(status_code=404, detail="User not found")
    return user