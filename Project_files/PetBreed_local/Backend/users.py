#users.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session, joinedload # Добавили joinedload, если еще не было
from typing import List, Optional # <--- ДОБАВИТЬ List сюда (Optional может уже быть)
import logging
# Импорты из ваших модулей
from database import get_db
from auth import create_access_token # Если /token используется
from auth import get_current_user # Если используется для защищенных эндпоинтов
from models import Announcement as AnnouncementModel, Pet as PetModel, User as UserModel
from schemas import UserCreate, User as UserSchema, Token, AnnouncementResponse # Добавили AnnouncementResponse
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix="/users", tags=["users"])

# --- НОВЫЙ ЭНДПОИНТ "МОИ ОБЪЯВЛЕНИЯ" (ИСПРАВЛЕННЫЙ) ---
@router.get("/me/announcements", response_model=List[AnnouncementResponse])
# VVV --- ДОБАВЛЯЕМ ЗАВИСИМОСТЬ get_current_user --- VVV
def get_my_announcements(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user) # Получаем текущего пользователя
):
    # --- УДАЛЯЕМ ИЛИ КОММЕНТИРУЕМ ТЕСТОВЫЙ БЛОК ---
    # MOCK_DB_USER_ID = 1
    # user_id_to_filter = MOCK_DB_USER_ID
    # print(f"Запрос 'Мои объявления' для тестового пользователя ID: {user_id_to_filter}")
    # --- КОНЕЦ УДАЛЯЕМОГО БЛОКА ---

    # Используем ID реального пользователя из токена
    user_id_to_filter = current_user.id
    print(f"Запрос 'Мои объявления' для пользователя ID: {user_id_to_filter}") # Теперь будет правильный ID

    try:
        my_announcements = db.query(AnnouncementModel).options(
            joinedload(AnnouncementModel.user), # Загружаем юзера для ответа
            joinedload(AnnouncementModel.pet)   # Загружаем питомца для ответа
        ).filter(
            # VVV --- ФИЛЬТРУЕМ ПО РЕАЛЬНОМУ user_id --- VVV
            AnnouncementModel.user_id == user_id_to_filter
        ).order_by(
            AnnouncementModel.timestamp.desc() # Сначала новые
        ).all()

        print(f"Найдено {len(my_announcements)} объявлений для пользователя {user_id_to_filter}")
        return my_announcements # Возвращаем список найденных объявлений

    except Exception as e:
        print(f"Ошибка при получении 'моих' объявлений для user ID {user_id_to_filter}: {e}")
        raise HTTPException(status_code=500, detail="Не удалось получить список ваших объявлений")
# --- КОНЕЦ ЭНДПОИНТА ---

# --- НОВЫЙ ЭНДПОИНТ: Получить ИЗБРАННЫЕ объявления ---
@router.get("/me/favorites", response_model=List[AnnouncementResponse])
def get_my_favorites(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user) # <--- ИСПОЛЬЗУЕМ РЕАЛЬНОГО ПОЛЬЗОВАТЕЛЯ
):
    # Блок с MOCK_DB_USER_ID УДАЛЕН

    # Проверяем, что пользователь загружен (хотя Depends должен это гарантировать)
    if not current_user:
        # Эта ситуация маловероятна, если get_current_user работает
        raise HTTPException(status_code=404, detail="Пользователь не найден (ошибка зависимости)")

    logger.info(f"Запрос 'Избранное' для пользователя ID: {current_user.id}.") # Используем logger

    # SQLAlchemy автоматически загрузит связанные favorite_announcements
    # благодаря настройкам relationship в models.py (lazy='selectin' или дефолтный select).
    # Нет необходимости делать здесь отдельный запрос с joinedload, если get_current_user
    # уже вернул пользователя, привязанного к сессии db.

    # Просто возвращаем список избранных объявлений ТЕКУЩЕГО пользователя
    # Pydantic сам преобразует их в List[AnnouncementResponse] благодаря response_model
    # Убедись, что в User.favorite_announcements загружаются связанные Pet и User для AnnouncementResponse
    # Можно явно подгрузить, если нужно:
    # fav_announcements = current_user.favorite_announcements # Получаем список
    # # Опционально: Убедимся, что все связи загружены (если не настроена eager loading)
    # for ann in fav_announcements:
    #     db.refresh(ann, attribute_names=['pet', 'user'])
    # return fav_announcements

    # Обычно достаточно просто вернуть коллекцию:
    return current_user.favorite_announcements

def create_access_token(data: dict):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@router.post("/register", response_model=UserSchema)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(UserModel).filter(UserModel.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    db_user = db.query(UserModel).filter(UserModel.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = pwd_context.hash(user.password)
    new_user = UserModel(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        status=user.status
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.username == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}
