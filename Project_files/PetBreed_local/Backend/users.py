#users.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
import logging
from database import get_db
from auth import create_access_token
from auth import get_current_user 
from models import Announcement as AnnouncementModel, Pet as PetModel, User as UserModel
from schemas import UserCreate, User as UserSchema, Token, AnnouncementResponse
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

@router.get("/me/announcements", response_model=List[AnnouncementResponse])
def get_my_announcements(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user) 
):

    # Используем ID реального пользователя из токена
    user_id_to_filter = current_user.id
    print(f"Запрос 'Мои объявления' для пользователя ID: {user_id_to_filter}") 

    try:
        my_announcements = db.query(AnnouncementModel).options(
            joinedload(AnnouncementModel.user), # Загружаем юзера для ответа
            joinedload(AnnouncementModel.pet)   # Загружаем питомца для ответа
        ).filter(
            AnnouncementModel.user_id == user_id_to_filter
        ).order_by(
            AnnouncementModel.timestamp.desc() 
        ).all()

        print(f"Найдено {len(my_announcements)} объявлений для пользователя {user_id_to_filter}")
        return my_announcements # Возвращаем список найденных объявлений

    except Exception as e:
        print(f"Ошибка при получении 'моих' объявлений для user ID {user_id_to_filter}: {e}")
        raise HTTPException(status_code=500, detail="Не удалось получить список ваших объявлений")

@router.get("/me/favorites", response_model=List[AnnouncementResponse])
def get_my_favorites(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user) 
):
    if not current_user:

        raise HTTPException(status_code=404, detail="Пользователь не найден (ошибка зависимости)")

    logger.info(f"Запрос 'Избранное' для пользователя ID: {current_user.id}.") 

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
