# schemas.py
from pydantic import BaseModel, EmailStr # Импортируем EmailStr для валидации
from typing import Optional, List

# --- User Schemas ---
class UserBase(BaseModel):
    # Оставим email/username опциональными в базовой схеме
    username: Optional[str] = None
    email: Optional[EmailStr] = None # Используем EmailStr для валидации
    status: str # Статус обязателен

class UserCreate(UserBase):
    # При создании через Telegram ID обязателен
    telegram_id: int
    # Пароль больше не нужен для TWA создания
    # password: str

class User(UserBase):
    id: int # DB ID
    telegram_id: Optional[int] = None # Может быть null у старых юзеров/админов

    class Config:
        from_attributes = True # Было orm_mode=True в Pydantic v1

# --- Token Schema (без изменений) ---
class Token(BaseModel):
    access_token: str
    token_type: str

# --- Pet Schemas (Обновлено) ---
class PetBase(BaseModel):
    animal_type: str
    name: Optional[str] = None
    gender: str
    age: Optional[int] = None
    breed: Optional[str] = None
    color: Optional[str] = None
    # Добавляем новые поля
    isNeutered: Optional[bool] = None
    isVaccinated: Optional[bool] = None
    size: Optional[str] = None # Для собак

# --- НОВАЯ СХЕМА для результата определения породы ---
class BreedIdentificationResult(BaseModel):
    animal_type: Optional[str] = None # Используем 'animal_type' для единообразия
    breed: Optional[str] = None
    recommendations: Optional[str] = None

class PetCreate(PetBase):
    pass # Пока совпадает с базой

class Pet(PetBase):
    id: int

    class Config:
        from_attributes = True

# --- Announcement Schemas (Обновлено) ---
class AnnouncementBase(BaseModel):
    keywords: Optional[str] = None
    description: Optional[str] = None
    # Добавляем город
    city: Optional[str] = None

class AnnouncementCreate(AnnouncementBase):
    pass # Пока совпадает с базой

class AnnouncementResponse(AnnouncementBase): # Переименовал для ясности, что это ответ API
    id: int
    user_id: int
    pet_id: int
    status: str
    timestamp: str # В идеале datetime, но соответствует модели
    image_path: Optional[str] = None # Сделаем опциональным
    # Вложенные схемы для user и pet
    user: User # Используем обновленную схему User
    pet: Pet   # Используем обновленную схему Pet

    class Config:
        from_attributes = True

# --- SearchRequest Schema (Обновлено) ---
class SearchRequest(BaseModel):
    animal_type: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None # Позже можно min/max
    breed: Optional[str] = None
    color: Optional[str] = None
    city: Optional[str] = None # Добавлено
    isNeutered: Optional[bool] = None # Добавлено
    isVaccinated: Optional[bool] = None # Добавлено
    keywords: Optional[List[str]] = None

# --- Message Schemas (без существенных изменений) ---
class MessageBase(BaseModel):
    content: str

class MessageCreate(MessageBase):
    # Эти поля будут установлены на сервере или получены из контекста
    # sender_id: int # Устанавливается из current_user
    receiver_id: int # Определяется из объявления
    announcement_id: int # Из пути URL
    # timestamp: str # Устанавливается на сервере

    # Оставляем только content, или если фронт шлет все - то все поля
    pass # Пересмотрим при реализации отправки

class Message(MessageBase):
    id: int
    sender_id: int
    receiver_id: int
    announcement_id: int
    timestamp: str # В идеале datetime
    # Вложенные данные об отправителе/получателе для отображения
    sender: Optional[User] = None
    receiver: Optional[User] = None

    class Config:
        from_attributes = True

# --- Схема для входных данных Telegram Auth ---
class TelegramInitData(BaseModel):
    init_data: str # Строка initData из Telegram.WebApp.initData
    # Можно добавить поле status при первом входе, если фронт его определит
    # status: Optional[str] = None # "Усыновитель" или "Пристраивающий"


class PetMateInput(BaseModel):
    # Поля, которые придут из формы (кроме фото)
    animal_type: str # Кошка или Собака
    gender: str      # Пол СВОЕГО питомца (М или Ж)
    nickname: Optional[str] = None
    age: int         # Возраст СВОЕГО питомца (обязателен, >=1?)
    breed: str       # Порода СВОЕГО питомца (обязательна?)
    size: Optional[str] = None # Размер (для собак)
    city: Optional[str] = None
    pedigree: Optional[bool] = False # Была ли отметка "родословная"?
    vaccinations: Optional[bool] = False # Была ли отметка "прививки"?
    experience: Optional[bool] = False # Была ли отметка "опыт вязки"?
    notes: Optional[str] = None