# schemas.py
from pydantic import BaseModel, EmailStr 
from typing import Optional, List

# --- User Schemas ---
class UserBase(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None 
    status: str 

class UserCreate(UserBase):
    telegram_id: int


class User(UserBase):
    id: int 
    telegram_id: Optional[int] = None 

    class Config:
        from_attributes = True 

# --- Token Schema ---
class Token(BaseModel):
    access_token: str
    token_type: str

# --- Pet Schemas ---
class PetBase(BaseModel):
    animal_type: str
    name: Optional[str] = None
    gender: str
    age: Optional[int] = None
    breed: Optional[str] = None
    color: Optional[str] = None
    isNeutered: Optional[bool] = None
    isVaccinated: Optional[bool] = None
    size: Optional[str] = None 

# --- Результат определения породы ---
class BreedIdentificationResult(BaseModel):
    animal_type: Optional[str] = None 
    breed: Optional[str] = None
    recommendations: Optional[str] = None

class PetCreate(PetBase):
    pass 

class Pet(PetBase):
    id: int

    class Config:
        from_attributes = True

# --- Announcement Schemas ---
class AnnouncementBase(BaseModel):
    keywords: Optional[str] = None
    description: Optional[str] = None
    city: Optional[str] = None

class AnnouncementCreate(AnnouncementBase):
    pass 

class AnnouncementResponse(AnnouncementBase): #
    id: int
    user_id: int
    pet_id: int
    status: str
    timestamp: str 
    image_path: Optional[str] = None 
    user: User 
    pet: Pet   

    class Config:
        from_attributes = True

# --- SearchRequest Schema ---
class SearchRequest(BaseModel):
    animal_type: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None 
    breed: Optional[str] = None
    color: Optional[str] = None
    city: Optional[str] = None 
    isNeutered: Optional[bool] = None 
    isVaccinated: Optional[bool] = None 
    keywords: Optional[List[str]] = None

# --- Message Schemas ---
class MessageBase(BaseModel):
    content: str

class MessageCreate(MessageBase):
    receiver_id: int 
    announcement_id: int 
    pass 

class Message(MessageBase):
    id: int
    sender_id: int
    receiver_id: int
    announcement_id: int
    timestamp: str 
    sender: Optional[User] = None
    receiver: Optional[User] = None

    class Config:
        from_attributes = True

# --- Схема для входных данных Telegram Auth ---
class TelegramInitData(BaseModel):
    init_data: str


class PetMateInput(BaseModel):
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