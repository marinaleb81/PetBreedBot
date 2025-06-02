# models.py

from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, Table # Добавь Table сюда
from sqlalchemy.orm import relationship
from database import Base

# --- Ассоциативная таблица для Избранного ---
# Связывает users и announcements (многие ко многим)
favorites_table = Table('favorites', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('announcement_id', Integer, ForeignKey('announcements.id'), primary_key=True)
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=True)
    username = Column(String, index=True, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=True)
    status = Column(String, nullable=False)

    # Relationships
    announcements = relationship("Announcement", back_populates="user")
    sent_messages = relationship("Message", foreign_keys="Message.sender_id", back_populates="sender")
    received_messages = relationship("Message", foreign_keys="Message.receiver_id", back_populates="receiver")

    favorite_announcements = relationship(
        "Announcement",
        secondary=favorites_table, 
        back_populates="favorited_by_users" 
    )



class Pet(Base):
    __tablename__ = "pets"
    id = Column(Integer, primary_key=True, index=True)
    animal_type = Column(String, index=True)
    name = Column(String, nullable=True)
    gender = Column(String)
    age = Column(Integer, nullable=True)
    breed = Column(String, nullable=True, index=True)
    color = Column(String, nullable=True)
    isNeutered = Column(Boolean, nullable=True)
    isVaccinated = Column(Boolean, nullable=True)
    size = Column(String, nullable=True)

    announcements = relationship("Announcement", back_populates="pet")


class Announcement(Base):
    __tablename__ = "announcements"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    pet_id = Column(Integer, ForeignKey("pets.id", ondelete="CASCADE"))
    keywords = Column(String, nullable=True)
    description = Column(String, nullable=True)
    status = Column(String, index=True, default="опубликовано")
    timestamp = Column(String)
    image_path = Column(String, nullable=True)
    city = Column(String, index=True, nullable=True)

    user = relationship("User", back_populates="announcements")
    pet = relationship("Pet", back_populates="announcements", cascade="all, delete-orphan", single_parent=True)
    messages = relationship("Message", back_populates="announcement", cascade="all, delete-orphan")

    favorited_by_users = relationship(
        "User",
        secondary=favorites_table, 
        back_populates="favorite_announcements" 
    )



class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))
    receiver_id = Column(Integer, ForeignKey("users.id"))
    announcement_id = Column(Integer, ForeignKey("announcements.id"))
    content = Column(String)
    timestamp = Column(String)
    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    receiver = relationship("User", foreign_keys=[receiver_id], back_populates="received_messages")
    announcement = relationship("Announcement", back_populates="messages")

