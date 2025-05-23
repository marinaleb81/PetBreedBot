#chat.py

from fastapi import APIRouter, WebSocket, Depends, HTTPException, Form
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
from typing import List, Dict
from models import User, Message as MessageModel, Announcement
from schemas import Message as MessageSchema
from database import get_db
from auth import get_current_user
import json
import logging

router = APIRouter(prefix="", tags=["chat"])

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Хранилище активных WebSocket-соединений
connected_clients: Dict[int, WebSocket] = {}

@router.websocket("/ws/chat/{announcement_id}")
async def websocket_endpoint(websocket: WebSocket, announcement_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Проверяем, существует ли объявление
    announcement = db.query(Announcement).filter(Announcement.id == announcement_id).first()
    if not announcement:
        logger.warning(f"Announcement not found: announcement_id={announcement_id}")
        await websocket.close(code=1008, reason="Announcement not found")
        return

    sender_id = current_user.id
    receiver_id = announcement.user_id  # Владелец объявления

    # Проверяем, что отправитель не является владельцем объявления
    if sender_id == receiver_id:
        logger.warning(f"User {sender_id} cannot chat with themselves")
        await websocket.close(code=1008, reason="Cannot chat with yourself")
        return

    # Проверяем, существует ли получатель
    receiver = db.query(User).filter(User.id == receiver_id).first()
    if not receiver:
        logger.warning(f"Receiver not found: receiver_id={receiver_id}")
        await websocket.close(code=1008, reason="Receiver not found")
        return

    await websocket.accept()
    connected_clients[sender_id] = websocket
    logger.info(f"User {sender_id} connected to WebSocket for chat with {receiver_id} about announcement {announcement_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Создаём объект SQLAlchemy MessageModel
            new_message = MessageModel(
                sender_id=sender_id,
                receiver_id=receiver_id,
                announcement_id=announcement_id,
                content=message_data["message"],
                timestamp=datetime.now().isoformat()
            )
            db.add(new_message)
            db.commit()
            db.refresh(new_message)
            logger.info(f"Message saved (WebSocket): from {sender_id} to {receiver_id}, content: {message_data['message']}")

            # Формируем ответное сообщение
            message_response = {
                "sender_id": sender_id,
                "message": message_data["message"],
                "timestamp": new_message.timestamp,
                "announcement_id": announcement_id
            }

            # Отправляем сообщение получателю, если он онлайн
            if receiver_id in connected_clients:
                await connected_clients[receiver_id].send_text(json.dumps(message_response))
                logger.info(f"Message sent to receiver {receiver_id} via WebSocket")
            # Отправляем копию отправителю
            await websocket.send_text(json.dumps(message_response))
            logger.info(f"Message sent to sender {sender_id} via WebSocket")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if sender_id in connected_clients:
            del connected_clients[sender_id]
            logger.info(f"User {sender_id} disconnected from WebSocket")
        await websocket.close()

@router.post("/chat/{announcement_id}")
async def send_message(
    announcement_id: int,
    message: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Проверяем, существует ли объявление
    announcement = db.query(Announcement).filter(Announcement.id == announcement_id).first()
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found")

    receiver_id = announcement.user_id  # Владелец объявления
    sender_id = current_user.id

    # Проверяем, что отправитель не является владельцем объявления
    if sender_id == receiver_id:
        raise HTTPException(status_code=400, detail="Cannot chat with yourself")

    # Проверяем, существует ли получатель
    receiver = db.query(User).filter(User.id == receiver_id).first()
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")

    # Создаём объект SQLAlchemy MessageModel
    new_message = MessageModel(
        sender_id=sender_id,
        receiver_id=receiver_id,
        announcement_id=announcement_id,
        content=message,
        timestamp=datetime.now().isoformat()
    )
    db.add(new_message)
    db.commit()
    db.refresh(new_message)
    logger.info(f"Message saved (REST): from {sender_id} to {receiver_id}, content: {message}")

    # Формируем ответное сообщение
    message_response = {
        "sender_id": sender_id,
        "message": message,
        "timestamp": new_message.timestamp,
        "announcement_id": announcement_id
    }

    # Отправляем сообщение через WebSocket, если пользователи онлайн
    if receiver_id in connected_clients:
        await connected_clients[receiver_id].send_text(json.dumps(message_response))
        logger.info(f"Message sent to receiver {receiver_id} via WebSocket (REST)")
    if sender_id in connected_clients:
        await connected_clients[sender_id].send_text(json.dumps(message_response))
        logger.info(f"Message sent to sender {sender_id} via WebSocket (REST)")

    return {"status": "Message sent"}

@router.get("/chat/{announcement_id}", response_model=List[MessageSchema])
def get_chat(
    announcement_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Проверяем, существует ли объявление
    announcement = db.query(Announcement).filter(Announcement.id == announcement_id).first()
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found")

    user_id = current_user.id

    # Проверяем, что пользователь участвует в переписке (либо как отправитель, либо как получатель)
    # Проверяем, является ли пользователь владельцем объявления или отправителем хотя бы одного сообщения
    is_owner = (announcement.user_id == user_id)
    has_messages = db.query(MessageModel).filter(
        (MessageModel.announcement_id == announcement_id) &
        (MessageModel.sender_id == user_id)
    ).first() is not None

    if not (is_owner or has_messages):
        raise HTTPException(status_code=403, detail="You are not authorized to view this chat")

    # Получаем все сообщения, связанные с объявлением, где текущий пользователь участвует
    messages = db.query(MessageModel).options(
        joinedload(MessageModel.sender),
        joinedload(MessageModel.receiver)
    ).filter(
        (MessageModel.announcement_id == announcement_id) &
        ((MessageModel.sender_id == user_id) | (MessageModel.receiver_id == user_id))
    ).order_by(MessageModel.timestamp.asc()).all()

    if not messages:
        logger.info(f"No messages found for user {user_id} about announcement {announcement_id}")
        return []

    logger.info(f"Retrieved {len(messages)} messages for user {user_id} about announcement {announcement_id}")
    return messages
