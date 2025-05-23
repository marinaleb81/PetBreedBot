from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import os
from dotenv import load_dotenv
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

# Проверка токенов
if not BOT_TOKEN:
    logger.error("BOT_TOKEN не найден в .env")
    raise ValueError("BOT_TOKEN не задан")
if not MISTRAL_TOKEN:
    logger.error("MISTRAL_TOKEN не найден в .env")
    raise ValueError("MISTRAL_TOKEN не задан")

def get_feeding_recommendation(breed):
    mistral_url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_TOKEN}",
        "Content-Type": "application/json"
    }

    prompt = (
    f"Составь точные и реалистичные рекомендации по кормлению для {breed} от имени ветеринарного бота. "
    f"Ответ должен быть на русском, 50–100 слов, без использования местоимения 'я', "
    f"с указанием дозировки, питательных веществ и режима питания. "
    f"Избегай преувеличенных количеств. "
    f"В конце добавь: 'Для индивидуальных рекомендаций проконсультируйтесь с ветеринаром.' "
    f"Ответ должен быть грамматически правильным и содержать только русские слова, в том числе название породы и рекомендаций по кормлению."
    f"Разбивай ответ на абзацы с отступами"
)

    payload = {
        "model": "pixtral-12b-2409",
        "messages": [
            {"role": "system", "content": "Ты ветеринар-диетолог."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        response = requests.post(mistral_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        logger.info(f"Mistral response for {breed}: {text}")
        return text
    except Exception as e:
        logger.error(f"Ошибка Mistral API для {breed}: {str(e)}")
        return f"Не удалось получить рекомендации: {str(e)}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пришли фото собаки или кошки!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        file_bytes = await file.download_as_bytearray()
        response = requests.post(f"{FASTAPI_URL}/predict", files={"file": file_bytes}, timeout=10)
        if response.status_code != 200:
            logger.error(f"FastAPI ошибка: {response.status_code} - {response.text}")
            await update.message.reply_text("Ошибка при определении породы.")
            return
        result = response.json()
        breed = result["breed"]
        probability = result["probability"]
        recommendation = get_feeding_recommendation(breed)
        await update.message.reply_text(
            f"Порода: {breed}, вероятность {probability*100:.1f}%\n{recommendation}"
        )
        user_id = update.effective_user.id
        logger.info(f"Успешно отправлено сообщение пользователю {user_id}")
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {str(e)}")
        await update.message.reply_text("Произошла ошибка при обработке фото.")

def main():
    application = ApplicationBuilder().token(BOT_TOKEN).read_timeout(30).connect_timeout(30).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    logger.info("Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main()

