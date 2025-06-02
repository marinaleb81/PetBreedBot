import logging
import os
from dotenv import load_dotenv  

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING) 
logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.error("Токен TELEGRAM_BOT_TOKEN не найден в .env файле!")
    exit()

# --- ВАЖНО: Замените на URL вашего веб-приложения ---
WEB_APP_URL = "https://feb5-67-220-95-210.ngrok-free.app"  
# ----------------------------------------------------

# --- Ссылки для кнопок ---
SUPPORT_URL = "https://t.me/telegram_pet_bot_111"
ADVERTISE_URL = "https://t.me/telegram_pet_bot_111"


# --------------------------


async def set_bot_commands(application: Application):
    """Устанавливает команды для кнопки 'Меню' в Telegram."""
    commands = [
        BotCommand("start", "🚀 Запустить бота / Показать меню"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Команды меню установлены.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение и показывает меню."""
    user = update.effective_user
    logger.info(f"Пользователь {user.username} ({user.id}) запустил бота.")

    # Создаем клавиатуру с кнопками
    keyboard = [
        [InlineKeyboardButton("🐾 Открыть приложение", web_app=WebAppInfo(url=WEB_APP_URL))],
        [InlineKeyboardButton("📞 Служба поддержки", url=SUPPORT_URL)],
        [InlineKeyboardButton("📢 Разместить рекламу", url=ADVERTISE_URL)],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_html(
        rf"👋 Привет, {user.mention_html()}! Ищешь питомца или пристраиваешь своего? Я помогу! А еще умею определять породу и находить пару для твоего любимца.",
        reply_markup=reply_markup,
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений (можно добавить логику)."""
    user_text = update.message.text
    logger.info(f"Получено сообщение от {update.effective_user.username}: {user_text}")


def main() -> None:
    """Запуск бота."""
    # Создаем объект Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", start))

    # Регистрируем обработчик текстовых сообщений (не команд)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Устанавливаем команды для кнопки "Меню" после инициализации
    # Используем `post_init` для асинхронного вызова после старта event loop
    application.post_init = set_bot_commands

    # Запускаем бота (режим опроса - polling)
    logger.info("Запуск бота...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()