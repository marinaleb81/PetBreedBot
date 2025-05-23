print("--- DEBUG: main.py - Начало выполнения ---")

# --- Импорты ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, HTMLResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from database import Base, engine
from users import router as users_router
from classify import router as classify_router
from chat import router as chat_router
from auth_telegram import router as auth_telegram_router
from announcements import router as announcements_router
from pathlib import Path
from fastapi.staticfiles import StaticFiles

print("--- DEBUG: main.py - Импорты завершены ---")

# --- Создание таблиц в базе данных ---
try:
    Base.metadata.create_all(bind=engine)
    print("Таблицы БД успешно созданы/проверены.")
except Exception as e:
    print(f"Ошибка при создании таблиц БД: {e}")

# --- Создание приложения FastAPI ---
app = FastAPI(
    title="Pet Adoption API",
    description="API for pet adoption platform with image classification",
    version="1.0.0",
    openapi_tags=[
        {"name": "users", "description": "Operations with users"},
        {"name": "classify", "description": "Image classification and announcement operations"},
        {"name": "chat", "description": "Chat operations"},
    ],
)

# --- Определение путей с помощью pathlib ---
# Определяем путь к директории, где находится main.py (Backend)
script_path_obj = Path(__file__).parent  # Path('E:/PetPreed/PetPreed/Backend')

# Определяем путь к корневой папке проекта (на уровень выше)
project_root = script_path_obj.parent  # Path('E:/PetPreed/PetPreed')

# Определяем пути к нужным папкам и файлам относительно КОРНЯ ПРОЕКТА
icon_dir_path = project_root / "icon"  # Path('E:/PetPreed/PetPreed/icon')
images_dir_path = project_root / "images"  # Path('E:/PetPreed/PetPreed/images')
index_path_obj = project_root / "index.html"
script_js_path_obj = project_root / "script.js"
style_css_path_obj = project_root / "style.css"

print(f"--- Рассчитанные пути (Pathlib) ---")
print(f"Корень проекта: {project_root}")
print(f"Папка icon:   {icon_dir_path}")
print(f"Папка images: {images_dir_path}")
print(f"index.html:   {index_path_obj}")
print(f"script.js:    {script_js_path_obj}")
print(f"style.css:    {style_css_path_obj}")
print(f"----------------------------------")

# --- Монтирование статических папок ---
# Используем объекты Path
if icon_dir_path.is_dir():
    app.mount("/icon", StaticFiles(directory=icon_dir_path), name="icon")
    print(f"Папка /icon смонтирована из {icon_dir_path}")
else:
    print(f"ПРЕДУПРЕЖДЕНИЕ: Папка icon не найдена по пути {icon_dir_path}")

if images_dir_path.is_dir():
    app.mount("/images", StaticFiles(directory=images_dir_path), name="images")
    print(f"Папка /images смонтирована из {images_dir_path}")
else:
    print(f"ПРЕДУПРЕЖДЕНИЕ: Папка images не найдена по пути {images_dir_path}")

# --- Явные маршруты для JS и CSS из родительской папки ---
@app.get("/script.js", include_in_schema=False)
async def serve_script():
    if not script_js_path_obj.is_file():
        print(f"ОШИБКА: script.js не найден по пути {script_js_path_obj}")
        raise StarletteHTTPException(status_code=404, detail="script.js not found")
    return FileResponse(script_js_path_obj, media_type="application/javascript")

@app.get("/style.css", include_in_schema=False)
async def serve_style():
    if not style_css_path_obj.is_file():
        print(f"ОШИБКА: style.css не найден по пути {style_css_path_obj}")
        raise StarletteHTTPException(status_code=404, detail="style.css not found")
    return FileResponse(style_css_path_obj, media_type="text/css")

# --- Настройка CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Настройте для продакшена!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Подключаем API роутеры ---
print("--- DEBUG: main.py - Подключение API роутеров ---")
app.include_router(announcements_router)
app.include_router(users_router)
app.include_router(auth_telegram_router)
app.include_router(classify_router)
app.include_router(chat_router)
print("--- DEBUG: main.py - API роутеры подключены ---")

# --- Простой эндпоинт для проверки работы API ---
@app.get("/api/status")
async def read_api_status():
    return {"message": "Pet Adoption API is running"}

# --- Маршрут для index.html ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index_html():
    print(f"Запрос на /, отдаю index.html из {index_path_obj}")
    if not index_path_obj.is_file():
        print(f"ОШИБКА: index.html не найден по пути {index_path_obj}")
        raise HTTPException(status_code=404, detail="index.html not found")
    try:
        html_content = index_path_obj.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        print(f"ОШИБКА чтения index.html: {e}")
        raise HTTPException(status_code=500, detail="Could not read index.html")

@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
async def catch_all(full_path: str):
    # Проверяем, начинается ли путь с известных префиксов
    if full_path == "":
        # Если путь пустой (корень), уже обработан маршрутом выше
        raise HTTPException(status_code=404, detail="Not found")
    if full_path.startswith(("icon/", "images/", "script.js", "style.css", "api/")):
        # Эти пути должны обрабатываться StaticFiles или другими маршрутами
        raise HTTPException(status_code=404, detail="Not found - handled by static files or other routes")
    # Для всех остальных путей возвращаем index.html (для SPA)
    print(f"Запрос на /{full_path}, отдаю index.html из {index_path_obj}")
    if not index_path_obj.is_file():
        print(f"ОШИБКА: index.html не найден по пути {index_path_obj}")
        raise HTTPException(status_code=404, detail="index.html not found")
    try:
        html_content = index_path_obj.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        print(f"ОШИБКА чтения index.html: {e}")
        raise HTTPException(status_code=500, detail="Could not read index.html")

print("--- DEBUG: main.py - Завершение настроек приложения ---")