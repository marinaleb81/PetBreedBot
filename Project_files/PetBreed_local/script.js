// --- script.js ---

// --- Глобальные переменные состояния и для API ---
//const BACKEND_BASE_URL = 'https://i-love-pets.ru/'; // <-- УСТАНОВИТЕ ПРАВИЛЬНЫЙ URL ДЛЯ ТЕСТА
//const BACKEND_BASE_URL = ''; // <-- Для продакшена на том же домене
const BACKEND_BASE_URL = 'http://127.0.0.1:8000';
let authToken = null;
let isAuthenticated = false;
let userRole = null;
let animalType = null;
let currentFilters = {};
let currentSort = 'newest';
//let loadedAnnouncements = [];
//let isDataLoading = false;
let isCityFilterPopulated = false;
//const MOCK_CURRENT_USER_ID = 1;
let selectedBreedFiles = { front: null, side: null, top: null };
let currentlyOpenMenu = null;
let favoriteAnnouncementIds = new Set();
let isMenuOpen = false; // Глобальная переменная состояния меню
// -------------------------------------------------


// === ГЛОБАЛЬНЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
// (Эти функции доступны везде и могут быть объявлены здесь)

function showScreen(screenId) {
    const mainMenu = document.getElementById('main-menu');
    const menuOverlay = document.getElementById('menu-overlay');
    const isMenuOpenNow = mainMenu?.classList.contains('is-open');

    document.querySelectorAll('.screen').forEach(screen => {
        if (screen.id !== 'splash-screen' || document.getElementById('splash-screen')?.classList.contains('hidden')) {
            screen.classList.add('hidden');
        }
    });

    const screenToShow = document.getElementById(screenId);
    if (screenToShow) {
        screenToShow.classList.remove('hidden');
        window.scrollTo(0, 0);
    } else {
        console.error(`Экран с id "${screenId}" не найден.`);
    }

    if (isMenuOpenNow && typeof closeMenu === 'function') {
        console.log("Закрываем меню при переходе на экран:", screenId);
        const visibleHamburger = document.querySelector(`.screen:not(.hidden) .app-header .hamburger-button`);
        closeMenu(visibleHamburger);
        // Дополнительная проверка
        if (menuOverlay?.classList.contains('is-visible')) {
            console.warn("Оверлей все еще видим после закрытия меню, исправляем...");
            menuOverlay.classList.remove('is-visible');
        }
    }
}

// --- Вспомогательная функция для запросов к API с токеном ---
async function fetchWithAuth(url, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) {
       headers['Content-Type'] = 'application/json';
    }
    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    } else {
        console.warn(`Запрос ${url} выполняется без токена авторизации.`);
    }

    const finalOptions = { ...options, headers: headers };
    if (options.body && (options.body instanceof FormData)) {
         delete finalOptions.headers['Content-Type'];
     }

    try {
        console.log(`Выполняется запрос: ${finalOptions.method || 'GET'} ${BACKEND_BASE_URL}${url}`);
        const response = await fetch(`${BACKEND_BASE_URL}${url}`, finalOptions);

        if (response.status === 401 && url !== '/auth/telegram') {
             console.error("Запрос не авторизован (401). Токен невалиден или истек.");
             alert("Ваша сессия истекла. Пожалуйста, перезапустите приложение.");
             authToken = null;
             isAuthenticated = false;
             favoriteAnnouncementIds.clear();
             showScreen('role-selection-screen');
             throw new Error("Unauthorized");
        }
        return response;
    } catch (error) {
        console.error(`Ошибка при выполнении запроса ${url}:`, error);
        if (error.message !== "Unauthorized") {
             alert(`Ошибка сети или сервера при запросе ${url}. Пожалуйста, проверьте соединение и попробуйте снова.`);
        }
        throw error;
    }
}

async function fetchFavoriteIds() {
    if (!isAuthenticated || !authToken) {
        console.log("Пользователь не аутентифицирован, ID избранных не загружены.");
        favoriteAnnouncementIds.clear();
        return;
    }
    console.log("Загрузка ID избранных объявлений...");
    try {
        const response = await fetchWithAuth('/users/me/favorites');
        if (response.ok) {
            const favorites = await response.json();
            favoriteAnnouncementIds = new Set(favorites.map(ann => ann.id));
            console.log(`Загружено ${favoriteAnnouncementIds.size} ID избранных:`, favoriteAnnouncementIds);
        } else {
            console.error("Не удалось загрузить ID избранных:", response.status, await response.text());
            favoriteAnnouncementIds.clear();
        }
    } catch (error) {
        console.error("Ошибка при запросе ID избранных:", error);
        favoriteAnnouncementIds.clear();
    }
}

async function handleFavoriteClick(_e) {
    const button = _e.currentTarget;
    const announcementId = button.dataset.petId;
    if (!announcementId) {
        console.error("Не удалось получить ID объявления из data-pet-id");
        return;
    }
    if (!isAuthenticated || !authToken) {
        alert("Пожалуйста, войдите, чтобы добавлять в избранное.");
        return;
    }
    const annIdInt = parseInt(announcementId);
    const isCurrentlyFavorite = button.classList.contains('is-favorite');
    const method = isCurrentlyFavorite ? 'DELETE' : 'POST';
    const apiUrl = `/announcements/${announcementId}/favorite`;

    console.log(`${isCurrentlyFavorite ? 'Удаляем из' : 'Добавляем в'} избранное (API): ID ${announcementId}`);
    button.disabled = true;

    try {
        const response = await fetchWithAuth(apiUrl, { method: method });

        if ((method === 'POST' && response.ok) || (method === 'DELETE' && response.status === 204)) {
            const nowFavorite = !isCurrentlyFavorite;

            const iconImage = button.querySelector('.favorite-icon');

            if (iconImage) {
                iconImage.src = nowFavorite ? 'icon/heart-filled.svg' : 'icon/heart-icon.svg';
                iconImage.alt = nowFavorite ? 'В избранном' : 'В избранное';
                console.log(`Иконка обновлена: src=${iconImage.src}`);
            } else {
                console.warn("Не найдена иконка .favorite-icon внутри кнопки для обновления src.");
            }

            button.classList.toggle('is-favorite', nowFavorite);
            console.log(`Успех: ${isCurrentlyFavorite ? 'удалено из' : 'добавлено в'} избранное (ID: ${announcementId})`);

            if (isCurrentlyFavorite) {
                favoriteAnnouncementIds.delete(annIdInt);
            } else {
                favoriteAnnouncementIds.add(annIdInt);
            }
            console.log("Обновленный Set ID избранных:", favoriteAnnouncementIds);


            const currentScreen = document.querySelector('.screen:not(.hidden)');
            if (currentScreen?.id === 'favorites-screen' && method === 'DELETE') {
                const cardToRemove = button.closest('.pet-card');
                if (cardToRemove) {
                    cardToRemove.remove();
                    const favoritesListContainer = document.getElementById('favorites-list');
                    if (favoritesListContainer && favoritesListContainer.querySelectorAll('.pet-card').length === 0) {
                        favoritesListContainer.innerHTML = '<p>Вы пока никого не добавили в избранное.</p>';
                    }
                }
            }
        } else {
            let errorDetail = `Ошибка ${response.status}`;
            try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (_e) { /* ignore */ }
            console.error(`Не удалось ${isCurrentlyFavorite ? 'удалить из' : 'добавить в'} избранное: ${errorDetail}`);
            alert(`Не удалось обновить избранное: ${errorDetail}`);
        }
    } catch (error) {
        console.error(`Сетевая ошибка или другая проблема при обновлении избранного:`, error);
        alert(`Произошла ошибка сети при обновлении избранного.`);
    } finally {
        button.disabled = false;
    }
}

// --- Функция аутентификации через Telegram ---
async function performAuthentication() {
    const urlParams = new URLSearchParams(window.location.search);
    const testMode = urlParams.get('test_mode');

    if (testMode === 'true' || window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost') {
        console.warn("ТЕСТОВЫЙ РЕЖИМ (или локальный): Пропуск аутентификации Telegram, установка тестового токена.");
        // --- Генерация токена для теста https://i-love-pets.ru/auth/generate_test_token/3
        const MOCK_AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIzIiwiZXhwIjoxNzQ3NjUzMzAxfQ.RGd-JZlWXtrhvP_VBTBc3w4Eqg5pjMTOlCr1i_Fsf3Y"; // User ID 3
        if (!MOCK_AUTH_TOKEN) {
             console.error("Тестовый токен НЕ установлен!"); alert("Ошибка конфигурации: Тестовый токен не настроен.");
             isAuthenticated = false; showScreen('role-selection-screen'); return;
        }
        authToken = MOCK_AUTH_TOKEN; isAuthenticated = true;
        console.log("Тестовый токен УСТАНОВЛЕН:", authToken);
        await fetchFavoriteIds();
        showScreen('role-selection-screen');
        return;
    }

    console.log("Попытка аутентификации Telegram...");
    const tg = window.Telegram.WebApp;
    const initDataStr = tg.initData;
    console.log("--- ДЕБАГ initData ---");
    console.log("Тип initData:", typeof initDataStr); console.log("Длина initData:", initDataStr?.length);
    console.log("--- КОНЕЦ ДЕБАГА ---");

    if (!initDataStr || typeof initDataStr !== 'string' || initDataStr.length === 0 ) {
        console.error("Ошибка: initData не является строкой или отсутствует!");
        alert("Ошибка: Не удалось получить данные для аутентификации от Telegram.");
        isAuthenticated = false; favoriteAnnouncementIds.clear();
        showScreen('role-selection-screen');
        return;
    }
    try {
        const response = await fetch(`${BACKEND_BASE_URL}/auth/telegram`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
            body: JSON.stringify({ init_data: initDataStr })
        });
        if (response.ok) {
            const data = await response.json();
            authToken = data.access_token; isAuthenticated = true;
            console.log('УСПЕШНАЯ АУТЕНТИФИКАЦИЯ (Telegram). Токен установлен:', authToken);
            await fetchFavoriteIds();
            showScreen('role-selection-screen');
        } else {
            const errorData = await response.json().catch(() => ({ detail: `Статус ошибки: ${response.status}` }));
            console.error("Ошибка аутентификации на бэкенде:", response.status, errorData);
            alert(`Ошибка аутентификации: ${errorData.detail || response.statusText}.`);
            isAuthenticated = false; favoriteAnnouncementIds.clear();
            showScreen('role-selection-screen');
        }
    } catch (error) {
        console.error("Сетевая ошибка или другая проблема при аутентификации:", error);
        alert("Не удалось связаться с сервером для аутентификации.");
        isAuthenticated = false; favoriteAnnouncementIds.clear();
        showScreen('role-selection-screen');
    }
}

//// === Функции управления меню (ГЛОБАЛЬНЫЕ) ===

function openMenu(buttonElement) {
    const mainMenu = document.getElementById('main-menu');
    const menuOverlay = document.getElementById('menu-overlay');
    if (!mainMenu || !menuOverlay || isMenuOpen) return;

    mainMenu.classList.add('is-open');
    menuOverlay.classList.add('is-visible');
    document.querySelectorAll('.hamburger-button').forEach(btn => btn.classList.remove('is-active'));
    if (buttonElement) {
        buttonElement.classList.add('is-active');
        buttonElement.setAttribute('aria-expanded', 'true');
    }
    mainMenu.setAttribute('aria-hidden', 'false');
    isMenuOpen = true;
    console.log("Меню открыто, overlay is-visible:", menuOverlay.classList.contains('is-visible'));
}

function closeMenu(buttonElement) {
    const mainMenu = document.getElementById('main-menu');
    const menuOverlay = document.getElementById('menu-overlay');
    if (!mainMenu || !menuOverlay || !isMenuOpen) return;

    // Снимаем фокус с активного menu-item перед закрытием
    const focusedMenuItem = mainMenu.querySelector('.menu-item:focus');
    if (focusedMenuItem) focusedMenuItem.blur();

    mainMenu.classList.remove('is-open');
    menuOverlay.classList.remove('is-visible');
    if (buttonElement) {
        buttonElement.classList.remove('is-active');
        buttonElement.setAttribute('aria-expanded', 'false');
    } else {
        const activeButton = document.querySelector('.hamburger-button.is-active');
        if (activeButton) {
            activeButton.classList.remove('is-active');
            activeButton.setAttribute('aria-expanded', 'false');
        }
    }
    mainMenu.setAttribute('aria-hidden', 'true');
    isMenuOpen = false;
    console.log("Меню закрыто, overlay is-visible:", menuOverlay.classList.contains('is-visible'));
}

function toggleMenu(buttonElement) {
    if (isMenuOpen) {
        closeMenu(buttonElement);
    } else {
        openMenu(buttonElement);
    }
}
// === Конец ГЛОБАЛЬНЫХ функций ===


// =============================================
// === ОСНОВНОЙ КОД ПОСЛЕ ЗАГРУЗКИ DOM ===
// =============================================
document.addEventListener('DOMContentLoaded', () => {

    // --- Фаза 1: Инициализация и Выборка Элементов ---
    const tg = window.Telegram.WebApp;
    tg.ready();
    tg.expand();
    console.log("DOM загружен, Telegram Web App инициализируется...");

    // Основные элементы интерфейса
    const splashScreen = document.getElementById('splash-screen');
    const mainMenu = document.getElementById('main-menu');
    //const menuOverlay = document.getElementById('menu-overlay');
    const userGreetingElement = document.getElementById('user-greeting');

    // Элементы экрана выбора роли
    const identifyBreedButton = document.getElementById('identify-breed-button');
    const findMateButton = document.getElementById('find-mate-button');
    const findPetButton = document.getElementById('find-pet-button');
    const placePetButton = document.getElementById('place-pet-button');
    const myPetsNavButton = document.getElementById('my-pets-nav-button');
    const favoritesNavButton = document.getElementById('favorites-nav-button');

    // Элементы выбора типа животного
    const selectDogButtonFind = document.getElementById('select-dog-button-find');
    const selectCatButtonFind = document.getElementById('select-cat-button-find');
    const selectDogButtonPlace = document.getElementById('select-dog-button-place');
    const selectCatButtonPlace = document.getElementById('select-cat-button-place');

    // Элементы формы объявления
    const adFormContainer = document.getElementById('ad-form-container');
    const createAdButton = document.getElementById('create-ad-button');
    // (Другие поля формы выбираются по ID внутри обработчиков)

    // Элементы экрана результатов
    const resultsListContainer = document.getElementById('results-list');
    const filterButton = document.getElementById('filter-button');
    const favoritesButton = document.getElementById('favorites-button'); // В футере
    const sortButton = document.getElementById('sort-button');
    const sortMenu = document.getElementById('sort-menu');

    // Элементы экрана фильтров
    const filtersContainer = document.querySelector('#filter-screen .filters-container');
    const applyFiltersButton = document.getElementById('apply-filters-button');
    const resetFiltersButton = document.getElementById('reset-filters-button');
    const keywordsTagsContainer = document.getElementById('keywords-filter-tags');

    // Элементы экрана "Мои объявления"
    const myPetsListContainer = document.getElementById('my-pets-list');

    // Элементы экрана "Избранное"
    const favoritesList = document.getElementById('favorites-list'); // Контейнер списка
    const viewToggleButton = document.getElementById('view-toggle-button'); // Кнопка переключения вида

    // Элементы формы "Найти пару"
    const findMateFormContainer = document.getElementById('find-mate-form-container');
    const findMateSizeGroup = document.getElementById('find-mate-size-group');
    const findMateSubmitButton = document.getElementById('find-mate-submit-button');

    // Элементы экрана "Определить породу" (мульти-фото)
    const angleLabels = {
        front: document.getElementById('upload-label-front-breed'),
        side: document.getElementById('upload-label-side-breed'),
        top: document.getElementById('upload-label-top-breed')
    };
    const anglePreviews = {
        front: document.getElementById('preview-front'),
        side: document.getElementById('preview-side'),
        top: document.getElementById('preview-top')
    };
    const frontInput = document.getElementById('image-input-front');
    const sideInput = document.getElementById('image-input-side');
    const topInput = document.getElementById('image-input-top');
    const submitBreedMultiButton = document.getElementById('submit-breed-multi-button');
    const breedResultOutputMulti = document.getElementById('breed-result-output');


    // --- Фаза 2: Определение Локальных Функций (внутри DOMContentLoaded) ---

    // --- Функции создания карточек ---
    function createListPetCardElement(announcementData, options = {}) {
        const card = document.createElement('div');
        card.className = 'pet-card';
        const pet = announcementData.pet || {};
        //const user = announcementData.user || {};
        const city = announcementData.city || 'Город не указан';
        const petName = pet.name || 'Имя не указано';
        const petBreed = pet.breed || 'Порода не указана';
        const petDescription = announcementData.description || '';

        const relativeImagePath = announcementData.image_path ? `/${announcementData.image_path.replace(/\\/g, '/')}` : null;
        const imagePath = relativeImagePath ? `${BACKEND_BASE_URL}${relativeImagePath}` : 'images/placeholder_pet.png';

        const petGender = pet.gender;
        const isNeutered = pet.isNeutered;
        let neuteredText = '';
        if (isNeutered === true) neuteredText = (petGender === 'Ж') ? 'Стерилизована' : 'Кастрирован';
        else if (isNeutered === false) neuteredText = (petGender === 'Ж') ? 'Не стерилизована' : 'Не кастрирован';

        const isVaccinated = pet.isVaccinated;
        let vaccinatedText = '';
        if (isVaccinated === true) vaccinatedText = (petGender === 'Ж') ? 'Привита' : 'Привит';
        else if (isVaccinated === false) vaccinatedText = (petGender === 'Ж') ? 'Не привита' : 'Не привит';
        const healthStatusText = [neuteredText, vaccinatedText].filter(Boolean).join(', ');

        let ageString = 'Возраст не указан';
        if (pet.age !== null && pet.age !== undefined) {
            let yearsText = 'лет'; const age = Number(pet.age);
            if (!isNaN(age)) {
                 if (age === 1 || (age % 10 === 1 && age % 100 !== 11)) yearsText = 'год';
                 else if (([2, 3, 4].includes(age % 10)) && ![12, 13, 14].includes(age % 100)) yearsText = 'года';
                 if (age === 0) ageString = 'меньше года'; else ageString = `${age} ${yearsText}`;
            }
        }

        const annIdInt = parseInt(announcementData.id);
        // Определяем, является ли текущее объявление избранным
        // Используем options.isFavoritesScreen для экрана избранного ИЛИ проверяем в глобальном Set
        const isFav = options.isFavoritesScreen || favoriteAnnouncementIds.has(annIdInt);

        // Устанавливаем путь и alt текст для ЕДИНСТВЕННОЙ иконки
        const heartIconSrc = isFav ? 'icon/heart-filled.svg' : 'icon/heart-icon.svg';
        const heartIconAlt = isFav ? 'В избранном' : 'В избранное';

        card.innerHTML = `
            <img src="${imagePath}" alt="Фото ${petName}" class="pet-card-image" loading="lazy">
            <div class="pet-card-info">
                <h3 class="pet-card-name">${petName} <span class="pet-card-city">(${city})</span></h3>
                <p class="pet-card-details">Возраст: ${ageString}</p>
                 ${healthStatusText ? `<p class="pet-card-health-status">${healthStatusText}</p>` : ''}
                 <p class="pet-card-details"> Порода: ${petBreed}${ (pet.animal_type === 'Собака' && pet.size) ? `, Размер: ${pet.size}` : '' } </p>
                 ${petDescription ? `<p class="pet-card-description">${petDescription}</p>` : ''}
                 <button class="contact-button" data-announcement-id="${announcementData.id}" style="width: auto; padding: 6px 12px; font-size: 0.9em; margin-top: 10px;">
                    Связаться
                 </button>
            </div>
            <button class="favorite-button ${isFav ? 'is-favorite' : ''}" data-pet-id="${announcementData.id}">
                 <img src="${heartIconSrc}" alt="${heartIconAlt}" class="favorite-icon">
            </button>
        `;
        // --- КОНЕЦ ИЗМЕНЕНИЙ в innerHTML ---

        const favButton = card.querySelector('.favorite-button');
        if (favButton) {
            const annIdInt = parseInt(announcementData.id);
            if (options.isFavoritesScreen) {
                 favButton.classList.add('is-favorite');
            } else {
                 if (favoriteAnnouncementIds.has(annIdInt)) {
                     favButton.classList.add('is-favorite');
                 }
            }
            favButton.addEventListener('click', handleFavoriteClick); // Глобальная
        }
        return card;
    }

    function createMyPetCardElement(announcementData) {
        const card = document.createElement('div');
        card.className = 'my-pet-card';
        card.dataset.announcementId = announcementData.id;
        const pet = announcementData.pet || {};
        const city = announcementData.city || 'Город не указан';
        let ageString = '?';
        if (pet.age !== null && pet.age !== undefined) {
            let yearsText = 'лет';
            if (pet.age === 1) yearsText = 'год';
            else if ([2, 3, 4].includes(pet.age % 10) && ![12, 13, 14].includes(pet.age % 100)) yearsText = 'года';
            else if (pet.age === 0) yearsText = 'меньше года';
            if (pet.age > 0) ageString = `${pet.age} ${yearsText}`;
            else if (pet.age === 0) ageString = yearsText;
        }
        const petName = pet.name || 'Имя не указано';
        const petBreed = pet.breed || 'Не указана';
        const status = announcementData.status || 'Неизвестен';
        const petDescription = announcementData.description || '';

        const relativeImagePath = announcementData.image_path ? `/${announcementData.image_path.replace(/\\/g, '/')}` : null;
        const imagePath = relativeImagePath ? `${BACKEND_BASE_URL}${relativeImagePath}` : 'images/placeholder_pet.png';

        card.innerHTML = `
            <img src="${imagePath}" alt="Фото ${petName}" class="my-pet-card-image" loading="lazy">
            <div class="my-pet-card-info">
                <h3 class="my-pet-card-name">${petName} <span class="pet-card-city">(${city})</span></h3>
                <p class="my-pet-card-details">Возраст: ${ageString}, Порода: ${petBreed}</p>
                <p class="my-pet-card-details">Статус: ${status}</p>
                ${petDescription ? `<p class="my-pet-card-description">${petDescription}</p>` : ''}
            </div>
            <button type="button" class="my-pet-card-menu-toggle" aria-label="Меню объявления">
                <img src="icon/dots-vertical.svg" alt="Меню объявления">
            </button>
            <div class="my-pet-card-menu hidden">
                <button type="button" class="my-pet-menu-option edit-button" data-action="edit">Редактировать</button>
                <button type="button" class="my-pet-menu-option delete-button" data-action="delete">Удалить</button>
            </div>
        `;
        return card;
    }

    // --- Функции загрузки данных ---
    async function loadAndRenderResults() {
        const animalTypeRussian = (animalType === 'dog') ? 'Собака' : 'Кошка';
        console.log(`Загрузка/Рендеринг РЕЗУЛЬТАТОВ с API для: Тип=${animalType} (${animalTypeRussian}), Фильтры=`, currentFilters, `Сортировка=${currentSort}`);
        if (!resultsListContainer) { console.error("Контейнер #results-list не найден!"); return; }
        resultsListContainer.innerHTML = '<p>Загрузка питомцев...</p>';
        // isDataLoading = true;
        try {
            const params = new URLSearchParams();
            params.set('animal_type', animalTypeRussian);
            if (currentFilters.gender) params.set('gender', currentFilters.gender);
            if (currentFilters.age !== undefined && currentFilters.age !== '') params.set('age', currentFilters.age);
            if (currentFilters.breed) params.set('breed', currentFilters.breed);
            if (currentFilters.color) params.set('color', currentFilters.color);
            if (currentFilters.city) params.set('city', currentFilters.city);
            if (currentFilters.neuteredStatus) params.set('isNeutered', currentFilters.neuteredStatus === 'Да');
            if (currentFilters.vaccinatedStatus) params.set('isVaccinated', currentFilters.vaccinatedStatus === 'Да');
            if (currentFilters.keywords && currentFilters.keywords.length > 0) currentFilters.keywords.forEach(kw => params.append('keywords', kw));
            if (animalType === 'dog' && currentFilters.size) params.set('size', currentFilters.size); // Добавлено size

            let sortByValue;
            if (currentSort === 'oldest') sortByValue = 'timestamp_asc';
            else if (currentSort === 'distance') sortByValue = 'distance';
            else if (currentSort === 'relevance') sortByValue = 'relevance';
            else sortByValue = 'timestamp_desc'; // Default newest

            params.set('sort_by', sortByValue);
            params.set('skip', '0');
            params.set('limit', '50'); // Limit results

            const apiUrl = `/announcements?${params.toString()}`;
            console.log("Запрос к API:", apiUrl);
            const response = await fetchWithAuth(apiUrl); // Глобальная
            if (!response.ok) {
                let errorDetail = `Статус ошибки: ${response.status}`;
                try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (_e) {}
                throw new Error(errorDetail);
            }
            const announcementsFromApi = await response.json();
            resultsListContainer.innerHTML = ''; // Очистка перед рендерингом

            if (announcementsFromApi && announcementsFromApi.length > 0) {
                console.log(`Получено ${announcementsFromApi.length} объявлений с API.`);
                announcementsFromApi.forEach(ann => {
                    try { resultsListContainer.appendChild(createListPetCardElement(ann)); } // Локальная
                    catch(_e) { console.error("Ошибка при создании карточки для объявления:", ann, _e); }
                });
                if (!isCityFilterPopulated) { // Заполнение фильтра городов один раз
                    try {
                        const uniqueCities = getUniqueCities(announcementsFromApi); // Локальная
                        if (uniqueCities.length > 0) populateCityFilter(uniqueCities); // Локальная
                    } catch (_e) { console.error("Ошибка при заполнении фильтра городов из данных API:", _e); }
                }
            } else {
                console.log("С API не получено ни одного объявления по текущим фильтрам.");
                resultsListContainer.innerHTML = `<p>По вашему запросу (${animalTypeRussian}) с учетом фильтров ничего не найдено.</p>`;
            }
        } catch (error) {
            console.error("Ошибка при загрузке или отображении объявлений:", error);
            resultsListContainer.innerHTML = `<p>Не удалось загрузить питомцев. Ошибка: ${error.message || 'Неизвестная ошибка'}</p>`;
        } finally {
            // isDataLoading = false;
            console.log("Загрузка и рендеринг результатов завершены.");
        }
    }

    async function loadAndRenderFavorites() {
        console.log("Загрузка/Рендеринг ИЗБРАННОГО с API (/users/me/favorites)");
        const favoritesListContainer = document.getElementById('favorites-list'); // Используем другой ID
        if (!favoritesListContainer) { console.error("Контейнер #favorites-list не найден!"); return; }
        favoritesListContainer.innerHTML = '<p>Загрузка избранного...</p>';
        // isDataLoading = true;
        try {
            const apiUrl = '/users/me/favorites';
            console.log("Вызов fetchWithAuth для", apiUrl);
            const response = await fetchWithAuth(apiUrl); // Глобальная
            if (!response.ok) {
                let errorDetail = `Статус ошибки: ${response.status}`;
                try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (_e) { }
                throw new Error(errorDetail);
            }
            const favoriteAnnouncements = await response.json();
            favoritesListContainer.innerHTML = ''; // Очистка
            if (favoriteAnnouncements && favoriteAnnouncements.length > 0) {
                console.log(`Получено ${favoriteAnnouncements.length} избранных объявлений с API.`);
                favoriteAnnouncements.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)); // Сортировка по дате
                favoriteAnnouncements.forEach(ann => {
                    try { favoritesListContainer.appendChild(createListPetCardElement(ann, { isFavoritesScreen: true })); } // Локальная
                    catch (_e) { console.error("Ошибка при создании карточки избранного для объявления:", ann, _e); }
                });
            } else {
                console.log("С API не получено ни одного избранного объявления.");
                favoritesListContainer.innerHTML = '<p>Вы пока никого не добавили в избранное.</p>';
            }
        } catch (error) {
            console.error("Ошибка при загрузке или отображении избранного:", error);
            favoritesListContainer.innerHTML = `<p>Не удалось загрузить избранное. Ошибка: ${error.message || 'Неизвестная ошибка'}</p>`;
        } finally {
            // isDataLoading = false;
            console.log("Загрузка и рендеринг избранного завершены.");
        }
    }

    async function loadAndRenderMyPets() {
        console.log("Загрузка/Рендеринг МОИХ ОБЪЯВЛЕНИЙ с API");
        if (!myPetsListContainer) { console.error("Контейнер #my-pets-list не найден!"); return; }
        myPetsListContainer.innerHTML = '<p>Загрузка ваших объявлений...</p>';
        try {
            console.log("Вызов fetchWithAuth для /users/me/announcements");
            const response = await fetchWithAuth('/users/me/announcements'); // Глобальная
            if (!response.ok) {
                let errorDetail = `Статус ошибки: ${response.status}`;
                try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (_e) {}
                throw new Error(errorDetail);
            }
            const myAnnouncements = await response.json();
            myPetsListContainer.innerHTML = ''; // Очистка
            if (myAnnouncements && myAnnouncements.length > 0) {
                console.log(`Получено ${myAnnouncements.length} моих объявлений с API.`);
                myAnnouncements.forEach(ann => {
                    try { myPetsListContainer.appendChild(createMyPetCardElement(ann)); } // Локальная
                    catch(_e) { console.error("Ошибка при создании карточки 'моего' объявления:", ann, _e); }
                });
            } else {
                console.log("С API не получено ни одного 'моего' объявления.");
                myPetsListContainer.innerHTML = '<p>Вы еще не добавили ни одного объявления.</p>';
            }
        } catch (error) {
            console.error("Ошибка при загрузке или отображении 'моих' объявлений:", error);
            myPetsListContainer.innerHTML = `<p>Не удалось загрузить ваши объявления. Ошибка: ${error.message || 'Неизвестная ошибка'}</p>`;
        } finally {
            console.log("Загрузка и рендеринг 'моих' объявлений завершены.");
        }
    }

    // --- Вспомогательные функции (формы, города и т.д.) ---
     function getUniqueCities(announcements) {
         if (!Array.isArray(announcements)) return [];
         const cities = new Set();
         announcements.forEach(ann => {
             if (ann.city && typeof ann.city === 'string' && ann.city.trim() !== '') {
                 const formattedCity = ann.city.trim();
                 const capitalizedCity = formattedCity.charAt(0).toUpperCase() + formattedCity.slice(1).toLowerCase();
                 cities.add(capitalizedCity);
             }
         });
         return Array.from(cities).sort((a, b) => a.localeCompare(b, 'ru'));
     }

     function populateCityFilter(cities) {
         const citySelect = document.getElementById('city-filter');
         if (!citySelect) { console.error("Элемент <select> для городов не найден!"); return; }
         // Сохраняем текущее значение, если оно есть
         const currentValue = citySelect.value;
         // Очищаем старые опции (кроме первой "Любой город")
         while (citySelect.options.length > 1) citySelect.remove(1);
         // Добавляем новые
         cities.forEach(city => {
             const option = document.createElement('option');
             option.value = city;
             option.textContent = city;
             citySelect.appendChild(option);
         });
         // Восстанавливаем значение, если оно было среди новых опций
         if (cities.includes(currentValue)) {
             citySelect.value = currentValue;
         }
         console.log(`Фильтр городов заполнен ${cities.length} уникальными значениями.`);
         isCityFilterPopulated = true;
     }

     function updateAdFormHeader(){
         const adAnimalTypeSpan = document.getElementById('ad-form-animal-type');
         if(adAnimalTypeSpan) adAnimalTypeSpan.textContent = (animalType === 'dog' ? 'Собака' : 'Кошка');
     }



    function resetFilterForm(formContainer) {
    if (!formContainer) {
        console.warn("filtersContainer не найден, пропускаем сброс фильтров");
        return;
    }
    const genderRadio = formContainer.querySelector('input[name="gender-filter"][value=""]');
    if (genderRadio) genderRadio.checked = true;
    const ageInput = formContainer.querySelector('#age-filter');
    if (ageInput) ageInput.value = '';
    const breedInput = formContainer.querySelector('#breed-filter');
    if (breedInput) breedInput.value = '';
    const colorInput = formContainer.querySelector('#color-filter');
    if (colorInput) colorInput.value = '';
    const citySelect = formContainer.querySelector('#city-filter');
    if (citySelect) citySelect.value = '';
    const anyNeuteredRadio = formContainer.querySelector('input[name="neutered-filter"][value=""]');
    if (anyNeuteredRadio) anyNeuteredRadio.checked = true;
    const anyVaccinatedRadio = formContainer.querySelector('input[name="vaccinated-filter"][value=""]');
    if (anyVaccinatedRadio) anyVaccinatedRadio.checked = true;
    const anySizeRadio = formContainer.querySelector('input[name="size-filter"][value=""]');
    if (anySizeRadio) anySizeRadio.checked = true;
    const allKeywordButtons = formContainer.querySelectorAll('.keyword-tag-button');
    allKeywordButtons.forEach(button => button.classList.remove('active'));
}



    function resetAdFormFields(formContainer) {
        if (!formContainer) {
            console.warn("adFormContainer не найден, пропускаем сброс формы");
            return;
        }
        const titleInput = formContainer.querySelector('#ad-title');
        if (titleInput) titleInput.value = '';
        const descriptionInput = formContainer.querySelector('#ad-description');
        if (descriptionInput) descriptionInput.value = '';
        const citySelect = formContainer.querySelector('#city-ad');
        if (citySelect) citySelect.value = '';
        const genderRadio = formContainer.querySelector('input[name="gender"][value=""]');
        if (genderRadio) genderRadio.checked = true;
        const ageInput = formContainer.querySelector('#age');
        if (ageInput) ageInput.value = '';
        const breedInput = formContainer.querySelector('#breed');
        if (breedInput) breedInput.value = '';
        const colorInput = formContainer.querySelector('#color');
        if (colorInput) colorInput.value = '';
        const neuteredRadio = formContainer.querySelector('input[name="neutered"][value=""]');
        if (neuteredRadio) neuteredRadio.checked = true;
        const vaccinatedRadio = formContainer.querySelector('input[name="vaccinated"][value=""]');
        if (vaccinatedRadio) vaccinatedRadio.checked = true;
        const sizeRadio = formContainer.querySelector('input[name="size"][value=""]');
        if (sizeRadio) sizeRadio.checked = true;
        const photoInput = formContainer.querySelector('#photos');
        if (photoInput) photoInput.value = '';
        const previewContainer = formContainer.querySelector('#photo-preview');
        if (previewContainer) previewContainer.innerHTML = '';
        const keywordButtons = formContainer.querySelectorAll('.keyword-tag-button');
        keywordButtons.forEach(button => button.classList.remove('active'));
        const submitButton = formContainer.querySelector('#submit-ad-button');
        if (submitButton) submitButton.disabled = false;
    }

    function toggleDogSizeField() {
        const selectedType = document.querySelector('input[name="find-mate-animal-type"]:checked');
        if (findMateSizeGroup && selectedType) {
            if (selectedType.value === 'Собака') findMateSizeGroup.style.display = 'block';
            else findMateSizeGroup.style.display = 'none';
            // Сброс значения, если выбрана кошка
            if (selectedType.value !== 'Собака') {
                const defaultDogSize = findMateSizeGroup.querySelector('input[value="Средний"]');
                if (defaultDogSize) defaultDogSize.checked = true;
            }
        } else if(findMateSizeGroup) {
            findMateSizeGroup.style.display = 'none'; // Скрыть по умолчанию
        }
    }

    function resetFindMateForm() {
        if (findMateFormContainer) {
            const photoInput = findMateFormContainer.querySelector('#image-input-find-mate'); // НОВЫЙ ID
            const photoLabel = findMateFormContainer.querySelector('#upload-label-find-mate'); // Находим лейбл
            const photoPreview = findMateFormContainer.querySelector('#preview-find-mate'); // Находим превью
            if (photoInput) photoInput.value = ''; // Очищаем значение инпута
            if (photoLabel) photoLabel.classList.remove('has-file'); // Убираем класс с лейбла
            if (photoPreview) photoPreview.style.backgroundImage = 'none'; // Убираем фон превью

            const catRadio = findMateFormContainer.querySelector('input[name="find-mate-animal-type"][value="Кошка"]'); if (catRadio) catRadio.checked = true;
            const maleRadio = findMateFormContainer.querySelector('input[name="find-mate-gender"][value="М"]'); if (maleRadio) maleRadio.checked = true;
            findMateFormContainer.querySelector('#find-mate-nickname').value = '';
            findMateFormContainer.querySelector('#find-mate-age').value = '';
            findMateFormContainer.querySelector('#find-mate-breed').value = '';
            findMateFormContainer.querySelector('#find-mate-city').value = '';
            findMateFormContainer.querySelectorAll('input[name="find-mate-checkbox"]').forEach(cb => cb.checked = false);
            findMateFormContainer.querySelector('#find-mate-notes').value = '';
            toggleDogSizeField(); // Обновить видимость поля размера
            console.log("Форма 'Найти пару' сброшена.");
        } else console.warn("Контейнер формы 'Найти пару' не найден для сброса.");
    }

    // --- Функции редактирования/удаления ---
    function closeMyPetMenu() {
        if (currentlyOpenMenu) {
            currentlyOpenMenu.classList.add('hidden');
            currentlyOpenMenu = null;
        }
    }

    async function startEditAnnouncement(announcementId) {
        console.log(`Загрузка данных для редактирования объявления ID: ${announcementId}`);
        try {
            console.log(`Вызов fetchWithAuth для /announcements/${announcementId}`);
            const response = await fetchWithAuth(`/announcements/${announcementId}`); // Глобальная
            if (!response.ok) {
                let errorDetail = `Статус ошибки: ${response.status}`;
                try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (_e) {}
                throw new Error(`Ошибка загрузки данных: ${errorDetail}`);
            }
            const data = await response.json();
            console.log("Получены данные для редактирования с API:", data);
            if (!data || !data.pet) throw new Error("Получены некорректные данные с сервера.");

            const form = document.getElementById('ad-form-container'); // Используем adFormContainer, выбранный ранее
            if (!form) { console.error("Контейнер формы #ad-form-container не найден!"); return; }

            animalType = (data.pet.animal_type === 'Собака') ? 'dog' : 'cat';
            updateAdFormHeader(); // Локальная

            form.querySelector('#ad-pet-name').value = data.pet.name || '';
            const genderRadio = form.querySelector(`input[name="ad-pet-gender"][value="${data.pet.gender}"]`); if (genderRadio) genderRadio.checked = true;
            form.querySelector('#ad-pet-age').value = data.pet.age !== null ? data.pet.age : '';
            form.querySelector('#ad-pet-breed').value = data.pet.breed || '';
            form.querySelector('#ad-pet-color').value = data.pet.color || '';
            form.querySelector('#ad-pet-city').value = data.city || '';

            const neuteredValue = data.pet.isNeutered === true ? 'Да' : (data.pet.isNeutered === false ? 'Нет' : '');
            const vaccinatedValue = data.pet.isVaccinated === true ? 'Да' : (data.pet.isVaccinated === false ? 'Нет' : '');
            const neuteredRadio = form.querySelector(`input[name="ad-pet-neutered"][value="${neuteredValue}"]`); if (neuteredRadio) neuteredRadio.checked = true; else form.querySelector('input[name="ad-pet-neutered"][value="Нет"]').checked = true; // Default
            const vaccinatedRadio = form.querySelector(`input[name="ad-pet-vaccinated"][value="${vaccinatedValue}"]`); if (vaccinatedRadio) vaccinatedRadio.checked = true; else form.querySelector('input[name="ad-pet-vaccinated"][value="Да"]').checked = true; // Default

            const sizeFormGroup = form.querySelector('#ad-form-size-group');
            if (sizeFormGroup) {
                if (animalType === 'dog') {
                    sizeFormGroup.classList.remove('hidden');
                    const sizeValue = data.pet.size || "Средний"; // Default to "Средний" if null/undefined
                    const sizeRadio = form.querySelector(`input[name="ad-pet-size"][value="${sizeValue}"]`);
                    if (sizeRadio) sizeRadio.checked = true;
                    else form.querySelector('input[name="ad-pet-size"][value="Средний"]').checked = true; // Fallback
                } else {
                    sizeFormGroup.classList.add('hidden');
                    // Ensure a default is checked even if hidden, maybe not necessary
                    // form.querySelector('input[name="ad-pet-size"][value="Средний"]').checked = true;
                }
            }

            const currentKeywords = data.keywords ? data.keywords.toLowerCase().split(',').map(kw => kw.trim()) : [];
            // Находим все кнопки тегов ВНУТРИ формы
            form.querySelectorAll('#keywords-filter-tags .keyword-tag-button').forEach(button => {
                // Проверяем, есть ли значение кнопки в списке текущих ключевых слов
                const keywordValue = button.dataset.keywordValue;
                if (keywordValue && currentKeywords.includes(keywordValue)) {
                    button.classList.add('active'); // Делаем кнопку активной
                } else {
                    button.classList.remove('active'); // Делаем неактивной
                }
            });
            form.querySelector('#ad-description').value = data.description || '';

            // --- Обработчик кликов по тегам внутри ФОРМЫ ОБЪЯВЛЕНИЯ ---
            const adFormKeywordsContainer = document.querySelector('#ad-form-container #keywords-filter-tags');
            if (adFormKeywordsContainer) {
                adFormKeywordsContainer.addEventListener('click', (_e) => {
                    if (event.target.classList.contains('keyword-tag-button')) {
                        event.target.classList.toggle('active'); // Переключаем класс active
                    }
                });
            } else {
                console.warn("Контейнер тегов #keywords-filter-tags ВНУТРИ формы объявления не найден.");
            }

            const imageInput = form.querySelector('#ad-image-upload'); if (imageInput) imageInput.value = ''; // Clear file input

            const previewElement = form.querySelector('#preview-front');
            const labelElement = form.querySelector('#upload-label-front');

//            if (previewElement && labelElement) {
//                if (data.image_path) {
//                    const relativeImagePath = `/${data.image_path.replace(/\\/g, '/')}`;
//                    //const fullImageUrl = `<span class="math-inline">\{BACKEND\_BASE\_URL\}</span>{relativeImagePath}`;
//                    const fullImageUrl = `${BACKEND_BASE_URL}${relativeImagePath}`;
//                    previewElement.style.backgroundImage = `url('${fullImageUrl}')`; // Устанавливаем фон
//                    labelElement.classList.add('has-file'); // Показываем, что файл есть
//                    console.log("Превью текущего изображения установлено:", fullImageUrl);
//                } else {
//                    previewElement.style.backgroundImage = 'none'; // Убираем фон
//                    labelElement.classList.remove('has-file'); // Показываем, что файла нет
//                    console.log("Текущее изображение отсутствует, превью скрыто.");
//                }
//            }

            if (previewElement && labelElement) {
                if (data.image_path) {
                    const relativeImagePath = `/${data.image_path.replace(/\\/g, '/')}`;
                    const fullImageUrl = `${BACKEND_BASE_URL}${relativeImagePath}`;
                    previewElement.style.backgroundImage = `url('${fullImageUrl}')`;
                    labelElement.classList.add('has-file');
                    console.log("Превью текущего изображения установлено:", fullImageUrl);
                } else {
                    previewElement.style.backgroundImage = 'none';
                    labelElement.classList.remove('has-file');
                    console.log("Текущее изображение отсутствует, превью скрыто.");
                }
            }

            const submitButton = form.querySelector('#create-ad-button'); if (submitButton) submitButton.textContent = 'Сохранить изменения';
            form.dataset.mode = 'edit';
            form.dataset.editId = announcementId;
            showScreen('ad-form-screen'); // Глобальная

        } catch (error) {
            console.error("Ошибка при подготовке к редактированию:", error);
            alert(`Не удалось загрузить данные для редактирования: ${error.message}`);
        } finally {
            console.log("Подготовка формы к редактированию завершена.");
        }
    }

    async function deleteAnnouncement(announcementId, cardElement) {
        console.log(`>>> НАЧАЛО УДАЛЕНИЯ: Объявление ID: ${announcementId}`);
        console.log(`>>> Используемый authToken ПЕРЕД ЗАПРОСОМ УДАЛЕНИЯ:`, authToken);
        console.log(`Удаление объявления ID: ${announcementId} через API...`);

        const deleteBtn = cardElement?.querySelector('.delete-button');
        if (cardElement) cardElement.style.opacity = '0.5';
        if (deleteBtn) deleteBtn.disabled = true;

        try {
            const apiUrl = `/announcements/${announcementId}`;
            const response = await fetchWithAuth(apiUrl, { method: 'DELETE' }); // Глобальная

            if (response.status === 204) {
                if (cardElement) cardElement.remove();
                console.log(`Объявление ID: ${announcementId} успешно удалено.`);
                // Check if container is empty
                if (myPetsListContainer && myPetsListContainer.children.length === 0) {
                    myPetsListContainer.innerHTML = '<p>Вы еще не добавили ни одного объявления.</p>';
                }
            } else {
                let errorDetail = `Ошибка удаления: Статус ${response.status}`;
                try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; }
                catch (_e) { console.warn("Не удалось получить тело ошибки при удалении, используется статус."); }
                throw new Error(`${errorDetail} (Объявление ID: ${announcementId})`);
            }
        } catch (error) {
            console.error("Ошибка при удалении объявления:", error);
            alert(`Не удалось удалить объявление: ${error.message}`);
            if (cardElement) cardElement.style.opacity = '1';
            if (deleteBtn) deleteBtn.disabled = false;
        } finally {
             console.log(`>>> КОНЕЦ УДАЛЕНИЯ: Объявление ID: ${announcementId}`);
        }
    }

    // Функция для обработки выбора файла для одного ракурса (ваш код)
    const handleAngleFileSelect = (_e) => {
        const input = event.target;
        const angle = input.dataset.angle;
        const file = input.files[0];
        const label = angleLabels[angle]; // Используем объект angleLabels
        const preview = anglePreviews[angle]; // Используем объект anglePreviews

        console.log('Input:', input);
        console.log('Angle (из data-angle):', angle);
        console.log('Выбранный файл:', file);
        console.log('Найденный Label:', label);
        console.log('Найденный Preview:', preview);

        if (file && angle && label && preview) {
            selectedBreedFiles[angle] = file;
            console.log(`Файл для ракурса '${angle}' выбран:`, file.name);
            const reader = new FileReader();
            reader.onload = (_e) => {
                preview.style.backgroundImage = `url('${e.target.result}')`;
                label.classList.add('has-file');
            };
            reader.onerror = (_e) => {
                 console.error(`Ошибка чтения файла для превью (${angle}):`, _e);
                 alert(`Не удалось прочитать файл для предпросмотра.`);
                 selectedBreedFiles[angle] = null;
                 preview.style.backgroundImage = 'none';
                 label.classList.remove('has-file');
            };
            reader.readAsDataURL(file);
        } else {
            console.log(`Выбор файла для ракурса '${angle}' отменен или элементы не найдены.`);
            selectedBreedFiles[angle] = null;
            if(preview) preview.style.backgroundImage = 'none';
            if(label) label.classList.remove('has-file');
        }
    };


    // --- Фаза 3: Назначение Обработчиков Событий ---

    // --- Обработчики кнопок выбора роли ---
    if (findPetButton) findPetButton.addEventListener('click', () => { userRole = 'find'; showScreen('animal-type-selection-find-screen'); });
    if (placePetButton) placePetButton.addEventListener('click', () => { userRole = 'place'; showScreen('animal-type-selection-place-screen'); });
    if (findMateButton) { findMateButton.addEventListener('click', () => { console.log("Кнопка 'Найти пару' (на экране ролей) нажата!"); resetFindMateForm(); showScreen('find-mate-form-screen'); toggleDogSizeField(); }); } else { console.error("Кнопка #find-mate-button не найдена!"); }
    if (identifyBreedButton) { identifyBreedButton.addEventListener('click', () => { // Сброс состояния экрана определения породы
        selectedBreedFiles = { front: null, side: null, top: null }; // Сброс выбранных файлов
        if (angleLabels) Object.values(angleLabels).forEach(lbl => lbl?.classList.remove('has-file'));
        if (anglePreviews) Object.values(anglePreviews).forEach(prv => { if (prv) prv.style.backgroundImage = 'none'; });
        if (breedResultOutputMulti) breedResultOutputMulti.innerHTML = ''; // Очистка результата
        // Очистка file inputs (опционально, но полезно)
        if(frontInput) frontInput.value = ''; if(sideInput) sideInput.value = ''; if(topInput) topInput.value = '';
        showScreen('identify-breed-screen');
    }); }

    // --- Обработчики НОВЫХ кнопок на экране выбора роли ---
    if (myPetsNavButton) {
        myPetsNavButton.addEventListener('click', () => {
            if (!isAuthenticated) { alert("Пожалуйста, войдите в систему, чтобы просмотреть свои объявления."); return; }
            showScreen('my-pets-screen');
            loadAndRenderMyPets();
        });
    } else { console.error("Кнопка #my-pets-nav-button не найдена!"); }

    if (favoritesNavButton) {
        favoritesNavButton.addEventListener('click', () => {
             if (!isAuthenticated) { alert("Пожалуйста, войдите в систему, чтобы просмотреть избранное."); return; }
            showScreen('favorites-screen');
            loadAndRenderFavorites();
        });
    } else { console.error("Кнопка #favorites-nav-button не найдена!"); }

    // --- Обработчик для кнопки "Связаться" (Делегирование) ---
     if (resultsListContainer) { // Используем resultsListContainer, не resultsContainerForDelegation
         resultsListContainer.addEventListener('click', async (_e) => {
             if (event.target.classList.contains('contact-button')) {
                 const button = event.target;
                 const announcementId = button.dataset.announcementId;
                 if (!announcementId) { console.error("Не удалось получить ID объявления из data-announcement-id"); alert("Произошла ошибка, не удалось определить объявление."); return; }
                 if (!isAuthenticated || !authToken) { alert("Пожалуйста, войдите в систему, чтобы связаться с владельцем."); return; }
                 console.log(`Нажата кнопка 'Связаться' для объявления ID: ${announcementId}`);
                 button.disabled = true; button.textContent = 'Отправка...';
                 const apiUrl = `/announcements/${announcementId}/request_contact`;
                 try {
                     const response = await fetchWithAuth(apiUrl, { method: 'POST' }); // Глобальная
                     if (response.ok) {
                         const result = await response.json();
                         alert(result.message || "Запрос успешно отправлен!");
                         console.log("Запрос на контакт успешно отправлен.");
                     } else {
                         let errorDetail = `Ошибка ${response.status}`;
                         try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (_e) { }
                         console.error(`Ошибка при отправке запроса на контакт: ${errorDetail}`);
                         alert(`Не удалось отправить запрос: ${errorDetail}`);
                     }
                 } catch (error) {
                     console.error("Сетевая ошибка или другая проблема при запросе на контакт:", error);
                     alert(`Произошла ошибка сети при отправке запроса: ${error.message}`);
                 } finally {
                     button.disabled = false; button.textContent = 'Связаться';
                 }
             }
         });
     } else { console.warn("Контейнер #results-list не найден для делегирования событий кнопки 'Связаться'"); }

    // --- УНИВЕРСАЛЬНЫЙ ОБРАБОТЧИК ПРЕВЬЮ через делегирование ---
    document.body.addEventListener('change', (_e) => {
        // Проверяем, что событие произошло на input[type=file]
        // И что этот input находится внутри элемента с классом 'angle-upload-button'
        if (event.target.type === 'file' && event.target.closest('.angle-upload-button')) {
            // Если да, вызываем нашу универсальную функцию
            handleUniversalImagePreview(_e);
        }
    });
    console.log("Универсальный обработчик превью изображений назначен через делегирование.");


    /**
     * Универсальная функция для обработки выбора файла изображения,
     * отображения превью И СОХРАНЕНИЯ ФАЙЛА в selectedBreedFiles.
     */
    function handleUniversalImagePreview(_e) {
        const imageInput = event.target;
        const file = imageInput.files[0];
        const label = imageInput.closest('.angle-upload-button');
        const previewElement = label ? label.querySelector('.angle-upload-preview') : null;
        const angle = imageInput.dataset.angle;

        if (!label || !previewElement) {
            console.error("Не найдены элементы превью или label для", imageInput);
            return;
        }

        previewElement.style.backgroundImage = 'none';
        label.classList.remove('has-file');

        if (angle && Object.prototype.hasOwnProperty.call(selectedBreedFiles, angle)) {
            selectedBreedFiles[angle] = null;
        }

        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewElement.style.backgroundImage = `url('${e.target.result}')`;
                label.classList.add('has-file');
                if (angle && Object.prototype.hasOwnProperty.call(selectedBreedFiles, angle)) {
                    selectedBreedFiles[angle] = file;
                    console.log('Обновлен selectedBreedFiles:', selectedBreedFiles);
                }
            };
            reader.onerror = () => { // Исправлено: убрали параметр e
                console.error("Ошибка чтения файла для превью.");
                if (angle && Object.prototype.hasOwnProperty.call(selectedBreedFiles, angle)) {
                    selectedBreedFiles[angle] = null;
                }
            };
            reader.readAsDataURL(file);
        } else if (file) {
            console.warn("Выбранный файл не является изображением:", file.type);
            imageInput.value = '';
        }
    }

    function handleAnimalTypeSelectionFind(type) {
        animalType = type; currentFilters = {}; currentSort = 'newest';
        const sizeFilterGroup = document.getElementById('filter-form-size-group');
        if (sizeFilterGroup) {
            if (animalType === 'dog') sizeFilterGroup.classList.remove('hidden');
            else {
                sizeFilterGroup.classList.add('hidden');
                const anySizeRadio = filtersContainer?.querySelector('input[name="size-filter"][value=""]');
                if (anySizeRadio) anySizeRadio.checked = true;
            }
        }
        showScreen('results-screen'); // Сначала показываем экран
        if (filtersContainer) resetFilterForm(filtersContainer); // Затем сбрасываем фильтры
        loadAndRenderResults();
    }
    if (selectDogButtonFind) selectDogButtonFind.addEventListener('click', () => handleAnimalTypeSelectionFind('dog'));
    if (selectCatButtonFind) selectCatButtonFind.addEventListener('click', () => handleAnimalTypeSelectionFind('cat'));
    if (selectDogButtonPlace) selectDogButtonPlace.addEventListener('click', () => { animalType = 'dog'; resetAdFormFields(adFormContainer); document.getElementById('ad-form-size-group')?.classList.remove('hidden'); showScreen('ad-form-screen'); updateAdFormHeader(); }); // Используются локальные/глобальные
    if (selectCatButtonPlace) selectCatButtonPlace.addEventListener('click', () => { animalType = 'cat'; resetAdFormFields(adFormContainer); document.getElementById('ad-form-size-group')?.classList.add('hidden'); showScreen('ad-form-screen'); updateAdFormHeader(); }); // Используются локальные/глобальные

    // --- Обработчики футера экрана результатов ---
    if (filterButton) filterButton.addEventListener('click', () => { showScreen('filter-screen'); });
    if (favoritesButton) favoritesButton.addEventListener('click', () => {
        if (!isAuthenticated) { alert("Пожалуйста, войдите в систему, чтобы просмотреть избранное."); return; }
        showScreen('favorites-screen');
        loadAndRenderFavorites(); // Локальная
    });
    if (sortButton && sortMenu) {
        sortButton.addEventListener('click', (_e) => {
            event.stopPropagation();
            console.log("Нажата кнопка сортировки!");
            if (sortMenu.classList.contains('hidden')) { // Обновляем активную опцию только при открытии
                const options = sortMenu.querySelectorAll('.sort-option');
                options.forEach(option => {
                    option.classList.toggle('active', option.dataset.sortValue === currentSort);
                });
            }
            sortMenu.classList.toggle('hidden');
        });
    }
    if (sortMenu) {
        sortMenu.addEventListener('click', (_e) => {
            const targetOption = event.target.closest('.sort-option');
            if (!targetOption) return;
            const newSortValue = targetOption.dataset.sortValue;
            console.log(`Выбрана опция сортировки: ${newSortValue}`);
            if (newSortValue && newSortValue !== currentSort) {
                currentSort = newSortValue;
                // Обновляем результаты, если мы на экране результатов
                const currentScreen = document.querySelector('.screen:not(.hidden)');
                if (currentScreen?.id === 'results-screen') loadAndRenderResults(); // Локальная
            }
            sortMenu.classList.add('hidden'); // Закрываем меню
        });
    }

    // --- Обработчики экрана фильтров ---
     if (applyFiltersButton && filtersContainer) {
         applyFiltersButton.addEventListener('click', () => {
             currentFilters = {}; // Начинаем с чистого объекта
             const genderSelected = filtersContainer.querySelector('input[name="gender-filter"]:checked'); if (genderSelected && genderSelected.value) currentFilters.gender = genderSelected.value;
             const ageInput = filtersContainer.querySelector('#age-filter'); if (ageInput && ageInput.value !== '') currentFilters.age = ageInput.value;
             const breedInput = filtersContainer.querySelector('#breed-filter'); if (breedInput && breedInput.value.trim() !== '') currentFilters.breed = breedInput.value.trim();
             const colorInput = filtersContainer.querySelector('#color-filter'); if (colorInput && colorInput.value.trim() !== '') currentFilters.color = colorInput.value.trim();
             const citySelect = filtersContainer.querySelector('#city-filter'); if (citySelect && citySelect.value) currentFilters.city = citySelect.value;
             const neuteredSelected = filtersContainer.querySelector('input[name="neutered-filter"]:checked'); if (neuteredSelected && neuteredSelected.value) currentFilters.neuteredStatus = neuteredSelected.value; // Store 'Да'/'Нет'
             const vaccinatedSelected = filtersContainer.querySelector('input[name="vaccinated-filter"]:checked'); if (vaccinatedSelected && vaccinatedSelected.value) currentFilters.vaccinatedStatus = vaccinatedSelected.value; // Store 'Да'/'Нет'

             // Удаляем 'size' и добавляем только если выбрана собака и есть значение
             delete currentFilters.size;
             if (animalType === 'dog') {
                 const sizeSelected = filtersContainer.querySelector('input[name="size-filter"]:checked');
                 if (sizeSelected && sizeSelected.value) {
                     currentFilters.size = sizeSelected.value;
                 }
             }

             const selectedKeywordButtons = filtersContainer.querySelectorAll('.keyword-tag-button.active');
             const selectedKeywords = Array.from(selectedKeywordButtons).map(button => button.dataset.keywordValue);
             if (selectedKeywords.length > 0) currentFilters.keywords = selectedKeywords;

             console.log("Применяемые фильтры:", currentFilters);
             showScreen('results-screen'); // Глобальная
             loadAndRenderResults(); // Локальная
         });
     }
     if (resetFiltersButton && filtersContainer) {
         resetFiltersButton.addEventListener('click', () => {
             currentFilters = {};
             resetFilterForm(filtersContainer); // Локальная
             console.log("Фильтры сброшены");
             showScreen('results-screen'); // Глобальная
             loadAndRenderResults(); // Локальная
         });
     }
     if (keywordsTagsContainer) {
         keywordsTagsContainer.addEventListener('click', (_e) => {
             if (event.target.classList.contains('keyword-tag-button')) {
                 event.target.classList.toggle('active');
             }
         });
     } else console.warn("Контейнер для тегов #keywords-filter-tags не найден.");

    // --- Обработчик формы создания/редактирования объявления ---
    if (createAdButton && adFormContainer) {
         createAdButton.addEventListener('click', async () => {
             const editId = adFormContainer.dataset.editId;

             const formMode = adFormContainer.dataset.mode;
             console.log(`--- DEBUG: Клик по кнопке Сохранить/Создать --- Режим: '${formMode}', ID Редактирования: ${editId}`);
             createAdButton.disabled = true; createAdButton.textContent = 'Сохранение...';

             // Сбор данных из формы
             const petName = adFormContainer.querySelector('#ad-pet-name').value.trim();
             const petGenderSelected = adFormContainer.querySelector('input[name="ad-pet-gender"]:checked'); const petGender = petGenderSelected ? petGenderSelected.value : null;
             const petAgeInput = adFormContainer.querySelector('#ad-pet-age').value; const petAge = (petAgeInput !== '') ? parseInt(petAgeInput, 10) : null; // Преобразуем в число
             const petBreed = adFormContainer.querySelector('#ad-pet-breed').value.trim();
             const petColor = adFormContainer.querySelector('#ad-pet-color').value.trim();
             const petCityInput = adFormContainer.querySelector('#ad-pet-city'); const petCity = petCityInput ? petCityInput.value.trim() : null;
             const petNeuteredSelected = adFormContainer.querySelector('input[name="ad-pet-neutered"]:checked'); const isNeuteredStr = petNeuteredSelected ? petNeuteredSelected.value : null; // Оставляем Да/Нет
             const petVaccinatedSelected = adFormContainer.querySelector('input[name="ad-pet-vaccinated"]:checked'); const isVaccinatedStr = petVaccinatedSelected ? petVaccinatedSelected.value : null; // Оставляем Да/Нет
             const petSizeSelected = adFormContainer.querySelector('input[name="ad-pet-size"]:checked'); const petSize = (animalType === 'dog' && petSizeSelected) ? petSizeSelected.value : null; // Только для собак
             const description = adFormContainer.querySelector('#ad-description').value.trim();
             const imageInput = adFormContainer.querySelector('#image-input-front'); // Используем ID или другой уникальный селектор ВНУТРИ ФОРМЫ
             const imageFile = imageInput?.files[0];
             console.log("Файл из формы объявления для отправки:", imageFile); // Отладка
             // Валидация файла (обязателен при создании)

             if (formMode !== 'edit' && !imageFile) {
                 alert("Фотография обязательна при создании"); // Используем ваш метод показа ошибок
                 // Не забыть разблокировать кнопку и выйти
                 createAdButton.disabled = false; createAdButton.textContent = 'Создать объявление';
                 return;
             }
             const selectedKeywordButtons = adFormContainer.querySelectorAll('#keywords-filter-tags .keyword-tag-button.active');
             const selectedKeywords = Array.from(selectedKeywordButtons).map(button => button.dataset.keywordValue);
             const keywordsString = selectedKeywords.join(', ');

             // Клиентская валидация
             let clientErrors = [];
             const consentCheckbox = adFormContainer.querySelector('#ad-consent-checkbox'); // Получаем чекбокс по его ID

             if (!consentCheckbox || !consentCheckbox.checked) clientErrors.push("Необходимо согласие на получение запросов");
             if (formMode !== 'edit' && !imageFile) clientErrors.push("Фотография обязательна при создании");
             if (!petGender) clientErrors.push("Пол обязателен");
             if (!description) clientErrors.push("Описание обязательно");
             if (!animalType) clientErrors.push("Тип животного не выбран (ошибка)");
             if (isNeuteredStr === null) clientErrors.push("Укажите статус стерилизации");
             if (isVaccinatedStr === null) clientErrors.push("Укажите статус прививок");
             if (petAge !== null && isNaN(petAge)) clientErrors.push("Возраст должен быть числом"); // Проверка NaN для возраста

             if (clientErrors.length > 0) {
                 alert("Пожалуйста, заполните поля:\n- " + clientErrors.join("\n- "));
                 createAdButton.disabled = false; createAdButton.textContent = (formMode === 'edit') ? 'Сохранить изменения' : 'Создать объявление';
                 return;
             }

             // Создание FormData
             const formData = new FormData();
             formData.append('animal_type', animalType === 'dog' ? 'Собака' : 'Кошка');
             if (petName) formData.append('name', petName); // Отправляем только если есть
             formData.append('gender', petGender);
             if (petAge !== null) formData.append('age', petAge); // Отправляем только если есть
             if (petBreed) formData.append('breed', petBreed);
             if (petColor) formData.append('color', petColor);
             if (petCity) formData.append('city', petCity);
             formData.append('isNeutered', isNeuteredStr); // Отправляем Да/Нет
             formData.append('isVaccinated', isVaccinatedStr); // Отправляем Да/Нет
             if (petSize) formData.append('size', petSize);
             if (keywordsString) formData.append('keywords', keywordsString);
             formData.append('description', description);
             if (imageFile) {
                 formData.append('image', imageFile, imageFile.name);
                 console.log("Новый файл изображения будет отправлен.");
             } else {
                 console.log("Новый файл изображения НЕ выбран (при редактировании).");
             }
             // Согласие на бэкенде не обрабатывается явно, но проверено на фронте

             // Отправка запроса
             try {
                 let response;
                 let apiUrl;
                 let method;

                 if (formMode === 'edit' && editId) {
                     method = 'PUT';
                     apiUrl = `/announcements/${editId}`;
                     console.log(`Отправка FormData на ${apiUrl} (${method})`);
                     response = await fetchWithAuth(apiUrl, { method: method, body: formData }); // Глобальная
                 } else {
                     method = 'POST';
                     // Убедитесь, что этот эндпоинт существует и принимает такие данные
                     apiUrl = '/create_announcement'; // Имя эндпоинта в classify.py
                     console.log(`Отправка FormData на ${apiUrl} (${method})`);
                     response = await fetchWithAuth(apiUrl, { method: method, body: formData }); // Глобальная
                 }

                 if (response.ok) {
                     const resultData = await response.json();
                     console.log(`Успех (${formMode === 'edit' ? 'Редактирование' : 'Создание'}):`, resultData);
                     alert(`Объявление успешно ${formMode === 'edit' ? 'обновлено' : 'создано'}!`);
                     resetAdFormFields(adFormContainer); // Локальная
                     showScreen('my-pets-screen'); // Глобальная
                     loadAndRenderMyPets(); // Локальная
                 } else {
                     let errorDetail = `Ошибка сервера: ${response.status}`;
                     try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (_e) {}
                     console.error(`Ошибка при ${formMode === 'edit' ? 'редактировании' : 'создании'} объявления:`, errorDetail);
                     alert(`Не удалось ${formMode === 'edit' ? 'сохранить изменения' : 'создать объявление'}: ${errorDetail}`);
                 }
             } catch (error) {
                 console.error(`Критическая ошибка при ${formMode === 'edit' ? 'редактировании' : 'создании'} :`, error);
                 alert(`Произошла ошибка при отправке данных: ${error.message}`);
             } finally {
                 createAdButton.disabled = false;
                 // Сброс текста кнопки только если мы НЕ в режиме редактирования (т.к. при ошибке лучше оставить 'Сохранить изменения')
                 if (formMode !== 'edit') {
                     createAdButton.textContent = 'Создать объявление';
                 } else {
                      createAdButton.textContent = 'Сохранить изменения';
                 }
                 // Сбрасывать mode/editId при ошибке не стоит, чтобы пользователь мог попробовать снова
             }
         });
     }

    // --- Обработчик формы "Найти пару" ---
    if (findMateSubmitButton && findMateFormContainer) {
        findMateSubmitButton.addEventListener('click', async () => {
            console.log("Нажата кнопка 'Найти пару'");
            const photoInput = findMateFormContainer.querySelector('#image-input-find-mate');
            const photoFile = photoInput?.files[0];
            console.log("Файл из формы 'Найти пару':", photoFile);
            const animalTypeSelected = findMateFormContainer.querySelector('input[name="find-mate-animal-type"]:checked');
            const animalTypeValue = animalTypeSelected ? animalTypeSelected.value : null;
            const genderSelected = findMateFormContainer.querySelector('input[name="find-mate-gender"]:checked');
            const genderValue = genderSelected ? genderSelected.value : null;
            const nicknameValue = findMateFormContainer.querySelector('#find-mate-nickname').value.trim();
            const ageValue = findMateFormContainer.querySelector('#find-mate-age').value;
            const breedValue = findMateFormContainer.querySelector('#find-mate-breed').value.trim();
            const sizeSelected = findMateFormContainer.querySelector('input[name="find-mate-size"]:checked');
            const sizeValue = (animalTypeValue === 'Собака' && sizeSelected) ? sizeSelected.value : null;
            const cityValue = findMateFormContainer.querySelector('#find-mate-city').value.trim();

            let clientErrors = [];
            if (!photoFile) {
                clientErrors.push("Фотография обязательна");
            }
            if (!animalTypeValue) clientErrors.push("Тип животного");
            if (!genderValue) clientErrors.push("Пол");
            if (!ageValue || parseInt(ageValue) < 1) clientErrors.push("Возраст (минимум 1 год)");
            if (!breedValue) clientErrors.push("Порода");

            if (clientErrors.length > 0) {
                alert("Пожалуйста, заполните обязательные поля: " + clientErrors.join(', '));
                return;
            }

            findMateSubmitButton.disabled = true;
            findMateSubmitButton.textContent = 'Поиск...';

            const formData = new FormData();
            formData.append('image', photoFile, photoFile.name); // Оставляем только одно добавление
            formData.append('animal_type', animalTypeValue);
            formData.append('gender', genderValue);
            formData.append('age', ageValue);
            formData.append('breed', breedValue);
            if (nicknameValue) formData.append('name', nicknameValue);
            if (sizeValue) formData.append('size', sizeValue);
            if (cityValue) formData.append('city', cityValue);

            const apiUrl = '/announcements/pets/find_mate';
            try {
                console.log(`Отправка FormData на ${apiUrl} (POST)`);
                const response = await fetchWithAuth(apiUrl, { method: 'POST', body: formData });
                // ... остальной код ...
            } catch (error) {
                console.error("Критическая ошибка при поиске пары:", error);
                alert(`Произошла ошибка при отправке данных: ${error.message}`);
            } finally {
                findMateSubmitButton.disabled = false;
                findMateSubmitButton.textContent = 'Найти пару';
            }
        });
    }

    // --- Обработчики меню на карточках "Мои объявления" (Делегирование) ---
     if (myPetsListContainer) {
         myPetsListContainer.addEventListener('click', async (_e) => {
             const toggleButton = event.target.closest('.my-pet-card-menu-toggle');
             const editButton = event.target.closest('.edit-button');
             const deleteButton = event.target.closest('.delete-button');

             if (toggleButton) {
                 event.stopPropagation();
                 const card = toggleButton.closest('.my-pet-card');
                 const menu = card?.querySelector('.my-pet-card-menu');
                 if (!menu) return;
                 if (menu === currentlyOpenMenu) closeMyPetMenu(); // Локальная
                 else { closeMyPetMenu(); menu.classList.remove('hidden'); currentlyOpenMenu = menu; } // Используем глобальную currentlyOpenMenu
             } else if (editButton) {
                 const card = editButton.closest('.my-pet-card');
                 const announcementId = card?.dataset.announcementId;
                 if (announcementId) {
                     console.log(`Нажато 'Редактировать' для ID: ${announcementId}`);
                     closeMyPetMenu(); // Локальная
                     await startEditAnnouncement(announcementId); // Локальная
                 }
             } else if (deleteButton) {
                 const card = deleteButton.closest('.my-pet-card');
                 const announcementId = card?.dataset.announcementId;
                 if (announcementId) {
                     console.log(`Нажато 'Удалить' для ID: ${announcementId}`);
                     closeMyPetMenu(); // Локальная
                     if (confirm('Вы уверены?')) await deleteAnnouncement(announcementId, card); // Локальная
                 }
             }
         });
     }

    document.getElementById('view-toggle-button-results').addEventListener('click', function () {
        const resultsList = document.getElementById('results-list');
        const viewMode = this.getAttribute('data-view-mode');
        const listIcon = this.querySelector('.view-icon-list');
        const gridIcon = this.querySelector('.view-icon-grid');

        if (viewMode === 'list') {
            // Switch to grid view
            resultsList.classList.remove('view-mode-list');
            resultsList.classList.add('view-mode-grid');
            this.setAttribute('data-view-mode', 'grid');
            listIcon.classList.add('hidden');
            gridIcon.classList.remove('hidden');
        } else {
            // Switch to list view
            resultsList.classList.remove('view-mode-grid');
            resultsList.classList.add('view-mode-list');
            this.setAttribute('data-view-mode', 'list');
            listIcon.classList.remove('hidden');
            gridIcon.classList.add('hidden');
        }
    });

    // --- Обработчик для Body (Закрытие меню по клику вне) ---
    document.body.addEventListener('click', (_e) => {
        const hamburgerClicked = event.target.closest('.hamburger-button');
        const overlayClicked = event.target.closest('#menu-overlay');
        const sortMenuElement = document.getElementById('sort-menu'); // Элемент выбран ранее
        const sortButtonElement = document.getElementById('sort-button'); // Элемент выбран ранее
        // const mainMenuElement = document.getElementById('main-menu'); // Элемент выбран ранее

        if (hamburgerClicked) {
            console.log("Нажата кнопка гамбургера (делегирование)");
            toggleMenu(hamburgerClicked); // Глобальная
        } else if (overlayClicked && isMenuOpen) { // Используем глобальную isMenuOpen
            console.log("Нажат оверлей, закрываем меню");
            const visibleHamburger = document.querySelector('.hamburger-button:not([style*="display: none"])');
            closeMenu(visibleHamburger); // Глобальная
        }

        // Закрытие меню карточки "Мои объявления"
        if (currentlyOpenMenu && !event.target.closest('.my-pet-card-menu')) { // Проверяем клик вне самого меню
             if (!event.target.closest('.my-pet-card-menu-toggle')) { // И вне кнопки открытия
                 closeMyPetMenu(); // Локальная
             }
        }
        // Закрытие меню сортировки
        if (sortMenuElement && !sortMenuElement.classList.contains('hidden')) { // Проверяем, что меню видимо
             if (!sortMenuElement.contains(event.target) && !sortButtonElement?.contains(event.target)) { // Клик вне меню и вне кнопки
                 sortMenuElement.classList.add('hidden');
             }
        }
    });

    // --- Обработчик кликов внутри основного меню ---
    if (mainMenu) {
         mainMenu.addEventListener('click', (_e) => {
             const target = event.target.closest('.menu-item');
             if (!target) return;
             const action = target.dataset.action;
             console.log("Menu action clicked:", action);
             const visibleHamburger = document.querySelector('.hamburger-button:not([style*="display: none"])');
             closeMenu(visibleHamburger); // Глобальная
             // Задержка для анимации закрытия меню
             setTimeout(() => {
                 switch (action) {
                      case 'find': userRole = 'find'; showScreen('animal-type-selection-find-screen'); break;
                      case 'place': userRole = 'place'; resetAdFormFields(adFormContainer); showScreen('animal-type-selection-place-screen'); break; // Локальная
                      case 'my-pets': showScreen('my-pets-screen'); loadAndRenderMyPets(); break; // Локальная
                      case 'favorites': showScreen('favorites-screen'); loadAndRenderFavorites(); break; // Локальная
                      case 'find-mate': console.log("Переход на форму 'Найти пару' из меню"); resetFindMateForm(); showScreen('find-mate-form-screen'); toggleDogSizeField(); break; // Локальные
                      case 'identify-breed':
                          // Сброс состояния экрана определения породы при переходе из меню
                          selectedBreedFiles = { front: null, side: null, top: null };
                          if (angleLabels) Object.values(angleLabels).forEach(lbl => lbl?.classList.remove('has-file'));
                          if (anglePreviews) Object.values(anglePreviews).forEach(prv => { if (prv) prv.style.backgroundImage = 'none'; });
                          if (breedResultOutputMulti) breedResultOutputMulti.innerHTML = '';
                          if(frontInput) frontInput.value = ''; if(sideInput) sideInput.value = ''; if(topInput) topInput.value = '';
                          showScreen('identify-breed-screen'); // Глобальная
                          break;
                      case 'feedback': window.open('https://t.me/telegram111', '_blank'); break;
                      default: console.warn("Unknown menu action:", action); showScreen('role-selection-screen'); // Глобальная
                 }
             }, 150); // Немного увеличил задержку
         });
     }

     // --- Обработчики кнопок "Назад" ---
    document.querySelectorAll('.back-button').forEach(button => {
         button.addEventListener('click', (_e) => {
             const buttonElement = event.currentTarget;
             const targetScreenId = buttonElement.dataset.target;
             if (!targetScreenId) return;
             // Сброс состояния при возврате на ключевые экраны
             if (targetScreenId === 'role-selection-screen') {
                 userRole = null; animalType = null; currentFilters = {}; currentSort = 'newest';
                 if (filtersContainer) resetFilterForm(filtersContainer); // Локальная
                 resetAdFormFields(adFormContainer); // Локальная
                 resetFindMateForm(); // Локальная
             }
             if (targetScreenId === 'animal-type-selection-find-screen' || targetScreenId === 'animal-type-selection-place-screen') {
                 animalType = null; currentFilters = {}; currentSort = 'newest';
                 if (filtersContainer) resetFilterForm(filtersContainer); // Локальная
             }
             if (targetScreenId === 'animal-type-selection-place-screen') {
                 resetAdFormFields(adFormContainer); // Локальная
             }
             // Переход на экран
             showScreen(targetScreenId); // Глобальная
             // Перезагрузка данных, если нужно (только для списков)
             if (targetScreenId === 'results-screen') loadAndRenderResults(); // Локальная
             if (targetScreenId === 'favorites-screen') loadAndRenderFavorites(); // Локальная
             if (targetScreenId === 'my-pets-screen') loadAndRenderMyPets(); // Локальная
         });
     });

    // Обработчик для кнопки submit
    if (submitBreedMultiButton && breedResultOutputMulti) {
        submitBreedMultiButton.addEventListener('click', async () => {
            console.log("Нажата кнопка 'Определить породу' (мульти-фото)");
            if (!selectedBreedFiles.front || !selectedBreedFiles.side || !selectedBreedFiles.top) {
                alert("Пожалуйста, загрузите фотографии со всех трех ракурсов (спереди, сбоку, сверху).");
                return;
            }
            submitBreedMultiButton.disabled = true; submitBreedMultiButton.textContent = 'Определение...';
            breedResultOutputMulti.innerHTML = '<p>Анализ изображений...</p>';
            const formData = new FormData();
            formData.append('image_front', selectedBreedFiles.front, selectedBreedFiles.front.name);
            formData.append('image_side', selectedBreedFiles.side, selectedBreedFiles.side.name);
            formData.append('image_top', selectedBreedFiles.top, selectedBreedFiles.top.name);
            //const apiUrl = `${BACKEND_BASE_URL}/identify_breed_multi`;
            const apiPath = '/identify_breed_multi';
            console.log('Токен ПЕРЕД запросом /identify_breed_multi:', authToken);

            try {
                // --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
                // Используем fetchWithAuth, чтобы добавить токен авторизации
                // const response = await fetchWithAuth(apiUrl, { method: 'POST', body: formData });
                const response = await fetchWithAuth(apiPath, { method: 'POST', body: formData });
                // --- КОНЕЦ ИЗМЕНЕНИЯ ---

                if (response.ok) {
                    // ... (обработка успешного ответа) ...
                     const result = await response.json();
                     console.log("Получен результат определения (мульти-фото):", result);
                     let resultHtml = '<p>Результат определения:</p>';
                     if (result && result.animal_type) {
                         resultHtml += `<p><strong>Тип:</strong> ${result.animal_type}</p>`;
                         resultHtml += `<p><strong>Порода:</strong> ${result.breed || 'Не определена'}</p>`;
                         if (result.recommendations) {
                             const formattedRecommendations = result.recommendations.replace(/\n/g, '<br>');
                             resultHtml += `<div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #eee; text-align: left;">`;
                             resultHtml += `<p><strong>Рекомендации по уходу:</strong></p>`;
                             resultHtml += `<p style="font-size: 0.9em; line-height: 1.5; font-weight: normal;">${formattedRecommendations}</p>`;
                             resultHtml += `</div>`;
                         }
                     } else {
                         resultHtml = '<p>Не удалось распознать животное на фото.</p>';
                     }
                     breedResultOutputMulti.innerHTML = resultHtml;

                } else {
                     // ---> ДОБАВЛЕНО: Обработка 401 ошибки здесь <---
                     if (response.status === 401) {
                          // Можно показать специфичное сообщение или вызвать общую логику 401 из fetchWithAuth
                          breedResultOutputMulti.innerHTML = `<p style="color: red;">Ошибка: Не авторизован. Попробуйте перезапустить приложение.</p>`;
                          // Возможно, стоит также сбросить флаг isAuthenticated и показать экран входа,
                          // но fetchWithAuth уже должен был это сделать, если ошибка не была перехвачена раньше.
                     } else {
                          // ---> КОНЕЦ ДОБАВЛЕНИЯ <---
                          let errorMsg = `Ошибка сервера ${response.status}: ${response.statusText}`;
                          try { const errorData = await response.json(); errorMsg = errorData.detail || errorMsg; } catch (_e) {}
                          breedResultOutputMulti.innerHTML = `<p style="color: red;">${errorMsg}</p>`;
                     }
                }
            } catch (error) {
                 // Если fetchWithAuth выбросил ошибку (например, "Unauthorized"), она попадет сюда
                console.error('Ошибка сети или другая при определении породы (мульти-фото):', error);
                 if (error.message === "Unauthorized") {
                      // fetchWithAuth уже должен был показать alert и перейти на экран входа
                      breedResultOutputMulti.innerHTML = `<p style="color: red;">Ошибка авторизации.</p>`;
                 } else {
                      breedResultOutputMulti.innerHTML = '<p style="color: red;">Ошибка сети или сервера при отправке запроса.</p>';
                 }
            } finally {
                submitBreedMultiButton.disabled = false; submitBreedMultiButton.textContent = 'Определить породу';
            }
        });
    } else {
        console.warn("Кнопка #submit-breed-multi-button или контейнер #breed-result-output не найдены при назначении обработчика.");
    }

    // --- Фаза 4: Начальные Действия ---
    // Приветствие пользователя
    const initDataUnsafe = tg.initDataUnsafe || {};
    const userDataUnsafe = initDataUnsafe.user;
    if (userDataUnsafe && userDataUnsafe.first_name && userGreetingElement) {
        userGreetingElement.textContent = `Привет, ${userDataUnsafe.first_name}!`;
    }
    console.log("Telegram Init Data (Unsafe для приветствия):", initDataUnsafe);

    // Логика инициализации и сплеш-скрина
    if (splashScreen && !splashScreen.classList.contains('hidden')) {
        console.log("Показываем сплеш-скрин...");
        setTimeout(() => {
            console.log("Сплеш-скрин скрыт, выполняем аутентификацию...");
            splashScreen.classList.add('hidden');
            performAuthentication(); // Глобальная
        }, 2000); // Задержка сплеш-скрина
    } else {
        console.log("Сплеш-скрин не найден/скрыт, выполняем аутентификацию...");
        performAuthentication(); // Глобальная
    }

    // --- Обработчик переключения вида на экране "Избранное" ---
    if (viewToggleButton && favoritesList) {
        viewToggleButton.addEventListener('click', () => {
            const isGridMode = favoritesList.classList.contains('view-mode-grid');

            // Переключаем классы у #favorites-list
            favoritesList.classList.toggle('view-mode-grid', !isGridMode);
            favoritesList.classList.toggle('view-mode-list', isGridMode);

            // Переключаем иконки в кнопке
            const gridIcon = viewToggleButton.querySelector('.view-icon-grid');
            const listIcon = viewToggleButton.querySelector('.view-icon-list');
            if (gridIcon && listIcon) {
                gridIcon.classList.toggle('hidden', !isGridMode);
                listIcon.classList.toggle('hidden', isGridMode);
            }

            // Обновляем атрибут data-view-mode
            viewToggleButton.setAttribute('data-view-mode', isGridMode ? 'list' : 'grid');
        });
    } else {
        console.warn("Кнопка #view-toggle-button или контейнер #favorites-list не найдены для переключения вида.");
    }
}); // --- Конец DOMContentLoaded ---

// --- Функция для обработки выбора файла и показа превью (НОВАЯ или АДАПТИРОВАННАЯ) ---
function handleAdFormFileSelect(event) {
    const fileInput = event.target;
    const file = fileInput.files[0];
    const parentLabel = fileInput.closest('.angle-upload-button');
    if (!parentLabel) {
        console.error("Не найден родительский .angle-upload-button для превью");
        return;
    }
    const previewElement = parentLabel.querySelector('.angle-upload-preview');
    if (!previewElement) {
        console.error("Не найден .angle-upload-preview внутри родительского элемента");
        return;
    }

    if (file) {
        if (!file.type.startsWith('image/')) {
            alert('Пожалуйста, выберите файл изображения.', 'error');
            fileInput.value = '';
            previewElement.style.backgroundImage = 'none';
            parentLabel.classList.remove('has-file');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(_e) {
            previewElement.style.backgroundImage = `url('${_e.target.result}')`;
            parentLabel.classList.add('has-file');
            console.log('Превью для формы объявления установлено.');
        };
        reader.onerror = function(_e) {
            console.error("Ошибка чтения файла:", _e);
            alert('Не удалось прочитать файл для превью.', 'error');
            previewElement.style.backgroundImage = 'none';
            parentLabel.classList.remove('has-file');
        };
        reader.readAsDataURL(file);
    } else {
        previewElement.style.backgroundImage = 'none';
        parentLabel.classList.remove('has-file');
        console.log('Выбор файла отменен, превью сброшено.');
    }
}


// Условный экспорт для Jest-тестов
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        showScreen,
        openMenu,
        closeMenu,
        toggleMenu,
    };
}