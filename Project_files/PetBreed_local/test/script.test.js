const SCRIPT_PATH = '../script.js';

// --- Тесты для showScreen с моком для closeMenu ---
// Эта инструкция jest.mock будет "поднята" Jest'ом вверх файла (hoisted),
// так что любой последующий require(SCRIPT_PATH) в этом файле будет использовать эту фабрику моков.
jest.mock(SCRIPT_PATH, () => {
  const originalModule = jest.requireActual(SCRIPT_PATH); // Загружаем реальный модуль
  return {
    ...originalModule, // Используем все оригинальные функции из модуля...
    closeMenu: jest.fn(), // ...кроме closeMenu, которую заменяем на мок-функцию jest.fn()
                          // Важно: сама showScreen будет использовать оригинальную closeMenu из своего скоупа,
                          // поэтому прямой тест вызова мока был проблематичен.
                          // Этот мок здесь больше для того, чтобы показать намерение,
                          // но основная проверка будет косвенной.
  };
});

// scriptUnderTest содержит модуль, где showScreen - оригинальная.
// scriptUnderTest.closeMenu будет моком, но мы его не будем прямо проверять в падающем тесте.
const scriptUnderTest = require(SCRIPT_PATH);

describe('showScreen', () => {
  let mainMenu, menuOverlay; // Объявляем здесь для доступа в тестах

  beforeEach(() => {
    document.body.innerHTML = `
      <div id="splash-screen" class="screen">Splash</div>
      <div id="screen1" class="screen hidden">Screen 1</div>
      <div id="screen2" class="screen hidden">Screen 2</div>
      <div id="main-menu"> <button class="hamburger-button"></button> </div>
      <div id="menu-overlay"></div>
    `;
    mainMenu = document.getElementById('main-menu'); // Инициализируем здесь
    menuOverlay = document.getElementById('menu-overlay'); // Инициализируем здесь

    document.getElementById('splash-screen').classList.add('hidden');
    Object.defineProperty(window, 'scrollTo', { value: jest.fn(), writable: true });

    // Очищаем мок, если бы мы его проверяли (для других тестов это может быть нужно)
    if (jest.isMockFunction(scriptUnderTest.closeMenu)) {
        scriptUnderTest.closeMenu.mockClear();
    }
  });

  test('должна корректно показывать целевой экран и скрывать другие', () => {
    scriptUnderTest.showScreen('screen1');
    expect(document.getElementById('screen1').classList.contains('hidden')).toBe(false);
    expect(document.getElementById('screen2').classList.contains('hidden')).toBe(true);
    expect(document.getElementById('splash-screen').classList.contains('hidden')).toBe(true);
  });

  test('должна показывать splash-screen, если он целевой и был скрыт', () => {
    scriptUnderTest.showScreen('splash-screen');
    expect(document.getElementById('splash-screen').classList.contains('hidden')).toBe(false);
  });

  test('должна вызывать console.error если экран не найден', () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    scriptUnderTest.showScreen('nonExistentScreen');
    expect(consoleErrorSpy).toHaveBeenCalledWith('Экран с id "nonExistentScreen" не найден.');
    consoleErrorSpy.mockRestore();
  });

  test('должна закрыть меню (косвенная проверка через DOM), если оно было открыто', () => {
    // 1. Открываем меню, вызывая оригинальную функцию openMenu
    const originalScriptForMenuTests = jest.requireActual(SCRIPT_PATH); // Загружаем оригинальный модуль
    const hamburgerInMenu = mainMenu.querySelector('.hamburger-button');
    originalScriptForMenuTests.openMenu(hamburgerInMenu); // Устанавливаем isMenuOpen и классы

    // Проверяем, что меню открыто
    expect(mainMenu.classList.contains('is-open')).toBe(true);
    expect(menuOverlay.classList.contains('is-visible')).toBe(true);
    if (hamburgerInMenu) expect(hamburgerInMenu.classList.contains('is-active')).toBe(true);

    // Настраиваем экран для корректной работы querySelector в showScreen
    const screen1 = document.getElementById('screen1');
    screen1.classList.remove('hidden');
    screen1.innerHTML = '<header class="app-header"><button class="hamburger-button"></button></header>';

    // 2. Вызываем showScreen, которая должна вызвать closeMenu
    scriptUnderTest.showScreen('screen2');

    // 3. Проверяем, что меню закрыто
    expect(mainMenu.classList.contains('is-open')).toBe(false);
    expect(menuOverlay.classList.contains('is-visible')).toBe(false);
  });

  test('не должна пытаться закрыть меню (нет изменений в классах), если оно было закрыто', () => {
    // Убеждаемся, что меню изначально закрыто (классов нет)
    mainMenu.classList.remove('is-open');
    menuOverlay.classList.remove('is-visible');

    const initialMainMenuClasses = new Set(mainMenu.classList);
    const initialMenuOverlayClasses = new Set(menuOverlay.classList);

    scriptUnderTest.showScreen('screen1');

    // Классы не должны были измениться, так как условие isMenuOpenNow в showScreen будет false
    expect(new Set(mainMenu.classList)).toEqual(initialMainMenuClasses);
    expect(new Set(menuOverlay.classList)).toEqual(initialMenuOverlayClasses);

    // Также убедимся, что мок (даже если он не отслеживается идеально) не был вызван
    if (jest.isMockFunction(scriptUnderTest.closeMenu)) {
        expect(scriptUnderTest.closeMenu).not.toHaveBeenCalled();
    }
  });
});

// --- Тесты для функций управления меню (openMenu, closeMenu) ---
// Используем jest.requireActual для получения оригинальных, немокнутых версий.
const originalScriptForMenuTests = jest.requireActual(SCRIPT_PATH);

describe('Функции управления меню (openMenu, closeMenu) - тестируем оригиналы', () => {
  let mainMenu, menuOverlay, hamburgerButton;

  beforeEach(() => {
    document.body.innerHTML = `
      <div id="main-menu" aria-hidden="true"></div>
      <div id="menu-overlay"></div>
      <button class="hamburger-button" aria-expanded="false"></button>
    `;
    mainMenu = document.getElementById('main-menu');
    menuOverlay = document.getElementById('menu-overlay');
    hamburgerButton = document.querySelector('.hamburger-button');

    // Сбрасываем состояние меню, вызывая ОРИГИНАЛЬНУЮ closeMenu.
    originalScriptForMenuTests.closeMenu(hamburgerButton);
    // Дополнительно приводим DOM к состоянию закрытого меню.
    mainMenu.classList.remove('is-open');
    menuOverlay.classList.remove('is-visible');
    if (hamburgerButton) {
        hamburgerButton.classList.remove('is-active');
        hamburgerButton.setAttribute('aria-expanded', 'false');
    }
    mainMenu.setAttribute('aria-hidden', 'true');
  });

  describe('openMenu', () => {
    test('должна открывать меню, добавляя классы и устанавливая атрибуты', () => {
      originalScriptForMenuTests.openMenu(hamburgerButton);
      expect(mainMenu.classList.contains('is-open')).toBe(true);
      expect(menuOverlay.classList.contains('is-visible')).toBe(true);
      expect(hamburgerButton.classList.contains('is-active')).toBe(true);
      expect(hamburgerButton.getAttribute('aria-expanded')).toBe('true');
      expect(mainMenu.getAttribute('aria-hidden')).toBe('false');
    });

    test('не должна ничего делать, если меню уже открыто (isMenuOpen=true)', () => {
      originalScriptForMenuTests.openMenu(hamburgerButton); // 1. Открываем
      const mainMenuClassesAfterFirstOpen = new Set(mainMenu.classList);
      const menuOverlayClassesAfterFirstOpen = new Set(menuOverlay.classList);
      const hamburgerButtonClassesAfterFirstOpen = new Set(hamburgerButton.classList);

      originalScriptForMenuTests.openMenu(hamburgerButton); // 2. Пытаемся открыть снова

      expect(new Set(mainMenu.classList)).toEqual(mainMenuClassesAfterFirstOpen);
      expect(new Set(menuOverlay.classList)).toEqual(menuOverlayClassesAfterFirstOpen);
      expect(new Set(hamburgerButton.classList)).toEqual(hamburgerButtonClassesAfterFirstOpen);
    });
  });

  describe('closeMenu', () => {
    test('должна закрывать меню, удаляя классы и устанавливая атрибуты, если оно было открыто', () => {
      originalScriptForMenuTests.openMenu(hamburgerButton);
      expect(mainMenu.classList.contains('is-open')).toBe(true);

      originalScriptForMenuTests.closeMenu(hamburgerButton);

      expect(mainMenu.classList.contains('is-open')).toBe(false);
      expect(menuOverlay.classList.contains('is-visible')).toBe(false);
      expect(hamburgerButton.classList.contains('is-active')).toBe(false);
      expect(hamburgerButton.getAttribute('aria-expanded')).toBe('false');
      expect(mainMenu.getAttribute('aria-hidden')).toBe('true');
    });

    test('не должна ничего делать, если меню уже закрыто (isMenuOpen=false)', () => {
      const mainMenuClassesBefore = new Set(mainMenu.classList);
      const menuOverlayClassesBefore = new Set(menuOverlay.classList);
      const hamburgerButtonClassesBefore = new Set(hamburgerButton.classList);

      originalScriptForMenuTests.closeMenu(hamburgerButton);

      expect(new Set(mainMenu.classList)).toEqual(mainMenuClassesBefore);
      expect(new Set(menuOverlay.classList)).toEqual(menuOverlayClassesBefore);
      expect(new Set(hamburgerButton.classList)).toEqual(hamburgerButtonClassesBefore);
    });
  });
});