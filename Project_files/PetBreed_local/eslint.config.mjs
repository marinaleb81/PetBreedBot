// eslint.config.mjs
import globals from "globals";
import eslintJs from "@eslint/js";

export default [
  // 1. Базовые рекомендуемые правила от ESLint для JavaScript
  eslintJs.configs.recommended,

  // 2. Общие настройки языка и твои кастомные правила
  // Эти правила будут применяться ко всем файлам, если не переопределены ниже
  {
    languageOptions: {
      ecmaVersion: 2021,
      globals: {
        ...globals.browser, // Глобальные переменные браузера по умолчанию
      }
    },
    linterOptions: {
      reportUnusedDisableDirectives: "warn"
    },
    rules: {
      "no-unused-vars": ["error", { "argsIgnorePattern": "^_", "caughtErrorsIgnorePattern": "^_" }],
      "no-empty": ["error", { "allowEmptyCatch": true }],
      "no-prototype-builtins": "warn",
      // Другие твои общие правила
    }
  },

  // 3. Специфичные настройки для твоего основного файла script.js
  {
    files: ["script.js"],
    languageOptions: {
      sourceType: "commonjs", // Так как у тебя есть module.exports в конце script.js
      globals: {
        // globals.browser УЖЕ НАСЛЕДУЮТСЯ из блока выше
        Telegram: "readonly",   // Для window.Telegram.WebApp
        // 'module' и 'exports' для блока if (typeof module !== 'undefined'...)
        // должны стать доступны благодаря sourceType: "commonjs" здесь
        // и правильному распознаванию этого типа файла ESLint-ом.
      }
    }
  },

  // 4. Специфичные настройки для ТЕСТОВЫХ файлов (*.test.js)
  {
    files: ["test/**/*.test.js", "tests/**/*.test.js"], // Путь к твоим тестовым файлам
    languageOptions: {
      sourceType: "commonjs", // Тестовые файлы используют require()
      globals: {
        ...globals.node,  // Определяет 'require', 'module' и др. для Node.js
        ...globals.jest,  // Определяет 'describe', 'test', 'expect', 'jest' и т.д.
      }
    }
  },

  // 5. Специфичные настройки для jest.config.js (Node.js окружение)
  {
    files: ["jest.config.js"], // Применяем ТОЛЬКО к jest.config.js
    languageOptions: {
      sourceType: "commonjs",
      globals: {
        ...globals.node, // Определяет 'module', 'exports', 'require' и т.д.
      }
    },
    rules: {
      "no-unused-vars": "off" // В конфигурационных файлах это часто нормально
    }
  }
  // Сам файл eslint.config.mjs будет корректно обработан, так как
  // eslintJs.configs.recommended по умолчанию считает .mjs файлы ES-модулями,
  // и мы больше не переопределяем sourceType для него на "commonjs".
];