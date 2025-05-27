// eslint.config.mjs
import globals from "globals";
import eslintJs from "@eslint/js"; // Стандартный импорт

export default [
  // 1. Базовые рекомендуемые правила от ESLint для JavaScript.
  // Этот конфиг по умолчанию использует sourceType: 'module' для .js и .mjs файлов,
  // что правильно для самого eslint.config.mjs.
  eslintJs.configs.recommended,

  // 2. Твои общие правила, которые будут применяться поверх рекомендованных.
  // Они не должны менять sourceType глобально на "commonjs".
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
    }
  },

  // 3. Специфичные настройки для твоего основного файла script.js
  {
    files: ["script.js"],
    languageOptions: {
      sourceType: "commonjs", // Важно, так как у тебя есть module.exports в конце script.js
      globals: {
        // globals.browser УЖЕ НАСЛЕДУЮТСЯ из блока выше
        Telegram: "readonly",   // Для window.Telegram.WebApp
        // 'module' и 'exports' должны быть доступны благодаря sourceType: "commonjs"
        // и globals.node, если бы он был применен здесь.
        // Чтобы точно работало для `if (typeof module !== 'undefined' && module.exports)`:
        module: "readonly",
        exports: "readonly",
      }
    }
  },

  // 4. Специфичные настройки для ТЕСТОВЫХ файлов (*.test.js)
  {
    files: ["test/**/*.test.js", "tests/**/*.test.js"],
    languageOptions: {
      sourceType: "commonjs", // Тестовые файлы используют require()
      globals: {
        ...globals.node,  // Определяет 'require', 'module' и др.
        ...globals.jest,  // Определяет 'describe', 'test', 'expect' и т.д.
      }
    }
  },

  // 5. Специфичные настройки для jest.config.js (Node.js окружение)
  {
    files: ["jest.config.js"], // ТОЛЬКО для jest.config.js
    languageOptions: {
      sourceType: "commonjs",
      globals: {
        ...globals.node, // Определяет 'module', 'exports', 'require' и т.д.
      }
    },
    rules: {
      "no-unused-vars": "off"
    }
  }
  // Сам файл eslint.config.mjs (этот самый файл) теперь НЕ указан в блоке выше,
  // и для него будет использоваться sourceType: 'module' из eslintJs.configs.recommended,
  // что корректно для его синтаксиса с import/export.
];