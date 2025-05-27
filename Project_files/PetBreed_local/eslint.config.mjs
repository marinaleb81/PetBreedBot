// eslint.config.mjs
import globals from "globals";
import eslintJs from "@eslint/js";

export default [
  // 1. Базовые рекомендуемые правила от ESLint для JavaScript.
  // Он по умолчанию правильно обработает .mjs как ES-модуль.
  eslintJs.configs.recommended,

  // 2. Общие настройки языка и твои кастомные правила
  {
    languageOptions: {
      ecmaVersion: 2021,
      globals: {
        ...globals.browser,
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

  // 3. Специфичные настройки для script.js
  {
    files: ["script.js"],
    languageOptions: {
      sourceType: "commonjs", // Для блока module.exports в script.js
      globals: {
        // globals.browser УЖЕ НАСЛЕДУЮТСЯ
        Telegram: "readonly",
        module: "readonly", // Явно разрешаем module и exports
        exports: "readonly",
      }
    }
  },

  // 4. Специфичные настройки для ТЕСТОВЫХ файлов (*.test.js)
  {
    files: ["test/**/*.test.js", "tests/**/*.test.js"],
    languageOptions: {
      sourceType: "commonjs", // Тесты используют require()
      globals: {
        ...globals.node,
        ...globals.jest,
      }
    }
  },

  // 5. Специфичные настройки для jest.config.js
  {
    files: ["jest.config.js"], // ТОЛЬКО для jest.config.js
    languageOptions: {
      sourceType: "commonjs",
      globals: {
        ...globals.node,
      }
    },
    rules: {
      "no-unused-vars": "off"
    }
  }
  // Сам eslint.config.mjs НЕ ДОЛЖЕН быть здесь с sourceType: "commonjs"
];
