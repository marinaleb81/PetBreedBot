// eslint.config.mjs
import globals from "globals";
import pluginJs from "@eslint/js";
// Если будешь использовать плагин для Jest, раскомментируй и установи:
// import pluginJest from "eslint-plugin-jest";

export default [
  pluginJs.configs.recommended, // Добавляем базовые рекомендуемые правила ESLint
  {
    // Глобальные настройки для всех JS файлов (можно оставить пустым, если все в pluginJs.configs.recommended)
    // Но languageOptions и linterOptions лучше оставить для явности
    languageOptions: {
      ecmaVersion: 2021,
      sourceType: "commonjs", // По умолчанию
      globals: {
        ...globals.browser,
        // Telegram: "readonly", // Можно определить глобально, если используется везде
      }
    },
    linterOptions: {
      reportUnusedDisableDirectives: "warn"
    },
    rules: {
      "no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
      "no-empty": ["error", { "allowEmptyCatch": true }],
      // "no-prototype-builtins": "warn", // Лучше исправить код, как обсуждали
    }
  },
  {
    // Настройки для основного скрипта script.js
    files: ["script.js"],
    languageOptions: {
      sourceType: "commonjs", // У script.js есть module.exports в конце
      globals: {
        ...globals.browser,
        Telegram: "readonly", // Для window.Telegram.WebApp
      }
    },
    rules: {
      // Сюда можно добавить правила, специфичные только для script.js
       "no-undef": ["error", { "typeof": true }] // Чтобы typeof module не вызывал ошибку
    }
  },
  {
    // Настройки для тестовых файлов
    files: ["test/**/*.test.js", "tests/**/*.test.js"],
    languageOptions: {
      globals: {
        ...globals.jest, // describe, test, expect, jest, beforeEach и т.д.
        ...globals.node, // Для require в тестах
      }
    },
    // Если НЕ используешь плагин eslint-plugin-jest, можно временно оставить это:
    // rules: {
    //     "no-undef": "off" // Отключает no-undef ТОЛЬКО для тестов
    // }
    // Но лучше, чтобы globals.jest и globals.node сработали.
    // Если после исправления конфига все еще будут ошибки no-undef в тестах,
    // тогда можно вернуть rules: { "no-undef": "off" } сюда как временное решение.
  },
  {
    // Настройки для конфигурационных файлов Node.js
    files: ["jest.config.js", "eslint.config.mjs"],
    languageOptions: {
      sourceType: "commonjs",
      globals: {
        ...globals.node, // module, exports, require, __dirname, process и т.д.
      }
    },
    rules: {
        "no-unused-vars": "off" // В конфигах часто бывают неиспользуемые переменные
    }
  }
];