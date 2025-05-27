// eslint.config.mjs
import globals from "globals";
import pluginJs from "@eslint/js";
// Если будешь использовать плагин для Jest, раскомментируй и установи:
// import pluginJest from "eslint-plugin-jest";

export default [
  {
    // Глобальные настройки для всех JS файлов
    languageOptions: {
      ecmaVersion: 2021, // или новее, например, 2022
      sourceType: "commonjs", // По умолчанию для большинства файлов, если не используешь import/export везде
      globals: {
        ...globals.browser, // Глобальные переменные браузера (window, document, etc.)
        // Сюда можно добавить свои глобальные переменные, если они есть и используются во многих файлах
        // Telegram: true, // Если window.Telegram используется глобально
      }
    },
    linterOptions: {
      reportUnusedDisableDirectives: "warn" // или true
    },
    rules: {
      // Здесь могут быть твои общие правила
      "no-unused-vars": ["error", { "argsIgnorePattern": "^_" }], // Предупреждать о неиспользуемых переменных, но игнорировать те, что начинаются с _
      "no-empty": ["error", { "allowEmptyCatch": true }], // Разрешить пустые catch, если нужно
      // "no-prototype-builtins": "off", // Если хочешь временно отключить это правило, но лучше исправить код
    }
  },
  {
    // Настройки для основного скрипта script.js
    files: ["script.js"],
    languageOptions: {
      sourceType: "commonjs", // У script.js есть module.exports в конце
      globals: {
        ...globals.browser, // document, window, fetch, alert, etc.
        Telegram: "readonly", // Предполагая, что window.Telegram.WebApp доступно
        // Если есть другие глобальные объекты, добавь их
      }
    },
    rules: {
      // Специфичные правила для script.js, если нужны
    }
  },
  {
    // Настройки для тестовых файлов
    files: ["test/**/*.test.js", "tests/**/*.test.js"], // Путь к твоим тестовым файлам
    // Если используешь плагин для Jest:
    // plugins: {
    //   jest: pluginJest,
    // },
    languageOptions: {
      globals: {
        ...globals.jest, // describe, test, expect, jest, beforeEach и т.д.
        ...globals.node, // Для require в тестах
      }
    },
    // Если используешь плагин для Jest:
    // rules: {
    //   ...pluginJest.configs.recommended.rules,
    //   // Твои переопределения правил Jest, если нужны
    // }
    // Если НЕ используешь плагин, но хочешь убрать ошибки no-undef для Jest:
    rules: {
        "no-undef": "off" // Временно отключает проверку на неопределенные переменные ТОЛЬКО для тестовых файлов.
                          // Это быстрый фикс, если не хочешь настраивать плагин Jest сейчас.
                          // Но лучше настроить плагин или явно указать jest:true в env.
    }
  },
  {
    // Настройки для конфигурационных файлов Node.js
    files: ["jest.config.js", "eslint.config.mjs"], // Добавь другие, если есть
    languageOptions: {
      sourceType: "commonjs",
      globals: {
        ...globals.node, // module, exports, require, __dirname, process и т.д.
      }
    },
    rules: {
        "no-unused-vars": "off" // В конфигах часто бывают неиспользуемые переменные из примеров
    }
  }
];