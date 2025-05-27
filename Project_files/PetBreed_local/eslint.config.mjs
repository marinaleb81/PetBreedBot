// eslint.config.mjs
import globals from "globals";
import pluginJs from "@eslint/js";
// import pluginJest from "eslint-plugin-jest"; // Если решишь использовать

export default [
  pluginJs.configs.recommended, // <-- ВАЖНО: Базовые рекомендуемые правила ESLint

  // Общие правила для всех файлов можно оставить здесь, если они не конфликтуют с recommended
  // или переопределить точечно ниже.
  {
    linterOptions: {
      reportUnusedDisableDirectives: "warn"
    },
    rules: {
      "no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
      "no-empty": ["error", { "allowEmptyCatch": true }],
      "no-prototype-builtins": "warn", // Лучше исправлять код, но можно "off" для начала
      // Другие твои общие правила
    }
  },

  {
    // Настройки для основного скрипта script.js
    files: ["script.js"],
    languageOptions: {
      sourceType: "commonjs", // Так как у тебя есть module.exports в конце
      globals: {
        ...globals.browser,    // document, window, fetch, alert, etc.
        Telegram: "readonly",   // Для window.Telegram.WebApp
      }
    },
    rules: {
      // Если 'module' в script.js (в блоке exports) все еще не определяется,
      // можно попробовать добавить 'node: true' в globals здесь,
      // но sourceType: "commonjs" с pluginJs.configs.recommended должно помочь.
    }
  },
  {
    // Настройки для тестовых файлов
    files: ["test/**/*.test.js", "tests/**/*.test.js"],
    // plugins: { // Если используешь eslint-plugin-jest
    //   jest: pluginJest,
    // },
    languageOptions: {
      globals: {
        ...globals.jest,     // describe, test, expect, jest, beforeEach и т.д.
        ...globals.node,     // Для require, module (если вдруг понадобится в тестах)
      }
    },
    // rules: pluginJest ? pluginJest.configs.recommended.rules : {}, // Если используешь eslint-plugin-jest
  },
  {
    // Настройки для конфигурационных файлов Node.js
    files: ["jest.config.js", "eslint.config.mjs"],
    languageOptions: {
      sourceType: "commonjs",
      globals: {
        ...globals.node,       // module, exports, require, __dirname, process и т.д.
      }
    },
    rules: {
        "no-unused-vars": "off" // В конфигах часто бывают неиспользуемые переменные
    }
  }
];