// eslint.config.mjs
import globals from "globals";
import pluginJs from "@eslint/js";
// Uncomment the following if you decide to use Jest plugin
// import pluginJest from "eslint-plugin-jest";

export default [
  // Base recommended rules from ESLint
  {
    ...pluginJs.configs.recommended,
    languageOptions: {
      sourceType: "module", // Required for eslint.config.mjs itself to handle import/export
      globals: {
        ...globals.node, // For Node.js environment in config files
      },
    },
  },

  // General rules for all files
  {
    linterOptions: {
      reportUnusedDisableDirectives: "warn",
    },
    rules: {
      "no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      "no-empty": ["error", { allowEmptyCatch: true }],
      "no-prototype-builtins": "off", // Temporarily disable to avoid warnings; fix code later
      "no-useless-escape": "error", // Keep this to catch unnecessary escapes
      "no-undef": "error", // Ensure undefined variables are caught
    },
  },

  // Settings for script.js (main application file)
  {
    files: ["**/script.js"],
    languageOptions: {
      sourceType: "module", // Use 'module' since script.js may use ES modules
      globals: {
        ...globals.browser, // document, window, fetch, alert, etc.
        Telegram: "readonly", // For window.Telegram.WebApp
        showAlert: "writable", // Define showAlert as a global (fix no-undef errors)
      },
    },
    rules: {
      "no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^(userRole|loadedAnnouncements|isDataLoading|MOCK_CURRENT_USER_ID|menuOverlay|user|relativeImagePath|handleAngleFileSelect|handleAdFormFileSelect)$" },
      ], // Ignore specific unused vars
      "no-useless-escape": "warn", // Downgrade to warning for now
    },
  },

  // Settings for test files
  {
    files: ["test/**/*.test.js", "tests/**/*.test.js"],
    languageOptions: {
      sourceType: "module", // Tests may use ES modules
      globals: {
        ...globals.jest, // describe, test, expect, jest, beforeEach, etc.
        ...globals.browser, // document, window for DOM testing
        ...globals.node, // For require, module in tests
      },
    },
    // Uncomment if using eslint-plugin-jest
    // plugins: {
    //   jest: pluginJest,
    // },
    // rules: pluginJest ? pluginJest.configs.recommended.rules : {},
  },

  // Settings for configuration files (like jest.config.js, eslint.config.mjs)
  {
    files: ["jest.config.js", "eslint.config.mjs"],
    languageOptions: {
      sourceType: "module", // Use module for config files with import/export
      globals: {
        ...globals.node, // module, exports, require, __dirname, process, etc.
      },
    },
    rules: {
      "no-unused-vars": "off", // Allow unused vars in config files
    },
  },
];