// eslint.config.mjs
import globals from "globals";
import pluginJs from "@eslint/js";

export default [
  {
    ...pluginJs.configs.recommended,
    languageOptions: {
      sourceType: "module",
      globals: {
        ...globals.node,
      },
    },
  },

  {
    linterOptions: {
      reportUnusedDisableDirectives: "warn",
    },
    rules: {
      "no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", },
      ],
      "no-empty": ["error", { allowEmptyCatch: true }],
      "no-prototype-builtins": "off",
      "no-useless-escape": "error",
    },
  },

  {
    files: ["**/script.js"],
    languageOptions: {
      sourceType: "module",
      globals: {
        ...globals.browser,
        Telegram: "readonly",
        showAlert: "writable",
      },
    },
    rules: {
      "no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^(userRole|loadedAnnouncements|isDataLoading|MOCK_CURRENT_USER_ID|menuOverlay|user|relativeImagePath|handleAngleFileSelect|handleAdFormFileSelect)$",
          args: "after-used",
          ignoreRestSiblings: true,
        },
      ],
      "no-useless-escape": "warn",
    },
  },

  {
    files: ["test/**/*.test.js", "tests/**/*.test.js"],
    languageOptions: {
      globals: {
        ...globals.jest,
        ...globals.browser,
        ...globals.node,
      },
    },
  },

  {
    files: ["jest.config.js", "eslint.config.mjs"],
    languageOptions: {
      sourceType: "module",
      globals: {
        ...globals.node,
      },
    },
    rules: {
      "no-unused-vars": "off",
    },
  },
];