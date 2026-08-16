// Next 16 ships native ESLint flat configs, so the @eslint/eslintrc FlatCompat
// bridge this file previously used is no longer needed. Keeping it caused
// eslint to crash inside the legacy config validator.
import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

export default [
  ...nextCoreWebVitals,
  ...nextTypescript,
  {
    ignores: [".next/**", "node_modules/**", "next-env.d.ts"],
  },
];
