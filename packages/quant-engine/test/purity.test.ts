import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { priceFixture } from "../src/priceFixture.js";

const SRC = fileURLToPath(new URL("../src", import.meta.url));

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return full.endsWith(".ts") ? [full] : [];
  });
}

// The project tsconfig declares "node" in compilerOptions.types so that this
// test file (which legitimately needs node:fs/node:path/node:url to do the
// scanning) typechecks. That declaration is project-wide: it also means
// `import ... from "node:crypto"` (or any other Node builtin) inside src/
// now typechecks with exit 0 — tsc no longer provides any backstop against
// Node builtins leaking into the pricing engine. This text scan is therefore
// the ONLY remaining guard, so the banned list below must stay broad and any
// addition of ambient types must be paired with a corresponding addition
// here.
const BANNED_IMPORTS = [
  "node:fs",
  "node:http",
  "node:https",
  "node:crypto",
  "node:path",
  "node:url",
  "node:os",
  "node:child_process",
  "node:worker_threads",
  "node:process",
  "axios",
  "node-fetch",
  "fetch(",
];

function findOffenders(files: string[], predicate: (text: string) => boolean): string[] {
  return files.filter((f) => predicate(readFileSync(f, "utf8")));
}

describe("engine purity", () => {
  const files = sourceFiles(SRC);

  it("finds source files to scan", () => {
    expect(files.length).toBeGreaterThan(5);
  });

  it("contains no Math.random anywhere in src", () => {
    // Defect D3: the previous engine had Math.random in the pricing path, so
    // odds changed on every refresh and could be re-rolled by the bettor.
    const offenders = findOffenders(files, (text) => text.includes("Math.random"));
    expect(offenders, `Math.random found in: ${offenders.join(", ")}`).toEqual([]);
  });

  it("contains no .random( call anywhere in src (catches aliasing of Math.random)", () => {
    // Broader than the Math.random check above: catches Math["random"](),
    // `const r = Math; r.random()`, or any other object exposing a
    // .random(...) method, not just the literal "Math.random" spelling.
    const offenders = findOffenders(files, (text) => text.includes(".random("));
    expect(offenders, `.random( found in: ${offenders.join(", ")}`).toEqual([]);
  });

  it("contains no Date.now or new Date in src", () => {
    const offenders = findOffenders(
      files,
      (text) => text.includes("Date.now(") || text.includes("new Date("),
    );
    expect(offenders, `Date.now/new Date found in: ${offenders.join(", ")}`).toEqual([]);
  });

  it("contains no performance.now or crypto. usage in src", () => {
    // performance.now() is another wall-clock, non-reproducible source.
    // "crypto." catches crypto.getRandomValues / crypto.randomUUID however
    // they are reached (global WebCrypto, node:crypto, or an alias).
    const offenders = findOffenders(
      files,
      (text) => text.includes("performance.now(") || text.includes("crypto."),
    );
    expect(offenders, `performance.now/crypto. found in: ${offenders.join(", ")}`).toEqual([]);
  });

  it("imports no node builtins or network clients in src", () => {
    const offenders = findOffenders(files, (text) => BANNED_IMPORTS.some((b) => text.includes(b)));
    expect(offenders, `banned import found in: ${offenders.join(", ")}`).toEqual([]);
  });

  it("prices identically across 100 repeated calls", () => {
    const first = JSON.stringify(priceFixture({ home: 1.72, away: 1.03 }));
    for (let i = 0; i < 100; i += 1) {
      expect(JSON.stringify(priceFixture({ home: 1.72, away: 1.03 }))).toBe(first);
    }
  });
});
