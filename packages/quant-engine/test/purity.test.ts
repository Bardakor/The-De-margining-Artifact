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

describe("engine purity", () => {
  const files = sourceFiles(SRC);

  it("finds source files to scan", () => {
    expect(files.length).toBeGreaterThan(5);
  });

  it("contains no Math.random anywhere in src", () => {
    // Defect D3: the previous engine had Math.random in the pricing path, so
    // odds changed on every refresh and could be re-rolled by the bettor.
    const offenders = files.filter((f) => readFileSync(f, "utf8").includes("Math.random"));
    expect(offenders).toEqual([]);
  });

  it("contains no Date.now or new Date in src", () => {
    const offenders = files.filter((f) => {
      const text = readFileSync(f, "utf8");
      return text.includes("Date.now(") || text.includes("new Date(");
    });
    expect(offenders).toEqual([]);
  });

  it("imports no node builtins or network clients in src", () => {
    const banned = ["node:fs", "node:http", "node:https", "axios", "node-fetch", "fetch("];
    const offenders = files.filter((f) => {
      const text = readFileSync(f, "utf8");
      return banned.some((b) => text.includes(b));
    });
    expect(offenders).toEqual([]);
  });

  it("prices identically across 100 repeated calls", () => {
    const first = JSON.stringify(priceFixture({ home: 1.72, away: 1.03 }));
    for (let i = 0; i < 100; i += 1) {
      expect(JSON.stringify(priceFixture({ home: 1.72, away: 1.03 }))).toBe(first);
    }
  });
});
