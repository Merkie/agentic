import { describe, expect, it, vi } from "vitest";
import { createOpenRouterCatalog, trimModelForClient } from "../src/index.js";

describe("OpenRouter catalog", () => {
  it("caches model fetches and de-dupes concurrent requests", async () => {
    let calls = 0;
    const fetchImpl = vi.fn(async () => {
      calls += 1;
      await new Promise((resolve) => setTimeout(resolve, 10));
      return Response.json({
        data: [{ id: "a/model", name: "A", context_length: 123, pricing: {} }],
      });
    });
    const catalog = createOpenRouterCatalog({ fetchImpl, ttlMs: 60_000 });

    const [a, b] = await Promise.all([catalog.getModels(), catalog.getModels()]);

    expect(a).toHaveLength(1);
    expect(b).toHaveLength(1);
    expect(calls).toBe(1);
    expect(await catalog.getModelContextLength("a/model")).toBe(123);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("serves stale cache on refresh failure and validates fail-open", async () => {
    let shouldFail = false;
    const fetchImpl = vi.fn(async () => {
      if (shouldFail) return new Response("bad", { status: 500 });
      return Response.json({ data: [{ id: "ok/model", name: "OK" }] });
    });
    const catalog = createOpenRouterCatalog({ fetchImpl, ttlMs: 0 });

    await expect(catalog.getModels()).resolves.toHaveLength(1);
    shouldFail = true;
    await expect(catalog.getModels()).resolves.toHaveLength(1);

    const emptyCatalog = createOpenRouterCatalog({
      ttlMs: 0,
      fetchImpl: async () => new Response("bad", { status: 500 }),
    });
    await expect(emptyCatalog.isValidOpenRouterModel("anything")).resolves.toBe(true);
  });

  it("trims client model shape", () => {
    expect(
      trimModelForClient({
        id: "x",
        name: "X",
        top_provider: { context_length: 10 },
        pricing: { prompt: "1", completion: "2" },
      }),
    ).toEqual({
      id: "x",
      name: "X",
      description: "",
      contextLength: 10,
      pricing: { prompt: "1", completion: "2" },
      created: 0,
    });
  });
});
