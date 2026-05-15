import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import type { OpenRouterModel, OpenRouterModelSummary } from "./types.js";

const OPENROUTER_MODELS_URL =
  "https://openrouter.ai/api/v1/models?input_modalities=text,image";
const DEFAULT_CACHE_TTL_MS = 5 * 60 * 1000;

export function createOpenRouterProvider(apiKey: string) {
  return createOpenRouter({
    apiKey,
    extraBody: { usage: { include: true } } as any,
  });
}

interface CacheState {
  models: OpenRouterModel[];
  byId: Map<string, OpenRouterModel>;
  fetchedAt: number;
}

export interface OpenRouterCatalogOptions {
  ttlMs?: number;
  fetchImpl?: typeof fetch;
}

export function createOpenRouterCatalog(options: OpenRouterCatalogOptions = {}) {
  const ttlMs = options.ttlMs ?? DEFAULT_CACHE_TTL_MS;
  const fetchImpl = options.fetchImpl ?? fetch;
  let cache: CacheState | null = null;
  let inflight: Promise<OpenRouterModel[]> | null = null;

  async function fetchFromOpenRouter(): Promise<OpenRouterModel[]> {
    const res = await fetchImpl(OPENROUTER_MODELS_URL, {
      signal: AbortSignal.timeout(15_000),
    });
    if (!res.ok) {
      throw new Error(`OpenRouter models API returned ${res.status}`);
    }
    const json = (await res.json()) as { data?: OpenRouterModel[] };
    return json.data ?? [];
  }

  async function getModels(): Promise<OpenRouterModel[]> {
    const now = Date.now();
    if (cache && now - cache.fetchedAt < ttlMs) return cache.models;
    if (inflight) return inflight;

    inflight = (async () => {
      try {
        const models = await fetchFromOpenRouter();
        cache = {
          models,
          byId: new Map(models.map((model) => [model.id, model])),
          fetchedAt: Date.now(),
        };
        return models;
      } catch (error) {
        if (cache) return cache.models;
        throw error;
      } finally {
        inflight = null;
      }
    })();

    return inflight;
  }

  async function getModelById(id: string): Promise<OpenRouterModel | null> {
    await getModels();
    return cache?.byId.get(id) ?? null;
  }

  async function getModelContextLength(id: string): Promise<number | null> {
    const model = await getModelById(id);
    return model?.context_length ?? model?.top_provider?.context_length ?? null;
  }

  async function isValidOpenRouterModel(id: string): Promise<boolean> {
    try {
      return (await getModelById(id)) !== null;
    } catch {
      return true;
    }
  }

  async function searchModels(opts: {
    search?: string;
    offset?: number;
    limit?: number;
  } = {}): Promise<{
    models: OpenRouterModel[];
    hasMore: boolean;
    nextOffset: number | null;
    total: number;
  }> {
    const all = await getModels();
    const search = (opts.search ?? "").trim().toLowerCase();
    const offset = Math.max(0, opts.offset ?? 0);
    const limit = Math.max(1, Math.min(100, opts.limit ?? 30));
    const filtered = search
      ? all.filter((model) => {
          return (
            model.id.toLowerCase().includes(search) ||
            model.name?.toLowerCase().includes(search) ||
            model.description?.toLowerCase().includes(search)
          );
        })
      : all;
    const page = filtered.slice(offset, offset + limit);
    return {
      models: page,
      hasMore: offset + limit < filtered.length,
      nextOffset: offset + limit < filtered.length ? offset + limit : null,
      total: filtered.length,
    };
  }

  return {
    getModels,
    getModelById,
    getModelContextLength,
    isValidOpenRouterModel,
    searchModels,
  };
}

export function trimModelForClient(model: OpenRouterModel): OpenRouterModelSummary {
  return {
    id: model.id,
    name: model.name,
    description: model.description ?? "",
    contextLength: model.context_length ?? model.top_provider?.context_length ?? 0,
    pricing: {
      prompt: model.pricing?.prompt ?? "0",
      completion: model.pricing?.completion ?? "0",
    },
    created: model.created ?? 0,
  };
}

export const openRouterCatalog = createOpenRouterCatalog();

export const getOpenRouterModels = openRouterCatalog.getModels;
export const getOpenRouterModelById = openRouterCatalog.getModelById;
export const getModelContextLength = openRouterCatalog.getModelContextLength;
export const isValidOpenRouterModel = openRouterCatalog.isValidOpenRouterModel;
export const searchOpenRouterModels = openRouterCatalog.searchModels;
