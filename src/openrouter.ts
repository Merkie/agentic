import {
	createOpenRouter as baseCreateOpenRouter,
	type OpenRouterProvider,
	type OpenRouterProviderSettings,
} from "@openrouter/ai-sdk-provider";

/**
 * Drop-in replacement for `createOpenRouter` from `@openrouter/ai-sdk-provider`.
 *
 * It behaves identically to the original, with two conveniences baked in:
 *
 *  1. `apiKey` defaults to `process.env.OPENROUTER_API_KEY` when you don't pass
 *     one, so the common case is just `createOpenRouter()`.
 *  2. `extraBody.usage.include` defaults to `true`, which tells OpenRouter to
 *     return cost + token accounting on every response. This is what powers the
 *     mid-run cost/usage tracking in {@link logStream}.
 *
 * Every option you pass through wins over the defaults — including `usage` — so
 * this is 1:1 compatible with the upstream `createOpenRouter`. You can override
 * the api key, add your own `extraBody`, flip `usage.include` off, etc.
 *
 * @example
 * ```ts
 * import { createOpenRouter } from "agentlib";
 *
 * // Reads OPENROUTER_API_KEY from the environment, usage tracking on.
 * const openrouter = createOpenRouter();
 *
 * const model = openrouter("openai/gpt-4o-mini");
 * ```
 */
export function createOpenRouter(
	settings: OpenRouterProviderSettings = {},
): OpenRouterProvider {
	const { apiKey, extraBody, ...rest } = settings;

	const userUsage =
		extraBody && typeof extraBody === "object" && "usage" in extraBody
			? (extraBody as Record<string, unknown>).usage
			: undefined;

	return baseCreateOpenRouter({
		apiKey: apiKey ?? process.env.OPENROUTER_API_KEY,
		...rest,
		extraBody: {
			...extraBody,
			usage: {
				include: true,
				...(userUsage && typeof userUsage === "object"
					? (userUsage as Record<string, unknown>)
					: {}),
			},
		},
	});
}

export type { OpenRouterProvider, OpenRouterProviderSettings };
