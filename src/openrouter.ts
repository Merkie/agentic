import {
	createOpenRouter as baseCreateOpenRouter,
	type OpenRouterProvider,
	type OpenRouterProviderSettings,
} from "@openrouter/ai-sdk-provider";

/**
 * Drop-in replacement for `createOpenRouter` from `@openrouter/ai-sdk-provider`.
 *
 * It behaves identically to the original — same options, same defaults
 * (including reading `OPENROUTER_API_KEY` from the environment) — with exactly
 * one thing added: `extraBody.usage.include` defaults to `true`, which tells
 * OpenRouter to return cost + token accounting on every response. That's what
 * powers the harness's per-step cost/usage tracking.
 *
 * Object-valued `usage` options win over that default, so you can add your own
 * `extraBody`, flip `usage.include` off, etc. A primitive `usage` value is
 * replaced with the valid `{ include: true }` shape.
 *
 * @example
 * ```ts
 * import { createOpenRouter } from "@merkie/agentic";
 *
 * // Same as the upstream factory, just with usage tracking switched on.
 * const openrouter = createOpenRouter();
 *
 * const model = openrouter("openai/gpt-4o-mini");
 * ```
 */
export function createOpenRouter(settings: OpenRouterProviderSettings = {}): OpenRouterProvider {
	const { extraBody, ...rest } = settings;

	const userUsage =
		extraBody && typeof extraBody === "object" && "usage" in extraBody
			? (extraBody as Record<string, unknown>).usage
			: undefined;

	return baseCreateOpenRouter({
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
