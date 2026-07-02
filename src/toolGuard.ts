import type { ToolSet } from "ai";

// Backstop so it is structurally impossible for any single tool result to
// overflow the model's context window, no matter which tool or what bug fed
// it. Two modes: "discard" replaces the whole result with a short error
// (right for lookup tools the model can live without); "truncate" keeps a
// head slice so diagnostics still reach the agent (right for build tools).

export interface ToolGuardOptions {
	maxBytes: number;
	mode: "truncate" | "discard";
	/** Telemetry hook fired when the cap trips. */
	onOversize?: (toolName: string, bytes: number) => void;
}

export function guardToolResultSizes<T extends ToolSet>(tools: T, opts: ToolGuardOptions): T {
	const guarded: Record<string, unknown> = {};
	for (const [name, def] of Object.entries(tools)) {
		const original = (def as { execute?: unknown }).execute;
		if (typeof original !== "function") {
			guarded[name] = def;
			continue;
		}
		guarded[name] = {
			...(def as object),
			execute: async (input: unknown, options: unknown) => {
				const result = await (original as (...a: unknown[]) => unknown)(input, options);
				let serialized: string;
				try {
					serialized = JSON.stringify(result) ?? "";
				} catch {
					return result; // non-serializable — let the SDK deal with it
				}
				const bytes = serialized.length;
				if (bytes <= opts.maxBytes) return result;
				opts.onOversize?.(name, bytes);
				if (opts.mode === "discard") {
					return {
						ok: false as const,
						error:
							`The result of ${name} was too large to use and was discarded so this ` +
							`conversation stays healthy. Don't retry the same call — continue with ` +
							`what you already know.`,
					};
				}
				return {
					ok: false as const,
					truncated: true as const,
					bytes,
					note:
						`The result of ${name} was ${bytes} bytes — too large to return whole, so ` +
						`only the first ${opts.maxBytes} bytes are shown below. Don't re-run it ` +
						`expecting more; narrow your next step instead.`,
					head: serialized.slice(0, opts.maxBytes),
				};
			},
		};
	}
	return guarded as T;
}
