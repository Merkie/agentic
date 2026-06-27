import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TextStreamPart, ToolSet } from "ai";
import { logStream } from "./logStream.js";

// Force chalk to emit plain strings so assertions don't have to match ANSI codes.
process.env.FORCE_COLOR = "0";

async function* toStream(parts: Array<Partial<TextStreamPart<ToolSet>>>) {
	for (const part of parts) {
		yield part as TextStreamPart<ToolSet>;
	}
}

describe("logStream", () => {
	let out: string;
	let logSpy: { mockRestore: () => void };
	let writeSpy: { mockRestore: () => void };

	beforeEach(() => {
		out = "";
		logSpy = vi.spyOn(console, "log").mockImplementation((...args) => {
			out += `${args.join(" ")}\n`;
		});
		writeSpy = vi
			.spyOn(process.stdout, "write")
			.mockImplementation((chunk: unknown) => {
				out += String(chunk);
				return true;
			});
		// Avoid real network calls for context-window lookups.
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => ({
				ok: true,
				json: async () => ({ data: { context_length: 1000 } }),
			})),
		);
	});

	afterEach(() => {
		logSpy.mockRestore();
		writeSpy.mockRestore();
		vi.unstubAllGlobals();
	});

	it("renders text deltas and a finish summary", async () => {
		await logStream(
			toStream([
				{ type: "start-step", request: { body: { model: "x/y" } } } as never,
				{ type: "text-delta", text: "hello world" } as never,
				{
					type: "finish-step",
					usage: { inputTokens: 10, outputTokens: 5 },
					response: { modelId: "x/y" },
					providerMetadata: { openrouter: { usage: { cost: 0.0001 } } },
				} as never,
				{ type: "finish", finishReason: "stop" } as never,
			]),
		);

		expect(out).toContain("hello world");
		expect(out).toContain("step");
		expect(out).toContain("done (stop)");
		// cost surfaced from providerMetadata
		expect(out).toContain("cost $0.000100");
		// context window percentage computed from usage + fetched window
		expect(out).toContain("1.50% context used (15 / 1,000 tokens)");
	});

	it("renders tool calls and results", async () => {
		await logStream(
			toStream([
				{ type: "tool-call", toolName: "search", input: { q: "cats" } } as never,
				{ type: "tool-result", toolName: "search", output: { hits: 3 } } as never,
				{ type: "finish", finishReason: "stop" } as never,
			]),
		);

		expect(out).toContain("tool call  search");
		expect(out).toContain('{"q":"cats"}');
		expect(out).toContain("tool result search");
		expect(out).toContain('{"hits":3}');
	});

	it("does not crash on an empty stream", async () => {
		await expect(logStream(toStream([]))).resolves.toBeUndefined();
	});
});
