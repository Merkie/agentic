import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock the upstream provider so we can inspect exactly what options our wrapper
// forwards to it.
const baseCreateOpenRouter = vi.fn((opts: unknown) => opts);
vi.mock("@openrouter/ai-sdk-provider", () => ({
	createOpenRouter: (opts: unknown) => baseCreateOpenRouter(opts),
}));

import { createOpenRouter } from "./openrouter.js";

type ForwardedOptions = {
	apiKey?: string;
	extraBody?: { usage?: Record<string, unknown> } & Record<string, unknown>;
};

function lastCall(): ForwardedOptions {
	return baseCreateOpenRouter.mock.calls.at(-1)?.[0] as ForwardedOptions;
}

describe("createOpenRouter", () => {
	const ORIGINAL_KEY = process.env.OPENROUTER_API_KEY;

	beforeEach(() => {
		baseCreateOpenRouter.mockClear();
		process.env.OPENROUTER_API_KEY = "env-key";
	});

	afterEach(() => {
		if (ORIGINAL_KEY === undefined) delete process.env.OPENROUTER_API_KEY;
		else process.env.OPENROUTER_API_KEY = ORIGINAL_KEY;
	});

	it("does not touch apiKey — leaves env handling to the upstream factory", () => {
		createOpenRouter();
		expect("apiKey" in lastCall()).toBe(false);
	});

	it("turns on usage accounting by default", () => {
		createOpenRouter();
		expect(lastCall().extraBody?.usage).toEqual({ include: true });
	});

	it("forwards an explicit apiKey unchanged", () => {
		createOpenRouter({ apiKey: "explicit-key" });
		expect(lastCall().apiKey).toBe("explicit-key");
	});

	it("preserves caller-provided extraBody fields", () => {
		createOpenRouter({ extraBody: { transforms: ["middle-out"] } });
		const { extraBody } = lastCall();
		expect(extraBody?.transforms).toEqual(["middle-out"]);
		// usage default still applied alongside the caller's fields.
		expect(extraBody?.usage).toEqual({ include: true });
	});

	it("merges caller usage options on top of the default", () => {
		createOpenRouter({ extraBody: { usage: { foo: "bar" } } });
		expect(lastCall().extraBody?.usage).toEqual({ include: true, foo: "bar" });
	});

	it("lets the caller turn usage.include off", () => {
		createOpenRouter({ extraBody: { usage: { include: false } } });
		expect(lastCall().extraBody?.usage).toEqual({ include: false });
	});

	it("forwards arbitrary settings unchanged", () => {
		createOpenRouter({ baseURL: "https://example.com" } as never);
		expect((lastCall() as Record<string, unknown>).baseURL).toBe("https://example.com");
	});
});
