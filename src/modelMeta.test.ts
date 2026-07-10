import { afterEach, describe, expect, it, vi } from "vitest";
import { getContextWindow } from "./modelMeta.js";

afterEach(() => {
	vi.restoreAllMocks();
});

describe("getContextWindow", () => {
	it("retries after a transient network failure instead of caching null", async () => {
		const fetchMock = vi
			.spyOn(globalThis, "fetch")
			.mockRejectedValueOnce(new Error("temporary network failure"))
			.mockResolvedValueOnce(
				new Response(JSON.stringify({ data: { context_length: 131_072 } }), {
					status: 200,
					headers: { "content-type": "application/json" },
				}),
			);

		await expect(getContextWindow("test/network-retry")).resolves.toBeNull();
		await expect(getContextWindow("test/network-retry")).resolves.toBe(131_072);
		expect(fetchMock).toHaveBeenCalledTimes(2);
		expect(fetchMock.mock.calls[0]?.[1]?.signal).toBeInstanceOf(AbortSignal);
	});

	it("retries transient HTTP failures but memoizes a successful lookup", async () => {
		const fetchMock = vi
			.spyOn(globalThis, "fetch")
			.mockResolvedValueOnce(new Response(null, { status: 503 }))
			.mockResolvedValueOnce(
				new Response(JSON.stringify({ data: { top_provider: { context_length: "65536" } } }), {
					status: 200,
					headers: { "content-type": "application/json" },
				}),
			);

		await expect(getContextWindow("test/http-retry")).resolves.toBeNull();
		await expect(getContextWindow("test/http-retry")).resolves.toBe(65_536);
		await expect(getContextWindow("test/http-retry")).resolves.toBe(65_536);
		expect(fetchMock).toHaveBeenCalledTimes(2);
	});

	it("memoizes a stable unknown-model response", async () => {
		const fetchMock = vi
			.spyOn(globalThis, "fetch")
			.mockResolvedValue(new Response(null, { status: 404 }));

		await expect(getContextWindow("test/unknown-model")).resolves.toBeNull();
		await expect(getContextWindow("test/unknown-model")).resolves.toBeNull();
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});
});
