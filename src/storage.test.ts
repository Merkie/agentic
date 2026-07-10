import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { decodeEvent, encodeEvent } from "./serialize.js";
import { fileStorage, memoryStorage } from "./storage.js";
import type { StoredEvent } from "./types.js";

const tempDirs: string[] = [];

afterEach(async () => {
	await Promise.all(tempDirs.splice(0).map((dir) => fsp.rm(dir, { recursive: true, force: true })));
});

type UserMessageEvent = Extract<StoredEvent, { type: "user-message" }>;

function multimodalEvent(image: Uint8Array, audio: ArrayBuffer): UserMessageEvent {
	return {
		type: "user-message",
		at: "t",
		message: {
			role: "user",
			content: [
				{ type: "image", image, mediaType: "image/png" },
				{ type: "file", data: audio, mediaType: "audio/wav" },
			],
		},
	};
}

function binaryParts(event: StoredEvent): [Uint8Array, Uint8Array] {
	if (event.type !== "user-message" || !Array.isArray(event.message.content)) {
		throw new Error("expected a multimodal user message");
	}
	const [imagePart, audioPart] = event.message.content as Array<Record<string, unknown>>;
	return [imagePart.image as Uint8Array, audioPart.data as Uint8Array];
}

describe("event serialization", () => {
	it("round-trips Uint8Array, Buffer, and ArrayBuffer values as Uint8Array", () => {
		const event = multimodalEvent(Uint8Array.of(137, 80, 78, 71), Uint8Array.of(1, 2, 3).buffer);
		const content = event.message.content as Array<Record<string, unknown>>;
		content.push({
			type: "file",
			data: Buffer.from([4, 5, 6]),
			mediaType: "application/octet-stream",
		});

		const decoded = decodeEvent(encodeEvent(event));
		const [image, audio] = binaryParts(decoded);
		if (decoded.type !== "user-message" || !Array.isArray(decoded.message.content)) {
			throw new Error("expected a multimodal user message");
		}
		const buffer = (decoded.message.content as Array<Record<string, unknown>>)[2]
			.data as Uint8Array;

		expect(image).toBeInstanceOf(Uint8Array);
		expect([...image]).toEqual([137, 80, 78, 71]);
		expect(audio).toBeInstanceOf(Uint8Array);
		expect([...audio]).toEqual([1, 2, 3]);
		expect(buffer).toBeInstanceOf(Uint8Array);
		expect(Buffer.isBuffer(buffer)).toBe(false);
		expect([...buffer]).toEqual([4, 5, 6]);
	});

	it("escapes ordinary objects that own codec marker keys", () => {
		const markerObjects = {
			exactBinaryEnvelope: { $agenticBinary: "AQ==" },
			binaryMarkerWithData: { $agenticBinary: "AQ==", label: "user data" },
			exactEscapeEnvelope: { $agenticEscaped: { nested: true } },
			nested: { value: { $agenticBinary: "Ag==" } },
		};
		const event: StoredEvent = {
			type: "user-message",
			at: "t",
			message: { role: "user", content: "marker collision" },
			meta: markerObjects,
		};

		const decoded = decodeEvent(encodeEvent(event));
		expect(decoded).toEqual(event);
		if (decoded.type !== "user-message") throw new Error("expected user-message");
		expect(decoded.meta?.exactBinaryEnvelope).not.toBeInstanceOf(Uint8Array);
	});
});

describe("memoryStorage", () => {
	it("preserves binary data and isolates stored snapshots from caller mutation", async () => {
		const image = Uint8Array.of(1, 2, 3);
		const event = multimodalEvent(image, Uint8Array.of(4, 5, 6).buffer);
		const storage = memoryStorage();
		await storage.append("media", event);

		image[0] = 99;
		const first = await storage.load("media");
		const [loadedImage] = binaryParts(first[0]);
		expect([...loadedImage]).toEqual([1, 2, 3]);

		loadedImage[1] = 88;
		const [reloadedImage] = binaryParts((await storage.load("media"))[0]);
		expect([...reloadedImage]).toEqual([1, 2, 3]);
	});
});

describe("fileStorage", () => {
	it("round-trips multimodal bytes through JSONL", async () => {
		const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "agentic-storage-"));
		tempDirs.push(dir);
		const storage = fileStorage(dir);
		await storage.append(
			"chat:media/1",
			multimodalEvent(Uint8Array.of(9, 8, 7), Uint8Array.of(6, 5, 4).buffer),
		);

		const events = await storage.load("chat:media/1");
		const [image, audio] = binaryParts(events[0]);
		expect([...image]).toEqual([9, 8, 7]);
		expect([...audio]).toEqual([6, 5, 4]);
	});

	it("drops only an invalid non-newline-terminated final record", async () => {
		const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "agentic-storage-"));
		tempDirs.push(dir);
		const storage = fileStorage(dir);
		const complete: StoredEvent = {
			type: "run-start",
			at: "t1",
			runId: "r1",
			model: "m",
		};
		await fsp.writeFile(
			path.join(dir, "torn.jsonl"),
			`${encodeEvent(complete)}\n{"type":"step","at":"t2"`,
			"utf8",
		);

		expect(await storage.load("torn")).toEqual([complete]);
	});

	it("rejects an invalid middle record", async () => {
		const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "agentic-storage-"));
		tempDirs.push(dir);
		const storage = fileStorage(dir);
		const first: StoredEvent = {
			type: "run-start",
			at: "t1",
			runId: "r1",
			model: "m",
		};
		const last: StoredEvent = {
			type: "run-end",
			at: "t3",
			runId: "r1",
			status: "completed",
		};
		await fsp.writeFile(
			path.join(dir, "middle-corrupt.jsonl"),
			`${encodeEvent(first)}\nnot-json\n${encodeEvent(last)}\n`,
			"utf8",
		);

		await expect(storage.load("middle-corrupt")).rejects.toThrow();
	});

	it("rejects an invalid newline-terminated final record", async () => {
		const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "agentic-storage-"));
		tempDirs.push(dir);
		const storage = fileStorage(dir);
		await fsp.writeFile(path.join(dir, "final-corrupt.jsonl"), "not-json\n", "utf8");

		await expect(storage.load("final-corrupt")).rejects.toThrow();
	});

	it("continues a session append chain after a failed write", async () => {
		const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "agentic-storage-"));
		tempDirs.push(dir);
		const storage = fileStorage(dir);
		await fsp.rm(dir, { recursive: true });
		await expect(
			storage.append("recover", {
				type: "run-start",
				at: "t1",
				runId: "r1",
				model: "m",
			}),
		).rejects.toThrow();

		await fsp.mkdir(dir, { recursive: true });
		await storage.append("recover", {
			type: "run-start",
			at: "t2",
			runId: "r2",
			model: "m",
		});
		expect(await storage.load("recover")).toEqual([
			expect.objectContaining({ type: "run-start", runId: "r2" }),
		]);
	});

	it("uses distinct reversible filenames for Unicode ids that collided in the legacy codec", async () => {
		const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "agentic-storage-"));
		tempDirs.push(dir);
		const storage = fileStorage(dir);
		const ids = ["\u0100", "\u000100", "chat:用户/😀"];
		for (const [index, id] of ids.entries()) {
			await storage.append(id, {
				type: "run-start",
				at: "t",
				runId: `r${index}`,
				model: "m",
			});
		}

		for (const [index, id] of ids.entries()) {
			expect(await storage.load(id)).toEqual([
				expect.objectContaining({ type: "run-start", runId: `r${index}` }),
			]);
		}
		expect(new Set(await storage.listSessions?.())).toEqual(new Set(ids));
		const names = await fsp.readdir(dir);
		expect(names).toHaveLength(ids.length);
		expect(names.every((name) => name.startsWith("~"))).toBe(true);
	});

	it("loads and continues existing legacy ASCII session files", async () => {
		const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "agentic-storage-"));
		tempDirs.push(dir);
		const first: StoredEvent = {
			type: "run-start",
			at: "t1",
			runId: "r1",
			model: "m",
		};
		await fsp.writeFile(
			path.join(dir, "chat%3auser%2f123.jsonl"),
			`${encodeEvent(first)}\n`,
			"utf8",
		);
		const storage = fileStorage(dir);
		expect(await storage.load("chat:user/123")).toEqual([first]);

		await storage.append("chat:user/123", {
			type: "run-end",
			at: "t2",
			runId: "r1",
			status: "completed",
		});
		expect(await storage.load("chat:user/123")).toHaveLength(2);
		expect(await storage.listSessions?.()).toContain("chat:user/123");
		expect(await fsp.readdir(dir)).toEqual(["chat%3auser%2f123.jsonl"]);
	});
});
