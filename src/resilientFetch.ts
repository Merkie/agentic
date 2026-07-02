// Stall detection for hung provider connections. Two independent timers:
// HEADER = time to first response headers; SSE CHUNK = max idle gap between
// stream bytes (re-armed per chunk, so long healthy streams are unaffected —
// only genuine silence trips it). Both errors classify as transient, so the
// run loop retries instead of hanging forever on a dead connection.

export class HeaderTimeoutError extends Error {
	override readonly name = "HeaderTimeoutError";
	constructor(ms: number) {
		super(`Response headers timed out after ${ms}ms`);
	}
}

export class StreamIdleTimeoutError extends Error {
	override readonly name = "StreamIdleTimeoutError";
	constructor(ms: number) {
		super(`SSE stream produced no bytes for ${ms}ms`);
	}
}

export interface ResilientFetchOptions {
	/** Max wait for response headers. Default 60s; 0 disables. */
	headerTimeoutMs?: number;
	/** Max idle gap between SSE chunks. Default 120s; 0 disables. */
	streamIdleTimeoutMs?: number;
	fetch?: typeof fetch;
}

function startAbortTimer(ms: number, error: Error) {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(error), ms);
	(timer as { unref?: () => void }).unref?.();
	return { controller, clear: () => clearTimeout(timer) };
}

type ReadResult = Awaited<ReturnType<ReadableStreamDefaultReader<Uint8Array>["read"]>>;

async function readWithTimeout(
	reader: ReadableStreamDefaultReader<Uint8Array>,
	ms: number,
): Promise<ReadResult> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	try {
		return await Promise.race([
			reader.read(),
			new Promise<never>((_, reject) => {
				timer = setTimeout(() => reject(new StreamIdleTimeoutError(ms)), ms);
				(timer as { unref?: () => void }).unref?.();
			}),
		]);
	} finally {
		if (timer) clearTimeout(timer);
	}
}

function wrapSseBody(
	response: Response,
	ms: number,
	idleController: AbortController,
): Response {
	if (!response.body || ms <= 0) return response;
	const contentType = response.headers.get("content-type") ?? "";
	if (!contentType.toLowerCase().includes("text/event-stream")) return response;

	const body = new ReadableStream<Uint8Array>({
		async start(controller) {
			const reader = response.body!.getReader();
			try {
				while (true) {
					const chunk = await readWithTimeout(reader, ms);
					if (chunk.done) {
						controller.close();
						return;
					}
					controller.enqueue(chunk.value);
				}
			} catch (err) {
				if (err instanceof StreamIdleTimeoutError) idleController.abort(err);
				try {
					await reader.cancel(err);
				} catch {
					// stream is already failing; preserve the original error
				}
				controller.error(err);
			}
		},
	});

	return new Response(body, {
		status: response.status,
		statusText: response.statusText,
		headers: response.headers,
	});
}

/**
 * Wrap fetch with header + SSE-idle stall detection. Pass the result as the
 * `fetch` option of `createOpenRouter`. Errors thrown here classify as
 * transient in {@link classifyFailure}, so the agent loop retries them.
 */
export function createResilientFetch(options: ResilientFetchOptions = {}): typeof fetch {
	const fetchImpl = options.fetch ?? fetch;
	const headerTimeoutMs = options.headerTimeoutMs ?? 60_000;
	const streamIdleTimeoutMs = options.streamIdleTimeoutMs ?? 120_000;

	return (async (input, init) => {
		const headerTimer =
			headerTimeoutMs > 0
				? startAbortTimer(headerTimeoutMs, new HeaderTimeoutError(headerTimeoutMs))
				: undefined;
		const idleController = streamIdleTimeoutMs > 0 ? new AbortController() : undefined;

		const signals: AbortSignal[] = [];
		if (init?.signal) signals.push(init.signal as AbortSignal);
		if (headerTimer) signals.push(headerTimer.controller.signal);
		if (idleController) signals.push(idleController.signal);

		try {
			const response = await fetchImpl(input, {
				...init,
				signal: signals.length > 0 ? AbortSignal.any(signals) : undefined,
			});
			headerTimer?.clear();
			if (!idleController) return response;
			return wrapSseBody(response, streamIdleTimeoutMs, idleController);
		} catch (err) {
			if (headerTimer?.controller.signal.aborted) throw headerTimer.controller.signal.reason;
			if (idleController?.signal.aborted) throw idleController.signal.reason;
			throw err;
		} finally {
			headerTimer?.clear();
		}
	}) as typeof fetch;
}
