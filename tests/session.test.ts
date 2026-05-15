import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { createAgentic, formatSystemNotification, readJsonl } from "../src/index.js";
import type { StreamTextImpl } from "../src/index.js";

function tmpFile(name: string): string {
  return path.join(fs.mkdtempSync(path.join(os.tmpdir(), "agentic-session-")), name);
}

function installFakeProvider() {
  (globalThis as any).__agenticCreateOpenRouter = () => ({
    chat: (model: string) => ({ provider: "fake", modelId: model }),
  });
  return () => {
    delete (globalThis as any).__agenticCreateOpenRouter;
  };
}

describe("AgenticSession", () => {
  it("runs streamText, logs step messages, and emits early tool events", async () => {
    const cleanup = installFakeProvider();
    const file = tmpFile("chat.jsonl");
    const streamTextImpl: StreamTextImpl = (options) => {
      (options.onStepFinish as any)({
        stepNumber: 1,
        finishReason: "stop",
        providerMetadata: { openrouter: { usage: { cost: 0.001 } } },
        response: { messages: [{ role: "assistant", content: "Hello" }] },
      });
      return {
        fullStream: (async function* () {
          yield { type: "text-start" };
          yield { type: "text-delta", delta: "Hello" };
          yield { type: "tool-input-start", toolName: "demo", id: "call_1" };
          yield { type: "tool-input-end", id: "call_1" };
          yield { type: "text-end" };
        })(),
        finishReason: Promise.resolve("stop"),
        usage: Promise.resolve({ inputTokens: 3, outputTokens: 2, totalTokens: 5 }),
      };
    };

    try {
      const agentic = createAgentic({
        openRouterApiKey: "test",
        streamTextImpl,
        getModelContextLength: () => 100,
      });
      const session = agentic.getSession({ id: "s1", file });
      const events: string[] = [];
      session.onEvent((event) => {
        events.push(event.type);
      });

      const result = await session.send("hi");

      expect(result.text).toBe("Hello");
      expect(result.cost.total).toBe(0.001);
      expect(result.context.usedPct).toBe(0.05);
      expect(events).toContain("tool-input-start");
      expect(readJsonl(file).some((line) => line.kind === "step_messages")).toBe(true);
    } finally {
      cleanup();
    }
  });

  it("groups queued messages into the next run while a stream is active", async () => {
    const cleanup = installFakeProvider();
    const file = tmpFile("chat.jsonl");
    let releaseFirst: (() => void) | undefined;
    const seenInputLengths: number[] = [];
    const streamTextImpl: StreamTextImpl = (options) => {
      const inputMessages = options.messages as unknown[];
      const runNumber = seenInputLengths.length + 1;
      seenInputLengths.push(inputMessages.length);
      (options.onStepFinish as any)({
        stepNumber: 1,
        finishReason: "stop",
        response: { messages: [{ role: "assistant", content: `run ${runNumber}` }] },
      });
      return {
        fullStream: (async function* () {
          yield { type: "text-start" };
          if (runNumber === 1) {
            await new Promise<void>((resolve) => {
              releaseFirst = resolve;
            });
          }
          yield { type: "text-delta", delta: `run ${runNumber}` };
          yield { type: "text-end" };
        })(),
        finishReason: "stop",
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
      };
    };

    try {
      const agentic = createAgentic({
        openRouterApiKey: "test",
        streamTextImpl,
        getModelContextLength: () => null,
      });
      const session = agentic.getSession({ id: "s1", file });

      const first = session.send("first");
      await vi.waitFor(() => expect(releaseFirst).toBeTypeOf("function"));
      const second = session.send("second");
      const third = session.send({ role: "user", content: formatSystemNotification("third") });

      releaseFirst?.();
      await Promise.all([first, second, third]);

      const runStarts = readJsonl(file).filter((line) => line.kind === "run_start");
      expect(runStarts).toHaveLength(2);
      expect(runStarts[1]?.inputMessages).toHaveLength(2);
      expect(seenInputLengths).toEqual([1, 4]);
    } finally {
      cleanup();
    }
  });

  it("aborts an active stream and persists a run_aborted line", async () => {
    const cleanup = installFakeProvider();
    const file = tmpFile("chat.jsonl");
    let releaseStream: (() => void) | undefined;
    const streamTextImpl: StreamTextImpl = (options) => {
      return {
        fullStream: (async function* () {
          yield { type: "text-start" };
          yield { type: "text-delta", delta: "Partial" };
          await new Promise<void>((resolve) => {
            releaseStream = resolve;
          });
          if ((options.abortSignal as AbortSignal).aborted) {
            (options.onAbort as any)({ steps: [] });
            yield { type: "abort" };
          }
        })(),
        finishReason: "stop",
        usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
      };
    };

    try {
      const agentic = createAgentic({
        openRouterApiKey: "test",
        streamTextImpl,
        getModelContextLength: () => null,
      });
      const session = agentic.getSession({ id: "s1", file });
      const resultPromise = session.send("start");
      await vi.waitFor(() => expect(releaseStream).toBeTypeOf("function"));

      expect(session.abort("stop")).toBe(true);
      releaseStream?.();
      const result = await resultPromise;

      expect(result.aborted).toBe(true);
      expect(result.text).toBe("Partial");
      expect(readJsonl(file).some((line) => line.kind === "run_aborted")).toBe(true);
    } finally {
      cleanup();
    }
  });
});
