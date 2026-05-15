import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { JsonlLogger, countJsonlStreamEvents, replayJsonl } from "../src/index.js";

function tmpFile(name: string): string {
  return path.join(fs.mkdtempSync(path.join(os.tmpdir(), "agentic-jsonl-")), name);
}

describe("jsonl replay", () => {
  it("reconstructs multi-turn ModelMessage history from raw JSONL", () => {
    const file = tmpFile("chat.jsonl");
    const logger = new JsonlLogger(file);
    logger.append({
      kind: "run_start",
      model: "test/model",
      system: "system",
      inputMessages: [{ role: "user", content: "first" }],
    });
    logger.append({
      kind: "stream_event",
      event: { type: "tool-input-start", toolName: "demo", id: "call_1" },
    });
    logger.append({
      kind: "step_messages",
      message: { role: "assistant", content: "answer one" },
    });
    logger.append({ kind: "run_end", finishReason: "stop" });
    logger.append({
      kind: "run_start",
      model: "test/model",
      inputMessages: [{ role: "user", content: "second" }],
    });
    logger.append({
      kind: "step_messages",
      message: { role: "assistant", content: "answer two" },
    });
    logger.append({ kind: "run_end", finishReason: "stop" });
    logger.close();

    const replayed = replayJsonl(file);
    expect(replayed.model).toBe("test/model");
    expect(replayed.system).toBe("system");
    expect(replayed.fullMessages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(countJsonlStreamEvents(file, "tool-input-start")).toBe(1);
  });
});
