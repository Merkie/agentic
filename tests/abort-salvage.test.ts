import { describe, expect, it } from "vitest";
import {
  INTERRUPTED_TOOL_RESULT_VALUE,
  collectAbortedRunMessages,
  createStreamRecorder,
  recordTextDelta,
  recordToolCall,
} from "../src/index.js";

describe("abort salvage", () => {
  it("creates replayable tool results for dangling tool calls", () => {
    const recorder = createStreamRecorder();
    recordTextDelta(recorder, "Partial text.");
    recordToolCall(recorder, "call_1", "write_file", { path: "app.ts" });

    const messages = collectAbortedRunMessages([], recorder);

    expect(messages).toHaveLength(2);
    expect(messages[0]?.role).toBe("assistant");
    expect(messages[1]?.role).toBe("tool");
    const toolMessage = messages[1] as any;
    expect(toolMessage.content[0].output.value).toBe(INTERRUPTED_TOOL_RESULT_VALUE);
  });
});
