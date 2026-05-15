import { describe, expect, it } from "vitest";
import { createTools, formatSystemNotification } from "../src/index.js";

describe("tool helpers", () => {
  it("binds run context through a closure factory", async () => {
    const factory = createTools((ctx) => ({
      whoami: {
        execute: async () => ({ userId: ctx.context.userId, runId: ctx.runId }),
      },
    }));

    const tools = factory({
      sessionId: "s1",
      runId: "r1",
      context: { userId: "u1" },
    });

    await expect(tools.whoami.execute()).resolves.toEqual({ userId: "u1", runId: "r1" });
  });

  it("formats system notifications as explicit prompt blocks", () => {
    expect(formatSystemNotification("Task done", { taskId: "t1" })).toContain(
      "<SystemNotification>",
    );
    expect(formatSystemNotification("Task done", { taskId: "t1" })).toContain('"taskId": "t1"');
  });
});
