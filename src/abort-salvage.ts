import type { ModelMessage } from "ai";

export interface StreamRecorder {
  textBuffer: string;
  toolCalls: Array<{ toolCallId: string; toolName: string; input: unknown }>;
  toolResults: Map<string, unknown>;
}

export const INTERRUPTED_TOOL_RESULT_VALUE =
  "Interrupted by user before tool completed. Awaiting further instruction.";

export function createStreamRecorder(): StreamRecorder {
  return { textBuffer: "", toolCalls: [], toolResults: new Map() };
}

export function recordTextDelta(recorder: StreamRecorder, text: string): void {
  recorder.textBuffer += text;
}

export function recordToolCall(
  recorder: StreamRecorder,
  toolCallId: string,
  toolName: string,
  input: unknown,
): void {
  recorder.toolCalls.push({ toolCallId, toolName, input });
}

export function recordToolResult(
  recorder: StreamRecorder,
  toolCallId: string,
  output: unknown,
): void {
  recorder.toolResults.set(toolCallId, output);
}

export function resetRecorderForNextStep(recorder: StreamRecorder): void {
  recorder.textBuffer = "";
  recorder.toolCalls = [];
  recorder.toolResults.clear();
}

export function collectAbortedRunMessages(
  recordedStepsOnAbort: any[],
  recorder: StreamRecorder,
): ModelMessage[] {
  const lastStep = recordedStepsOnAbort[recordedStepsOnAbort.length - 1];
  const fromSteps: ModelMessage[] = Array.isArray(lastStep?.response?.messages)
    ? lastStep.response.messages
    : [];
  return [...fromSteps, ...buildSalvageMessages(recorder)];
}

function buildSalvageMessages(recorder: StreamRecorder): ModelMessage[] {
  const assistantParts: any[] = [];
  if (recorder.textBuffer) {
    assistantParts.push({ type: "text", text: recorder.textBuffer });
  }

  for (const toolCall of recorder.toolCalls) {
    assistantParts.push({
      type: "tool-call",
      toolCallId: toolCall.toolCallId,
      toolName: toolCall.toolName,
      input: toolCall.input,
    });
  }

  if (assistantParts.length === 0) return [];

  const messages: ModelMessage[] = [{ role: "assistant", content: assistantParts }];
  if (recorder.toolCalls.length === 0) return messages;

  const toolParts = recorder.toolCalls.map((toolCall) => {
    const observed = recorder.toolResults.get(toolCall.toolCallId);
    if (observed !== undefined) {
      return {
        type: "tool-result",
        toolCallId: toolCall.toolCallId,
        toolName: toolCall.toolName,
        output: { type: "json", value: observed },
      };
    }
    return {
      type: "tool-result",
      toolCallId: toolCall.toolCallId,
      toolName: toolCall.toolName,
      output: { type: "error-text", value: INTERRUPTED_TOOL_RESULT_VALUE },
    };
  });

  messages.push({ role: "tool", content: toolParts as any });
  return messages;
}
