import type { ModelMessage } from "ai";
import type { ToolFactory } from "./tools.js";

export type AgenticToolSet = Record<string, any>;

export type AgenticInput =
  | string
  | ModelMessage
  | ModelMessage[]
  | AgenticInputMessage;

export interface AgenticInputMessage {
  role?: "user" | "system" | "assistant" | "tool";
  content: string;
  metadata?: Record<string, unknown>;
}

export interface AgenticSessionOptions {
  id: string;
  system?: string;
  model?: string;
  tools?: AgenticToolSet | ToolFactory<AgenticToolSet>;
  maxSteps?: number;
  logFile?: string;
  context?: Record<string, unknown>;
}

export interface AgenticConfig {
  openRouterApiKey: string;
  defaultModel?: string;
  sessionsDir?: string;
  maxSteps?: number;
  streamTextImpl?: StreamTextImpl;
  getModelContextLength?: (model: string) => Promise<number | null> | number | null;
}

export type StreamTextImpl = (options: Record<string, unknown>) => StreamTextLike;

export interface StreamTextLike {
  fullStream: AsyncIterable<any>;
  text?: PromiseLike<string> | string;
  usage?: PromiseLike<AgenticUsageLike> | AgenticUsageLike;
  finishReason?: PromiseLike<string> | string;
  response?: PromiseLike<{ messages?: ModelMessage[] }> | { messages?: ModelMessage[] };
}

export interface AgenticUsageLike {
  inputTokens?: number | undefined;
  outputTokens?: number | undefined;
  totalTokens?: number | undefined;
  promptTokens?: number | undefined;
  completionTokens?: number | undefined;
  [key: string]: unknown;
}

export type AgenticStreamEvent =
  | { type: "run-start"; sessionId: string; runId: string; messages: ModelMessage[] }
  | { type: "text-start"; sessionId: string; runId: string; raw: unknown }
  | { type: "text-delta"; sessionId: string; runId: string; text: string; raw: unknown }
  | { type: "text-end"; sessionId: string; runId: string; raw: unknown }
  | {
      type: "tool-input-start";
      sessionId: string;
      runId: string;
      toolName: string;
      toolCallId: string;
      raw: unknown;
    }
  | {
      type: "tool-input-delta";
      sessionId: string;
      runId: string;
      toolCallId: string;
      delta: string;
      raw: unknown;
    }
  | { type: "tool-input-end"; sessionId: string; runId: string; toolCallId: string; raw: unknown }
  | {
      type: "tool-call";
      sessionId: string;
      runId: string;
      toolName: string;
      toolCallId: string;
      input: unknown;
      raw: unknown;
    }
  | {
      type: "tool-result";
      sessionId: string;
      runId: string;
      toolName: string;
      toolCallId: string;
      output: unknown;
      raw: unknown;
    }
  | { type: "step-finish"; sessionId: string; runId: string; raw: unknown }
  | { type: "finish"; sessionId: string; runId: string; finishReason: string; raw?: unknown }
  | { type: "abort"; sessionId: string; runId: string; reason?: unknown; raw?: unknown }
  | { type: "error"; sessionId: string; runId: string; error: unknown; raw?: unknown };

export interface AgenticRunResult {
  sessionId: string;
  runId: string;
  model: string;
  messages: ModelMessage[];
  newMessages: ModelMessage[];
  text: string;
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  cost: {
    llm: number;
    total: number;
    entries: Array<{ source: string; cost: number; metadata?: Record<string, unknown> }>;
  };
  context: {
    contextLength: number | null;
    usedTokens: number;
    usedPct: number | null;
    remainingPct: number | null;
  };
  finishReason: string;
  aborted: boolean;
  logFile: string;
}

export type AgenticEventHandler = (event: AgenticStreamEvent) => void | Promise<void>;

export interface OpenRouterModel {
  id: string;
  canonical_slug?: string;
  name: string;
  description?: string;
  created?: number;
  context_length?: number;
  architecture?: {
    modality?: string;
    input_modalities?: string[];
    output_modalities?: string[];
    tokenizer?: string;
  };
  pricing?: {
    prompt?: string;
    completion?: string;
    [key: string]: unknown;
  };
  top_provider?: {
    context_length?: number;
    max_completion_tokens?: number;
    is_moderated?: boolean;
  };
  [key: string]: unknown;
}

export interface OpenRouterModelSummary {
  id: string;
  name: string;
  description: string;
  contextLength: number;
  pricing: {
    prompt: string;
    completion: string;
  };
  created: number;
}
