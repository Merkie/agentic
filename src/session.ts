import path from "node:path";
import { streamText, stepCountIs, type ModelMessage } from "ai";
import type { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { JsonlLogger, replayJsonl } from "./jsonl.js";
import {
  collectAbortedRunMessages,
  createStreamRecorder,
  recordTextDelta,
  recordToolCall,
  recordToolResult,
  resetRecorderForNextStep,
} from "./abort-salvage.js";
import { calculateContextUsage, extractOpenRouterStepCost, normalizeUsage } from "./context.js";
import type { ToolFactory } from "./tools.js";
import type {
  AgenticConfig,
  AgenticEventHandler,
  AgenticInput,
  AgenticRunResult,
  AgenticSessionOptions,
  AgenticStreamEvent,
  AgenticToolSet,
  StreamTextImpl,
} from "./types.js";

const DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview";
const DEFAULT_SESSIONS_DIR = "sessions";
const DEFAULT_MAX_STEPS = 8;

type OpenRouterProvider = ReturnType<typeof createOpenRouter>;

interface PendingTurn {
  messages: ModelMessage[];
  resolve: (result: AgenticRunResult) => void;
  reject: (error: unknown) => void;
}

export interface AgenticRuntime {
  getSession(options: AgenticSessionOptions): AgenticSession;
  getOpenRouterProvider(): OpenRouterProvider;
}

export function createAgentic(config: AgenticConfig): AgenticRuntime {
  const streamTextImpl = config.streamTextImpl ?? ((options) => streamText(options as any));
  const sessions = new Map<string, AgenticSession>();
  let provider: OpenRouterProvider | null = null;

  function getOpenRouterProvider(): OpenRouterProvider {
    if (!provider) {
      provider = createProvider(config.openRouterApiKey);
    }
    return provider;
  }

  return {
    getSession(options) {
      const existing = sessions.get(options.id);
      if (existing) {
        existing.configure(options);
        return existing;
      }
      const session = new AgenticSession({
        config,
        options,
        streamTextImpl,
        getOpenRouterProvider,
      });
      sessions.set(options.id, session);
      return session;
    },
    getOpenRouterProvider,
  };
}

function createProvider(apiKey: string): OpenRouterProvider {
  return (globalThis as any).__agenticCreateOpenRouter
    ? (globalThis as any).__agenticCreateOpenRouter(apiKey)
    : // imported lazily by dynamic import would complicate the sync public API;
      // this require-like hook is only used in tests. Production uses the real import below.
      realCreateOpenRouter({
        apiKey,
        extraBody: { usage: { include: true } } as any,
      });
}

import { createOpenRouter as realCreateOpenRouter } from "@openrouter/ai-sdk-provider";

export class AgenticSession {
  private options: AgenticSessionOptions;
  private listeners = new Set<AgenticEventHandler>();
  private running = false;
  private pendingQueue: PendingTurn[] = [];
  private abortController: AbortController | null = null;
  private messages: ModelMessage[] = [];
  private initializedFromLog = false;

  constructor(
    private readonly runtime: {
      config: AgenticConfig;
      options: AgenticSessionOptions;
      streamTextImpl: StreamTextImpl;
      getOpenRouterProvider: () => OpenRouterProvider;
    },
  ) {
    this.options = runtime.options;
  }

  get id(): string {
    return this.options.id;
  }

  get logFile(): string {
    return (
      this.options.logFile ??
      path.join(
        this.runtime.config.sessionsDir ?? DEFAULT_SESSIONS_DIR,
        `${sanitizeFileName(this.options.id)}.jsonl`,
      )
    );
  }

  get history(): ModelMessage[] {
    this.ensureInitializedFromLog();
    return [...this.messages];
  }

  get isRunning(): boolean {
    return this.running;
  }

  configure(options: AgenticSessionOptions): void {
    this.options = { ...this.options, ...options };
  }

  onEvent(handler: AgenticEventHandler): () => void {
    this.listeners.add(handler);
    return () => {
      this.listeners.delete(handler);
    };
  }

  send(input: AgenticInput): Promise<AgenticRunResult> {
    return this.enqueue(normalizeInput(input));
  }

  abort(reason: unknown = "Session aborted"): boolean {
    if (!this.abortController) return false;
    this.abortController.abort(reason);
    this.abortController = null;
    return true;
  }

  private enqueue(messages: ModelMessage[]): Promise<AgenticRunResult> {
    return new Promise((resolve, reject) => {
      this.pendingQueue.push({ messages, resolve, reject });
      void this.drainQueue();
    });
  }

  private async drainQueue(): Promise<void> {
    if (this.running) return;
    this.ensureInitializedFromLog();

    while (this.pendingQueue.length > 0) {
      const pending = this.pendingQueue.splice(0);
      const messages = pending.flatMap((item) => item.messages);
      this.running = true;
      try {
        const result = await this.runTurn(messages);
        for (const item of pending) item.resolve(result);
      } catch (error) {
        for (const item of pending) item.reject(error);
      } finally {
        this.running = false;
      }
    }
  }

  private ensureInitializedFromLog(): void {
    if (this.initializedFromLog) return;
    const replayed = replayJsonl(this.logFile);
    this.messages = replayed.fullMessages;
    this.initializedFromLog = true;
  }

  private async runTurn(newMessages: ModelMessage[]): Promise<AgenticRunResult> {
    const runId = createRunId();
    const model = this.options.model ?? this.runtime.config.defaultModel ?? DEFAULT_MODEL;
    const maxSteps = this.options.maxSteps ?? this.runtime.config.maxSteps ?? DEFAULT_MAX_STEPS;
    const logger = new JsonlLogger(this.logFile);
    const abortController = new AbortController();
    this.abortController = abortController;

    const inputMessages = [...this.messages, ...newMessages];
    this.messages = inputMessages;

    const costEntries: Array<{ source: string; cost: number; metadata?: Record<string, unknown> }> = [];
    const stepMessages: ModelMessage[] = [];
    const recorder = createStreamRecorder();
    let recordedStepsOnAbort: any[] = [];
    let sawAbortChunk = false;
    let fullText = "";

    const emit = async (event: AgenticStreamEvent) => {
      for (const listener of this.listeners) {
        await listener(event);
      }
    };

    logger.append({
      kind: "run_start",
      sessionId: this.id,
      runId,
      model,
      system: this.options.system,
      inputMessages: newMessages,
    });

    await emit({ type: "run-start", sessionId: this.id, runId, messages: newMessages });

    try {
      const tools = resolveTools({
        tools: this.options.tools,
        sessionId: this.id,
        runId,
        abortSignal: abortController.signal,
        context: this.options.context ?? {},
      });
      const provider = this.runtime.getOpenRouterProvider();
      const result = this.runtime.streamTextImpl({
        model: provider.chat(model),
        system: this.options.system,
        messages: inputMessages,
        tools,
        stopWhen: stepCountIs(maxSteps),
        abortSignal: abortController.signal,
        onAbort: ({ steps }: any) => {
          recordedStepsOnAbort = steps ?? [];
        },
        onStepFinish: (step: any) => {
          const cost = extractOpenRouterStepCost(step);
          if (cost > 0) {
            costEntries.push({
              source: "llm",
              cost,
              metadata: { model, stepNumber: step?.stepNumber },
            });
          }
          for (const message of step?.response?.messages ?? []) {
            stepMessages.push(message);
            logger.append({
              kind: "step_messages",
              sessionId: this.id,
              runId,
              stepNumber: step?.stepNumber,
              finishReason: step?.finishReason,
              message,
            });
          }
        },
      });

      for await (const part of result.fullStream) {
        logger.append({ kind: "stream_event", sessionId: this.id, runId, event: part });
        switch (part.type) {
          case "text-start":
            await emit({ type: "text-start", sessionId: this.id, runId, raw: part });
            break;
          case "text-delta": {
            const text = part.text ?? part.delta ?? "";
            fullText += text;
            recordTextDelta(recorder, text);
            await emit({ type: "text-delta", sessionId: this.id, runId, text, raw: part });
            break;
          }
          case "text-end":
            await emit({ type: "text-end", sessionId: this.id, runId, raw: part });
            break;
          case "tool-input-start":
            await emit({
              type: "tool-input-start",
              sessionId: this.id,
              runId,
              toolName: part.toolName,
              toolCallId: part.id,
              raw: part,
            });
            break;
          case "tool-input-delta":
            await emit({
              type: "tool-input-delta",
              sessionId: this.id,
              runId,
              toolCallId: part.id,
              delta: part.delta ?? "",
              raw: part,
            });
            break;
          case "tool-input-end":
            await emit({
              type: "tool-input-end",
              sessionId: this.id,
              runId,
              toolCallId: part.id,
              raw: part,
            });
            break;
          case "tool-call":
            recordToolCall(recorder, part.toolCallId, part.toolName, part.input);
            await emit({
              type: "tool-call",
              sessionId: this.id,
              runId,
              toolName: part.toolName,
              toolCallId: part.toolCallId,
              input: part.input,
              raw: part,
            });
            break;
          case "tool-result":
            recordToolResult(recorder, part.toolCallId, part.output);
            await emit({
              type: "tool-result",
              sessionId: this.id,
              runId,
              toolName: part.toolName,
              toolCallId: part.toolCallId,
              output: part.output,
              raw: part,
            });
            break;
          case "finish-step":
            resetRecorderForNextStep(recorder);
            await emit({ type: "step-finish", sessionId: this.id, runId, raw: part });
            break;
          case "abort":
            sawAbortChunk = true;
            await emit({
              type: "abort",
              sessionId: this.id,
              runId,
              reason: abortController.signal.reason,
              raw: part,
            });
            break;
          case "error":
            await emit({ type: "error", sessionId: this.id, runId, error: part.error, raw: part });
            break;
        }
      }

      const aborted = sawAbortChunk || abortController.signal.aborted;
      const alreadyLoggedStepMessageCount = stepMessages.length;
      const newModelMessages = aborted
        ? collectAbortedRunMessages(recordedStepsOnAbort, recorder)
        : stepMessages.length > 0
          ? stepMessages
          : ((await resolveMaybe(result.response))?.messages ?? []);

      for (const message of newModelMessages.slice(alreadyLoggedStepMessageCount)) {
        logger.append({ kind: "step_messages", sessionId: this.id, runId, message });
      }

      this.messages = [...inputMessages, ...newModelMessages];
      const finishReason = aborted
        ? "abort"
        : ((await resolveMaybe(result.finishReason)) ?? "unknown");
      const usage = normalizeUsage(await resolveMaybe(result.usage));
      const contextLength = await getContextLengthBestEffort(
        model,
        this.runtime.config.getModelContextLength,
      );
      const context = calculateContextUsage({
        usedTokens: usage.totalTokens,
        contextLength,
      });
      const llmCost = costEntries.reduce((sum, entry) => sum + entry.cost, 0);
      const cost = { llm: llmCost, total: llmCost, entries: costEntries };

      const resultPayload: AgenticRunResult = {
        sessionId: this.id,
        runId,
        model,
        messages: this.messages,
        newMessages: newModelMessages,
        text: fullText || ((await resolveMaybe(result.text)) ?? ""),
        usage,
        cost,
        context,
        finishReason,
        aborted,
        logFile: this.logFile,
      };

      logger.append({
        kind: aborted ? "run_aborted" : "run_end",
        sessionId: this.id,
        runId,
        finishReason,
        usage,
        cost,
      });
      logger.append({ kind: "cost_summary", sessionId: this.id, runId, ...cost });

      await emit({
        type: aborted ? "abort" : "finish",
        sessionId: this.id,
        runId,
        ...(aborted
          ? { reason: abortController.signal.reason }
          : { finishReason, raw: resultPayload }),
      } as AgenticStreamEvent);

      return resultPayload;
    } catch (error) {
      logger.append({ kind: "run_error", sessionId: this.id, runId, error });
      await emit({ type: "error", sessionId: this.id, runId, error });
      throw error;
    } finally {
      if (this.abortController === abortController) {
        this.abortController = null;
      }
      logger.close();
    }
  }
}

function normalizeInput(input: AgenticInput): ModelMessage[] {
  if (typeof input === "string") return [{ role: "user", content: input }];
  if (Array.isArray(input)) return input;
  if ("role" in input && input.role) {
    return [{ role: input.role as any, content: input.content as any }];
  }
  return [input as ModelMessage];
}

function resolveTools(opts: {
  tools: AgenticToolSet | ToolFactory<AgenticToolSet> | undefined;
  sessionId: string;
  runId: string;
  abortSignal: AbortSignal;
  context: Record<string, unknown>;
}): AgenticToolSet | undefined {
  if (!opts.tools) return undefined;
  if (typeof opts.tools === "function") {
    return opts.tools({
      sessionId: opts.sessionId,
      runId: opts.runId,
      abortSignal: opts.abortSignal,
      context: opts.context,
    });
  }
  return opts.tools;
}

async function resolveMaybe<T>(value: PromiseLike<T> | T | undefined): Promise<T | undefined> {
  return value === undefined ? undefined : await value;
}

async function getContextLengthBestEffort(
  model: string,
  override?: (model: string) => Promise<number | null> | number | null,
): Promise<number | null> {
  try {
    if (override) return await override(model);
    const { getModelContextLength } = await import("./openrouter.js");
    return await getModelContextLength(model);
  } catch {
    return null;
  }
}

function createRunId(): string {
  return `run_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function sanitizeFileName(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]/g, "_");
}
