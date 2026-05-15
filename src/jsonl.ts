import fs from "node:fs";
import path from "node:path";
import type { ModelMessage } from "ai";

export type JsonlRecord = Record<string, any>;

export class JsonlLogger {
  private fd: number;
  public lineCount = 0;

  constructor(public readonly file: string) {
    fs.mkdirSync(path.dirname(path.resolve(file)), { recursive: true });
    this.fd = fs.openSync(file, "a");
  }

  append(record: JsonlRecord): void {
    fs.writeSync(this.fd, `${JSON.stringify({ t: Date.now(), ...record }, jsonReplacer)}\n`);
    this.lineCount += 1;
  }

  close(): void {
    fs.closeSync(this.fd);
  }
}

export function appendJsonl(file: string, record: JsonlRecord): void {
  const logger = new JsonlLogger(file);
  try {
    logger.append(record);
  } finally {
    logger.close();
  }
}

export function readJsonl(file: string): JsonlRecord[] {
  if (!fs.existsSync(file)) return [];
  const text = fs.readFileSync(file, "utf8");
  return text
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as JsonlRecord);
}

export interface ReplayedJsonlSession {
  model: string | undefined;
  system: string | undefined;
  inputMessages: ModelMessage[];
  stepMessages: ModelMessage[];
  fullMessages: ModelMessage[];
  finishReason: string | undefined;
  aborted: boolean;
}

export function replayJsonl(file: string): ReplayedJsonlSession {
  const lines = readJsonl(file);
  const firstRunStart = lines.find((line) => line.kind === "run_start");
  const endLines = lines.filter(
    (line) => line.kind === "run_end" || line.kind === "run_aborted",
  );
  const lastEnd = endLines[endLines.length - 1];

  const fullMessages: ModelMessage[] = [];
  const inputMessages: ModelMessage[] = [];
  const stepMessages: ModelMessage[] = [];

  for (const line of lines) {
    if (line.kind === "run_start") {
      const next = (line.inputMessages ?? []) as ModelMessage[];
      inputMessages.push(...next);
      fullMessages.push(...next);
      continue;
    }
    if (line.kind === "step_messages") {
      const message = line.message as ModelMessage | undefined;
      if (message) {
        stepMessages.push(message);
        fullMessages.push(message);
      }
    }
  }

  return {
    model: firstRunStart?.model,
    system: firstRunStart?.system,
    inputMessages,
    stepMessages,
    fullMessages,
    finishReason: lastEnd?.finishReason,
    aborted: lastEnd?.kind === "run_aborted",
  };
}

export function countJsonlStreamEvents(file: string, type: string): number {
  return readJsonl(file).filter(
    (line) => line.kind === "stream_event" && line.event?.type === type,
  ).length;
}

function jsonReplacer(_key: string, value: unknown): unknown {
  if (value instanceof Error) {
    return { name: value.name, message: value.message, stack: value.stack };
  }
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "function") return undefined;
  return value;
}
