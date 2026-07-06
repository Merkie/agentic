import chalk from "chalk";
import type { AgenticEvent, EventListener } from "./types.js";

function fmtCost(cost: number): string {
	return `$${cost.toFixed(6)}`;
}

function fmtTokens(tokens: number): string {
	return tokens.toLocaleString("en-US");
}

function shortId(id: string): string {
	return id.length > 8 ? id.slice(0, 8) : id;
}

/**
 * The default console logger for the observability firehose — one colored
 * line per {@link AgenticEvent}. Turn it on with `createAgentic({ logs: true })`,
 * or pass it (or wrap it) as `onEvent` yourself.
 */
export const logEvents: EventListener = (event: AgenticEvent): void => {
	switch (event.type) {
		case "run-start":
			console.log(
				chalk.dim(
					`▶ run-start   ${event.model} · session=${event.sessionId} · run=${shortId(event.runId)}`,
				),
			);
			break;
		case "step": {
			const context = (event.usage.inputTokens ?? 0) + (event.usage.outputTokens ?? 0);
			const cost = event.usage.billedCost ?? event.usage.cost;
			const tools = event.toolCalls.map((t) => t.toolName).join(", ");
			console.log(
				chalk.cyan(
					`· step        ${event.finishReason} · context=${fmtTokens(context)}t` +
						(cost != null ? ` · ${fmtCost(cost)}` : "") +
						(tools ? ` · tools: ${tools}` : ""),
				),
			);
			break;
		}
		case "retry":
			console.log(
				chalk.yellow(
					`↻ retry       ${event.attempt}/${event.maxAttempts} in ${event.delayMs}ms — ${event.error.message}`,
				),
			);
			break;
		case "compaction":
			console.log(
				chalk.magenta(
					`✂ compaction  ${
						event.beforeTokens != null ? `${fmtTokens(event.beforeTokens)}t` : "?t"
					} → ${event.summaryChars}-char summary`,
				),
			);
			break;
		case "poke":
			console.log(
				chalk.yellow(
					`✎ poke        ${event.poke}/${event.maxPokes} — turn ended without a terminal tool`,
				),
			);
			break;
		case "queued-message":
			console.log(
				chalk.blue(
					`✉ queued      message queued into live run${event.runId ? ` ${shortId(event.runId)}` : ""}`,
				),
			);
			break;
		case "run-end": {
			const line =
				`■ run-end     ${event.status} · ${event.totals.steps} steps · ` +
				fmtCost(event.totals.cost) +
				(event.error ? ` — ${event.error.message}` : "");
			console.log(event.status === "completed" ? chalk.green(line) : chalk.red(line));
			break;
		}
	}
};
