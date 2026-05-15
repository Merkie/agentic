export interface ToolFactoryContext<TContext extends Record<string, unknown> = Record<string, unknown>> {
  sessionId: string;
  runId: string;
  abortSignal?: AbortSignal;
  context: TContext;
}

export type ToolFactory<
  TTools extends Record<string, any>,
  TContext extends Record<string, unknown> = Record<string, unknown>,
> = (ctx: ToolFactoryContext<TContext>) => TTools;

export function createTools<
  TTools extends Record<string, any>,
  TContext extends Record<string, unknown> = Record<string, unknown>,
>(factory: ToolFactory<TTools, TContext>): ToolFactory<TTools, TContext> {
  return factory;
}

export function formatSystemNotification(content: string, metadata?: Record<string, unknown>): string {
  const metadataBlock =
    metadata && Object.keys(metadata).length > 0
      ? `\n\nMetadata:\n${JSON.stringify(metadata, null, 2)}`
      : "";
  return `<SystemNotification>\n${content}${metadataBlock}\n</SystemNotification>`;
}
