import { defineConfig } from "tsup";

export default defineConfig({
	entry: ["src/index.ts"],
	format: ["esm", "cjs"],
	dts: true,
	clean: true,
	sourcemap: true,
	target: "node18",
	// These are provided by the consuming app, never bundled in.
	external: ["ai", "@openrouter/ai-sdk-provider", "chalk"],
});
