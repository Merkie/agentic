import { dynamicTool, stepCountIs, streamText, tool } from "ai";
import { z } from "zod";
import type {
	createOpenRouter as createOpenRouterSource,
	logStream as logStreamSource,
} from "../../src/index.js";
import { MOCK_PRODUCTS } from "./mock.js";

type AgenticApi = {
	createOpenRouter: typeof createOpenRouterSource;
	logStream: typeof logStreamSource;
};

export async function runProductDemo({
	createOpenRouter,
	logStream,
}: AgenticApi) {
	const openrouter = createOpenRouter();

	const result = streamText({
		model: openrouter("xiaomi/mimo-v2.5"),
		stopWhen: [stepCountIs(999)],
		tools: {
			get_products: dynamicTool({
				description: "Get the list of products available in the store.",
				inputSchema: z.object({}),
				execute: async () => {
					return MOCK_PRODUCTS.map((product) => ({
						id: product.id,
						name: product.name,
						price: product.price,
					}));
				},
			}),
			get_product_details: tool({
				description: "Get the details of a specific product by its ID.",
				inputSchema: z.object({
					productId: z.number().int(),
				}),
				execute: async ({ productId }) => {
					const product = MOCK_PRODUCTS.find((p) => p.id === productId);
					if (!product) {
						throw new Error(`Product with ID ${productId} not found.`);
					}

					return {
						id: product.id,
						name: product.name,
						price: product.price,
						description: product.description,
						color: product.color,
						stock: product.stock,
					};
				},
			}),
			get_product_reviews: tool({
				description: "Get the reviews of a specific product by its ID.",
				inputSchema: z.object({
					productId: z.number().int(),
				}),
				execute: async ({ productId }) => {
					const product = MOCK_PRODUCTS.find((p) => p.id === productId);
					if (!product) {
						throw new Error(`Product with ID ${productId} not found.`);
					}

					return product.reviews;
				},
			}),
		},
		system: [
			"You are a helpful assistant that provides information about products in our web store to users.",
			"You can use your provided tools to see a list of our products, get detailed information about a product, and get the reviews for a product.",
			"If a user asks a question that requires looking through multiple products, use the tools as needed to provide the best answer.",
		].join(" "),
		prompt:
			"I am looking for a keyboard but I am not sure which one I should get, can you help me?",
	});

	await logStream(result.fullStream);
}
