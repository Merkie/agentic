# agentic

Opinionated framework for building AI agents in JS

### What & Why

The best foundation for building AI agents (in my opinion) is currently the [AI SDK](https://www.npmjs.com/package/ai) by Vercel with the [OpenRouter Provider](https://www.npmjs.com/package/@openrouter/ai-sdk-provider). This gives you a great translation layer that you could take to other providers and you get every model with OpenRouter with just a string. Whereas using native provider packages require installing new packages, etc. You could dynamically update an OpenRouter model string without any code changes theoretically. So it's way more flexible and future-proof.

But even when you use those packages, you are still missing basic things. Like for example, listing the models available to you. You need to hit `https://openrouter.ai/api/v1/models?input_modalities=text,image` directly or install the entire native openrouter sdk on top of the ai provider package. And the API isn't even typed since it's just a fetch. That's one thing I aim to solve with this package, just put everything in one WITH functions to work with the model list.

On that note as well, there's no way in the AI SDK to see the context window that's been used as a %, you have to fetch the model info from openrouter or take it from the list endpoint in openrouter, then rig up the math yourself based on the tokens used in the conversation.

There's also no way to track conversations really, or save them properly and replay them, you have to do that all yourself. And by doing this it can track things like total tokens used, take that and divide by the token context window size and get the % used. We also want to track the cost like this:

```ts
createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
  // Optional: opt in to per-call cost reporting on providerMetadata.
  extraBody: { usage: { include: true } } as any,
});
```

And that will return the cost of each step, and we need to add that up as a cumulative value for conversations as well. And for this package we should probably call them sessions instead of conversations since they can be more than just conversations, they will be holding state and be used for like agentic workflows, etc.

We will also track the abort signal based on this abstraction as well, since right now doing chat aborts via an abort controller is a total bring-your-own situation, and it could easily be intertwined with the session/conversation state management like the mesasges coming in, the tokens used, the cost, etc. So we can have a `session.abort()` function that will trigger the abort controller and then we can handle that properly in the provider implementation and also in the session state management.

Other patterns we might want to use:

- createTools() type tool factory that has shared state
- This fix for google models:

```ts
const result = await generateText({
  model: openrouter(input.model, {
    // Google AI Studio's Gemini endpoint drops thought signatures across the
    // OpenRouter→Google translation, then rejects the next turn with
    // "Corrupted thought signature". Route to any other provider.
    extraBody: { provider: { ignore: ["google-ai-studio"] } },
  }),
  system: input.system,
  prompt: input.prompt,
});
```

- queue'd messages, like if the agent is currently running and you want to send a new message to it (e.g. a system notification) you can just send it and it will be queued and sent right when it's done with the current message. and it would send all queued messages together when it's done so it can address everything all together, not 1-by-1. This would be really good for like agentic workflows where you have a bunch of steps and you want to send updates to the agent as it works through those steps, but you don't want to interrupt it in the middle of processing something. I think any message sent to the agent should have to go through this queue system.

I think we should define the agent group or chats or whatever as some sort of global scope or scoped thing that we can import and export across files, like one file can send a message and the other file can recieve the stream if that makes sense.

We will need extensive testing to make sure the framework logic is sound before we start converting it into a proper package. And we can capture chat replay for tests so we dont need to like keep calling the API directly.
