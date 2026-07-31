/**
 * Tests for VisionBridgeGuardrail.
 * Uses dependency injection to avoid SQLite dependency.
 */

import test from "node:test";
import assert from "node:assert/strict";

const { VisionBridgeGuardrail } = await import("../../../src/lib/guardrails/visionBridge.ts");
const { resetGuardrailsForTests } = await import("../../../src/lib/guardrails/registry.ts");
const { getResolvedModelCapabilities } = await import("../../../src/lib/modelCapabilities.ts");
const { translateRequest } = await import("../../../open-sse/translator/index.ts");
const { FORMATS } = await import("../../../open-sse/translator/formats.ts");
import type { GuardrailContext } from "../../../src/lib/guardrails/base.ts";
import type { VisionModelConfig } from "../../../src/lib/guardrails/visionBridgeHelpers.ts";

// ── Mock state ──────────────────────────────────────────────────────────────

let mockSettings: Record<string, unknown> = {
  visionBridgeEnabled: true,
  visionBridgeModel: "openai/gpt-4o-mini",
  visionBridgePrompt: "Describe this image concisely.",
  visionBridgeTimeout: 30000,
  visionBridgeMaxImages: 10,
};

let mockVisionResponse = "A beautiful sunset over the ocean";
let shouldVisionFail = false;
let visionCallCount = 0;

function createGuardrail(options?: Parameters<typeof VisionBridgeGuardrail>[0]) {
  return new VisionBridgeGuardrail({
    ...options,
    deps: {
      getSettings: async () => mockSettings,
      callVisionModel: async (_imageDataUri: string, _config: VisionModelConfig) => {
        visionCallCount++;
        if (shouldVisionFail) {
          throw new Error("Vision model failed");
        }
        return mockVisionResponse;
      },
      // Fail-open (null) so classic VB-S01/S07/S10 reroute tests keep working without a
      // live credential DB. Credential-aware cases inject an explicit mock.
      hasUsableCredentials: async () => null,
      ...(options?.deps ?? {}),
    },
  });
}

test.beforeEach(() => {
  resetGuardrailsForTests({ registerDefaults: false });
  visionCallCount = 0;
  shouldVisionFail = false;
  mockSettings = {
    visionBridgeEnabled: true,
    visionBridgeModel: "openai/gpt-4o-mini",
    visionBridgePrompt: "Describe this image concisely.",
    visionBridgeTimeout: 30000,
    visionBridgeMaxImages: 10,
  };
});

// ── Helpers ─────────────────────────────────────────────────────────────────

function createContext(overrides: Partial<GuardrailContext> = {}): GuardrailContext {
  return {
    model: "minimax/minimax-01",
    log: console,
    ...overrides,
  };
}

function createPayload(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model: "minimax/minimax-01",
    messages: [{ role: "user", content: "Hello" }],
    ...overrides,
  };
}

// ── Basic Properties ────────────────────────────────────────────────────────

test("VisionBridgeGuardrail has correct name and priority", () => {
  const guardrail = createGuardrail();
  assert.strictEqual(guardrail.name, "vision-bridge");
  assert.strictEqual(guardrail.priority, 5);
});

test("VisionBridgeGuardrail is enabled by default", () => {
  const guardrail = createGuardrail();
  assert.strictEqual(guardrail.enabled, true);
});

test("VisionBridgeGuardrail can be disabled via constructor", () => {
  const guardrail = createGuardrail({ enabled: false });
  assert.strictEqual(guardrail.enabled, false);
});

// ── VB-S05: Vision Bridge disabled via settings ────────────────────────────

test("VB-S05: passthroughs when visionBridgeEnabled is false", async () => {
  mockSettings.visionBridgeEnabled = false;
  const guardrail = createGuardrail();

  const payload = createPayload({
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is in this image?" },
          { type: "image_url", image_url: { url: "https://example.com/image.png" } },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext());
  assert.strictEqual(result.block, false);
  assert.strictEqual(result.modifiedPayload, undefined);
});

// ── VB-S06: Disabled via context ────────────────────────────────────────────

test("VB-S06: skips when disabledGuardrails includes vision-bridge", async () => {
  const guardrail = createGuardrail();
  const payload = createPayload();
  const context = createContext({ disabledGuardrails: ["vision-bridge"] });

  const result = await guardrail.preCall(payload, context);
  assert.strictEqual(result.block, false);
  assert.strictEqual(result.modifiedPayload, undefined);
});

// ── VB-S02: Vision-capable model passthrough ────────────────────────────────

test("VB-S02: passthroughs for vision-capable model (gpt-4o)", async () => {
  const guardrail = createGuardrail();
  const payload = createPayload({
    model: "openai/gpt-4o",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/image.png" },
          },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "openai/gpt-4o" }));

  // If supportsVision is true, it should passthrough (no modification)
  // If supportsVision is null/undefined (no sync data), it will process — that's correct behavior
  const capabilities = getResolvedModelCapabilities("openai/gpt-4o");
  if (capabilities.supportsVision === true) {
    assert.strictEqual(result.block, false);
    assert.strictEqual(result.modifiedPayload, undefined);
  } else {
    // Without sync data, supportsVision is null — guardrail processes the image
    // This is correct fail-open behavior for unknown model capabilities
    assert.strictEqual(result.block, false);
  }
});

test("VB-S02b: respects native vision support for GPT-family models", async () => {
  const guardrail = createGuardrail();

  for (const model of ["gpt-5.5", "gpt-5.5-high", "codex/gpt-5.5", "openai/gpt-4o-mini"]) {
    visionCallCount = 0;

    const payload = createPayload({
      model,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What is this?" },
            {
              type: "image_url",
              image_url: { url: "https://example.com/image.png" },
            },
          ],
        },
      ],
    });

    const result = await guardrail.preCall(payload, createContext({ model }));

    assert.strictEqual(result.block, false, `expected passthrough for ${model}`);
    assert.strictEqual(visionCallCount, 0, `expected no bridge call for ${model}`);

    // If supportsVision is true, payload should be unmodified.
    // If supportsVision is null, the guardrail reroutes (modifiedPayload defined, model changed).
    // Both are correct behavior — the key invariant is no describe call.
    const caps = getResolvedModelCapabilities(model);
    if (caps.supportsVision === true) {
      assert.strictEqual(
        result.modifiedPayload,
        undefined,
        `expected unmodified payload for ${model}`
      );
    }
  }
});

test("VB-S02: model capabilities returns supportsVision for known models", () => {
  const gpt4oCaps = getResolvedModelCapabilities("openai/gpt-4o");
  // supportsVision may be true (if sync data exists) or null (if not synced)
  assert.ok(gpt4oCaps.supportsVision === true || gpt4oCaps.supportsVision === null);
});

// ── VB-S04: No images passthrough ──────────────────────────────────────────

test("VB-S04: passthroughs when no images in messages", async () => {
  const guardrail = createGuardrail();
  const payload = createPayload({
    messages: [{ role: "user", content: "Hello, how are you?" }],
  });

  const result = await guardrail.preCall(payload, createContext());
  assert.strictEqual(result.block, false);
  assert.strictEqual(result.modifiedPayload, undefined);
});

test("VB-S04: passthroughs when messages array is empty", async () => {
  const guardrail = createGuardrail();
  const payload = createPayload({ messages: [] });
  const result = await guardrail.preCall(payload, createContext());
  assert.strictEqual(result.block, false);
});

// ── VB-S12: Auto-prefix skip ────────────────────────────────────────────────

test("VB-S12: reroutes auto/ prefix model to vision model (auto/vision)", async () => {
  const guardrail = createGuardrail();

  const payload = createPayload({
    model: "auto/vision",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is in this image?" },
          { type: "image_url", image_url: { url: "https://example.com/image.png" } },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "auto/vision" }));
  assert.strictEqual(result.block, false);
  assert.ok(result.modifiedPayload, "auto/vision should reroute to vision model");
  assert.strictEqual(
    result.modifiedPayload?.model,
    "openai/gpt-4o-mini",
    "should reroute to configured vision model"
  );
  assert.strictEqual(result.meta?.rerouted, true, "rerouted meta should be set");
  assert.strictEqual(visionCallCount, 0, "should NOT call vision API (reroute, not describe)");
});

test("VB-S12b: reroutes auto prefix to best vision model when images present", async () => {
  const guardrail = createGuardrail();

  const payload = createPayload({
    model: "auto",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is in this image?" },
          { type: "image_url", image_url: { url: "https://example.com/image.png" } },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "auto" }));
  assert.strictEqual(result.block, false);
  assert.ok(result.modifiedPayload, "auto should reroute to vision model");
  assert.strictEqual(
    result.modifiedPayload?.model,
    "openai/gpt-4o-mini",
    "should reroute to configured vision model"
  );
  assert.strictEqual(result.meta?.rerouted, true, "rerouted meta should be set");
});

// ── VB-S01: Single image → reroute (individual non-vision model) ───────────

test("VB-S01: reroutes non-vision model with images to best vision model", async () => {
  const guardrail = createGuardrail();

  const payload = createPayload({
    model: "minimax/minimax-01",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is in this image?" },
          { type: "image_url", image_url: { url: "https://example.com/image.png" } },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "minimax/minimax-01" }));

  assert.strictEqual(result.block, false);
  assert.ok(result.modifiedPayload);

  // Model should be rerouted to best vision-capable model (auto-selected from providers)
  const modified = result.modifiedPayload as {
    model?: string;
    messages: Array<{ content: unknown[] }>;
  };
  assert.ok(modified.model, "rerouted model should be set");
  assert.notStrictEqual(
    modified.model,
    "minimax/minimax-01",
    "model should be different from original"
  );

  // Images should be KEPT since the vision model handles them natively
  const content = modified.messages[0].content as Array<{ type: string; [key: string]: unknown }>;
  const imagePart = content.find((p) => p.type === "image_url");
  assert.ok(imagePart, "original image_url part must be preserved for rerouted vision model");

  // Meta should indicate reroute occurred
  const meta = result.meta as Record<string, unknown>;
  assert.strictEqual(meta.rerouted, true);
  assert.strictEqual(meta.fromModel, "minimax/minimax-01");
  assert.ok(
    typeof meta.toModel === "string" && meta.toModel.length > 0,
    "toModel should be a non-empty string"
  );
  assert.notStrictEqual(meta.toModel, "minimax/minimax-01", "toModel should differ from original");
  assert.strictEqual(meta.imagesKept, 1);
  assert.strictEqual(visionCallCount, 0, "should NOT call vision API for description");
});

// ── VB-S13: Reroute preserves multiple images ──────────────────────────────

test("VB-S13: reroutes with multiple images, all preserved", async () => {
  const guardrail = createGuardrail();

  const payload = createPayload({
    model: "minimax/minimax-01",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Describe these images" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/cat.png" },
          },
          {
            type: "image_url",
            image_url: { url: "https://example.com/dog.png" },
          },
          {
            type: "image_url",
            image_url: { url: "https://example.com/bird.png" },
          },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "minimax/minimax-01" }));

  assert.strictEqual(result.block, false);
  assert.ok(result.modifiedPayload);

  // All 3 images should be present in the rerouted payload
  const modified = result.modifiedPayload as {
    model?: string;
    messages: Array<{ content: unknown[] }>;
  };
  const content = modified.messages[0].content as Array<{ type: string; [key: string]: unknown }>;
  const images = content.filter((p) => p.type === "image_url");
  assert.strictEqual(images.length, 3, "all 3 images should be preserved");
  assert.strictEqual(visionCallCount, 0, "should NOT call vision API");
});

// ── VB-S07: Base64 image format → reroute ──────────────────────────────────

test("VB-S07: reroutes base64 image to vision model", async () => {
  const guardrail = createGuardrail();

  const payload = createPayload({
    model: "minimax/minimax-01",
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/png",
              data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            },
          },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "minimax/minimax-01" }));

  assert.strictEqual(result.block, false);
  assert.ok(result.modifiedPayload);
  const modified = result.modifiedPayload as { model?: string };
  assert.ok(modified.model, "rerouted model should be set");
  assert.notStrictEqual(
    modified.model,
    "minimax/minimax-01",
    "model should be different from original"
  );
  // Don't assert a specific model — auto-router picks the best available vision model
  assert.strictEqual(visionCallCount, 0, "should NOT call vision API");
});

// ── VB-S03: Fail-open on vision error (via combo mapping path) ────────────

test("VB-S03: preserves the original image when the vision API fails (#4012)", async () => {
  shouldVisionFail = true;
  const guardrail = createGuardrail({
    deps: {
      checkModelHasComboMapping: async (_model: string) => true,
    },
  });

  const payload = createPayload({
    model: "openai/gpt-4o",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/image.png" },
          },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "openai/gpt-4o" }));

  assert.strictEqual(result.block, false);

  const modified = (result.modifiedPayload ?? payload) as {
    messages: Array<{ content: unknown[] }>;
  };
  const content = modified.messages[0].content as Array<{
    type: string;
    text?: string;
  }>;

  // #4012: a failed describe must NOT replace the image with an "(unavailable)"
  // stub — the original image is preserved so a vision-capable upstream can see it.
  const imagePart = content.find((p) => p.type === "image_url");
  assert.ok(imagePart, "original image_url part must be preserved on describe failure");
  const unavailPart = content.find((p) => p.type === "text" && p.text?.includes("unavailable"));
  assert.strictEqual(unavailPart, undefined);
});

test("VB-DS-01: ds/deepseek-v4-pro replaces images with bridge text", async () => {
  const guardrail = createGuardrail({
    deps: {
      hasUsableCredentials: async (model: string) => (model === "ds/deepseek-v4-pro" ? true : null),
    },
  });

  const payload = createPayload({
    model: "ds/deepseek-v4-pro",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/png",
              data: "aW1hZ2UtYnl0ZXM=",
            },
          },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "ds/deepseek-v4-pro" }));
  assert.strictEqual(result.block, false);
  assert.strictEqual(visionCallCount, 1);

  const modified = result.modifiedPayload as {
    messages: Array<{ content: Array<{ type: string; text?: string }> }>;
  };
  const content = modified.messages[0].content;
  assert.equal(
    content.some((part) => part.type === "image"),
    false
  );
  assert.equal(
    content.some(
      (part) => part.type === "text" && part.text === `[Image 1]: ${mockVisionResponse}`
    ),
    true
  );
});

test("VB-DS-02: a failed bridge never forwards raw images to text-only DeepSeek", async () => {
  shouldVisionFail = true;
  const guardrail = createGuardrail({
    deps: {
      hasUsableCredentials: async (model: string) => (model === "ds/deepseek-v4-pro" ? true : null),
    },
  });

  const payload = createPayload({
    model: "ds/deepseek-v4-pro",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/image.png" },
          },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "ds/deepseek-v4-pro" }));
  assert.strictEqual(result.block, false);

  const modified = result.modifiedPayload as {
    messages: Array<{ content: Array<{ type: string; text?: string }> }>;
  };
  const content = modified.messages[0].content;
  assert.equal(
    content.some((part) => part.type === "image_url"),
    false
  );
  assert.equal(
    content.some((part) => part.type === "text" && part.text === "[Image 1]: unavailable"),
    true
  );
});

test("VB-DS-03: final DeepSeek payload contains no image_url from nested Claude tool results", async () => {
  const guardrail = createGuardrail({
    deps: {
      hasUsableCredentials: async (model: string) => (model === "ds/deepseek-v4-pro" ? true : null),
    },
  });

  const payload = createPayload({
    model: "ds/deepseek-v4-pro",
    messages: [
      {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "tool-screenshot",
            name: "screenshot",
            input: {},
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "tool-screenshot",
            content: [
              { type: "text", text: "Screenshot captured." },
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: "image/png",
                  data: "aGlzdG9yaWNhbC1zY3JlZW5zaG90",
                },
              },
            ],
          },
        ],
      },
      {
        role: "user",
        content: [
          { type: "text", text: "Compare that screenshot with this image." },
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/png",
              data: "Y3VycmVudC1pbWFnZQ==",
            },
          },
        ],
      },
    ],
  });

  const bridged = await guardrail.preCall(payload, createContext({ model: "ds/deepseek-v4-pro" }));
  assert.strictEqual(bridged.block, false);

  const translated = translateRequest(
    FORMATS.CLAUDE,
    FORMATS.OPENAI,
    "deepseek-v4-pro",
    bridged.modifiedPayload ?? payload,
    true,
    null,
    "deepseek"
  );
  const finalJson = JSON.stringify(translated);

  assert.equal(visionCallCount, 2, "both historical and current images must be described");
  assert.equal(
    finalJson.includes('"type":"image_url"'),
    false,
    `final DeepSeek payload still contains image_url: ${finalJson}`
  );
  assert.equal(
    finalJson.includes("aGlzdG9yaWNhbC1zY3JlZW5zaG90"),
    false,
    "historical screenshot base64 must not survive translation"
  );
  assert.equal(
    finalJson.includes("Y3VycmVudC1pbWFnZQ=="),
    false,
    "current image base64 must not survive translation"
  );
});

test("VB-S03: logs warning when vision API fails (via combo mapping)", async () => {
  shouldVisionFail = true;
  let warningLogged = false;
  const guardrail = createGuardrail({
    deps: {
      checkModelHasComboMapping: async (_model: string) => true,
    },
  });

  const payload = createPayload({
    model: "openai/gpt-4o",
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: { url: "https://example.com/image.png" },
          },
        ],
      },
    ],
  });

  const mockLog = {
    warn: (_tag: string, msg: string) => {
      if (msg.includes("Failed to get description")) {
        warningLogged = true;
      }
    },
  };

  await guardrail.preCall(
    payload,
    createContext({
      model: "openai/gpt-4o",
      log: mockLog as GuardrailContext["log"],
    })
  );

  assert.strictEqual(warningLogged, true);
});

// ── VB-S09: Image count limit for text-only targets ────────────────────────

test("VB-S09: respects maxImages without leaking remaining images to a text-only target", async () => {
  mockSettings.visionBridgeMaxImages = 2;
  const guardrail = createGuardrail({
    deps: {
      hasUsableCredentials: async (model: string) => (model === "ds/deepseek-v4-pro" ? true : null),
    },
  });

  const images = Array.from({ length: 5 }, (_, i) => ({
    type: "image_url" as const,
    image_url: { url: `https://example.com/image${i}.png` },
  }));

  const payload = createPayload({
    model: "ds/deepseek-v4-pro",
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: "Describe these" }, ...images],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "ds/deepseek-v4-pro" }));

  // Should only call vision API for 2 images (maxImages=2)
  assert.strictEqual(visionCallCount, 2);
  const finalJson = JSON.stringify(result.modifiedPayload);
  assert.equal(
    finalJson.includes('"type":"image_url"'),
    false,
    "images beyond maxImages must become safe text placeholders for a text-only target"
  );
  assert.equal(
    finalJson.includes("skipped (limit)"),
    true,
    "the payload should disclose why remaining images were not described"
  );
});

// ── VB-S10: Meta information returned (reroute path) ───────────────────────

test("VB-S10: returns meta with reroute info for individual non-vision model", async () => {
  const guardrail = createGuardrail();

  const payload = createPayload({
    model: "minimax/minimax-01",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/a.png" },
          },
          {
            type: "image_url",
            image_url: { url: "https://example.com/b.png" },
          },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "minimax/minimax-01" }));

  assert.ok(result.meta);
  assert.ok(typeof result.meta === "object");

  const meta = result.meta as Record<string, unknown>;
  assert.strictEqual(meta.rerouted, true);
  assert.strictEqual(meta.fromModel, "minimax/minimax-01");
  assert.ok(
    typeof meta.toModel === "string" && meta.toModel.length > 0,
    "toModel should be a non-empty string"
  );
  assert.notStrictEqual(meta.toModel, "minimax/minimax-01", "toModel should differ from original");
  assert.strictEqual(meta.imagesKept, 2);
});

// ── VB-S01b: Describe images via combo mapping path ────────────────────────

test("VB-S01b: describes images when combo mapping forces process path", async () => {
  mockVisionResponse = "A cat sitting on a windowsill";
  const guardrail = createGuardrail({
    deps: {
      checkModelHasComboMapping: async (_model: string) => true,
    },
  });

  const payload = createPayload({
    model: "openai/gpt-4o",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/cat.png" },
          },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "openai/gpt-4o" }));

  assert.strictEqual(result.block, false);
  assert.ok(result.modifiedPayload);

  const modified = result.modifiedPayload as { messages: Array<{ content: unknown[] }> };
  const content = modified.messages[0].content as Array<{ type: string; text?: string }>;

  // Images should be replaced with text descriptions (combo path)
  const imagePart = content.find((p) => p.type === "image_url");
  assert.strictEqual(imagePart, undefined, "image should be replaced by description");

  const descriptionPart = content.find((p) => p.type === "text" && p.text?.includes("cat"));
  assert.ok(descriptionPart, "description should be present");
  assert.ok(visionCallCount > 0, "vision API should have been called for description");
});

// ── VB-S11: Combo mapping forces vision processing despite vision-capable model ──

test("VB-S11: processes images when vision-capable model has combo mapping", async () => {
  mockVisionResponse = "A description from combo-mapped vision bridge";
  const guardrail = createGuardrail({
    deps: {
      checkModelHasComboMapping: async (_model: string) => true,
    },
  });

  const payload = createPayload({
    model: "openai/gpt-4o",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/image.png" },
          },
        ],
      },
    ],
  });

  const startCallCount = visionCallCount;
  const result = await guardrail.preCall(payload, createContext({ model: "openai/gpt-4o" }));

  // Vision bridge should have processed the image
  assert.strictEqual(result.block, false);
  assert.ok(visionCallCount > startCallCount, "Expected vision model to be called");
  assert.ok(
    result.modifiedPayload !== undefined,
    "Expected modifiedPayload when combo mapping forces vision bridge"
  );
});

test("VB-S11b: passthroughs when vision-capable model has NO combo mapping", async () => {
  const guardrail = createGuardrail({
    deps: {
      checkModelHasComboMapping: async (_model: string) => false,
    },
  });

  const payload = createPayload({
    model: "openai/gpt-4o",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/image.png" },
          },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "openai/gpt-4o" }));

  // Vision bridge should skip (passthrough) since model supports vision + no combo mapping
  assert.strictEqual(result.block, false);
  assert.strictEqual(result.modifiedPayload, undefined);
  assert.strictEqual(visionCallCount, 0);
});

// ── Credential-aware whole-request reroute (combo zai hijack fix) ───────────

test("VB-CRED-01: does NOT whole-request-reroute when original model has usable credentials", async () => {
  // Repro: OpenCode 94-msg body with images + combo target zai/glm-5.2 was
  // hijacked to opencode-zen/gpt-5.4 (priority 0, noauth) → 401 Missing API key.
  const guardrail = createGuardrail({
    deps: {
      hasUsableCredentials: async (m: string) =>
        m.startsWith("zai/") ? true : m.startsWith("opencode-") ? false : null,
    },
  });

  const payload = createPayload({
    model: "zai/glm-5.2",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is in this screenshot?" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/shot.png" },
          },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "zai/glm-5.2" }));
  assert.strictEqual(result.block, false);
  // Must NOT swap model to opencode-zen / openai vision target
  if (result.modifiedPayload) {
    const modified = result.modifiedPayload as { model?: string };
    assert.strictEqual(
      modified.model,
      "zai/glm-5.2",
      "credentialed original model must not be whole-request-rerouted"
    );
  }
  const meta = result.meta as Record<string, unknown> | undefined;
  assert.notStrictEqual(meta?.rerouted, true, "must not set rerouted meta for credentialed model");
});

test("VB-CRED-02: does NOT reroute to a vision model known to lack credentials", async () => {
  mockSettings.visionBridgeModel = "opencode-zen/gpt-5.4";
  const guardrail = createGuardrail({
    deps: {
      // Original unusable, best vision model also unusable
      hasUsableCredentials: async () => false,
    },
  });

  const payload = createPayload({
    model: "minimax/minimax-01",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Describe" },
          { type: "image_url", image_url: { url: "https://example.com/a.png" } },
        ],
      },
    ],
  });

  const result = await guardrail.preCall(payload, createContext({ model: "minimax/minimax-01" }));
  assert.strictEqual(result.block, false);
  const meta = result.meta as Record<string, unknown> | undefined;
  assert.notStrictEqual(meta?.rerouted, true, "must not reroute to unusable vision model");
});

test("isProviderConnectionUsable rejects noauth without api key", async () => {
  const { isProviderConnectionUsable } =
    await import("../../../src/lib/guardrails/visionBridge.ts");
  assert.strictEqual(isProviderConnectionUsable({ authType: "noauth", apiKey: null }), false);
  assert.strictEqual(isProviderConnectionUsable({ authType: "apikey", apiKey: "sk-real" }), true);
  assert.strictEqual(isProviderConnectionUsable({ authType: "oauth", refreshToken: "rt" }), true);
  assert.strictEqual(
    isProviderConnectionUsable({ authType: "apikey", apiKey: "x", testStatus: "banned" }),
    false
  );
});
