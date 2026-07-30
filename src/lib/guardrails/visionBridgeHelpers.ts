/**
 * Vision Bridge helper functions for image processing.
 */
import { fetchRemoteImage } from "@/shared/network/remoteImageFetch";
import { getRuntimePorts } from "@/lib/runtime/ports";
import { getBestVisionModel, getFallbackModels, recordLatency } from "./visionBridgeRouter";
/**
 * Provider to environment variable mapping for API key resolution.
 */
const PROVIDER_API_KEY_MAP: Record<string, string> = {
  anthropic: "ANTHROPIC_API_KEY",
  google: "GOOGLE_API_KEY",
  openai: "OPENAI_API_KEY",
};

/**
 * Resolve API key based on model provider (issue #2232).
 *
 * Priority:
 *   1. `explicitKey` argument (caller override)
 *   2. `VISION_BRIDGE_API_KEY` env var — operator-set, takes precedence over
 *      per-provider env vars. Used when the operator wants every vision-bridge
 *      call to go through a single OpenAI-compatible endpoint (e.g.,
 *      OmniRoute itself, OpenRouter, a Gemini-OpenAI-compat URL).
 *   3. Per-provider env var (`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
 *      `OPENAI_API_KEY`) based on the `provider/` prefix in the model id.
 *   4. `OPENAI_API_KEY` as final fallback when the prefix is unrecognized.
 *
 * @param model - Model identifier (e.g., "anthropic/claude-3-haiku", "openai/gpt-4o-mini")
 * @param explicitKey - Explicit API key passed as argument (takes precedence)
 * @returns Resolved API key string
 */
export function resolveProviderApiKey(model: string, explicitKey?: string): string {
  if (explicitKey) return explicitKey;
  const isAnthropic = model.startsWith("anthropic/");
  // VISION_BRIDGE_API_KEY only applies to the OpenAI-compatible branch — the
  // Anthropic branch keeps its dedicated key, since the wire format differs.
  if (!isAnthropic) {
    const bridgeKey = (process.env.VISION_BRIDGE_API_KEY || "").trim();
    if (bridgeKey) return bridgeKey;
  }
  const provider = model.includes("/") ? model.split("/")[0] : "";
  const envVar = PROVIDER_API_KEY_MAP[provider] || "OPENAI_API_KEY";
  return process.env[envVar] || "";
}

/**
 * Resolve the OpenAI-compatible base URL for non-Anthropic vision bridge calls
 * (issue #2232).
 *
 * Priority:
 *   1. `VISION_BRIDGE_BASE_URL` env var — operator-set, e.g. point this at
 *      OmniRoute's own `/v1` so the vision model can be any provider
 *      registered in OmniRoute (`google/gemini-2.0-flash`,
 *      `openrouter/...`, etc.) instead of being limited to OpenAI/Anthropic.
 *   2. `OPENAI_API_URL` env var (legacy)
 *   3. OmniRoute self-loop (`http://localhost:20128/v1`) — auto-detected when
 *      the model uses a known OmniRoute-internal provider (e.g. `kr/`, `if/`,
 *      `pol/`, `groq/`, etc.) instead of a direct OpenAI/Anthropic endpoint.
 *   4. `https://api.openai.com/v1` (fallback when the model is `openai/*` or
 *      unprefixed — works only when the operator actually has an OpenAI
 *      account and OPENAI_API_KEY set)
 *
 * @param model - Optional model identifier used to detect non-standard providers
 *                that require OmniRoute self-loop routing.
 */
export function resolveVisionBridgeBaseUrl(model?: string): string {
  const explicit = (process.env.VISION_BRIDGE_BASE_URL || "").trim();
  if (explicit) return explicit.replace(/\/+$/, "");
  const legacy = (process.env.OPENAI_API_URL || "").trim();
  if (legacy) return legacy.replace(/\/+$/, "");

  // When the model has a non-standard provider prefix (not openai/ or
  // anthropic/), it can only be resolved through OmniRoute's own router,
  // not through a direct OpenAI/Anthropic endpoint. Use the operator-configured
  // port via OMNIROUTE_PORT / PORT env vars, falling back to the default 20128.
  if (model && model.includes("/")) {
    const provider = model.split("/")[0].toLowerCase();
    if (provider !== "openai" && provider !== "anthropic") {
      const { port } = getRuntimePorts();
      return `http://localhost:${port}/v1`;
    }
  }

  return "https://api.openai.com/v1";
}

export interface ImagePart {
  messageIndex: number;
  partIndex: number;
  /** Index path within message.content, including nested tool_result.content arrays. */
  contentPath: number[];
  imageUrl: string;
  imageType: "image_url" | "image" | "input_image";
}

export interface RequestMessage {
  role?: string;
  content?: string | RequestContentPart[];
}

export type RequestContentPart =
  | { type: "text"; text: string }
  | { type: "input_text"; text: string }
  | { type: "image_url"; image_url: string | { url: string; detail?: string } }
  | {
      type: "image";
      source?: { type: "base64"; media_type: string; data: string } | { type: "url"; url: string };
      image?: string;
    }
  | { type: "input_image"; image_url: string | { url: string } }
  | { type: "tool_result"; tool_use_id?: string; content?: string | RequestContentPart[] };

type ImageInput = Pick<ImagePart, "imageUrl" | "imageType">;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function readImageUrl(value: unknown): string | null {
  if (typeof value === "string" && value.length > 0) return value;
  if (!isRecord(value)) return null;
  return typeof value.url === "string" && value.url.length > 0 ? value.url : null;
}

/**
 * Normalize the image content-block shapes accepted elsewhere in OmniRoute:
 * - OpenAI image_url (object and shorthand string forms)
 * - Anthropic image source (base64 and URL forms)
 * - AI SDK image string
 * - Responses-style input_image
 */
function readImageInput(part: unknown): ImageInput | null {
  if (!isRecord(part) || typeof part.type !== "string") return null;

  if (part.type === "image_url") {
    const imageUrl = readImageUrl(part.image_url);
    return imageUrl ? { imageUrl, imageType: "image_url" } : null;
  }

  if (part.type === "input_image") {
    const imageUrl = readImageUrl(part.image_url);
    return imageUrl ? { imageUrl, imageType: "input_image" } : null;
  }

  if (part.type !== "image") return null;

  const source = isRecord(part.source) ? part.source : null;
  if (source?.type === "base64" && typeof source.data === "string" && source.data.length > 0) {
    const mediaType =
      typeof source.media_type === "string" && source.media_type.length > 0
        ? source.media_type
        : "image/png";
    return {
      imageUrl: `data:${mediaType};base64,${source.data}`,
      imageType: "image",
    };
  }

  if (source?.type === "url") {
    const imageUrl = readImageUrl(source.url);
    return imageUrl ? { imageUrl, imageType: "image" } : null;
  }

  if (typeof part.image === "string" && part.image.length > 0) {
    return { imageUrl: part.image, imageType: "image" };
  }

  return null;
}

/**
 * Extract image parts from messages array.
 * Recurses through nested content arrays because Claude Code commonly carries
 * screenshots inside tool_result.content. The Claude-to-OpenAI translator later
 * lifts those screenshots into new image_url messages, so a top-level-only scan
 * leaves text-only providers with an invalid final payload.
 */
export function extractImageParts(messages: RequestMessage[]): ImagePart[] {
  const results: ImagePart[] = [];

  if (!Array.isArray(messages)) {
    return results;
  }

  const visitContent = (
    content: unknown[],
    messageIndex: number,
    parentPath: number[],
    depth: number
  ) => {
    // HTTP JSON cannot contain cycles, but cap depth so a synthetic/internal
    // payload cannot force unbounded recursion.
    if (depth > 16) return;

    for (let partIndex = 0; partIndex < content.length; partIndex++) {
      const part = content[partIndex];
      const contentPath = [...parentPath, partIndex];
      const image = readImageInput(part);

      if (image) {
        results.push({
          messageIndex,
          // Preserve the historical field's top-level meaning for callers/tests.
          partIndex: contentPath[0] ?? partIndex,
          contentPath,
          ...image,
        });
        continue;
      }

      if (isRecord(part) && Array.isArray(part.content)) {
        visitContent(part.content, messageIndex, contentPath, depth + 1);
      }
    }
  };

  for (let messageIndex = 0; messageIndex < messages.length; messageIndex++) {
    const message = messages[messageIndex];
    if (message && Array.isArray(message.content)) {
      visitContent(message.content, messageIndex, [], 0);
    }
  }

  return results;
}

/**
 * Resolve image URL to data URI format for vision model.
 * - HTTP/HTTPS URLs: passed through as-is
 * - Data URIs: passed through as-is
 * - Base64 without media type: assumed PNG
 */
export function resolveImageAsDataUri(imageUrl: string): string {
  if (!imageUrl || typeof imageUrl !== "string") {
    throw new Error("Invalid image URL: must be a non-empty string");
  }

  // Already a data URI
  if (imageUrl.startsWith("data:")) {
    return imageUrl;
  }

  // HTTP/HTTPS URL - vision API will fetch it
  if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) {
    return imageUrl;
  }

  // Assume it's a base64 string without prefix
  // Add PNG as default media type
  return `data:image/png;base64,${imageUrl}`;
}

async function fetchRemoteImageAsDataUri(imageUrl: string, signal: AbortSignal): Promise<string> {
  const remoteImage = await fetchRemoteImage(imageUrl, { signal });
  const mediaType = remoteImage.contentType.split(";")[0]?.trim() || "image/png";
  return `data:${mediaType};base64,${remoteImage.buffer.toString("base64")}`;
}

async function normalizeVisionImageInput(
  imageInput: string,
  isAnthropic: boolean,
  signal: AbortSignal
): Promise<string> {
  const normalizedImage = resolveImageAsDataUri(imageInput);

  if (
    isAnthropic &&
    (normalizedImage.startsWith("http://") || normalizedImage.startsWith("https://"))
  ) {
    return fetchRemoteImageAsDataUri(normalizedImage, signal);
  }

  return normalizedImage;
}

/**
 * Extract text from the response shapes returned by OpenAI-compatible
 * gateways. Most return Chat Completions content as a string, but some
 * adapters return content-part arrays or a Responses-style output envelope.
 */
function extractOpenAICompatibleText(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;

  const root = data as Record<string, unknown>;
  const choices = Array.isArray(root.choices) ? root.choices : [];
  const firstChoice =
    choices[0] && typeof choices[0] === "object" ? (choices[0] as Record<string, unknown>) : null;
  const message =
    firstChoice?.message && typeof firstChoice.message === "object"
      ? (firstChoice.message as Record<string, unknown>)
      : null;

  const collectText = (value: unknown): string[] => {
    if (typeof value === "string") {
      return value.trim() ? [value] : [];
    }
    if (!Array.isArray(value)) return [];

    return value.flatMap((part) => {
      if (!part || typeof part !== "object") return [];
      const record = part as Record<string, unknown>;
      if (typeof record.text === "string" && record.text.trim()) {
        return [record.text];
      }
      if (typeof record.output_text === "string" && record.output_text.trim()) {
        return [record.output_text];
      }
      return collectText(record.content);
    });
  };

  const chatText = collectText(message?.content).join("\n").trim();
  if (chatText) return chatText;

  if (typeof root.output_text === "string" && root.output_text.trim()) {
    return root.output_text.trim();
  }

  const responsesText = collectText(root.output).join("\n").trim();
  return responsesText || null;
}

export interface VisionModelConfig {
  model: string;
  prompt: string;
  timeoutMs: number;
  maxImages: number;
}

/**
 * Call the vision model to get an image description.
 * Supports both OpenAI-compatible and Anthropic API formats.
 * Uses auto-routing to select the fastest available model.
 */
export async function callVisionModel(
  imageDataUri: string,
  config: VisionModelConfig,
  apiKey?: string,
  routerConfig?: Partial<import("./visionBridgeRouter").VisionBridgeRouterConfig>
): Promise<string> {
  // Auto-select the best vision model if not explicitly configured
  const modelToUse = await getBestVisionModel({
    fixedModel: config.model,
    ...routerConfig,
  });
  let lastError: Error | null = null;

  // Try primary model + fallbacks
  const modelsToTry = [modelToUse, ...(await getFallbackModels(modelToUse, routerConfig))];
  const maxAttempts = Math.min(modelsToTry.length, routerConfig?.maxFallbackAttempts ?? 3);

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const currentModel = modelsToTry[attempt];
    const attemptStart = Date.now();
    try {
      const result = await callVisionModelSingle(
        imageDataUri,
        { ...config, model: currentModel },
        apiKey
      );
      recordLatency(currentModel, Date.now() - attemptStart, true);
      return result;
    } catch (error) {
      recordLatency(currentModel, Date.now() - attemptStart, false);
      lastError = error instanceof Error ? error : new Error(String(error));
      // Continue to next model on failure
    }
  }

  // All models failed
  throw lastError || new Error("All vision models failed");
}

/**
 * Internal function to call a single vision model.
 */
async function callVisionModelSingle(
  imageDataUri: string,
  config: VisionModelConfig,
  apiKey?: string
): Promise<string> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), config.timeoutMs);

  // Resolve API key based on provider
  const resolvedApiKey = resolveProviderApiKey(config.model, apiKey);

  // Detect provider from model identifier
  const isAnthropic = config.model.startsWith("anthropic/");

  try {
    // Extract model name from provider/model format
    const modelName = config.model.includes("/") ? config.model.split("/")[1] : config.model;
    const normalizedImageInput = await normalizeVisionImageInput(
      imageDataUri,
      isAnthropic,
      controller.signal
    );

    let response: Response;

    if (isAnthropic) {
      // Anthropic API path
      const anthropicBaseUrl = process.env.ANTHROPIC_API_URL || "https://api.anthropic.com";

      // Parse data URI to extract media type and base64 data
      const matches = normalizedImageInput.match(/^data:([^;]+);base64,(.+)$/);
      let mediaType = "image/png";
      let base64Data = normalizedImageInput;

      if (matches) {
        mediaType = matches[1];
        base64Data = matches[2];
      }

      response = await fetch(`${anthropicBaseUrl}/v1/messages`, {
        method: "POST",
        signal: controller.signal,
        headers: {
          "x-api-key": resolvedApiKey,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: modelName,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image",
                  source: {
                    type: "base64",
                    media_type: mediaType,
                    data: base64Data,
                  },
                },
                {
                  type: "text",
                  text: config.prompt,
                },
              ],
            },
          ],
          max_tokens: 300,
        }),
      });
    } else {
      // OpenAI-compatible path (default) — issue #2232: honor
      // VISION_BRIDGE_BASE_URL so the vision-bridge call can be routed through
      // OmniRoute itself or any other OpenAI-compatible endpoint instead of
      // hardcoded api.openai.com.
      const baseUrl = resolveVisionBridgeBaseUrl(config.model);

      // When routing through the OmniRoute self-loop (non-standard provider),
      // keep the full provider-prefixed model ID so OmniRoute can resolve the
      // correct provider backend. Only strip the prefix for direct OpenAI calls.
      const useFullModelId =
        baseUrl.startsWith("http://localhost") &&
        config.model.includes("/") &&
        !config.model.startsWith("openai/");
      const requestModel = useFullModelId ? config.model : modelName;

      // Build headers with optional recursion guard for self-loop calls.
      // When routing through OmniRoute's own API, omit the vision-bridge
      // guardrail on the sub-request to prevent infinite recursion.
      // Use sk_omniroute as fallback for self-loop if no API key is resolved.
      const selfLoopApiKey = resolvedApiKey || "sk_omniroute";
      const headers: Record<string, string> = {
        Accept: "application/json",
        "Content-Type": "application/json",
        Authorization: `Bearer ${selfLoopApiKey}`,
      };
      if (useFullModelId) {
        headers["x-omniroute-disabled-guardrails"] = "vision-bridge";
      }

      response = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        signal: controller.signal,
        headers,
        body: JSON.stringify({
          model: requestModel,
          stream: false,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image_url",
                  image_url: {
                    url: normalizedImageInput,
                    detail: "low",
                  },
                },
                { type: "text", text: config.prompt },
              ],
            },
          ],
          max_tokens: 300,
        }),
      });
    }

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text().catch(() => "Unknown error");
      throw new Error(`Vision API error ${response.status}: ${errorText}`);
    }

    const data = await response.json();

    if (isAnthropic) {
      // Anthropic response format: { content: [{ type: "text", text: "..." }] }
      const anthropicData = data as {
        content?: Array<{ type?: string; text?: string }>;
        error?: { message?: string };
      };

      if (anthropicData.error) {
        throw new Error(
          `Vision API error: ${anthropicData.error.message || JSON.stringify(anthropicData.error)}`
        );
      }

      const textContent = anthropicData.content?.find((c) => c.type === "text");
      const content = textContent?.text;
      if (!content || typeof content !== "string") {
        throw new Error("Vision API returned empty or invalid response");
      }

      return content.trim();
    } else {
      // OpenAI-compatible gateways can return a Chat Completions string, a
      // content-part array, or a Responses-style output envelope.
      const openaiData = data as {
        error?: { message?: string };
      };

      if (openaiData.error) {
        throw new Error(
          `Vision API error: ${openaiData.error.message || JSON.stringify(openaiData.error)}`
        );
      }

      const content = extractOpenAICompatibleText(data);
      if (!content) {
        throw new Error("Vision API returned empty or invalid response");
      }

      return content;
    }
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Vision model call timed out");
    }

    throw error;
  }
}

export interface RequestBody {
  model?: string;
  messages?: RequestMessage[];
  [key: string]: unknown;
}

/**
 * Replace image content parts with text descriptions.
 * Concatenates descriptions with labels: "[Image 1]: ..."
 */
export function replaceImageParts(
  body: RequestBody,
  // #4012: a `null` entry means the describe call failed for that image — keep
  // the original image part instead of dropping it / stubbing "(unavailable)".
  descriptions: (string | null)[]
): RequestBody {
  if (!descriptions || descriptions.length === 0) {
    return body;
  }

  const result = structuredClone(body) as RequestBody;

  if (!Array.isArray(result.messages)) {
    return result;
  }

  let descriptionIndex = 0;

  const replaceContent = (content: unknown[], depth: number): unknown[] => {
    if (depth > 16) return content;

    return content.map((part) => {
      const image = readImageInput(part);
      if (image) {
        if (descriptionIndex >= descriptions.length) {
          // Preserve images outside the processed range. The guardrail supplies
          // explicit placeholders here for a known text-only target.
          return part;
        }

        const description = descriptions[descriptionIndex];
        descriptionIndex++;
        if (description == null) {
          // #4012: describe failed for this image — preserve the original
          // image so a vision-capable upstream can still process it.
          return part;
        }

        const replacement: Record<string, unknown> = {
          type: image.imageType === "input_image" ? "input_text" : "text",
          text: description,
        };
        if (isRecord(part) && part.cache_control !== undefined) {
          replacement.cache_control = part.cache_control;
        }
        return replacement;
      }

      if (isRecord(part) && Array.isArray(part.content)) {
        return {
          ...part,
          content: replaceContent(part.content, depth + 1),
        };
      }

      return part;
    });
  };

  for (const message of result.messages) {
    if (message && Array.isArray(message.content)) {
      message.content = replaceContent(message.content, 0) as RequestContentPart[];
    }
  }

  return result;
}
