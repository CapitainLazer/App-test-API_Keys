const express = require("express");
const path = require("path");
const dotenv = require("dotenv");
const OpenAI = require("openai");

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const DEFAULT_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";
const DEFAULT_GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-1.5-flash";
const DEFAULT_HF_MODEL = process.env.HF_MODEL || "facebook/detr-resnet-50";
const MAX_IMAGE_MB = Number(process.env.MAX_IMAGE_MB || 8);
const MAX_IMAGE_BYTES = MAX_IMAGE_MB * 1024 * 1024;
const GEMINI_MODELS_CACHE_TTL_MS = Number(process.env.GEMINI_MODELS_CACHE_TTL_MS || 10 * 60 * 1000);
const GEMINI_QUOTA_COOLDOWN_MS = Number(process.env.GEMINI_QUOTA_COOLDOWN_MS || 60 * 1000);
const GEMINI_AUTO_RETRY_MAX_WAIT_MS = Number(process.env.GEMINI_AUTO_RETRY_MAX_WAIT_MS || 15000);

const FIXED_PROMPT =
  "Peux-tu me donner les informations suivantes liées à l'objet principal mis en avant dans la photo que je fournis :\nType de matériau\nDimensions";

const geminiModelsCache = new Map();
const geminiQuotaCooldown = new Map();

app.use(express.json({ limit: `${MAX_IMAGE_MB + 2}mb` }));
app.use(express.static(path.join(__dirname, "public")));

function sanitizeKeys(rawKeys) {
  if (!Array.isArray(rawKeys)) return [];

  const unique = new Set();
  for (const raw of rawKeys) {
    const key = String(raw || "").trim();
    if (!key) continue;
    unique.add(key);
  }

  return [...unique];
}

function maskKey(key) {
  if (key.length <= 10) return `${key.slice(0, 2)}***`;
  return `${key.slice(0, 6)}...${key.slice(-4)}`;
}

function isGeminiModel(model) {
  const normalized = String(model || "").trim().toLowerCase();
  return normalized === "gemini" || normalized.startsWith("gemini-");
}

function isAutoGeneralModel(model) {
  return String(model || "").trim().toLowerCase() === "auto-general";
}

function isGeminiKey(apiKey) {
  return String(apiKey || "").trim().startsWith("AIza");
}

function resolveProvider({ apiKey, selectedModel }) {
  if (isAutoGeneralModel(selectedModel)) {
    if (isGeminiKey(apiKey)) {
      return {
        provider: "gemini",
        model: "gemini",
      };
    }

    return {
      provider: "openai",
      model: DEFAULT_MODEL,
    };
  }

  if (isGeminiModel(selectedModel)) {
    return {
      provider: "gemini",
      model: selectedModel,
    };
  }

  if (isGeminiKey(apiKey)) {
    return {
      provider: "gemini",
      model: "gemini",
    };
  }

  return {
    provider: "openai",
    model: selectedModel,
  };
}

function parseDataUrl(imageDataUrl) {
  const [meta, data] = String(imageDataUrl || "").split(",");
  const mimeMatch = meta.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64$/);
  if (!mimeMatch || !data) {
    throw new Error("Image data URL invalide.");
  }

  return {
    mimeType: mimeMatch[1],
    base64Data: data,
  };
}

function getRetryDelayMsFromMessage(message) {
  const text = String(message || "");
  const match = text.match(/retry in\s+([0-9]+(?:\.[0-9]+)?)s/i);
  if (!match) return null;

  const seconds = Number(match[1]);
  if (!Number.isFinite(seconds) || seconds <= 0) return null;
  return Math.ceil(seconds * 1000);
}

function isGeminiQuotaErrorMessage(message) {
  const text = String(message || "").toLowerCase();
  return text.includes("quota exceeded") || text.includes("rate limit") || text.includes("resource_exhausted");
}

function isGeminiTooManyRequestsMessage(message) {
  const text = String(message || "").toLowerCase();
  return text.includes("too many request") || text.includes("too many requests") || text.includes("429");
}

function isGeminiZeroFreeTierLimitMessage(message) {
  const text = String(message || "").toLowerCase();
  return text.includes("free_tier") && text.includes("limit: 0");
}

function getRetryDelayMsFromHeaders(headers) {
  const retryAfterRaw = headers?.get?.("retry-after");
  if (!retryAfterRaw) return null;

  const asNumber = Number(retryAfterRaw);
  if (Number.isFinite(asNumber) && asNumber > 0) {
    return Math.ceil(asNumber * 1000);
  }

  const asDate = Date.parse(retryAfterRaw);
  if (Number.isFinite(asDate)) {
    const delayMs = asDate - Date.now();
    return delayMs > 0 ? delayMs : null;
  }

  return null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function safeJsonStringify(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function summarizeHfDetectionOutput(output) {
  if (!Array.isArray(output) || output.length === 0) {
    return "Aucune détection exploitable renvoyée par le modèle Hugging Face.";
  }

  const detections = output
    .filter((item) => item && typeof item === "object")
    .filter((item) => item.label)
    .sort((a, b) => Number(b.score || 0) - Number(a.score || 0));

  if (!detections.length) {
    return `Réponse brute Hugging Face : ${safeJsonStringify(output).slice(0, 1000)}`;
  }

  const primary = detections[0];
  const scorePct = Math.round(Number(primary.score || 0) * 100);
  const box = primary.box || {};
  const hasBox =
    Number.isFinite(Number(box.xmin)) &&
    Number.isFinite(Number(box.xmax)) &&
    Number.isFinite(Number(box.ymin)) &&
    Number.isFinite(Number(box.ymax));

  const widthPx = hasBox ? Math.max(0, Math.round(Number(box.xmax) - Number(box.xmin))) : null;
  const heightPx = hasBox ? Math.max(0, Math.round(Number(box.ymax) - Number(box.ymin))) : null;

  return [
    `Objet principal détecté : ${primary.label} (confiance ${scorePct}%).`,
    "Type de matériau : non déterminable avec fiabilité depuis cette sortie de détection seule.",
    hasBox
      ? `Dimensions estimées dans l'image : ~${widthPx}px x ${heightPx}px (boîte de détection).`
      : "Dimensions : non disponibles dans la réponse du modèle.",
  ].join("\n");
}

async function testWithOpenAI({ apiKey, imageDataUrl, model, timeoutMs }) {
  const client = new OpenAI({ apiKey, timeout: timeoutMs });

  const response = await client.responses.create({
    model,
    input: [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: FIXED_PROMPT,
          },
          {
            type: "input_image",
            image_url: imageDataUrl,
          },
        ],
      },
    ],
  });

  const answer = (response.output_text || "").trim();

  return {
    status: "success",
    provider: "openai",
    model,
    answer: answer || "Réponse vide de l'API.",
  };
}

async function testWithHuggingFace({ apiKey, imageDataUrl, model, timeoutMs }) {
  const { mimeType, base64Data } = parseDataUrl(imageDataUrl);
  const endpoint = `https://api-inference.huggingface.co/models/${encodeURIComponent(model)}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": mimeType,
      },
      body: Buffer.from(base64Data, "base64"),
      signal: controller.signal,
    });

    const responseText = await response.text();
    let parsed = null;
    try {
      parsed = responseText ? JSON.parse(responseText) : null;
    } catch {
      parsed = responseText;
    }

    if (!response.ok) {
      const hfMessage =
        (parsed && typeof parsed === "object" && (parsed.error || parsed.message)) ||
        `Erreur Hugging Face (${response.status})`;
      const error = new Error(String(hfMessage));
      error.status = response.status;
      throw error;
    }

    const answer = summarizeHfDetectionOutput(parsed);

    return {
      status: "success",
      provider: "huggingface",
      model,
      answer,
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

async function testWithGemini({ apiKey, imageDataUrl, model, timeoutMs }) {
  const { mimeType, base64Data } = parseDataUrl(imageDataUrl);
  const now = Date.now();
  const quotaState = geminiQuotaCooldown.get(apiKey);
  if (quotaState && quotaState.retryAfter > now) {
    const waitSeconds = Math.max(1, Math.ceil((quotaState.retryAfter - now) / 1000));
    const error = new Error(`Quota Gemini dépassé pour cette clé. Réessaie dans ${waitSeconds}s.`);
    error.status = 429;
    throw error;
  }

  async function fetchJsonWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      const data = await response.json();
      return { response, data };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async function listGeminiGenerateModels() {
    const cacheKey = apiKey;
    const now = Date.now();
    const cached = geminiModelsCache.get(cacheKey);
    if (cached && cached.expiresAt > now && Array.isArray(cached.models) && cached.models.length > 0) {
      return cached.models;
    }

    const listUrl = `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(apiKey)}`;
    const { response, data } = await fetchJsonWithTimeout(listUrl, {
      method: "GET",
    });

    if (!response.ok) {
      const apiMessage = data?.error?.message || `Erreur Gemini (${response.status})`;
      const error = new Error(apiMessage);
      error.status = response.status;

      if (response.status === 429 || isGeminiQuotaErrorMessage(apiMessage)) {
        const retryDelayMs = getRetryDelayMsFromMessage(apiMessage) || GEMINI_QUOTA_COOLDOWN_MS;
        geminiQuotaCooldown.set(apiKey, {
          retryAfter: Date.now() + retryDelayMs,
        });

        const waitSeconds = Math.max(1, Math.ceil(retryDelayMs / 1000));
        error.message = `Quota Gemini dépassé pour cette clé (HTTP ${response.status}). Réessaie dans ${waitSeconds}s ou active la facturation du projet Google AI Studio.`;
        error.status = 429;
      }

      throw error;
    }

    const availableModels = (data?.models || [])
      .filter((item) => Array.isArray(item?.supportedGenerationMethods) && item.supportedGenerationMethods.includes("generateContent"))
      .map((item) => String(item.name || "").replace(/^models\//, ""))
      .filter(Boolean);

    geminiModelsCache.set(cacheKey, {
      models: availableModels,
      expiresAt: now + GEMINI_MODELS_CACHE_TTL_MS,
    });

    return availableModels;
  }

  function pickPreferredGeminiModel(availableModels) {
    const preferred = [
      "gemini-1.5-flash",
      "gemini-1.5-pro",
      "gemini-2.0-flash",
      "gemini-3-flash",
      DEFAULT_GEMINI_MODEL,
    ];

    return preferred.find((candidate) => availableModels.includes(candidate)) || availableModels[0] || null;
  }

  async function callGeminiGenerate(modelName) {
    const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(modelName)}:generateContent?key=${encodeURIComponent(apiKey)}`;

    const { response, data } = await fetchJsonWithTimeout(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        contents: [
          {
            role: "user",
            parts: [
              { text: FIXED_PROMPT },
              {
                inlineData: {
                  mimeType,
                  data: base64Data,
                },
              },
            ],
          },
        ],
      }),
    });

    if (!response.ok) {
      const apiMessage = data?.error?.message || `Erreur Gemini (${response.status})`;
      const error = new Error(apiMessage);
      error.status = response.status;

      if (response.status === 429 || isGeminiQuotaErrorMessage(apiMessage) || isGeminiTooManyRequestsMessage(apiMessage)) {
        const retryDelayMs =
          getRetryDelayMsFromHeaders(response.headers) ||
          getRetryDelayMsFromMessage(apiMessage) ||
          GEMINI_QUOTA_COOLDOWN_MS;
        geminiQuotaCooldown.set(apiKey, {
          retryAfter: Date.now() + retryDelayMs,
        });

        const waitSeconds = Math.max(1, Math.ceil(retryDelayMs / 1000));
        if (isGeminiZeroFreeTierLimitMessage(apiMessage)) {
          error.message = `Quota Gemini gratuit indisponible pour cette clé/projet (limit: 0). Essaie le modèle gemini-1.5-flash ou active la facturation Google AI Studio. Réessaie dans ${waitSeconds}s.`;
        } else {
          error.message = `Quota Gemini dépassé pour cette clé (HTTP ${response.status}). Réessaie dans ${waitSeconds}s ou active la facturation du projet Google AI Studio.`;
        }
        error.status = 429;
        error.retryDelayMs = retryDelayMs;
        error.isGeminiRateLimit = true;
        error.isGeminiZeroLimit = isGeminiZeroFreeTierLimitMessage(apiMessage);
      }

      throw error;
    }

    const answer = (data?.candidates || [])
      .flatMap((candidate) => candidate?.content?.parts || [])
      .map((part) => part?.text)
      .filter(Boolean)
      .join("\n")
      .trim();

    return {
      status: "success",
      provider: "gemini",
      model: modelName,
      answer: answer || "Réponse vide de l'API.",
    };
  }

  const requestedModel = String(model || "").trim() || "gemini";
  let effectiveModel = requestedModel;

  if (requestedModel === "gemini") {
    const availableModels = await listGeminiGenerateModels();
    const autoModel = pickPreferredGeminiModel(availableModels);
    if (!autoModel) {
      const noModelError = new Error("Aucun modèle Gemini compatible generateContent n'est disponible pour cette clé API.");
      noModelError.status = 400;
      throw noModelError;
    }
    effectiveModel = autoModel;
  }

  try {
    return await callGeminiGenerate(effectiveModel);
  } catch (error) {
    if (error?.isGeminiRateLimit && !error?.isGeminiZeroLimit) {
      const retryDelayMs = Number(error?.retryDelayMs || 0);
      if (retryDelayMs > 0 && retryDelayMs <= GEMINI_AUTO_RETRY_MAX_WAIT_MS) {
        await sleep(retryDelayMs);
        return callGeminiGenerate(effectiveModel);
      }
    }

    const message = String(error?.message || "").toLowerCase();
    const isUnknownModel = message.includes("not found") || message.includes("not supported for generatecontent");

    if (!isUnknownModel) {
      throw error;
    }

    const availableModels = await listGeminiGenerateModels();
    const fallbackModel = pickPreferredGeminiModel(availableModels);

    if (!fallbackModel) {
      const noModelError = new Error("Aucun modèle Gemini compatible generateContent n'est disponible pour cette clé API.");
      noModelError.status = 400;
      throw noModelError;
    }

    if (fallbackModel === effectiveModel) {
      throw error;
    }

    return callGeminiGenerate(fallbackModel);
  }
}

async function testSingleKey({ apiKey, imageDataUrl, selectedModel, timeoutMs }) {
  const { provider, model } = resolveProvider({ apiKey, selectedModel });

  if (provider === "gemini") {
    return testWithGemini({ apiKey, imageDataUrl, model, timeoutMs });
  }

  return testWithOpenAI({ apiKey, imageDataUrl, model, timeoutMs });
}

app.post("/api/test-keys", async (req, res) => {
  const { keys, imageDataUrl, model, timeoutMs } = req.body || {};

  const parsedKeys = sanitizeKeys(keys);
  const selectedModel = String(model || DEFAULT_MODEL).trim() || DEFAULT_MODEL;
  const parsedTimeout = Number(timeoutMs || 45000);
  const safeTimeout = Number.isFinite(parsedTimeout)
    ? Math.min(Math.max(parsedTimeout, 5000), 120000)
    : 45000;

  if (parsedKeys.length === 0) {
    return res.status(400).json({
      error: "Ajoute au moins une clé API.",
    });
  }

  if (!imageDataUrl || typeof imageDataUrl !== "string" || !imageDataUrl.startsWith("data:image/")) {
    return res.status(400).json({
      error: "Image invalide. Envoie une image au format data URL.",
    });
  }

  const base64Part = imageDataUrl.split(",")[1] || "";
  const imageSizeBytes = Buffer.byteLength(base64Part, "base64");
  if (imageSizeBytes > MAX_IMAGE_BYTES) {
    return res.status(400).json({
      error: `Image trop volumineuse (max ${MAX_IMAGE_MB}MB).`,
    });
  }

  const results = [];
  for (const key of parsedKeys) {
    const resolved = resolveProvider({ apiKey: key, selectedModel });

    try {
      const result = await testSingleKey({
        apiKey: key,
        imageDataUrl,
        selectedModel,
        timeoutMs: safeTimeout,
      });

      results.push({
        keyLabel: maskKey(key),
        ...result,
      });
    } catch (error) {
      const message = error?.error?.message || error?.message || "Erreur inconnue";
      const statusCode = error?.status || null;

      results.push({
        keyLabel: maskKey(key),
        status: "error",
        provider: resolved.provider,
        model: resolved.model,
        error: message,
        statusCode,
      });
    }
  }

  return res.json({
    prompt: FIXED_PROMPT,
    model: selectedModel,
    count: results.length,
    results,
  });
});

app.post("/api/test-hf-keys", async (req, res) => {
  const { keys, imageDataUrl, model, timeoutMs } = req.body || {};

  const parsedKeys = sanitizeKeys(keys);
  const selectedModel = String(model || DEFAULT_HF_MODEL).trim() || DEFAULT_HF_MODEL;
  const parsedTimeout = Number(timeoutMs || 45000);
  const safeTimeout = Number.isFinite(parsedTimeout)
    ? Math.min(Math.max(parsedTimeout, 5000), 120000)
    : 45000;

  if (parsedKeys.length === 0) {
    return res.status(400).json({
      error: "Ajoute au moins une clé Hugging Face.",
    });
  }

  if (!imageDataUrl || typeof imageDataUrl !== "string" || !imageDataUrl.startsWith("data:image/")) {
    return res.status(400).json({
      error: "Image invalide. Envoie une image au format data URL.",
    });
  }

  const base64Part = imageDataUrl.split(",")[1] || "";
  const imageSizeBytes = Buffer.byteLength(base64Part, "base64");
  if (imageSizeBytes > MAX_IMAGE_BYTES) {
    return res.status(400).json({
      error: `Image trop volumineuse (max ${MAX_IMAGE_MB}MB).`,
    });
  }

  const results = [];
  for (const key of parsedKeys) {
    try {
      const result = await testWithHuggingFace({
        apiKey: key,
        imageDataUrl,
        model: selectedModel,
        timeoutMs: safeTimeout,
      });

      results.push({
        keyLabel: maskKey(key),
        ...result,
      });
    } catch (error) {
      const message = error?.error?.message || error?.message || "Erreur inconnue";
      const statusCode = error?.status || null;

      results.push({
        keyLabel: maskKey(key),
        status: "error",
        provider: "huggingface",
        model: selectedModel,
        error: message,
        statusCode,
      });
    }
  }

  return res.json({
    model: selectedModel,
    count: results.length,
    results,
  });
});

app.get("/api/health", (req, res) => {
  res.json({ ok: true });
});

app.listen(port, () => {
  console.log(`App-test-api démarrée sur http://localhost:${port}`);
});