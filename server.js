const express = require("express");
const path = require("path");
const dotenv = require("dotenv");
const OpenAI = require("openai");

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const DEFAULT_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";
const DEFAULT_GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-1.5-flash";
const MAX_IMAGE_MB = Number(process.env.MAX_IMAGE_MB || 8);
const MAX_IMAGE_BYTES = MAX_IMAGE_MB * 1024 * 1024;
const GEMINI_MODELS_CACHE_TTL_MS = Number(process.env.GEMINI_MODELS_CACHE_TTL_MS || 10 * 60 * 1000);
const GEMINI_QUOTA_COOLDOWN_MS = Number(process.env.GEMINI_QUOTA_COOLDOWN_MS || 60 * 1000);

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

function isGeminiKey(apiKey) {
  return String(apiKey || "").trim().startsWith("AIza");
}

function resolveProvider({ apiKey, selectedModel }) {
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
      "gemini-3-flash",
      "gemini-2.0-flash",
      "gemini-1.5-flash",
      "gemini-1.5-pro",
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

app.get("/api/health", (req, res) => {
  res.json({ ok: true });
});

app.listen(port, () => {
  console.log(`App-test-api démarrée sur http://localhost:${port}`);
});