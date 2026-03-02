const express = require("express");
const path = require("path");
const dotenv = require("dotenv");
const OpenAI = require("openai");

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const DEFAULT_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";
const DEFAULT_GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-1.5-flash";
const DEFAULT_HF_MODEL_TEXT = process.env.HF_MODEL_TEXT || "moonshotai/Kimi-K2.5";
const DEFAULT_HF_MODEL_VISION = process.env.HF_MODEL_VISION || "facebook/detr-resnet-50";
const HF_INFERENCE_BASE_URL =
  process.env.HF_INFERENCE_BASE_URL || "https://router.huggingface.co/hf-inference/models";
const HF_ROUTER_CHAT_URL =
  process.env.HF_ROUTER_CHAT_URL || "https://router.huggingface.co/v1/chat/completions";
const MAX_IMAGE_MB = Number(process.env.MAX_IMAGE_MB || 8);
const MAX_IMAGE_BYTES = MAX_IMAGE_MB * 1024 * 1024;
const GEMINI_MODELS_CACHE_TTL_MS = Number(process.env.GEMINI_MODELS_CACHE_TTL_MS || 10 * 60 * 1000);
const GEMINI_QUOTA_COOLDOWN_MS = Number(process.env.GEMINI_QUOTA_COOLDOWN_MS || 60 * 1000);
const GEMINI_AUTO_RETRY_MAX_WAIT_MS = Number(process.env.GEMINI_AUTO_RETRY_MAX_WAIT_MS || 15000);

const FIXED_PROMPT = `Peux-tu me donner les informations suivantes liées à l'objet principal mis en avant dans la photo que je fournis :
Type de matériau
Dimensions

Réponds en 3 lignes maximum, sans explication.
Format exact :
Matériau: <1 mot ou 2 max>
Dimensions estimées: <L x l x e en cm, ou "Non estimable">
Poids estimés : < xx Kg, ou "Non estimable">
Etat : < Neuf / bon etat / abimé >
Confiance: <faible|moyenne|élevée>
Exemple de sortie attendue :

Matériau: bois (chêne)
Dimensions estimées: ~180 x 65 x 3 cm
Poids Estimées: ~60 kg
Etat : Bon
Confiance: moyenne

NE ME RENVOIE ABSOLUMENT PAS AUTRE CHOSE QUE CE AUE JE TE DEMANDE
Je veux une reponse courte et concise tel que decrite
renvoie moi ABSOLUMENT une réponse au format json`;

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

function truncateText(value, maxLength = 1200) {
  const text = String(value || "");
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength)}...`;
}

function normalizeFieldValue(value, fallback = "Non estimable") {
  const parsed = String(value || "").trim();
  return parsed || fallback;
}

function simplifyStateLabel(value) {
  const text = String(value || "").toLowerCase();
  if (text.includes("neuf")) return "Neuf";
  if (text.includes("abim") || text.includes("abîm")) return "abîmé";
  if (text.includes("bon")) return "bon etat";
  return "Non estimable";
}

function simplifyConfidenceLabel(value) {
  const text = String(value || "").toLowerCase();
  if (text.includes("faible")) return "faible";
  if (text.includes("elev") || text.includes("élev")) return "élevée";
  if (text.includes("moy")) return "moyenne";
  return "moyenne";
}

function normalizeJsonAnswerShape(input) {
  const object = input && typeof input === "object" ? input : {};
  const normalized = {};

  for (const [rawKey, rawValue] of Object.entries(object)) {
    const key = String(rawKey)
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");

    normalized[key] = rawValue;
  }

  return {
    materiau: normalizeFieldValue(
      normalized.materiau ?? normalized.typedemateriau ?? normalized.material,
      "Non estimable"
    ),
    dimensions_estimees: normalizeFieldValue(
      normalized.dimensionsestimees ?? normalized.dimensions ?? normalized.dimension,
      "Non estimable"
    ),
    poids_estime: normalizeFieldValue(
      normalized.poidsestimes ?? normalized.poidsestime ?? normalized.poids,
      "Non estimable"
    ),
    etat: simplifyStateLabel(normalized.etat),
    confiance: simplifyConfidenceLabel(normalized.confiance),
  };
}

function extractDimensionComponent(text, labelRegex) {
  const source = String(text || "");
  const regex = new RegExp(
    `${labelRegex}[^\\d]{0,20}(~?\\d+(?:[.,]\\d+)?(?:\\s*[-à]\\s*\\d+(?:[.,]\\d+)?)?)\\s*(cm|mm|m)`,
    "i"
  );
  const match = source.match(regex);
  if (!match) return null;

  const value = String(match[1] || "").replace(/\s+/g, " ").trim();
  const unit = String(match[2] || "cm").trim();
  if (!value) return null;
  return `${value} ${unit}`;
}

function extractDimensionEstimateFromText(text) {
  const source = String(text || "");

  const compactMatch = source.match(
    /(\d{2,4}(?:[.,]\d+)?\s*(?:x|×)\s*\d{2,4}(?:[.,]\d+)?(?:\s*(?:x|×)\s*\d{1,4}(?:[.,]\d+)?)?\s*(?:cm|mm|m)?)/i
  );
  if (compactMatch) {
    return compactMatch[1].replace(/\s+/g, " ").trim();
  }

  const height = extractDimensionComponent(source, "hauteur|height|h");
  const width = extractDimensionComponent(source, "largeur|width|l");
  const depth = extractDimensionComponent(source, "epaisseur|épaisseur|profondeur|depth|e");

  const parts = [];
  if (height) parts.push(`H ${height}`);
  if (width) parts.push(`L ${width}`);
  if (depth) parts.push(`E ${depth}`);

  if (parts.length > 0) {
    return parts.join(" • ");
  }

  return "Non estimable";
}

function extractWeightEstimateFromText(text) {
  const source = String(text || "");

  const explicitWeight = source.match(/poids[^\d]{0,25}(~?\d+(?:[.,]\d+)?)\s*(kg|g)/i);
  if (explicitWeight) {
    return `${explicitWeight[1]} ${explicitWeight[2]}`.replace(/\s+/g, " ").trim();
  }

  const genericKg = source.match(/(~?\d+(?:[.,]\d+)?)\s*(kg|g)/i);
  if (genericKg) {
    return `${genericKg[1]} ${genericKg[2]}`.replace(/\s+/g, " ").trim();
  }

  return "Non estimable";
}

function extractFirstJsonObjectFromText(text) {
  const source = String(text || "");
  for (let start = 0; start < source.length; start += 1) {
    if (source[start] !== "{") continue;

    let depth = 0;
    for (let end = start; end < source.length; end += 1) {
      const char = source[end];
      if (char === "{") depth += 1;
      if (char === "}") depth -= 1;

      if (depth === 0) {
        const candidate = source.slice(start, end + 1);
        try {
          const parsed = JSON.parse(candidate);
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            return parsed;
          }
        } catch {
          break;
        }
      }
    }
  }

  return null;
}

function buildHeuristicJsonFromText(text) {
  const source = String(text || "");

  const materialMatch = source.match(/(bois|ch[eê]ne|m[eé]tal|plastique|verre|c[ée]ramique|textile)/i);
  const stateMatch = source.match(/(neuf|bon(?:\s+[eé]tat)?|ab[iî]m[ée])/i);
  const confidenceMatch = source.match(/(faible|moyenne|[eé]lev[ée])/i);

  return {
    materiau: materialMatch ? materialMatch[1] : "Non estimable",
    dimensions_estimees: extractDimensionEstimateFromText(source),
    poids_estime: extractWeightEstimateFromText(source),
    etat: simplifyStateLabel(stateMatch ? stateMatch[1] : ""),
    confiance: simplifyConfidenceLabel(confidenceMatch ? confidenceMatch[1] : ""),
  };
}

function normalizeToStrictJsonString(text) {
  const fromJson = extractFirstJsonObjectFromText(text);
  const normalizedObject = fromJson
    ? normalizeJsonAnswerShape(fromJson)
    : buildHeuristicJsonFromText(text);

  return JSON.stringify(normalizedObject, null, 2);
}

function getHfEstimatedTimeSeconds(payload) {
  if (!payload || typeof payload !== "object") return null;
  const raw = payload.estimated_time ?? payload.estimatedTime;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) return null;
  return value;
}

function extractHfChatTextPayload(payload) {
  if (typeof payload === "string") {
    return payload.trim();
  }

  if (!payload || typeof payload !== "object") {
    return "";
  }

  const directTextCandidates = [
    payload.output_text,
    payload.generated_text,
    payload.text,
    payload.completion,
  ];
  for (const candidate of directTextCandidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim();
    }
  }

  const choice = payload?.choices?.[0];
  if (choice && typeof choice === "object") {
    if (typeof choice.text === "string" && choice.text.trim()) {
      return choice.text.trim();
    }

    if (typeof choice.message === "string" && choice.message.trim()) {
      return choice.message.trim();
    }

    const messageContent = choice?.message?.content;
    if (typeof messageContent === "string" && messageContent.trim()) {
      return messageContent.trim();
    }

    if (Array.isArray(messageContent)) {
      const collected = messageContent
        .map((part) => {
          if (typeof part === "string") return part;
          if (!part || typeof part !== "object") return "";
          if (typeof part.text === "string") return part.text;
          if (typeof part.content === "string") return part.content;
          return "";
        })
        .filter(Boolean)
        .join("\n")
        .trim();

      if (collected) return collected;
    }

    if (typeof choice?.delta?.content === "string" && choice.delta.content.trim()) {
      return choice.delta.content.trim();
    }

    const reasoning = choice?.message?.reasoning ?? choice?.reasoning;
    if (typeof reasoning === "string" && reasoning.trim()) {
      return reasoning.trim();
    }

    if (Array.isArray(reasoning)) {
      const collectedReasoning = reasoning
        .map((part) => {
          if (typeof part === "string") return part;
          if (!part || typeof part !== "object") return "";
          if (typeof part.text === "string") return part.text;
          if (typeof part.content === "string") return part.content;
          return "";
        })
        .filter(Boolean)
        .join("\n")
        .trim();

      if (collectedReasoning) return collectedReasoning;
    }
  }

  if (Array.isArray(payload) && payload.length > 0) {
    for (const item of payload) {
      if (typeof item === "string" && item.trim()) {
        return item.trim();
      }
      if (item && typeof item === "object") {
        const itemTextCandidates = [item.generated_text, item.output_text, item.text];
        for (const candidate of itemTextCandidates) {
          if (typeof candidate === "string" && candidate.trim()) {
            return candidate.trim();
          }
        }
      }
    }
  }

  return "";
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

async function testWithHuggingFace({ apiKey, imageDataUrl, model, timeoutMs, retryCount = 0 }) {
  const { mimeType, base64Data } = parseDataUrl(imageDataUrl);
  const endpoint = `${HF_INFERENCE_BASE_URL.replace(/\/$/, "")}/${encodeURIComponent(model)}`;

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
      const estimatedSec = getHfEstimatedTimeSeconds(parsed);
      if (response.status === 503 && retryCount === 0 && estimatedSec && estimatedSec <= 20) {
        await sleep(Math.ceil(estimatedSec * 1000));
        return testWithHuggingFace({ apiKey, imageDataUrl, model, timeoutMs, retryCount: 1 });
      }

      const hfMessage =
        (parsed && typeof parsed === "object" && (parsed.error || parsed.message)) ||
        `Erreur Hugging Face (${response.status})`;
      const error = new Error(String(hfMessage));
      error.status = response.status;

      if (response.status === 503) {
        const waitMsg = estimatedSec ? `Réessaie dans ~${Math.ceil(estimatedSec)}s.` : "Réessaie dans quelques secondes.";
        error.message = `Service Hugging Face temporairement indisponible (503, modèle en cours de chargement). ${waitMsg}`;
      }

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

async function testWithHuggingFaceText({ apiKey, model, timeoutMs, prompt, imageDataUrl, retryCount = 0 }) {
  const endpoint = HF_ROUTER_CHAT_URL;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: prompt,
              },
              {
                type: "image_url",
                image_url: {
                  url: imageDataUrl,
                },
              },
            ],
          },
        ],
        max_tokens: 300,
      }),
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
      const estimatedSec = getHfEstimatedTimeSeconds(parsed);
      if (response.status === 503 && retryCount === 0 && estimatedSec && estimatedSec <= 20) {
        await sleep(Math.ceil(estimatedSec * 1000));
        return testWithHuggingFaceText({ apiKey, model, timeoutMs, prompt, imageDataUrl, retryCount: 1 });
      }

      const hfMessage =
        (parsed && typeof parsed === "object" && (parsed.error || parsed.message)) ||
        `Erreur Hugging Face (${response.status})`;
      const error = new Error(String(hfMessage));
      error.status = response.status;

      if (response.status === 503) {
        const waitMsg = estimatedSec ? `Réessaie dans ~${Math.ceil(estimatedSec)}s.` : "Réessaie dans quelques secondes.";
        error.message = `Service Hugging Face temporairement indisponible (503, modèle en cours de chargement). ${waitMsg}`;
      }

      throw error;
    }

    const generatedText = extractHfChatTextPayload(parsed);
    const fallbackPayload = truncateText(safeJsonStringify(parsed));

    return {
      status: "success",
      provider: "huggingface",
      model,
      answer: normalizeToStrictJsonString(
        generatedText ||
          `Réponse texte non détectée dans le format retourné par l'API. Aperçu brut : ${fallbackPayload}`
      ),
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
        answer: normalizeToStrictJsonString(result.answer),
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
  const { keys, imageDataUrl, model, timeoutMs, task, prompt } = req.body || {};

  const parsedKeys = sanitizeKeys(keys);
  const selectedTask = String(task || "object-detection").trim().toLowerCase();
  const defaultHfModel = selectedTask === "text-generation" ? DEFAULT_HF_MODEL_TEXT : DEFAULT_HF_MODEL_VISION;
  const selectedModel = String(model || defaultHfModel).trim() || defaultHfModel;
  const selectedPrompt = String(prompt || FIXED_PROMPT).trim() || FIXED_PROMPT;
  const parsedTimeout = Number(timeoutMs || 45000);
  const safeTimeout = Number.isFinite(parsedTimeout)
    ? Math.min(Math.max(parsedTimeout, 5000), 120000)
    : 45000;

  if (parsedKeys.length === 0) {
    return res.status(400).json({
      error: "Ajoute au moins une clé Hugging Face.",
    });
  }

  if (!["object-detection", "text-generation"].includes(selectedTask)) {
    return res.status(400).json({
      error: "Mode Hugging Face invalide.",
    });
  }

  if (selectedTask === "object-detection" && (!imageDataUrl || typeof imageDataUrl !== "string" || !imageDataUrl.startsWith("data:image/"))) {
    return res.status(400).json({
      error: "Image invalide. Envoie une image au format data URL.",
    });
  }

  if (selectedTask === "text-generation" && (!imageDataUrl || typeof imageDataUrl !== "string" || !imageDataUrl.startsWith("data:image/"))) {
    return res.status(400).json({
      error: "Image invalide. Envoie une image au format data URL.",
    });
  }

  if (selectedTask === "object-detection" || selectedTask === "text-generation") {
    const base64Part = imageDataUrl.split(",")[1] || "";
    const imageSizeBytes = Buffer.byteLength(base64Part, "base64");
    if (imageSizeBytes > MAX_IMAGE_BYTES) {
      return res.status(400).json({
        error: `Image trop volumineuse (max ${MAX_IMAGE_MB}MB).`,
      });
    }
  }

  const results = [];
  for (const key of parsedKeys) {
    try {
      const result =
        selectedTask === "text-generation"
          ? await testWithHuggingFaceText({
              apiKey: key,
              model: selectedModel,
              timeoutMs: safeTimeout,
              prompt: selectedPrompt,
              imageDataUrl,
            })
          : await testWithHuggingFace({
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
    task: selectedTask,
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