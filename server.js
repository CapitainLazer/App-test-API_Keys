const express = require("express");
const path = require("path");
const dotenv = require("dotenv");
const OpenAI = require("openai");

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const DEFAULT_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";
const MAX_IMAGE_MB = Number(process.env.MAX_IMAGE_MB || 8);
const MAX_IMAGE_BYTES = MAX_IMAGE_MB * 1024 * 1024;

const FIXED_PROMPT =
  "Peux-tu me donner les informations suivantes liées à l'objet principal mis en avant dans la photo que je fournis :\nType de matériau\nDimensions";

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

async function testSingleKey({ apiKey, imageDataUrl, model, timeoutMs }) {
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
    model,
    answer: answer || "Réponse vide de l'API.",
  };
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
    try {
      const result = await testSingleKey({
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
        model: selectedModel,
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