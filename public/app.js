const keysInput = document.getElementById("keysInput");
const modelSelect = document.getElementById("modelSelect");
const timeoutInput = document.getElementById("timeoutInput");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const runBtn = document.getElementById("runBtn");
const statusBox = document.getElementById("status");
const resultsBox = document.getElementById("results");
const cooldownInfo = document.getElementById("cooldownInfo");

let imageDataUrl = "";
const geminiCooldownByKey = new Map();
let cooldownIntervalId = null;

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.style.color = isError ? "#ff8f8f" : "#b9d7ff";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderResults(results = []) {
  if (!results.length) {
    resultsBox.innerHTML = "<p>Aucun résultat pour le moment.</p>";
    return;
  }

  resultsBox.innerHTML = results
    .map((item) => {
      if (item.status === "success") {
        return `
          <article class="result-item success">
            <div class="meta"><strong>${escapeHtml(item.keyLabel)}</strong> • ${escapeHtml(item.provider || "openai")} • ${escapeHtml(item.model)}</div>
            <div>${escapeHtml(item.answer)}</div>
          </article>
        `;
      }

      return `
        <article class="result-item error">
          <div class="meta"><strong>${escapeHtml(item.keyLabel)}</strong> • ${escapeHtml(item.provider || "openai")} • ${escapeHtml(item.model)}${item.statusCode ? ` • HTTP ${item.statusCode}` : ""}</div>
          <div>${escapeHtml(item.error || "Erreur inconnue")}</div>
        </article>
      `;
    })
    .join("");
}

function parseKeys(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function maskKey(key) {
  const safeKey = String(key || "");
  if (safeKey.length <= 10) return `${safeKey.slice(0, 2)}***`;
  return `${safeKey.slice(0, 6)}...${safeKey.slice(-4)}`;
}

function isGeminiModel(model) {
  const value = String(model || "").trim().toLowerCase();
  return value === "gemini" || value.startsWith("gemini-");
}

function isGeminiKey(key) {
  return String(key || "").trim().startsWith("AIza");
}

function keyUsesGemini({ key, model }) {
  return isGeminiModel(model) || isGeminiKey(key);
}

function parseRetrySeconds(message) {
  const text = String(message || "");
  const match = text.match(/(?:réessaie|retry)\s+(?:dans|in)\s+([0-9]+(?:[.,][0-9]+)?)s/i);
  if (!match) return null;

  const parsed = Number(match[1].replace(",", "."));
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return Math.ceil(parsed);
}

function cleanupExpiredCooldowns() {
  const now = Date.now();
  for (const [key, retryAfter] of geminiCooldownByKey.entries()) {
    if (retryAfter <= now) {
      geminiCooldownByKey.delete(key);
    }
  }
}

function getActiveGeminiCooldowns(keys, model) {
  cleanupExpiredCooldowns();
  const now = Date.now();

  return keys
    .filter((key) => keyUsesGemini({ key, model }))
    .map((key) => {
      const retryAfter = geminiCooldownByKey.get(key);
      if (!retryAfter || retryAfter <= now) return null;
      const remainingSec = Math.max(1, Math.ceil((retryAfter - now) / 1000));
      return {
        key,
        remainingSec,
      };
    })
    .filter(Boolean);
}

function renderCooldownInfo() {
  const keys = parseKeys(keysInput.value);
  const model = modelSelect.value.trim();
  const activeCooldowns = getActiveGeminiCooldowns(keys, model);

  if (!activeCooldowns.length) {
    cooldownInfo.hidden = true;
    cooldownInfo.innerHTML = "";
    return;
  }

  cooldownInfo.hidden = false;
  cooldownInfo.innerHTML = `
    <div>Clés Gemini temporairement bloquées (quota/rate limit) :</div>
    <div class="cooldown-items">
      ${activeCooldowns
        .map(
          (item) =>
            `<span class="cooldown-item">${escapeHtml(maskKey(item.key))} • ${item.remainingSec}s</span>`
        )
        .join("")}
    </div>
  `;
}

function ensureCooldownRefresh() {
  if (cooldownIntervalId) return;

  cooldownIntervalId = setInterval(() => {
    cleanupExpiredCooldowns();
    renderCooldownInfo();

    if (geminiCooldownByKey.size === 0) {
      clearInterval(cooldownIntervalId);
      cooldownIntervalId = null;
    }
  }, 1000);
}

function updateCooldownsFromResults(submittedKeys, results) {
  const now = Date.now();

  results.forEach((result, index) => {
    const key = submittedKeys[index];
    if (!key) return;

    const provider = String(result?.provider || "").toLowerCase();
    if (provider !== "gemini") return;

    if (result.status === "success") {
      geminiCooldownByKey.delete(key);
      return;
    }

    const retryFromMessage = parseRetrySeconds(result.error || "");
    const isRateLimit = Number(result.statusCode) === 429;

    if (retryFromMessage || isRateLimit) {
      const waitSec = retryFromMessage || 60;
      geminiCooldownByKey.set(key, now + waitSec * 1000);
    }
  });

  if (geminiCooldownByKey.size > 0) {
    ensureCooldownRefresh();
  }

  renderCooldownInfo();
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Impossible de lire l'image."));
    reader.readAsDataURL(file);
  });
}

imageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    imageDataUrl = "";
    preview.hidden = true;
    return;
  }

  try {
    imageDataUrl = await readFileAsDataUrl(file);
    preview.src = imageDataUrl;
    preview.hidden = false;
    setStatus("Image chargée.");
  } catch (error) {
    imageDataUrl = "";
    preview.hidden = true;
    setStatus(error.message || "Erreur de chargement image.", true);
  }
});

runBtn.addEventListener("click", async () => {
  const keys = parseKeys(keysInput.value);
  const model = modelSelect.value.trim() || "gpt-4.1-mini";
  const timeoutMs = Number(timeoutInput.value || 45000);
  const activeCooldowns = getActiveGeminiCooldowns(keys, model);

  if (!keys.length) {
    setStatus("Ajoute au moins une clé API.", true);
    return;
  }

  if (activeCooldowns.length) {
    setStatus("Certaines clés Gemini sont en cooldown. Attends la fin du délai avant de relancer.", true);
    renderCooldownInfo();
    return;
  }

  if (!imageDataUrl) {
    setStatus("Ajoute une image avant de lancer le test.", true);
    return;
  }

  runBtn.disabled = true;
  renderResults([]);
  setStatus(`Test en cours sur ${keys.length} clé(s)...`);

  try {
    const response = await fetch("/api/test-keys", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        keys,
        model,
        timeoutMs,
        imageDataUrl,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Erreur backend");
    }

    renderResults(data.results || []);
    updateCooldownsFromResults(keys, data.results || []);
    setStatus(`Terminé : ${data.count || 0} clé(s) testée(s).`);
  } catch (error) {
    setStatus(error.message || "Erreur pendant le test.", true);
  } finally {
    runBtn.disabled = false;
  }
});

renderResults([]);
keysInput.addEventListener("input", renderCooldownInfo);
modelSelect.addEventListener("change", renderCooldownInfo);
renderCooldownInfo();