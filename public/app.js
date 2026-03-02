const tabMainBtn = document.getElementById("tabMainBtn");
const tabHfBtn = document.getElementById("tabHfBtn");
const panelMain = document.getElementById("panelMain");
const panelHf = document.getElementById("panelHf");

const keysInput = document.getElementById("keysInput");
const modelSelect = document.getElementById("modelSelect");
const timeoutInput = document.getElementById("timeoutInput");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const runBtn = document.getElementById("runBtn");
const statusBox = document.getElementById("status");
const resultsBox = document.getElementById("results");
const cooldownInfo = document.getElementById("cooldownInfo");

const hfKeysInput = document.getElementById("hfKeysInput");
const hfModelInput = document.getElementById("hfModelInput");
const hfTimeoutInput = document.getElementById("hfTimeoutInput");
const hfImageInput = document.getElementById("hfImageInput");
const hfPreview = document.getElementById("hfPreview");
const hfRunBtn = document.getElementById("hfRunBtn");
const hfStatusBox = document.getElementById("hfStatus");
const hfResultsBox = document.getElementById("hfResults");

let imageDataUrl = "";
let hfImageDataUrl = "";
const geminiCooldownByKey = new Map();
let cooldownIntervalId = null;

function setStatus(target, message, isError = false) {
  target.textContent = message;
  target.classList.toggle("error", isError);
  target.classList.toggle("ok", !isError);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderResults(target, results = [], defaultProvider = "openai") {
  if (!results.length) {
    target.innerHTML = "<p class=\"hint\">Aucun résultat pour le moment. Lance un test pour voir les réponses de chaque clé.</p>";
    return;
  }

  target.innerHTML = results
    .map((item) => {
      if (item.status === "success") {
        return `
          <article class="result-item success">
            <div class="meta"><strong>${escapeHtml(item.keyLabel)}</strong> • ${escapeHtml(item.provider || defaultProvider)} • ${escapeHtml(item.model)}</div>
            <div>${escapeHtml(item.answer)}</div>
          </article>
        `;
      }

      return `
        <article class="result-item error">
          <div class="meta"><strong>${escapeHtml(item.keyLabel)}</strong> • ${escapeHtml(item.provider || defaultProvider)} • ${escapeHtml(item.model)}${item.statusCode ? ` • HTTP ${item.statusCode}` : ""}</div>
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

function switchTab(tab) {
  const isMain = tab === "main";

  tabMainBtn.classList.toggle("active", isMain);
  tabHfBtn.classList.toggle("active", !isMain);
  panelMain.classList.toggle("active", isMain);
  panelHf.classList.toggle("active", !isMain);
  tabMainBtn.setAttribute("aria-selected", String(isMain));
  tabHfBtn.setAttribute("aria-selected", String(!isMain));
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
    setStatus(statusBox, "Image chargée.");
  } catch (error) {
    imageDataUrl = "";
    preview.hidden = true;
    setStatus(statusBox, error.message || "Erreur de chargement image.", true);
  }
});

hfImageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    hfImageDataUrl = "";
    hfPreview.hidden = true;
    return;
  }

  try {
    hfImageDataUrl = await readFileAsDataUrl(file);
    hfPreview.src = hfImageDataUrl;
    hfPreview.hidden = false;
    setStatus(hfStatusBox, "Image chargée.");
  } catch (error) {
    hfImageDataUrl = "";
    hfPreview.hidden = true;
    setStatus(hfStatusBox, error.message || "Erreur de chargement image.", true);
  }
});

runBtn.addEventListener("click", async () => {
  const keys = parseKeys(keysInput.value);
  const model = modelSelect.value.trim() || "gpt-4.1-mini";
  const timeoutMs = Number(timeoutInput.value || 45000);
  const activeCooldowns = getActiveGeminiCooldowns(keys, model);

  if (!keys.length) {
    setStatus(statusBox, "Ajoute au moins une clé API.", true);
    return;
  }

  if (activeCooldowns.length) {
    setStatus(statusBox, "Certaines clés Gemini sont en cooldown. Attends la fin du délai avant de relancer.", true);
    renderCooldownInfo();
    return;
  }

  if (!imageDataUrl) {
    setStatus(statusBox, "Ajoute une image avant de lancer le test.", true);
    return;
  }

  runBtn.disabled = true;
  runBtn.textContent = "Test en cours...";
  renderResults(resultsBox, []);
  setStatus(statusBox, `Test en cours sur ${keys.length} clé(s)...`);

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

    renderResults(resultsBox, data.results || [], "openai");
    updateCooldownsFromResults(keys, data.results || []);
    setStatus(statusBox, `Terminé : ${data.count || 0} clé(s) testée(s).`);
  } catch (error) {
    setStatus(statusBox, error.message || "Erreur pendant le test.", true);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Lancer le test";
  }
});

hfRunBtn.addEventListener("click", async () => {
  const keys = parseKeys(hfKeysInput.value);
  const model = hfModelInput.value.trim() || "facebook/detr-resnet-50";
  const timeoutMs = Number(hfTimeoutInput.value || 45000);

  if (!keys.length) {
    setStatus(hfStatusBox, "Ajoute au moins une clé Hugging Face.", true);
    return;
  }

  if (!hfImageDataUrl) {
    setStatus(hfStatusBox, "Ajoute une image avant de lancer le test.", true);
    return;
  }

  hfRunBtn.disabled = true;
  hfRunBtn.textContent = "Test HF en cours...";
  renderResults(hfResultsBox, [], "huggingface");
  setStatus(hfStatusBox, `Test Hugging Face en cours sur ${keys.length} clé(s)...`);

  try {
    const response = await fetch("/api/test-hf-keys", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        keys,
        model,
        timeoutMs,
        imageDataUrl: hfImageDataUrl,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Erreur backend Hugging Face");
    }

    renderResults(hfResultsBox, data.results || [], "huggingface");
    setStatus(hfStatusBox, `Terminé : ${data.count || 0} clé(s) HF testée(s).`);
  } catch (error) {
    setStatus(hfStatusBox, error.message || "Erreur pendant le test Hugging Face.", true);
  } finally {
    hfRunBtn.disabled = false;
    hfRunBtn.textContent = "Lancer le test Hugging Face";
  }
});

keysInput.addEventListener("input", renderCooldownInfo);
modelSelect.addEventListener("change", renderCooldownInfo);

 tabMainBtn.addEventListener("click", () => switchTab("main"));
 tabHfBtn.addEventListener("click", () => switchTab("hf"));

renderResults(resultsBox, []);
renderResults(hfResultsBox, [], "huggingface");
renderCooldownInfo();
switchTab("main");
