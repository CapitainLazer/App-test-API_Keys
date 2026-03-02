const keysInput = document.getElementById("keysInput");
const modelSelect = document.getElementById("modelSelect");
const timeoutInput = document.getElementById("timeoutInput");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const runBtn = document.getElementById("runBtn");
const statusBox = document.getElementById("status");
const resultsBox = document.getElementById("results");

let imageDataUrl = "";

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
            <div class="meta"><strong>${escapeHtml(item.keyLabel)}</strong> • ${escapeHtml(item.model)}</div>
            <div>${escapeHtml(item.answer)}</div>
          </article>
        `;
      }

      return `
        <article class="result-item error">
          <div class="meta"><strong>${escapeHtml(item.keyLabel)}</strong> • ${escapeHtml(item.model)}${item.statusCode ? ` • HTTP ${item.statusCode}` : ""}</div>
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

  if (!keys.length) {
    setStatus("Ajoute au moins une clé API.", true);
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
    setStatus(`Terminé : ${data.count || 0} clé(s) testée(s).`);
  } catch (error) {
    setStatus(error.message || "Erreur pendant le test.", true);
  } finally {
    runBtn.disabled = false;
  }
});

renderResults([]);