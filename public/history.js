const HISTORY_STORAGE_KEY = "api_test_history_cache_v1";

const historyList = document.getElementById("historyList");
const historyStatus = document.getElementById("historyStatus");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function readHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeHistory(items) {
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(items));
}

function formatDate(isoString) {
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return "Date inconnue";
  return date.toLocaleString("fr-FR");
}

function setStatus(message, isError = false) {
  historyStatus.textContent = message;
  historyStatus.classList.toggle("error", isError);
  historyStatus.classList.toggle("ok", !isError);
}

function renderHistory() {
  const items = readHistory();

  if (!items.length) {
    historyList.innerHTML = '<p class="hint">Aucune réponse en cache pour le moment.</p>';
    setStatus("Historique vide.");
    return;
  }

  historyList.innerHTML = items
    .map((item) => {
      const body = item.status === "success" ? item.answer || "" : item.error || "";
      return `
        <article class="history-item ${item.status === "success" ? "success" : "error"}">
          <div class="meta">
            <strong>${escapeHtml(item.keyLabel || "clé")}</strong> • ${escapeHtml(item.provider || "n/a")} • ${escapeHtml(item.model || "n/a")} • ${escapeHtml(formatDate(item.createdAt))}
          </div>
          <div>${escapeHtml(body)}</div>
        </article>
      `;
    })
    .join("");

  setStatus(`${items.length} entrée(s) dans le cache.`);
}

clearHistoryBtn.addEventListener("click", () => {
  writeHistory([]);
  renderHistory();
});

renderHistory();
