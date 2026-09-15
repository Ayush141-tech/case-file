/* =========================================================================
   CASEFILE — Frontend logic: loads model info, handles form → /api/predict
   ========================================================================= */

document.addEventListener("DOMContentLoaded", () => {
  loadModelInfo();
  populateAreaSelects();
  document.getElementById("case-form").addEventListener("submit", onSubmit);
});

/* ------------------------------------------------------------------ */
/*  Model info & dashboard                                            */
/* ------------------------------------------------------------------ */

async function loadModelInfo() {
  try {
    const res = await fetch("/api/model-info");
    const info = await res.json();
    renderCards(info);
    renderTable(info);
    document.getElementById("repeat-info").textContent =
      `${info.repeats} repeats | seeds: [${info.seeds.join(", ")}] | ${info.n_cases} cases | ${info.n_features} features`;
  } catch (err) {
    console.error("Failed to load model info:", err);
  }
}

function fmtPct(val) {
  return (val * 100).toFixed(1) + "%";
}
function fmtPctStd(mean, std) {
  return (mean * 100).toFixed(1) + "% ± " + (std * 100).toFixed(1) + "%";
}

function renderCards(info) {
  const best = info.models.find(m => m.model === info.best_model);
  if (!best) return;

  document.getElementById("card-best").textContent = best.model;
  document.getElementById("card-best-sub").textContent = `Top-1 mean: ${fmtPct(best.top1_mean)}`;
  document.getElementById("card-top1").textContent = fmtPctStd(best.top1_mean, best.top1_std);
  document.getElementById("card-top3").textContent = fmtPctStd(best.top3_mean, best.top3_std);
  document.getElementById("card-top5").textContent = fmtPctStd(best.top5_mean, best.top5_std);
}

function renderTable(info) {
  const tbody = document.getElementById("model-tbody");
  // Sort by top1_mean descending
  const sorted = [...info.models].sort((a, b) => b.top1_mean - a.top1_mean);
  const maxTop1 = sorted[0]?.top1_mean || 1;

  tbody.innerHTML = sorted.map(m => {
    const isBest = m.model === info.best_model;
    const barW = (m.top1_mean / maxTop1 * 100).toFixed(0);
    return `<tr class="${isBest ? "best-row" : ""}">
      <td>${m.model}</td>
      <td><span class="cell-bar" style="width:${barW}%; background:${isBest ? "#22c55e" : "#06b6d4"}"></span>${fmtPct(m.top1_mean)} ± ${(m.top1_std * 100).toFixed(1)}%</td>
      <td>${fmtPct(m.top3_mean)}</td>
      <td>${fmtPct(m.top5_mean)}</td>
      <td>${fmtPct(m.macro_f1_mean)}</td>
      <td>${fmtPct(m.weighted_f1_mean)}</td>
    </tr>`;
  }).join("");
}

/* ------------------------------------------------------------------ */
/*  Area selects                                                      */
/* ------------------------------------------------------------------ */

function populateAreaSelects() {
  const areas = [];
  for (let i = 0; i < 10; i++) {
    areas.push("Area_" + String.fromCharCode(65 + i)); // Area_A .. Area_J
  }
  const usual = document.getElementById("select-usual");
  const prev = document.getElementById("select-prev");
  areas.forEach((a, idx) => {
    usual.innerHTML += `<option value="${a}" ${idx === 4 ? "selected" : ""}>${a}</option>`;
    prev.innerHTML += `<option value="${a}" ${idx === 4 ? "selected" : ""}>${a}</option>`;
  });
}

/* ------------------------------------------------------------------ */
/*  Form submit → /api/predict                                        */
/* ------------------------------------------------------------------ */

async function onSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const btn = document.getElementById("btn-predict");
  const emptyEl = document.getElementById("results-empty");
  const contentEl = document.getElementById("results-content");
  const errorEl = document.getElementById("results-error");

  // Collect form values
  const data = {};
  const fields = form.querySelectorAll("input, select");
  fields.forEach(f => { data[f.name] = f.value; });

  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Predicting...`;
  emptyEl.hidden = true;
  contentEl.hidden = true;
  errorEl.hidden = true;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const json = await res.json();

    if (!res.ok) {
      throw new Error(json.error || "Prediction failed");
    }

    renderResults(json);
  } catch (err) {
    document.getElementById("error-text").textContent = err.message;
    errorEl.hidden = false;
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
      Generate Prediction Report`;
  }
}

function renderResults(json) {
  const contentEl = document.getElementById("results-content");

  // Case info
  const ci = json.case;
  document.getElementById("case-info").innerHTML =
    `<strong>${ci.Case_ID || "N/A"}</strong> | ${ci.Age_Group} ${ci.Gender} | ` +
    `${ci.Day} ${ci.Weather} | Usual: ${ci.Usual_Area} → Previous: ${ci.Previous_Area}`;

  // Predictions
  const list = document.getElementById("predictions-list");
  list.innerHTML = json.predictions.map(p => {
    return `
      <div class="pred-item" style="border-left-color: ${p.color}">
        <div class="pred-rank" style="color: ${p.color}">#${p.rank}</div>
        <div class="pred-details">
          <span class="pred-area">${p.area}</span>
          <div class="pred-bar-wrap">
            <div class="pred-bar" style="width: ${p.probability}%; background: ${p.color}"></div>
          </div>
        </div>
        <div class="pred-stats">
          <span class="pred-pct">${p.probability}%</span>
          <span class="pred-badge" style="background: ${p.color}22; color: ${p.color}; border: 1px solid ${p.color}44">${p.priority}</span>
        </div>
      </div>`;
  }).join("");

  // Ethical notice
  document.getElementById("ethical-notice").innerHTML =
    `<strong>Ethical Notice:</strong> ${json.ethical_notice}`;

  contentEl.hidden = false;
}