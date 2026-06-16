// Talk to a local backend when developing (file:// or localhost), otherwise the
// deployed instance. Override by setting window.BACKEND_URL before this script loads.
const BACKEND_URL =
  window.BACKEND_URL ||
  (location.protocol === "file:" ||
  location.hostname === "localhost" ||
  location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://tactical-scout.onrender.com");

const chatWindow = document.getElementById("chat-window");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const emptyState = document.getElementById("empty-state");
const suggestions = document.getElementById("suggestions");

// Initialize Marked with basic options
marked.use({ breaks: true, gfm: true });

let _msgSeq = 0;
let started = false;

const nextMsgId = () => "msg-" + ++_msgSeq;

/* ── Backend status & cold-start handling ───────────────────────────
   This demo is hosted on a free tier that spins the server down when
   idle. The first request after a sleep can take ~30–60s while the
   instance boots, so we probe /health on load, surface a clear "waking
   up" notice, and reflect the real connection state in the header. */

let backendReady = false;

const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");
const coldBanner = document.getElementById("cold-banner");
const coldTitle = document.getElementById("cold-title");
const coldDesc = document.getElementById("cold-desc");
const coldDismiss = document.getElementById("cold-dismiss");

const STATUS_TEXT = {
  connecting: "Connecting",
  live: "Live",
  waking: "Waking up",
  offline: "Offline",
};

function setStatus(state) {
  if (!statusDot) return;
  statusDot.className = "status-dot is-" + state;
  statusLabel.textContent = STATUS_TEXT[state] || state;
}

let bannerDismissed = false;

function showColdBanner({ offline = false } = {}) {
  if (bannerDismissed || !coldBanner) return;
  coldBanner.classList.remove("is-ready");
  coldBanner.classList.toggle("is-offline", offline);
  if (offline) {
    coldTitle.textContent = "Server unreachable";
    coldDesc.textContent =
      "The backend isn't responding yet — it may still be starting. You can send a message to retry; the first one can take up to a minute.";
  } else {
    coldTitle.textContent = "Waking the scout…";
    coldDesc.textContent =
      "This demo runs on a free tier that spins down when idle, so the first request can take up to a minute. Subsequent answers are fast.";
  }
  coldBanner.hidden = false;
}

function markBannerReady() {
  if (!coldBanner || coldBanner.hidden || bannerDismissed) return;
  coldBanner.classList.remove("is-offline");
  coldBanner.classList.add("is-ready");
  coldTitle.textContent = "Scout is ready";
  coldDesc.textContent = "Server is warm — ask about any outfield player.";
  setTimeout(() => {
    coldBanner.hidden = true;
  }, 2600);
}

function hideColdBanner() {
  if (coldBanner) coldBanner.hidden = true;
}

coldDismiss?.addEventListener("click", () => {
  bannerDismissed = true;
  hideColdBanner();
});

async function probeBackend() {
  setStatus("connecting");
  // If /health hasn't answered shortly, the instance is almost certainly cold —
  // surface the waking notice while the boot completes in the background.
  const coldTimer = setTimeout(() => {
    if (!backendReady) {
      setStatus("waking");
      showColdBanner();
    }
  }, 2500);

  try {
    const res = await fetch(BACKEND_URL + "/health", { method: "GET" });
    clearTimeout(coldTimer);
    if (res.ok) {
      backendReady = true;
      setStatus("live");
      markBannerReady();
      loadExplorer();
    } else {
      setStatus("offline");
      showColdBanner({ offline: true });
    }
  } catch (err) {
    clearTimeout(coldTimer);
    console.warn("Health probe failed:", err);
    setStatus("offline");
    showColdBanner({ offline: true });
  }
}

function hideWelcome() {
  if (started) return;
  started = true;
  chatWindow.classList.remove("is-empty");
  if (emptyState) emptyState.remove();
  if (suggestions) suggestions.classList.add("hidden");
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  hideWelcome();
  hideAutocomplete();
  appendMessage(text, "user-msg");
  userInput.value = "";

  // When the backend hasn't confirmed readiness, the first call may hit a cold
  // instance — tell the user the wait is expected rather than leaving a silent spinner.
  const cold = !backendReady;
  const loadingId = appendTypingIndicator(
    cold ? "Waking the server… the first request can take up to a minute." : "",
  );
  if (cold) {
    setStatus("waking");
    showColdBanner();
  }

  try {
    const response = await fetch(BACKEND_URL + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    // If the server returns a 500 or other error, jump to catch
    if (!response.ok) {
      throw new Error(`Server Error: ${response.status}`);
    }

    const data = await response.json();
    document.getElementById(loadingId)?.remove();

    // A successful reply proves the instance is warm.
    backendReady = true;
    setStatus("live");
    markBannerReady();
    loadExplorer();

    // Render player cards if they exist in the response
    if (data.players && Array.isArray(data.players.clones)) {
      appendCards(data.players);
    }

    // Safeguard: Ensure data.reply is a string before passing to Marked
    const botReply = data.reply || "I received a response, but it was empty.";
    appendMessage(botReply, "bot-msg");
  } catch (err) {
    console.error("sendMessage failed:", err);
    document.getElementById(loadingId)?.remove();
    setStatus("offline");
    showColdBanner({ offline: true });

    // Provide a hardcoded string so marked.parse doesn't receive 'undefined'
    const errorMsg =
      "**Unable to reach the backend.** The free-tier server may still be starting up — wait a few seconds and try again. If it persists, the instance may be down.";
    appendMessage(errorMsg, "bot-msg");
  }
}

function appendMessage(text, className) {
  const div = document.createElement("div");
  div.className = `message ${className}`;
  div.id = nextMsgId();

  if (className === "bot-msg") {
    // We use || '' to strictly ensure marked never receives null/undefined
    div.innerHTML = marked.parse(text || "");
  } else {
    div.textContent = text;
  }

  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div.id;
}

function appendTypingIndicator(note = "") {
  const div = document.createElement("div");
  div.className = "message bot-msg typing-indicator" + (note ? " has-note" : "");
  div.id = nextMsgId();
  div.innerHTML =
    `<span class="dots"><span></span><span></span><span></span></span>` +
    (note ? `<span class="typing-note">${escapeHtml(note)}</span>` : "");
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div.id;
}

/* ── Player cards ────────────────────────────────────────────────── */

function appendCards(payload) {
  if (!payload.clones || !payload.clones.length) return;

  const block = document.createElement("div");
  block.className = "cards-block";
  block.id = nextMsgId();

  const top = payload.clones[0];
  const query = payload.query;

  // 1. Engine telemetry — players scanned, candidates, latency, role/league.
  block.insertAdjacentHTML("beforeend", buildTelemetry(payload));

  // 2. Reference player (the queried player) as its own card.
  if (query) {
    block.insertAdjacentHTML("beforeend", sectionLabel("Reference player"));
    block.appendChild(buildCard(query, { isQuery: true }));
  }

  // 3. The headline pick.
  block.insertAdjacentHTML("beforeend", sectionLabel("Closest tactical clone"));
  block.appendChild(buildCard(top, { isQuery: false, isTopPick: true }));

  // 4. Per-90 head-to-head (raw values) — a different lens than the radar.
  if (query && query.stats && top.stats) {
    block.insertAdjacentHTML("beforeend", buildHeadToHead(query, top));
  }

  // 5. Percentile radar (shape comparison).
  if (query && (query.percentiles || top.percentiles)) {
    block.insertAdjacentHTML("beforeend", buildRadar(query, top));
  }

  // 6. Runners-up (#2–#5) as compact, scannable rows.
  const rest = payload.clones.slice(1);
  if (rest.length) {
    const runners = document.createElement("div");
    runners.className = "runners-block";
    runners.innerHTML =
      `<div class="runners-title">Other strong matches</div>` +
      rest.map(buildRunnerRow).join("");
    block.appendChild(runners);
  }

  chatWindow.appendChild(block);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function sectionLabel(text) {
  return `<div class="section-label">${escapeHtml(text)}</div>`;
}

/* ── Engine telemetry strip ── */

function buildTelemetry(payload) {
  const items = [];
  if (payload.search_time_ms != null)
    items.push({ val: `${payload.search_time_ms}`, unit: "ms", key: "Search", lead: true });
  if (payload.db_size != null)
    items.push({ val: Number(payload.db_size).toLocaleString(), key: "Players scanned" });
  if (payload.candidate_pool != null)
    items.push({ val: `${payload.candidate_pool}`, key: "Position-matched" });
  if (payload.primary_pos)
    items.push({ val: escapeHtml(payload.primary_pos), key: "Role" });
  if (payload.league_filter)
    items.push({ val: escapeHtml(payload.league_filter), key: "League" });

  if (!items.length) return "";

  const cells = items
    .map(
      (it) => `
      <div class="telem-item">
        <span class="telem-val">${it.lead ? '<span class="telem-spark">⚡</span>' : ""}${it.val}${it.unit ? `<span class="telem-unit">${it.unit}</span>` : ""}</span>
        <span class="telem-key">${it.key}</span>
      </div>`,
    )
    .join('<span class="telem-sep"></span>');

  return `<div class="telemetry"><span class="telem-eyebrow">Engine</span><div class="telem-grid">${cells}</div></div>`;
}

/* ── Per-90 head-to-head ── */

function buildHeadToHead(q, c) {
  const stats = RADAR_STATS.filter(
    (s) => (q.stats && s in q.stats) || (c.stats && s in c.stats),
  );
  if (!stats.length) return "";

  const rows = stats
    .map((s) => {
      const qv = Number(q.stats?.[s] ?? 0);
      const cv = Number(c.stats?.[s] ?? 0);
      const max = Math.max(qv, cv, 0.0001);
      const qw = (qv / max) * 100;
      const cw = (cv / max) * 100;
      const winCls = cv > qv ? "c-win" : cv < qv ? "q-win" : "tie";
      return `
        <div class="h2h-row">
          <span class="h2h-num h2h-num-q">${qv.toFixed(2)}</span>
          <span class="h2h-track h2h-track-q"><i style="width:${qw.toFixed(1)}%"></i></span>
          <span class="h2h-label">${escapeHtml(STAT_LABELS[s] || s)}</span>
          <span class="h2h-track h2h-track-c"><i style="width:${cw.toFixed(1)}%"></i></span>
          <span class="h2h-num h2h-num-c ${winCls}">${cv.toFixed(2)}</span>
        </div>`;
    })
    .join("");

  return `
    <div class="h2h-block">
      <div class="h2h-header">
        <div class="runners-title">Per-90 head-to-head</div>
        <div class="radar-legend">
          <span class="legend-item"><span class="dot dot-query"></span>${escapeHtml(q.name)}</span>
          <span class="legend-item"><span class="dot dot-clone"></span>${escapeHtml(c.name)}</span>
        </div>
      </div>
      <div class="h2h-rows">${rows}</div>
    </div>`;
}

function buildRunnerRow(p) {
  const hue = avatarHue(p.name);
  const ini = initials(p.name);
  const flag = p.nation?.flag_url
    ? `<img class="flag" src="${escapeHtml(p.nation.flag_url)}" alt="${escapeHtml(p.nation.label)}" loading="lazy">`
    : "";
  const metaBits = [];
  if (p.club) metaBits.push(escapeHtml(p.club));
  if (p.league) metaBits.push(escapeHtml(p.league));
  if (p.age != null) metaBits.push(`Age ${p.age}`);

  const pct = p.similarity != null ? `${(p.similarity * 100).toFixed(1)}%` : "";
  const pctBadge = pct
    ? `<span class="runner-pct ${confidenceClass(p.confidence)}">${pct}</span>`
    : "";

  return `
    <div class="runner-row">
      <span class="runner-rank">#${p.rank ?? ""}</span>
      <span class="avatar avatar-sm" style="background:hsl(${hue} 65% 92%);color:hsl(${hue} 55% 28%)">${escapeHtml(ini)}</span>
      <span class="runner-id">
        <span class="runner-name">${flag}${escapeHtml(p.name)}</span>
        <span class="runner-meta">${escapeHtml(p.position || "")}${metaBits.length ? " · " + metaBits.join(" · ") : ""}</span>
      </span>
      ${pctBadge}
    </div>`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(
    /[&<>"']/g,
    (ch) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[ch],
  );
}

function initials(name) {
  return String(name || "?")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0].toUpperCase())
    .join("");
}

function avatarHue(name) {
  let h = 0;
  for (const ch of String(name || "")) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return h;
}

function confidenceClass(label) {
  return (
    {
      Exceptional: "badge-exceptional",
      Strong: "badge-strong",
      Good: "badge-good",
      Moderate: "badge-moderate",
    }[label] || "badge-moderate"
  );
}

const STAT_LABELS = {
  Goals: "Goals",
  Assists: "Asst",
  xG: "xG",
  xAG: "xAG",
  PrgC: "PrgC",
  PrgP: "PrgP",
  Tkl: "Tkl",
  KP: "KP",
};

function buildCard(p, { isQuery, isTopPick }) {
  const card = document.createElement("div");
  card.className =
    "player-card" +
    (isQuery ? " is-query" : "") +
    (isTopPick ? " is-top-pick" : "");

  const hue = avatarHue(p.name);
  const ini = initials(p.name);
  const flag = p.nation?.flag_url
    ? `<img class="flag" src="${escapeHtml(p.nation.flag_url)}" alt="${escapeHtml(p.nation.label)}" loading="lazy">`
    : "";
  const nationLabel = p.nation?.label ? escapeHtml(p.nation.label) : "";

  const metaBits = [];
  if (flag || nationLabel)
    metaBits.push(
      `<span class="meta-nation">${flag}<span>${nationLabel}</span></span>`,
    );
  if (p.club) metaBits.push(`<span>${escapeHtml(p.club)}</span>`);
  if (p.league)
    metaBits.push(`<span class="meta-league">${escapeHtml(p.league)}</span>`);

  const subBits = [];
  if (p.age != null) subBits.push(`Age ${p.age}`);
  if (p.matches != null) subBits.push(`${p.matches} apps`);
  if (p.minutes != null) subBits.push(`${p.minutes.toLocaleString()} min`);

  const badge = isQuery
    ? `<div class="match-badge badge-query"><span class="badge-pct">REF</span><span class="badge-label">Query</span></div>`
    : p.similarity != null
      ? `<div class="match-badge ${confidenceClass(p.confidence)}">
                 <span class="badge-pct">${(p.similarity * 100).toFixed(1)}%</span>
                 <span class="badge-label">${escapeHtml(p.confidence || "")}</span>
               </div>`
      : "";

  const pcts = p.percentiles || {};
  const stats =
    p.stats && Object.keys(p.stats).length
      ? `<div class="stat-grid">${Object.entries(p.stats)
          .map(([k, v]) => {
            const pc = pcts[k];
            const bar =
              pc != null
                ? `<div class="stat-bar" title="${Number(pc).toFixed(0)}th percentile in role"><span style="width:${Math.max(0, Math.min(100, pc))}%"></span></div>`
                : "";
            return `
                <div class="stat-cell">
                    <div class="stat-key">${escapeHtml(STAT_LABELS[k] || k)}</div>
                    <div class="stat-val">${Number(v).toFixed(2)}</div>
                    ${bar}
                </div>`;
          })
          .join("")}</div>`
      : "";

  const rankBadge = isTopPick
    ? `<span class="rank-pill rank-top">TOP MATCH</span>`
    : !isQuery && p.rank
      ? `<span class="rank-pill">#${p.rank}</span>`
      : "";

  card.innerHTML = `
        <div class="card-top">
            <div class="avatar" style="background:hsl(${hue} 65% 92%);color:hsl(${hue} 55% 28%)">${escapeHtml(ini)}</div>
            <div class="card-id">
                <div class="card-name-row">
                    ${rankBadge}
                    <h4 class="card-name">${escapeHtml(p.name)}</h4>
                </div>
                <div class="card-pos">${escapeHtml(p.position || "")}</div>
                <div class="card-meta">${metaBits.join('<span class="meta-sep">·</span>')}</div>
                ${subBits.length ? `<div class="card-sub">${subBits.join(" · ")}</div>` : ""}
            </div>
            ${badge}
        </div>
        ${stats}
    `;
  return card;
}

/* ── Radar chart ── */

const RADAR_STATS = [
  "Goals",
  "Assists",
  "xG",
  "xAG",
  "PrgC",
  "PrgP",
  "Tkl",
  "KP",
];

function buildRadar(query, clone) {
  const size = 300;
  const center = size / 2;
  const maxRadius = center - 46;
  const step = (2 * Math.PI) / RADAR_STATS.length;
  const angle = (i) => -Math.PI / 2 + i * step;

  const point = (i, val) => {
    const r = (Math.max(0, Math.min(100, val)) / 100) * maxRadius;
    return [center + r * Math.cos(angle(i)), center + r * Math.sin(angle(i))];
  };

  const gridRings = [25, 50, 75, 100]
    .map((level) => {
      const pts = RADAR_STATS.map((_, i) => point(i, level))
        .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
        .join(" ");
      return `<polygon points="${pts}" class="radar-ring" />`;
    })
    .join("");

  const axes = RADAR_STATS.map((stat, i) => {
    const [ax, ay] = point(i, 100);
    const [lx, ly] = point(i, 118);
    return `
          <line x1="${center}" y1="${center}" x2="${ax.toFixed(1)}" y2="${ay.toFixed(1)}" class="radar-axis" />
          <text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" class="radar-label"
                text-anchor="middle" dominant-baseline="middle">${escapeHtml(STAT_LABELS[stat] || stat)}</text>`;
  }).join("");

  const polyFor = (player, cls) => {
    const pcts = player?.percentiles || {};
    const pts = RADAR_STATS.map((s, i) => point(i, pcts[s] || 0))
      .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
      .join(" ");
    return `<polygon points="${pts}" class="radar-poly ${cls}" />`;
  };

  return `
      <div class="radar-block">
        <div class="radar-header">
          <div class="radar-title">Tactical Profile · per-90 percentiles</div>
          <div class="radar-legend">
            <span class="legend-item"><span class="dot dot-query"></span>${escapeHtml(query.name)}</span>
            <span class="legend-item"><span class="dot dot-clone"></span>${escapeHtml(clone.name)}</span>
          </div>
        </div>
        <div class="radar-svg-wrap">
          <svg viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg" class="radar-svg">
            ${gridRings}
            ${axes}
            ${polyFor(query, "poly-query")}
            ${polyFor(clone, "poly-clone")}
          </svg>
        </div>
      </div>
    `;
}

// Suggestion chips — prefill the input so the user can tweak before sending.
document.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    userInput.value = chip.dataset.query;
    userInput.focus();
  });
});

// Featured-player chips (empty state) — send immediately for a one-tap demo.
document.querySelectorAll(".featured-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    userInput.value = chip.dataset.query;
    sendMessage();
  });
});

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  // Defer to the autocomplete keydown handler when an item is highlighted.
  if (e.key === "Enter" && (acEl.hidden || acActive < 0)) sendMessage();
});

// Theme toggle
document.getElementById("theme-toggle").addEventListener("click", () => {
  const next =
    document.documentElement.getAttribute("data-theme") === "dark"
      ? "light"
      : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
});

/* ════════════════════════════════════════════════════════════════════
   DATASET EXPLORER  —  sidebar engine stats + live leaderboard
   ════════════════════════════════════════════════════════════════════ */

let explorerLoaded = false;
let explorerBusy = false;
let explorerTries = 0;
let allPlayers = []; // roster for autocomplete

const leaderState = { metric: "xG", pos: "" };

function scoutPlayer(name) {
  userInput.value = `Find players similar to ${name}`;
  closeSidebar();
  sendMessage();
}

async function loadExplorer() {
  if (explorerLoaded || explorerBusy) return;
  explorerBusy = true;
  try {
    const [statsRes, playersRes] = await Promise.all([
      fetch(BACKEND_URL + "/stats"),
      fetch(BACKEND_URL + "/players"),
    ]);
    if (!statsRes.ok) throw new Error("stats " + statsRes.status);
    renderEngineStats(await statsRes.json());
    if (playersRes.ok) allPlayers = (await playersRes.json()).players || [];
    await loadLeaders();
    explorerLoaded = true;
  } catch (err) {
    console.warn("Explorer load failed:", err);
    explorerTries++;
    // A cold instance can take ~50s to boot — keep retrying for a while, then degrade.
    if (explorerTries < 8) {
      setTimeout(loadExplorer, 5000);
    } else {
      showExplorerUnavailable();
    }
  } finally {
    explorerBusy = false;
  }
}

function showExplorerUnavailable() {
  const tilesEl = document.getElementById("engine-tiles");
  if (tilesEl) {
    tilesEl.innerHTML = `<div class="panel-note">Live engine data unavailable. Start the backend (<code>uvicorn main:app</code>) or redeploy it, then reload.</div>`;
  }
  const distEl = document.getElementById("dist-positions");
  if (distEl) distEl.innerHTML = "";
  const listEl = document.getElementById("leaders-list");
  if (listEl) listEl.innerHTML = `<li class="leaders-empty">Leaderboard needs the backend running.</li>`;
}

/* ── Engine stats panel ── */

function renderEngineStats(s) {
  const fmt = (n) => Number(n).toLocaleString();
  const tiles = [
    { val: fmt(s.players), key: "Players" },
    { val: fmt(s.clubs), key: "Clubs" },
    { val: (s.leagues || []).length, key: "Leagues" },
    { val: s.avg_age, key: "Avg age" },
  ];
  const tilesEl = document.getElementById("engine-tiles");
  if (tilesEl) {
    tilesEl.innerHTML = tiles
      .map(
        (t) =>
          `<div class="tile"><span class="tile-val">${t.val}</span><span class="tile-key">${escapeHtml(t.key)}</span></div>`,
      )
      .join("");
  }

  const pill = document.getElementById("engine-pill");
  if (pill && s.latent_dim) pill.textContent = `${s.latent_dim}-dim latent`;

  // League distribution mini-bars
  const distEl = document.getElementById("dist-positions");
  if (distEl && Array.isArray(s.leagues)) {
    const max = Math.max(...s.leagues.map((l) => l.count), 1);
    distEl.innerHTML =
      `<div class="dist-title">League coverage</div>` +
      s.leagues
        .map(
          (l) => `
        <div class="dist-row">
          <span class="dist-label">${escapeHtml(l.name)}</span>
          <span class="dist-track"><i style="width:${((l.count / max) * 100).toFixed(1)}%"></i></span>
          <span class="dist-num">${l.count}</span>
        </div>`,
        )
        .join("");
  }

  // Metric tabs from the dataset's available metrics
  const tabsEl = document.getElementById("metric-tabs");
  if (tabsEl && Array.isArray(s.metrics)) {
    tabsEl.innerHTML = s.metrics
      .map(
        (m) =>
          `<button class="metric-tab${m.key === leaderState.metric ? " is-active" : ""}" data-metric="${escapeHtml(m.key)}" title="${escapeHtml(m.label)}">${escapeHtml(m.key)}</button>`,
      )
      .join("");
    tabsEl.querySelectorAll(".metric-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        leaderState.metric = btn.dataset.metric;
        tabsEl.querySelectorAll(".metric-tab").forEach((b) => b.classList.toggle("is-active", b === btn));
        loadLeaders();
      });
    });
  }
}

/* ── Leaderboard panel ── */

async function loadLeaders() {
  const listEl = document.getElementById("leaders-list");
  if (!listEl) return;
  listEl.classList.add("is-loading");
  try {
    const url = `${BACKEND_URL}/leaders?metric=${encodeURIComponent(leaderState.metric)}&limit=8${leaderState.pos ? `&pos=${leaderState.pos}` : ""}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("leaders " + res.status);
    const data = await res.json();
    renderLeaders(data);
  } catch (err) {
    console.warn("Leaders load failed:", err);
    listEl.innerHTML = `<li class="leaders-empty">Couldn't load leaders.</li>`;
  } finally {
    listEl.classList.remove("is-loading");
  }
}

function renderLeaders(data) {
  const listEl = document.getElementById("leaders-list");
  if (!listEl) return;
  const rows = data.leaders || [];
  if (!rows.length) {
    listEl.innerHTML = `<li class="leaders-empty">No players for this filter.</li>`;
    return;
  }
  const max = Math.max(...rows.map((r) => Number(r.value) || 0), 1);
  listEl.innerHTML = rows
    .map((r) => {
      const flag = r.flag
        ? `<img class="flag" src="${escapeHtml(r.flag)}" alt="${escapeHtml(r.nation || "")}" loading="lazy">`
        : "";
      const w = ((Number(r.value) / max) * 100).toFixed(1);
      return `
      <li class="leader-row" data-name="${escapeHtml(r.name)}" tabindex="0" role="button">
        <span class="leader-rank">${r.rank}</span>
        <span class="leader-id">
          <span class="leader-name">${flag}${escapeHtml(r.name)}</span>
          <span class="leader-meta">${escapeHtml(r.pos || "")} · ${escapeHtml(r.club || "")}</span>
        </span>
        <span class="leader-val">
          <span class="leader-num">${escapeHtml(String(r.value))}</span>
          <span class="leader-bar"><i style="width:${w}%"></i></span>
        </span>
      </li>`;
    })
    .join("");

  listEl.querySelectorAll(".leader-row").forEach((row) => {
    const go = () => scoutPlayer(row.dataset.name);
    row.addEventListener("click", go);
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        go();
      }
    });
  });
}

// Position filter (All / FW / MF / DF)
document.querySelectorAll("#pos-filter .pos-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    leaderState.pos = btn.dataset.pos;
    document.querySelectorAll("#pos-filter .pos-btn").forEach((b) => b.classList.toggle("is-active", b === btn));
    loadLeaders();
  });
});

/* ════════════════════════════════════════════════════════════════════
   PLAYER AUTOCOMPLETE  (accent-insensitive, keyboard-navigable)
   ════════════════════════════════════════════════════════════════════ */

const acEl = document.getElementById("autocomplete");
let acMatches = [];
let acActive = -1;

// Mirror of the backend's accent folding so "odegaard" matches "Ødegaard".
const AC_SPECIAL = { ø: "o", ł: "l", đ: "d", ß: "ss", æ: "ae", œ: "oe", ð: "d", þ: "th" };
function fold(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[øłđßæœðþ]/g, (c) => AC_SPECIAL[c] || c)
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, ""); // strip combining diacritical marks
}

function renderAutocomplete() {
  if (!acEl) return;
  if (!acMatches.length) {
    acEl.hidden = true;
    acEl.innerHTML = "";
    return;
  }
  acEl.innerHTML = acMatches
    .map((p, i) => {
      const flag = p.flag
        ? `<img class="flag" src="${escapeHtml(p.flag)}" alt="" loading="lazy">`
        : "";
      return `
      <button class="ac-item${i === acActive ? " is-active" : ""}" data-idx="${i}">
        <span class="ac-name">${flag}${escapeHtml(p.name)}</span>
        <span class="ac-meta">${escapeHtml(p.pos || "")}${p.club ? " · " + escapeHtml(p.club) : ""}</span>
      </button>`;
    })
    .join("");
  acEl.hidden = false;

  acEl.querySelectorAll(".ac-item").forEach((item) => {
    item.addEventListener("mousedown", (e) => {
      e.preventDefault(); // keep input focus
      scoutPlayer(acMatches[Number(item.dataset.idx)].name);
      hideAutocomplete();
    });
  });
}

function hideAutocomplete() {
  acMatches = [];
  acActive = -1;
  renderAutocomplete();
}

function updateAutocomplete() {
  const q = fold(userInput.value.trim());
  if (q.length < 2 || !allPlayers.length) {
    hideAutocomplete();
    return;
  }
  // Prefer name-start matches, then any substring.
  const starts = [];
  const contains = [];
  for (const p of allPlayers) {
    const f = fold(p.name);
    if (f.startsWith(q) || f.split(" ").some((t) => t.startsWith(q))) starts.push(p);
    else if (f.includes(q)) contains.push(p);
    if (starts.length >= 6) break;
  }
  acMatches = [...starts, ...contains].slice(0, 6);
  acActive = -1;
  renderAutocomplete();
}

userInput.addEventListener("input", updateAutocomplete);
userInput.addEventListener("blur", () => setTimeout(hideAutocomplete, 120));
userInput.addEventListener("keydown", (e) => {
  if (acEl.hidden || !acMatches.length) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    acActive = (acActive + 1) % acMatches.length;
    renderAutocomplete();
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    acActive = (acActive - 1 + acMatches.length) % acMatches.length;
    renderAutocomplete();
  } else if (e.key === "Enter" && acActive >= 0) {
    e.preventDefault();
    scoutPlayer(acMatches[acActive].name);
    hideAutocomplete();
  } else if (e.key === "Escape") {
    hideAutocomplete();
  }
});

/* ── Sidebar drawer (mobile) ── */

const appEl = document.getElementById("app");
function openSidebar() { appEl.classList.add("sidebar-open"); }
function closeSidebar() { appEl.classList.remove("sidebar-open"); }

document.getElementById("menu-btn")?.addEventListener("click", openSidebar);
document.getElementById("sidebar-close")?.addEventListener("click", closeSidebar);
document.getElementById("sidebar-backdrop")?.addEventListener("click", closeSidebar);

// Probe the backend on load so the cold-start notice appears before the
// user's first message rather than after a mysterious 60s wait.
probeBackend();

// Start loading the sidebar engine data immediately — it self-retries while a
// cold instance boots, and degrades to a clear message if it never answers.
loadExplorer();
