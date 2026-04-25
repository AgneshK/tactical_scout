const chatWindow  = document.getElementById('chat-window');
const userInput   = document.getElementById('user-input');
const sendBtn     = document.getElementById('send-btn');
const emptyState  = document.getElementById('empty-state');
const suggestions = document.getElementById('suggestions');

marked.use({ breaks: true, gfm: true });

let _msgSeq  = 0;
let started  = false;

const nextMsgId = () => 'msg-' + (++_msgSeq);

function hideWelcome() {
    if (started) return;
    started = true;
    chatWindow.classList.remove('is-empty');
    emptyState.remove();
    suggestions.classList.add('hidden');
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    hideWelcome();
    appendMessage(text, 'user-msg');
    userInput.value = '';

    const loadingId = appendTypingIndicator();

    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();
        document.getElementById(loadingId).remove();

        if (data.players && Array.isArray(data.players.clones)) {
            appendCards(data.players);
        }
        appendMessage(data.reply, 'bot-msg');

    } catch {
        document.getElementById(loadingId).remove();
        appendMessage('Unable to reach the backend. The server may be starting up — please try again in a moment.', 'bot-msg');
    }
}

function appendMessage(text, className) {
    const div = document.createElement('div');
    div.className = `message ${className}`;
    div.id = nextMsgId();

    if (className === 'bot-msg') {
        div.innerHTML = marked.parse(text);
    } else {
        div.textContent = text;
    }

    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div.id;
}

function appendTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message bot-msg typing-indicator';
    div.id = nextMsgId();
    div.innerHTML = '<span></span><span></span><span></span>';
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div.id;
}

/* ── Player cards ────────────────────────────────────────────────── */

function appendCards(payload) {
    if (!payload.clones || !payload.clones.length) return;

    const block = document.createElement('div');
    block.className = 'cards-block';
    block.id = nextMsgId();

    const top = payload.clones[0];
    block.appendChild(buildCard(top, { isQuery: false, isTopPick: true }));

    if (payload.query && (payload.query.percentiles || top.percentiles)) {
        block.insertAdjacentHTML('beforeend', buildRadar(payload.query, top));
    }

    chatWindow.appendChild(block);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
}

function initials(name) {
    return String(name || '?')
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 2)
        .map(w => w[0].toUpperCase())
        .join('');
}

function avatarHue(name) {
    let h = 0;
    for (const ch of String(name || '')) h = (h * 31 + ch.charCodeAt(0)) % 360;
    return h;
}

function confidenceClass(label) {
    return ({
        Exceptional: 'badge-exceptional',
        Strong:      'badge-strong',
        Good:        'badge-good',
        Moderate:    'badge-moderate',
    })[label] || 'badge-moderate';
}

const STAT_LABELS = {
    Goals: 'Goals', Assists: 'Asst', xG: 'xG', xAG: 'xAG',
    PrgC: 'PrgC',  PrgP: 'PrgP',   Tkl: 'Tkl',  KP: 'KP',
};

function buildCard(p, { isQuery, isTopPick }) {
    const card = document.createElement('div');
    card.className = 'player-card'
        + (isQuery ? ' is-query' : '')
        + (isTopPick ? ' is-top-pick' : '');

    const hue   = avatarHue(p.name);
    const ini   = initials(p.name);
    const flag  = p.nation?.flag_url
        ? `<img class="flag" src="${escapeHtml(p.nation.flag_url)}" alt="${escapeHtml(p.nation.label)}" loading="lazy">`
        : '';
    const nationLabel = p.nation?.label ? escapeHtml(p.nation.label) : '';

    const metaBits = [];
    if (flag || nationLabel) metaBits.push(`<span class="meta-nation">${flag}<span>${nationLabel}</span></span>`);
    if (p.club)              metaBits.push(`<span>${escapeHtml(p.club)}</span>`);
    if (p.league)            metaBits.push(`<span class="meta-league">${escapeHtml(p.league)}</span>`);

    const subBits = [];
    if (p.age != null)     subBits.push(`Age ${p.age}`);
    if (p.matches != null) subBits.push(`${p.matches} apps`);
    if (p.minutes != null) subBits.push(`${p.minutes.toLocaleString()} min`);

    const badge = isQuery
        ? `<div class="match-badge badge-query"><span class="badge-pct">REF</span><span class="badge-label">Query</span></div>`
        : (p.similarity != null
            ? `<div class="match-badge ${confidenceClass(p.confidence)}">
                 <span class="badge-pct">${(p.similarity * 100).toFixed(1)}%</span>
                 <span class="badge-label">${escapeHtml(p.confidence || '')}</span>
               </div>`
            : '');

    const stats = p.stats && Object.keys(p.stats).length
        ? `<div class="stat-grid">${
            Object.entries(p.stats).map(([k, v]) => `
                <div class="stat-cell">
                    <div class="stat-key">${escapeHtml(STAT_LABELS[k] || k)}</div>
                    <div class="stat-val">${Number(v).toFixed(2)}</div>
                </div>`).join('')
          }</div>`
        : '';

    const rankBadge = isTopPick
        ? `<span class="rank-pill rank-top">TOP MATCH</span>`
        : (!isQuery && p.rank ? `<span class="rank-pill">#${p.rank}</span>` : '');

    card.innerHTML = `
        <div class="card-top">
            <div class="avatar" style="background:hsl(${hue} 65% 92%);color:hsl(${hue} 55% 28%)">${escapeHtml(ini)}</div>
            <div class="card-id">
                <div class="card-name-row">
                    ${rankBadge}
                    <h4 class="card-name">${escapeHtml(p.name)}</h4>
                </div>
                <div class="card-pos">${escapeHtml(p.position || '')}</div>
                <div class="card-meta">${metaBits.join('<span class="meta-sep">·</span>')}</div>
                ${subBits.length ? `<div class="card-sub">${subBits.join(' · ')}</div>` : ''}
            </div>
            ${badge}
        </div>
        ${stats}
    `;
    return card;
}

/* ── Radar chart (custom SVG, percentile-ranked within position) ── */

const RADAR_STATS = ['Goals', 'Assists', 'xG', 'xAG', 'PrgC', 'PrgP', 'Tkl', 'KP'];

function buildRadar(query, clone) {
    const size      = 300;
    const center    = size / 2;
    const maxRadius = center - 46;
    const step      = (2 * Math.PI) / RADAR_STATS.length;
    const angle     = i => -Math.PI / 2 + i * step;

    const point = (i, val) => {
        const r = (Math.max(0, Math.min(100, val)) / 100) * maxRadius;
        return [
            center + r * Math.cos(angle(i)),
            center + r * Math.sin(angle(i)),
        ];
    };

    const gridRings = [25, 50, 75, 100].map(level => {
        const pts = RADAR_STATS.map((_, i) => point(i, level))
            .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
        return `<polygon points="${pts}" class="radar-ring" />`;
    }).join('');

    const axes = RADAR_STATS.map((stat, i) => {
        const [ax, ay] = point(i, 100);
        const [lx, ly] = point(i, 118);
        return `
          <line x1="${center}" y1="${center}" x2="${ax.toFixed(1)}" y2="${ay.toFixed(1)}" class="radar-axis" />
          <text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" class="radar-label"
                text-anchor="middle" dominant-baseline="middle">${escapeHtml(STAT_LABELS[stat] || stat)}</text>`;
    }).join('');

    const polyFor = (player, cls) => {
        const pcts = player?.percentiles || {};
        const pts  = RADAR_STATS.map((s, i) => point(i, pcts[s] || 0))
            .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
        return `<polygon points="${pts}" class="radar-poly ${cls}" />`;
    };

    return `
      <div class="radar-block">
        <div class="radar-header">
          <div class="radar-title">Tactical Profile · per-90 percentiles within position</div>
          <div class="radar-legend">
            <span class="legend-item"><span class="dot dot-query"></span>${escapeHtml(query.name)}</span>
            <span class="legend-item"><span class="dot dot-clone"></span>${escapeHtml(clone.name)}</span>
          </div>
        </div>
        <div class="radar-svg-wrap">
          <svg viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg" class="radar-svg" aria-label="Tactical comparison radar">
            ${gridRings}
            ${axes}
            ${polyFor(query, 'poly-query')}
            ${polyFor(clone, 'poly-clone')}
          </svg>
        </div>
      </div>
    `;
}

// Suggestion chips — populate input on click
document.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        userInput.value = chip.dataset.query;
        userInput.focus();
    });
});

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', e => {
    if (e.key === 'Enter') sendMessage();
});

// Theme toggle
document.getElementById('theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
});
