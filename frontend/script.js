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
        const response = await fetch('https://tactical-scout.onrender.com/chat', {
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
    const block = document.createElement('div');
    block.className = 'cards-block';
    block.id = nextMsgId();

    if (payload.query) {
        block.appendChild(buildCard(payload.query, { isQuery: true }));
    }
    payload.clones.forEach(c => {
        block.appendChild(buildCard(c, { isQuery: false }));
    });

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

function buildCard(p, { isQuery }) {
    const card = document.createElement('div');
    card.className = 'player-card' + (isQuery ? ' is-query' : '');

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

    const rankBadge = (!isQuery && p.rank)
        ? `<span class="rank-pill">#${p.rank}</span>` : '';

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
