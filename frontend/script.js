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
