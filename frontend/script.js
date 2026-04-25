const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Single newlines in agent output become <br> (GitHub-flavoured markdown style)
marked.use({ breaks: true, gfm: true });

// Monotonic counter — avoids duplicate IDs when two elements are created in the same ms
let _msgSeq = 0;
const nextMsgId = () => 'msg-' + (++_msgSeq);

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

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
        appendMessage(data.reply, 'bot-msg');

    } catch (error) {
        document.getElementById(loadingId).remove();
        appendMessage('Error connecting to backend.', 'bot-msg');
    }
}

function appendMessage(text, className) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${className}`;
    if (className === 'bot-msg') {
        msgDiv.innerHTML = marked.parse(text);
    } else {
        msgDiv.textContent = text;
    }
    const id = nextMsgId();
    msgDiv.id = id;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return id;
}

function appendTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message bot-msg typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    const id = nextMsgId();
    indicator.id = id;
    chatWindow.appendChild(indicator);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return id;
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// ── Theme toggle ──────────────────────────────────
const themeToggle = document.getElementById('theme-toggle');

themeToggle.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const next = isDark ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
});
