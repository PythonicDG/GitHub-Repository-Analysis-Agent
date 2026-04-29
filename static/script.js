document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    
    // Form Inputs
    const repoUrlInput = document.getElementById('repo-url-input');
    const analyzeRepoBtn = document.getElementById('analyze-repo-btn');
    const resetSessionBtn = document.getElementById('reset-session-btn');

    // UI State
    let isWaitingForAI = false;
    let currentRepo = null;

    // Navigation Logic
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetView = item.getAttribute('data-view');
            
            navItems.forEach(ni => ni.classList.remove('active'));
            item.classList.add('active');

            views.forEach(v => v.classList.remove('active'));
            document.getElementById(targetView).classList.add('active');

            const labels = {
                'chat-view': 'Chat Analysis',
                'add-repo-view': 'Analyze Repository'
            };
            document.getElementById('current-view-title').textContent = labels[targetView];
        });
    });

    // Chat UI Helpers
    const appendMessage = (role, text) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        // Improved Markdown formatting
        let formattedText = text
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/^\s*-\s+(.+)$/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
            .replace(/\n/g, '<br>');

        msgDiv.innerHTML = `<div class="msg-content">${formattedText}</div>`;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    };

    // Backend API Calls
    const analyzeRepository = async () => {
        const repoUrl = repoUrlInput.value.trim();
        if (!repoUrl) return;

        loadingOverlay.style.display = 'flex';
        loadingText.textContent = `Analyzing ${repoUrl}...`;

        try {
            const response = await fetch('/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ repo_url: repoUrl }),
                credentials: 'include'
            });

            const data = await response.json();
            if (response.ok) {
                currentRepo = {
                    name: data.repo_name,
                    ...data.repo_metadata
                };
                updateRepoUI();
                // Switch to chat view
                document.querySelector('[data-view="chat-view"]').click();
                appendMessage('system', `Repository **${currentRepo.name}** analyzed successfully! I've mapped the structure and key files. Ask me anything.`);
            } else {
                alert(`Error: ${data.detail}`);
            }
        } catch (error) {
            alert('Failed to connect to the server.');
        } finally {
            loadingOverlay.style.display = 'none';
        }
    };

    const handleSendMessage = async () => {
        const question = chatInput.value.trim();
        if (!question || isWaitingForAI) return;

        appendMessage('user', question);
        chatInput.value = '';
        chatInput.style.height = 'auto';
        
        isWaitingForAI = true;
        sendBtn.disabled = true;

        // Thinking state
        const aiMsgDiv = document.createElement('div');
        aiMsgDiv.className = 'message system thinking';
        aiMsgDiv.innerHTML = '<div class="msg-content">Processing...</div>';
        chatMessages.appendChild(aiMsgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question }),
                credentials: 'include'
            });

            const data = await response.json();
            chatMessages.removeChild(aiMsgDiv);

            if (response.ok) {
                appendMessage('system', data.answer);
            } else {
                appendMessage('system', `Error: ${data.detail}`);
            }
        } catch (error) {
            chatMessages.removeChild(aiMsgDiv);
            appendMessage('system', 'Error: Connection failed.');
        } finally {
            isWaitingForAI = false;
            sendBtn.disabled = false;
        }
    };

    const updateRepoUI = () => {
        if (!currentRepo) return;

        const card = document.getElementById('active-repo-card');
        card.className = 'repo-card-mini';
        card.innerHTML = `
            <span class="name">${currentRepo.name}</span>
            <span class="desc">${currentRepo.description || 'No description'}</span>
        `;

        document.getElementById('top-repo-stats').style.display = 'flex';
        document.getElementById('stat-stars').textContent = currentRepo.stars.toLocaleString();
        document.getElementById('stat-forks').textContent = currentRepo.forks.toLocaleString();
        document.getElementById('stat-lang').textContent = currentRepo.language || 'N/A';

        chatInput.disabled = false;
        sendBtn.disabled = false;
    };

    // Event Listeners
    analyzeRepoBtn.addEventListener('click', analyzeRepository);
    repoUrlInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') analyzeRepository();
    });

    sendBtn.addEventListener('click', handleSendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
    });

    resetSessionBtn.addEventListener('click', () => {
        window.location.reload();
    });
});
