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
    const repoNameInput = document.getElementById('repo-name-input');
    const loadRepoBtn = document.getElementById('load-repo-btn');
    const topicInput = document.getElementById('topic-input');
    const searchTopicBtn = document.getElementById('search-topic-btn');
    const newChatBtn = document.getElementById('new-chat-btn');

    // UI State
    let isIngesting = false;
    let isWaitingForAI = false;
    let currentRepo = null;

    // --- Navigation Logic ---
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetView = item.getAttribute('data-view');
            
            // Update Nav UI
            navItems.forEach(ni => ni.classList.remove('active'));
            item.classList.add('active');

            // Update View
            views.forEach(v => v.classList.remove('active'));
            document.getElementById(targetView).classList.add('active');

            // Update Title
            const labels = {
                'chat-view': 'Chat Analysis',
                'add-repo-view': 'Load Repository',
                'search-topic-view': 'Topic Discovery'
            };
            document.getElementById('current-view-title').textContent = labels[targetView];
        });
    });

    // --- Chat Logic ---
    const appendMessage = (role, text) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        // Simple markdown parsing for code blocks and inline code
        // For a production app, use a library like marked.js
        const formattedText = text
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>');

        msgDiv.innerHTML = `<div class="msg-content">${formattedText}</div>`;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const handleSendMessage = async () => {
        const question = chatInput.value.trim();
        if (!question || isWaitingForAI) return;

        appendMessage('user', question);
        chatInput.value = '';
        chatInput.style.height = 'auto';
        
        isWaitingForAI = true;
        sendBtn.disabled = true;

        // Show "AI is thinking" message placeholder or state
        const aiMsgDiv = document.createElement('div');
        aiMsgDiv.className = 'message system thinking';
        aiMsgDiv.innerHTML = '<div class="msg-content">Thinking...</div>';
        chatMessages.appendChild(aiMsgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            const data = await response.json();
            
            // Remove thinking state and add real response
            chatMessages.removeChild(aiMsgDiv);
            if (response.ok) {
                appendMessage('system', data.answer);
            } else {
                appendMessage('system', `Error: ${data.detail || 'Something went wrong'}`);
            }
        } catch (error) {
            chatMessages.removeChild(aiMsgDiv);
            appendMessage('system', 'Error: Failed to connect to server.');
        } finally {
            isWaitingForAI = false;
            sendBtn.disabled = false;
        }
    };

    sendBtn.addEventListener('click', handleSendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
    });

    // --- Ingestion Logic ---
    const startIngestion = async (payload) => {
        isIngesting = true;
        loadingOverlay.style.display = 'flex';
        loadingText.textContent = payload.repo_name 
            ? `Ingesting ${payload.repo_name}...` 
            : `Searching and ingesting topic: ${payload.topic}...`;

        try {
            const response = await fetch('/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (response.ok) {
                currentRepo = data.repo_metadata;
                updateRepoUI();
                // Switch to chat view
                document.querySelector('[data-view="chat-view"]').click();
                appendMessage('system', `Successfully loaded **${currentRepo.name}**. I am ready to answer your questions about this repository.`);
            } else {
                alert(`Error: ${data.detail}`);
            }
        } catch (error) {
            alert('Failed to ingest repository.');
        } finally {
            isIngesting = false;
            loadingOverlay.style.display = 'none';
        }
    };

    const updateRepoUI = () => {
        if (!currentRepo) return;

        // Update active repo card in sidebar
        const card = document.getElementById('active-repo-card');
        card.className = 'repo-card-mini';
        card.innerHTML = `
            <span class="name">${currentRepo.name}</span>
            <span class="desc">${currentRepo.description || 'No description'}</span>
        `;

        // Update Top Bar
        document.getElementById('top-repo-stats').style.display = 'flex';
        document.getElementById('stat-stars').textContent = currentRepo.stars.toLocaleString();
        document.getElementById('stat-forks').textContent = currentRepo.forks.toLocaleString();
        document.getElementById('stat-lang').textContent = currentRepo.language || 'N/A';

        // Enable chat input
        chatInput.disabled = false;
        sendBtn.disabled = false;
    };

    loadRepoBtn.addEventListener('click', () => {
        const repoName = repoNameInput.value.trim();
        if (repoName) startIngestion({ repo_name: repoName });
    });

    searchTopicBtn.addEventListener('click', () => {
        const topic = topicInput.value.trim();
        if (topic) startIngestion({ topic: topic });
    });

    newChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = `
            <div class="message system">
                <div class="msg-content">Session reset. Ask me anything about the currently loaded repository.</div>
            </div>
        `;
    });

    // Initialize state
    chatInput.disabled = true;
    sendBtn.disabled = true;
});
