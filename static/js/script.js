document.addEventListener('DOMContentLoaded', function () {
    const startBtn = document.getElementById('startBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const languageSelect = document.getElementById('languageSelect');
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const emptyState = document.getElementById('emptyState');
    const translationCards = document.getElementById('translationCards');
    const originalText = document.getElementById('originalText');
    const translatedText = document.getElementById('translatedText');
    const sourceLanguageName = document.getElementById('sourceLanguageName');
    const targetLanguageName = document.getElementById('targetLanguageName');

    let isTranslating = false;
    let isPaused = false;
    let currentLanguage = targetLanguageName.textContent;

    function updateStatus(status, text) {
        statusIndicator.className = `status-indicator ${status}`;
        statusText.textContent = text;
    }

    function showTranslationCards() {
        emptyState.style.display = 'none';
        translationCards.style.display = 'flex';
    }

    function hideTranslationCards() {
        emptyState.style.display = 'block';
        translationCards.style.display = 'none';
    }

    function updateOriginalText(text) {
        originalText.innerHTML = text?.trim()
            ? `<div class="actual-text">${text}</div>`
            : '<div class="placeholder-text">Your speech will appear here...</div>';
    }

    function updateTranslatedText(text) {
        translatedText.innerHTML = text?.trim()
            ? `<div class="actual-text">${text}</div>`
            : '<div class="placeholder-text">Translation will appear here...</div>';
    }

    function addTypingIndicator(element) {
        element.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
    }

    startBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/start', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
            const data = await res.json();
            if (data.status === 'started') {
                isTranslating = true;
                isPaused = false;
                startBtn.style.display = 'none';
                pauseBtn.style.display = 'flex';
                pauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
                updateStatus('listening', 'Listening...');
                showTranslationCards();
                pollTranslationStatus();
            }
        } catch (err) {
            console.error('Start error:', err);
            updateStatus('error', 'Failed to start');
        }
    });

    pauseBtn.addEventListener('click', async () => {
        const endpoint = isPaused ? '/resume' : '/pause';
        const pauseStatus = document.getElementById('pauseStatus');

        try {
            const res = await fetch(endpoint, { method: 'POST' });
            const data = await res.json();
            if (data.status === 'paused') {
                isPaused = true;
                updateStatus('paused', 'Paused');
                pauseBtn.innerHTML = '<i class="fas fa-play"></i>';
                pauseStatus.style.display = 'inline'; // Show pause message
            } else if (data.status === 'resumed') {
                isPaused = false;
                updateStatus('listening', 'Listening...');
                pauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
                pauseStatus.style.display = 'none'; // Hide pause message
                pollTranslationStatus();
            }
        } catch (err) {
            console.error('Pause/Resume error:', err);
        }
    });

    languageSelect.addEventListener('change', async function () {
        const newLanguage = this.value;
        try {
            const res = await fetch('/change_language', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ language: newLanguage })
            });
            const data = await res.json();
            if (data.status === 'language_changed') {
                currentLanguage = newLanguage;
                targetLanguageName.textContent = data.language_name;
                updateStatus('ready', 'Language changed');
            } else {
                languageSelect.value = currentLanguage;
            }
        } catch (err) {
            console.error('Language change error:', err);
            languageSelect.value = currentLanguage;
        }
    });

    async function pollTranslationStatus() {
        if (!isTranslating || isPaused) return;

        try {
            const res = await fetch('/status');
            const data = await res.json();

            if (data.source_lang) sourceLanguageName.textContent = data.source_lang;
            if (data.transcription) {
                updateOriginalText(data.transcription);
                if (!data.translation || data.translation !== window.lastTranslation) {
                    addTypingIndicator(translatedText);
                    updateStatus('processing', 'Processing...');
                }
            }

            if (data.translation) {
                updateTranslatedText(data.translation);
                window.lastTranslation = data.translation;
                updateStatus('listening', 'Listening...');
            }

            if (data.running && isTranslating && !isPaused) {
                setTimeout(pollTranslationStatus, 500);
            }
        } catch (err) {
            console.error('Polling error:', err);
            if (isTranslating && !isPaused) {
                setTimeout(pollTranslationStatus, 2000);
            }
        }
    }
});
