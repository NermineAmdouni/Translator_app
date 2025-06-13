document.addEventListener('DOMContentLoaded', function () {
    if (window.translatorScriptLoaded) {
        console.log('Translator script already loaded, skipping...');
        return;
    }
    window.translatorScriptLoaded = true;
    // 🚨 ONLY RUN ON TRANSLATOR PAGE
    const startBtn = document.getElementById('startBtn');
    if (!startBtn) {
        console.log('Not on translator page, skipping translator JS');
        return; // Exit if not on translator page
    }
    
    // 🔍 DEBUGGING SNIPPET TO DETECT INJECTED TRACKERS
    setInterval(() => {
        if (window.hybridaction?.zybTrackerStatisticsAction) {
            console.warn("⚠️ Detected zybTrackerStatisticsAction is injected!");
            console.log(window.hybridaction.zybTrackerStatisticsAction.toString());
        }
    }, 1000);

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
    
    // Initialize currentLanguage from the DOM on page load
    let currentLanguage = languageSelect.value;
    console.log('🚀 TRANSLATOR Initial language from DOM:', currentLanguage);
    console.log('🚀 TRANSLATOR Initial display name:', targetLanguageName.textContent);

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

    // Prevent multiple event listeners and add request deduplication
    let isChangingLanguage = false;
    let lastLanguageChangeTime = 0;
    
    // Remove any existing event listeners first
    languageSelect.removeEventListener('change', handleLanguageChange);
    
    async function handleLanguageChange() {
        const now = Date.now();
        const newLanguage = languageSelect.value;
        
        console.log('🔄 TRANSLATOR Language selection changed to:', newLanguage);
        console.log('🔄 TRANSLATOR Current language was:', currentLanguage);
        console.log('🔄 TRANSLATOR isChangingLanguage:', isChangingLanguage);
        console.log('🔄 TRANSLATOR Time since last change:', now - lastLanguageChangeTime, 'ms');

        // Prevent duplicate requests within 500ms or if already changing
        if (isChangingLanguage || (now - lastLanguageChangeTime < 500)) {
            console.log('⚠️ Ignoring duplicate language change request');
            return;
        }

        if (!newLanguage) {
            console.error('❌ No language selected or invalid value:', newLanguage);
            updateStatus('error', 'Invalid language selected');
            return;
        }

        // Don't send request if it's the same language
        if (newLanguage === currentLanguage) {
            console.log('⚠️ Same language selected, skipping request');
            return;
        }

        // Set flags to prevent duplicate requests
        isChangingLanguage = true;
        lastLanguageChangeTime = now;

        // Store the previous language in case we need to revert
        const previousLanguage = currentLanguage;
        const previousLanguageName = targetLanguageName.textContent;

        try {
            updateStatus('processing', 'Changing language...');
            
            const res = await fetch('/change_language', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ language: newLanguage })
            });

            console.log('📡 TRANSLATOR Response status:', res.status);

            if (!res.ok) {
                const errorText = await res.text();
                console.error(`❌ TRANSLATOR Language change failed: ${res.status}`, errorText);
                throw new Error(`HTTP error! status: ${res.status}`);
            }

            const data = await res.json();
            console.log('✅ TRANSLATOR Change language response:', data);

            if (data.status === 'language_changed') {
                // Update both the current language code and display name
                currentLanguage = newLanguage;
                targetLanguageName.textContent = data.language_name;
                updateStatus('ready', 'Language changed successfully');
                console.log('✅ Language successfully changed to:', newLanguage, '(' + data.language_name + ')');
            } else if (data.status === 'language_not_changed') {
                console.warn('⚠️ Backend says language not changed - reverting selection');
                languageSelect.value = previousLanguage;
                updateStatus('error', 'Language not changed');
            } else {
                console.warn('⚠️ Unexpected backend response:', data);
                languageSelect.value = previousLanguage;
                updateStatus('error', 'Unexpected response');
            }
        } catch (err) {
            console.error('❌ TRANSLATOR Language change error:', err);
            // Revert the selection on error
            languageSelect.value = previousLanguage;
            currentLanguage = previousLanguage;
            targetLanguageName.textContent = previousLanguageName;
            updateStatus('error', 'Failed to change language');
        } finally {
            // Reset the flag after a delay to allow for the next change
            setTimeout(() => {
                isChangingLanguage = false;
            }, 1000);
        }
    }
    
    // Add the event listener
    languageSelect.addEventListener('change', handleLanguageChange);

    async function pollTranslationStatus() {
        if (!isTranslating || isPaused) return;

        try {
            const res = await fetch('/status');
            const data = await res.json();

            // Add debugging to see what the backend thinks the current language is
            if (data.target_lang) {
                console.log('🔍 Backend current target language:', data.target_lang);
                console.log('🔍 Frontend current language:', currentLanguage);
                
                // Sync if there's a mismatch
                if (data.target_lang !== currentLanguage) {
                    console.warn('⚠️ Language mismatch detected - syncing frontend with backend');
                    currentLanguage = data.target_lang;
                    languageSelect.value = data.target_lang;
                }
            }

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