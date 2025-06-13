document.addEventListener('DOMContentLoaded', function() {
    
    // 🚨 ONLY RUN ON CHATBOT PAGE
    const startBtn = document.getElementById('startChatbotBtn');
    if (!startBtn) {
        console.log('Not on chatbot page, skipping chatbot JS');
        return; // Exit if not on chatbot page
    }
    
    const stopBtn = document.getElementById('stopChatbotBtn');
    const statusAlert = document.getElementById('chatbotStatusAlert');
    const chatHistory = document.getElementById('chatHistory');
    const userMessage = document.getElementById('userMessage');
    const botResponse = document.getElementById('botResponse');
    const languageSelect = document.getElementById('chatbotLanguageSelect');
    
    let isChatting = false;
    
    // Start chatbot conversation
    startBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/start_chatbot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    language: languageSelect.value
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'started') {
                isChatting = true;
                startBtn.disabled = true;
                stopBtn.disabled = false;
                statusAlert.textContent = 'Conversation active - speak now';
                statusAlert.className = 'alert alert-success';
                
                // Start polling for updates
                pollChatbotUpdates();
            }
        } catch (error) {
            console.error('Error starting chatbot:', error);
            statusAlert.textContent = 'Error starting conversation';
            statusAlert.className = 'alert alert-danger';
        }
    });
    
    // Stop chatbot conversation
    stopBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/stop_chatbot', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.status === 'stopped') {
                isChatting = false;
                startBtn.disabled = false;
                stopBtn.disabled = true;
                statusAlert.textContent = 'Conversation stopped';
                statusAlert.className = 'alert alert-info';
            }
        } catch (error) {
            console.error('Error stopping chatbot:', error);
        }
    });
    
    // Change language
    languageSelect.addEventListener('change', async function() {
        console.log('🔄 CHATBOT Language selection changed to:', this.value);
        
        if (isChatting) {
            try {
                const response = await fetch('/change_chatbot_language', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        language: languageSelect.value
                    })
                });
                
                console.log('📡 CHATBOT Response status:', response.status);
                const data = await response.json();
                console.log('✅ CHATBOT Change language response:', data);
                
                if (data.status !== 'language_changed') {
                    // Revert selection if change failed
                    languageSelect.value = data.previous_language || 'en';
                }
            } catch (error) {
                console.error('❌ CHATBOT Error changing language:', error);
            }
        }
    });
    
    // Poll for conversation updates
    async function pollChatbotUpdates() {
        if (!isChatting) return;
        
        try {
            const response = await fetch('/get_chatbot_status');
            const data = await response.json();
            
            if (data.user_message) {
                userMessage.textContent = data.user_message;
            }
            
            if (data.bot_response) {
                botResponse.textContent = data.bot_response;
                
                // Add to conversation history
                if (data.user_message && data.bot_response) {
                    const userDiv = document.createElement('div');
                    userDiv.className = 'user-message mb-2';
                    userDiv.innerHTML = `<strong>You:</strong> ${data.user_message}`;
                    
                    const botDiv = document.createElement('div');
                    botDiv.className = 'bot-message mb-3';
                    botDiv.innerHTML = `<strong>Bot:</strong> ${data.bot_response}`;
                    
                    chatHistory.appendChild(userDiv);
                    chatHistory.appendChild(botDiv);
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }
            }
            
            // Continue polling
            setTimeout(pollChatbotUpdates, 1000);
        } catch (error) {
            console.error('Error polling chatbot updates:', error);
            setTimeout(pollChatbotUpdates, 5000); // Retry after 5 seconds
        }
    }
});