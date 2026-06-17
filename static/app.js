document.addEventListener('DOMContentLoaded', () => {
    // API Endpoint Configuration
    const API_URL = window.location.origin;

    // DOM Elements
    const textInput = document.getElementById('textInput');
    const clearInputBtn = document.getElementById('clearInputBtn');
    const charCount = document.getElementById('charCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const sampleBtn = document.getElementById('sampleBtn');
    const loadingState = document.getElementById('loadingState');
    const resultsSection = document.getElementById('resultsSection');
    
    // Status Bar Elements
    const apiStatus = document.getElementById('apiStatus');
    const vaderStatus = document.getElementById('vaderStatus');
    const robertaStatus = document.getElementById('robertaStatus');
    
    // Verdict Elements
    const verdictBanner = document.getElementById('verdictBanner');
    const verdictIcon = document.getElementById('verdictIcon');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictText = document.getElementById('verdictText');
    
    // VADER Elements
    const vaderSentimentBadge = document.getElementById('vaderSentimentBadge');
    const vaderSentimentLabel = document.getElementById('vaderSentimentLabel');
    const vaderCompoundVal = document.getElementById('vaderCompoundVal');
    const vaderCompoundBar = document.getElementById('vaderCompoundBar');
    const vaderPosVal = document.getElementById('vaderPosVal');
    const vaderPosBar = document.getElementById('vaderPosBar');
    const vaderNeuVal = document.getElementById('vaderNeuVal');
    const vaderNeuBar = document.getElementById('vaderNeuBar');
    const vaderNegVal = document.getElementById('vaderNegVal');
    const vaderNegBar = document.getElementById('vaderNegBar');
    
    // RoBERTa Elements
    const robertaFallbackOverlay = document.getElementById('robertaFallbackOverlay');
    const robertaOutputContent = document.getElementById('robertaOutputContent');
    const robertaBreakdownContent = document.getElementById('robertaBreakdownContent');
    const robertaSentimentBadge = document.getElementById('robertaSentimentBadge');
    const robertaSentimentLabel = document.getElementById('robertaSentimentLabel');
    const robertaConfidenceVal = document.getElementById('robertaConfidenceVal');
    const robertaPosVal = document.getElementById('robertaPosVal');
    const robertaPosBar = document.getElementById('robertaPosBar');
    const robertaNeuVal = document.getElementById('robertaNeuVal');
    const robertaNeuBar = document.getElementById('robertaNeuBar');
    const robertaNegVal = document.getElementById('robertaNegVal');
    const robertaNegBar = document.getElementById('robertaNegBar');
    
    // History Elements
    const historyList = document.getElementById('historyList');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');

    // Preset Sample Reviews
    const sampleReviews = [
        "This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go.",
        "I bought this product expecting a miracle, but it was just average. Not terrible, but certainly not great either.",
        "This camera is absolutely amazing! The focus is lightning fast, and low light performance is spectacular. Highly recommended!",
        "Product arrived labeled as Jumbo Salted Peanuts... the peanuts were actually small sized unsalted. Not as Advertised",
        "The movie was supposedly a masterpiece, but I found it incredibly boring and fell asleep halfway through.",
        "I am so happy!"
    ];
    let sampleIndex = 0;

    // Initialize Page
    checkBackendStatus();
    loadHistory();

    // Event Listeners
    textInput.addEventListener('input', updateCharCount);
    clearInputBtn.addEventListener('click', () => {
        textInput.value = '';
        updateCharCount();
        resultsSection.classList.add('hidden');
        textInput.focus();
    });

    sampleBtn.addEventListener('click', () => {
        textInput.value = sampleReviews[sampleIndex];
        sampleIndex = (sampleIndex + 1) % sampleReviews.length;
        updateCharCount();
        textInput.focus();
    });

    analyzeBtn.addEventListener('click', performAnalysis);
    clearHistoryBtn.addEventListener('click', clearHistory);

    // Helpers
    function updateCharCount() {
        const length = textInput.value.length;
        charCount.textContent = `${length} / 1000 characters`;
    }

    // Check Backend Server Status
    async function checkBackendStatus() {
        try {
            const res = await fetch(`${API_URL}/api/status`);
            if (!res.ok) throw new Error('API response not OK');
            const data = await res.json();
            
            // Server Connected
            apiStatus.innerHTML = '<span class="dot active"></span>Online';
            apiStatus.className = 'status-badge';
            
            // VADER Status
            vaderStatus.innerHTML = '<span class="dot active"></span>Loaded';
            vaderStatus.className = 'status-badge';
            
            // RoBERTa Status
            if (data.roberta_status === 'loaded') {
                robertaStatus.innerHTML = '<span class="dot active"></span>Loaded';
                robertaStatus.className = 'status-badge';
            } else {
                robertaStatus.innerHTML = '<span class="dot failed"></span>Offline (Fallback)';
                robertaStatus.className = 'status-badge';
                console.warn("RoBERTa model failed to load in backend. Reason:", data.roberta_error);
            }
        } catch (err) {
            console.error('Error connecting to API Server:', err);
            apiStatus.innerHTML = '<span class="dot failed"></span>Offline';
            apiStatus.className = 'status-badge';
            vaderStatus.innerHTML = '<span class="dot failed"></span>Unavailable';
            vaderStatus.className = 'status-badge';
            robertaStatus.innerHTML = '<span class="dot failed"></span>Unavailable';
            robertaStatus.className = 'status-badge';
        }
    }

    // Call Sentiment Analysis API
    async function performAnalysis() {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }

        // Setup UI for Loading State
        loadingState.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        analyzeBtn.disabled = true;

        try {
            const response = await fetch(`${API_URL}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis request failed');
            }

            const result = await response.json();
            renderResults(result);
            saveToHistory(text, result);
        } catch (err) {
            console.error('Analysis Error:', err);
            alert(`Analysis failed: ${err.message}`);
        } finally {
            loadingState.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    }

    // Render results on the screen
    function renderResults(data) {
        resultsSection.classList.remove('hidden');

        // 1. Process VADER
        const v = data.vader;
        let vaderSentiment = 'neutral';
        if (v.compound >= 0.05) vaderSentiment = 'positive';
        else if (v.compound <= -0.05) vaderSentiment = 'negative';

        vaderSentimentLabel.textContent = vaderSentiment;
        vaderSentimentBadge.className = `sentiment-badge ${vaderSentiment.substring(0, 3)}`;
        
        // Compound formatting
        const compFormatted = v.compound >= 0 ? `+${v.compound.toFixed(2)}` : v.compound.toFixed(2);
        vaderCompoundVal.textContent = compFormatted;
        vaderCompoundVal.className = 'score-val ' + (v.compound >= 0.05 ? 'pos' : v.compound <= -0.05 ? 'neg' : 'neu');
        
        // VADER compound progress bar (Vader compound ranges from -1 to +1, normalize to 0 to 100%)
        const compoundPercent = ((v.compound + 1) / 2) * 100;
        vaderCompoundBar.style.width = `${compoundPercent}%`;
        vaderCompoundBar.style.backgroundColor = v.compound >= 0.05 ? 'var(--color-pos)' : v.compound <= -0.05 ? 'var(--color-neg)' : 'var(--color-neu)';
        
        // VADER probability breakdown
        vaderPosVal.textContent = `${(v.pos * 100).toFixed(0)}%`;
        vaderPosBar.style.width = `${v.pos * 100}%`;
        vaderNeuVal.textContent = `${(v.neu * 100).toFixed(0)}%`;
        vaderNeuBar.style.width = `${v.neu * 100}%`;
        vaderNegVal.textContent = `${(v.neg * 100).toFixed(0)}%`;
        vaderNegBar.style.width = `${v.neg * 100}%`;

        // 2. Process RoBERTa
        let robertaSentiment = 'neutral';
        if (data.roberta_available && data.roberta) {
            robertaFallbackOverlay.classList.add('hidden');
            robertaOutputContent.classList.remove('hidden');
            robertaBreakdownContent.classList.remove('hidden');

            const r = data.roberta;
            // The notebook classes are: 0: negative, 1: neutral, 2: positive
            const maxVal = Math.max(r.neg, r.neu, r.pos);
            
            if (maxVal === r.pos) robertaSentiment = 'positive';
            else if (maxVal === r.neg) robertaSentiment = 'negative';

            robertaSentimentLabel.textContent = robertaSentiment;
            robertaSentimentBadge.className = `sentiment-badge ${robertaSentiment.substring(0, 3)}`;

            const confidencePercent = maxVal * 100;
            robertaConfidenceVal.textContent = `${confidencePercent.toFixed(1)}%`;
            
            robertaPosVal.textContent = `${(r.pos * 100).toFixed(1)}%`;
            robertaPosBar.style.width = `${r.pos * 100}%`;
            robertaNeuVal.textContent = `${(r.neu * 100).toFixed(1)}%`;
            robertaNeuBar.style.width = `${r.neu * 100}%`;
            robertaNegVal.textContent = `${(r.neg * 100).toFixed(1)}%`;
            robertaNegBar.style.width = `${r.neg * 100}%`;
        } else {
            robertaFallbackOverlay.classList.remove('hidden');
            robertaOutputContent.classList.add('hidden');
            robertaBreakdownContent.classList.add('hidden');
            
            // Show detailed error if available
            const errorPara = robertaFallbackOverlay.querySelector('p');
            if (errorPara) {
                if (data.roberta_error) {
                    errorPara.innerHTML = `Failed to connect to the Hugging Face Inference API.<br><br><span style="color: #ff6b6b; font-family: monospace; font-size: 0.9rem; background: rgba(255, 107, 107, 0.1); padding: 8px 12px; border-radius: 4px; border: 1px solid rgba(255, 107, 107, 0.2); display: block; word-break: break-all; margin-top: 10px; text-align: left;">Error: ${escapeHtml(data.roberta_error)}</span>`;
                } else {
                    errorPara.textContent = 'Failed to connect to the Hugging Face Inference API. Please ensure the server has internet access, or check your HF_API_TOKEN environment variable configuration.';
                }
            }
        }

        // 3. Verdict Banner Render
        let verdictClass = 'neu';
        let consensusTitle = 'Models Consensus';
        let consensusText = '';

        if (data.roberta_available && data.roberta) {
            if (vaderSentiment === robertaSentiment) {
                verdictClass = vaderSentiment.substring(0, 3);
                consensusTitle = 'Models Agreement';
                consensusText = `Both models agree that the sentiment is overall <strong>${vaderSentiment.toUpperCase()}</strong>.`;
                verdictIcon.className = 'fa-solid fa-handshake';
            } else {
                verdictClass = 'neu';
                consensusTitle = 'Models Divergence';
                consensusText = `VADER predicts <strong>${vaderSentiment.toUpperCase()}</strong>, while RoBERTa predicts <strong>${robertaSentiment.toUpperCase()}</strong>. Deep learning (RoBERTa) is generally more accurate for nuanced contexts.`;
                verdictIcon.className = 'fa-solid fa-circle-half-stroke';
            }
        } else {
            // VADER only
            verdictClass = vaderSentiment.substring(0, 3);
            consensusTitle = 'VADER Analysis';
            consensusText = `Lexicon model indicates an overall <strong>${vaderSentiment.toUpperCase()}</strong> sentiment.`;
            verdictIcon.className = 'fa-solid fa-magnifying-glass-chart';
        }

        verdictBanner.className = `verdict-banner glass-card ${verdictClass}`;
        verdictTitle.textContent = consensusTitle;
        verdictText.innerHTML = consensusText;

        // Scroll results into view smoothly
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Local Storage History Management
    function saveToHistory(text, result) {
        let history = JSON.parse(localStorage.getItem('sentiment_history') || '[]');
        
        // Build history item
        const historyItem = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            text,
            result
        };

        // Prepend and limit size to 20
        history.unshift(historyItem);
        if (history.length > 20) history.pop();

        localStorage.setItem('sentiment_history', JSON.stringify(history));
        loadHistory();
    }

    function loadHistory() {
        const history = JSON.parse(localStorage.getItem('sentiment_history') || '[]');
        historyList.innerHTML = '';

        if (history.length === 0) {
            historyList.innerHTML = `
                <div class="no-history">
                    <p>No recent analyses. Try running some text!</p>
                </div>
            `;
            return;
        }

        history.forEach(item => {
            // Get simple label from result
            const v = item.result.vader;
            let vaderSentiment = 'neu';
            if (v.compound >= 0.05) vaderSentiment = 'pos';
            else if (v.compound <= -0.05) vaderSentiment = 'neg';

            let robertaSentiment = '';
            if (item.result.roberta_available && item.result.roberta) {
                const r = item.result.roberta;
                const maxVal = Math.max(r.neg, r.neu, r.pos);
                if (maxVal === r.pos) robertaSentiment = 'pos';
                else if (maxVal === r.neg) robertaSentiment = 'neg';
                else robertaSentiment = 'neu';
            }

            const itemDiv = document.createElement('div');
            itemDiv.className = 'history-item';
            
            // Build badge markup
            let badgesHtml = `<span class="history-badge-small ${vaderSentiment}">V: ${vaderSentiment}</span>`;
            if (robertaSentiment) {
                badgesHtml += ` <span class="history-badge-small ${robertaSentiment}">R: ${robertaSentiment}</span>`;
            }

            itemDiv.innerHTML = `
                <div class="history-item-header">
                    <span class="history-time">${item.timestamp}</span>
                    <div class="history-badges">
                        ${badgesHtml}
                    </div>
                </div>
                <div class="history-item-text">${escapeHtml(item.text)}</div>
            `;

            // Restore from history on click
            itemDiv.addEventListener('click', () => {
                textInput.value = item.text;
                updateCharCount();
                renderResults(item.result);
            });

            historyList.appendChild(itemDiv);
        });
    }

    function clearHistory() {
        if (confirm('Clear all history items?')) {
            localStorage.removeItem('sentiment_history');
            loadHistory();
        }
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }
});
