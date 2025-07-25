class ISLConverter {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.isCapturing = false;
        this.isRealtimeActive = false;
        this.realtimeInterval = null;
        this.realtimePollInterval = null;  // For polling real-time predictions
        
        // UI elements
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.signToTextBtn = document.getElementById('signToTextBtn');
        this.realtimeTextBtn = document.getElementById('realtimeTextBtn');
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.handCanvas = document.getElementById('handCanvas');
        this.predictedLetter = document.getElementById('predictedLetter');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.textArea = document.getElementById('textArea');
        this.clearBtn = document.getElementById('clearBtn');
        this.copyBtn = document.getElementById('copyBtn');
        this.status = document.getElementById('status');
        this.historyList = document.getElementById('historyList');
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
        this.loading = document.getElementById('loading');
        
        this.predictionHistory = [];
        this.currentMode = 'text'; // only text mode now
        this.handTrackingActive = false;
        this.handTrackingInterval = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupCanvas();
        this.checkCameraSupport();
        // Set up hand canvas to overlay on video
        this.setupHandCanvas();
    }
    
    setupHandCanvas() {
        // Make hand canvas same size as video container
        const videoContainer = document.querySelector('.video-container');
        this.handCanvas.width = 640;
        this.handCanvas.height = 480;
        this.handCanvas.style.width = '100%';
        this.handCanvas.style.height = 'auto';
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.signToTextBtn.addEventListener('click', () => this.captureAndPredict('text'));
        
        // Real-time prediction controls (if they exist)
        if (this.realtimeTextBtn) {
            this.realtimeTextBtn.addEventListener('click', () => this.toggleRealtime('text'));
        }
        
        this.clearBtn.addEventListener('click', () => this.clearText());
        this.copyBtn.addEventListener('click', () => this.copyText());
        
        // Modal close
        document.querySelector('.close').addEventListener('click', () => {
            this.errorModal.style.display = 'none';
        });
        
        // Close modal on outside click
        window.addEventListener('click', (e) => {
            if (e.target === this.errorModal) {
                this.errorModal.style.display = 'none';
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === '1' && e.ctrlKey) {
                e.preventDefault();
                this.captureAndPredict('text');
            } else if (e.key === '2' && e.ctrlKey) {
                e.preventDefault();
                this.toggleRealtime('text');
            } else if (e.key === 'Escape') {
                this.errorModal.style.display = 'none';
                this.stopRealtime();
            }
        });
    }
    
    setupCanvas() {
        this.canvas.width = 640;
        this.canvas.height = 480;
    }
    
    checkCameraSupport() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError('Camera not supported in this browser.');
            this.startBtn.disabled = true;
        }
    }
    
    async startCamera() {
        try {
            this.showLoading(true);
            this.updateStatus('Connecting to camera...');
            
            // Start the backend video feed first
            try {
                const response = await fetch('/start_video_feed', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                if (response.ok) {
                    // Show the video feed with hand detection overlay
                    const videoFeed = document.getElementById('videoFeed');
                    videoFeed.src = '/video_feed?' + new Date().getTime(); // Cache busting
                    videoFeed.style.display = 'block';
                    this.video.style.display = 'none';
                    
                    this.updateStatus('Camera active - Hand detection enabled');
                    this.startBtn.disabled = true;
                    this.stopBtn.disabled = false;
                    this.signToTextBtn.disabled = false;
                    if (this.realtimeTextBtn) this.realtimeTextBtn.disabled = false;
                    this.showLoading(false);
                } else {
                    throw new Error('Failed to start video feed');
                }
            } catch (error) {
                console.error('Error starting video feed:', error);
                this.showError('Failed to start camera. Please ensure camera is not being used by another application.');
                this.showLoading(false);
            }
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showError('Failed to access camera. Please ensure camera permissions are granted.');
            this.showLoading(false);
        }
    }
    
    stopCamera() {
        // Stop real-time prediction first
        if (this.isRealtimeActive) {
            this.stopRealtime();
        }
        
        // Stop polling interval if it exists
        if (this.realtimePollInterval) {
            clearInterval(this.realtimePollInterval);
            this.realtimePollInterval = null;
        }
        
        // Stop the backend video feed
        fetch('/stop_video_feed', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        }).catch(error => console.error('Error stopping video feed:', error));
        
        // Hide the video feed and show regular video element
        const videoFeed = document.getElementById('videoFeed');
        videoFeed.src = '';
        videoFeed.style.display = 'none';
        this.video.style.display = 'block';
        
        // Stop hand tracking and real-time prediction when camera stops
        this.stopHandTracking();
        
        this.updateStatus('Camera stopped');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.signToTextBtn.disabled = true;
        if (this.realtimeTextBtn) this.realtimeTextBtn.disabled = true;
    }
    
    startHandTracking() {
        if (!this.handTrackingActive) {
            this.handTrackingActive = true;
            this.handCanvas.style.display = 'block';
            this.detectHands();
            console.log('Hand tracking started automatically');
        }
    }
    
    stopHandTracking() {
        this.handTrackingActive = false;
        if (this.handCanvas) {
            this.handCanvas.style.display = 'none';
            const ctx = this.handCanvas.getContext('2d');
            ctx.clearRect(0, 0, this.handCanvas.width, this.handCanvas.height);
        }
        console.log('Hand tracking stopped');
    }
    
    async detectHands() {
        if (!this.handTrackingActive) return;
        
        if (this.video && this.video.videoWidth > 0 && this.video.videoHeight > 0) {
            // Set canvas dimensions to match video
            this.handCanvas.width = this.video.videoWidth;
            this.handCanvas.height = this.video.videoHeight;
            
            const ctx = this.handCanvas.getContext('2d');
            ctx.clearRect(0, 0, this.handCanvas.width, this.handCanvas.height);
            
            // Draw hand landmarks overlay
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 2;
            ctx.fillStyle = '#FF0000';
            
            // Here you would add MediaPipe hand detection results
            // For now, just show that hand tracking is active
            ctx.font = '16px Arial';
            ctx.fillStyle = '#00FF00';
            ctx.fillText('Hand Tracking Active', 10, 30);
        }
        
        // Continue hand tracking loop
        if (this.handTrackingActive) {
            requestAnimationFrame(() => this.detectHands());
        }
    }
    
    async captureFrame() {
        try {
            const response = await fetch('/capture_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                return result.image;
            } else {
                console.error('Error capturing frame:', result.error);
                return null;
            }
        } catch (error) {
            console.error('Error capturing frame:', error);
            return null;
        }
    }
    
    async captureAndPredict(mode = 'text') {
        if (this.isCapturing) return;
        
        // Check if camera is active
        const videoFeed = document.getElementById('videoFeed');
        if (!videoFeed || videoFeed.style.display === 'none' || !videoFeed.src) {
            this.showError('Please start the camera first');
            return;
        }
        
        try {
            this.isCapturing = true;
            this.currentMode = mode;
            this.showLoading(true);
            
            // Update status based on mode
            if (mode === 'text') {
                this.updateStatus('Analyzing hand sign for text...');
            } else {
                this.updateStatus('Analyzing hand sign for speech...');
            }
            
            const imageData = await this.captureFrame();
            if (!imageData) {
                throw new Error('Failed to capture frame');
            }
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    image: imageData,
                    mode: mode
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayPrediction(result.prediction, result.confidence, mode);
                this.addToHistory(result.prediction, result.confidence, mode);
                
                if (mode === 'text') {
                    this.updateStatus('Hand sign converted to text');
                } else {
                    if (result.spoken) {
                        this.updateStatus('Hand sign converted to speech - Speaking...');
                        // Reset status after estimated speaking time
                        setTimeout(() => {
                            this.updateStatus('Ready for next hand sign');
                        }, 2000);
                    } else {
                        this.updateStatus('Low confidence - try again for speech');
                    }
                }
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Failed to predict hand sign. Please try again.');
            this.updateStatus('Error occurred');
        } finally {
            this.isCapturing = false;
            this.showLoading(false);
        }
    }
    
    displayPrediction(letter, confidence, mode = 'text') {
        this.predictedLetter.textContent = letter;
        this.confidenceFill.style.width = `${confidence}%`;
        this.confidenceValue.textContent = `${confidence}%`;
        
        // Add visual indicator for mode
        const modeIndicator = mode === 'speech' ? ' 🔊' : ' 📝';
        this.predictedLetter.setAttribute('title', `Mode: ${mode === 'speech' ? 'Speech' : 'Text'}`);
        
        // Add letter to text area
        const currentText = this.textArea.value;
        this.textArea.value = currentText + letter;
        
        // Animate the prediction (only text mode now)
        this.predictedLetter.style.background = 'linear-gradient(45deg, #28a745, #20c997)';
        this.predictedLetter.style.color = 'white';
        
        this.predictedLetter.style.transform = 'scale(1.2)';
        setTimeout(() => {
            this.predictedLetter.style.transform = 'scale(1)';
            // Reset to default styling after animation
            setTimeout(() => {
                this.predictedLetter.style.background = '#f8f9fa';
                this.predictedLetter.style.color = '#667eea';
            }, 1000);
        }, 200);
    }
    
    addToHistory(letter, confidence, mode = 'text') {
        const timestamp = new Date().toLocaleTimeString();
        const modeIcon = '📝'; // Only text mode now
        this.predictionHistory.unshift({ letter, confidence, timestamp, mode, modeIcon });
        
        // Keep only last 10 predictions
        if (this.predictionHistory.length > 10) {
            this.predictionHistory = this.predictionHistory.slice(0, 10);
        }
        
        this.updateHistoryDisplay();
    }
    
    updateHistoryDisplay() {
        if (this.predictionHistory.length === 0) {
            this.historyList.innerHTML = '<p class="empty-history">No predictions yet</p>';
            return;
        }
        
        this.historyList.innerHTML = this.predictionHistory
            .map(item => `
                <div class="history-item">
                    <span class="history-letter">${item.modeIcon} ${item.letter}</span>
                    <span class="history-confidence">${item.confidence}% - ${item.timestamp}</span>
                </div>
            `).join('');
    }
    
    clearText() {
        this.textArea.value = '';
        this.predictedLetter.textContent = '-';
        this.confidenceFill.style.width = '0%';
        this.confidenceValue.textContent = '0%';
    }
    
    async copyText() {
        const text = this.textArea.value.trim();
        if (!text) {
            this.showError('No text to copy.');
            return;
        }
        
        try {
            await navigator.clipboard.writeText(text);
            this.updateStatus('Text copied to clipboard');
            
            // Reset status after 2 seconds
            setTimeout(() => {
                this.updateStatus('Ready');
            }, 2000);
            
        } catch (error) {
            console.error('Copy error:', error);
            // Fallback for older browsers
            this.textArea.select();
            document.execCommand('copy');
            this.updateStatus('Text copied to clipboard');
        }
    }
    
    updateStatus(message) {
        this.status.querySelector('span').textContent = message;
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.style.display = 'block';
    }
    
    // Real-time prediction methods (text only)
    toggleRealtime(mode = 'text') {
        if (this.isRealtimeActive) {
            this.stopRealtime();
        } else {
            this.startRealtime('text'); // Always use text mode
        }
    }
    
    async startRealtime(mode = 'text') {
        if (this.isRealtimeActive) return;
        
        // Check if camera is active
        const videoFeed = document.getElementById('videoFeed');
        if (!videoFeed || videoFeed.style.display === 'none' || !videoFeed.src) {
            this.showError('Please start the camera first');
            return;
        }
        
        try {
            const response = await fetch('/start_realtime', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ mode: 'text' }) // Always text mode
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isRealtimeActive = true;
                this.currentMode = 'text';
                
                // Update button state (only text button)
                if (this.realtimeTextBtn) {
                    this.realtimeTextBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Real-time Text';
                    this.realtimeTextBtn.classList.remove('btn-primary');
                    this.realtimeTextBtn.classList.add('btn-danger');
                }
                
                this.updateStatus('Real-time text prediction started - Watch the video feed for predictions');
                
                // Start polling for real-time predictions to update UI
                this.startRealtimePolling();
            }
        } catch (error) {
            console.error('Error starting real-time prediction:', error);
            this.showError('Failed to start real-time prediction');
        }
    }
    
    startRealtimePolling() {
        // Poll for real-time predictions every 500ms to update UI
        this.realtimePollInterval = setInterval(async () => {
            if (this.isRealtimeActive) {
                try {
                    const response = await fetch('/get_realtime_prediction');
                    const result = await response.json();
                    
                    if (result.success && result.hand_detected && result.confidence > 65) {
                        // Update UI with real-time prediction
                        this.displayPrediction(result.prediction, result.confidence, result.mode);
                        this.addToHistory(result.prediction, result.confidence, result.mode);
                    }
                } catch (error) {
                    // Silently ignore errors in polling to avoid spam
                    console.debug('Real-time polling error:', error);
                }
            }
        }, 500); // Poll every 500ms
    }
    
    async stopRealtime() {
        if (!this.isRealtimeActive) return;
        
        try {
            const response = await fetch('/stop_realtime', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isRealtimeActive = false;
                
                // Stop polling for real-time predictions
                if (this.realtimePollInterval) {
                    clearInterval(this.realtimePollInterval);
                    this.realtimePollInterval = null;
                }
                
                // Reset button state (only text button)
                if (this.realtimeTextBtn) {
                    this.realtimeTextBtn.innerHTML = '<i class="fas fa-play"></i> Real-time Text';
                    this.realtimeTextBtn.classList.remove('btn-danger');
                    this.realtimeTextBtn.classList.add('btn-primary');
                }
                
                this.updateStatus('Real-time prediction stopped');
            }
        } catch (error) {
            console.error('Error stopping real-time prediction:', error);
        }
    }
    
    showLoading(show) {
        this.loading.style.display = show ? 'flex' : 'none';
    }
    
    // Health check
    async checkHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            console.log('App health:', health);
            
            if (!health.model_loaded) {
                this.showError('Model not loaded. Please check server configuration.');
            }
            
        } catch (error) {
            console.error('Health check failed:', error);
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new ISLConverter();
    
    // Perform health check
    app.checkHealth();
    
    // Add helpful instructions
    console.log('ISL Converter loaded!');
    console.log('Keyboard shortcuts:');
    console.log('- Ctrl + 1: Capture hand sign and predict text');
    console.log('- Ctrl + 2: Capture hand sign and predict with speech');
    console.log('- Ctrl + 3: Toggle real-time text prediction');
    console.log('- Ctrl + 4: Toggle real-time speech prediction');
    console.log('- Ctrl + 5: Toggle between detection view and regular video');
    console.log('- Escape: Close error modal / Stop real-time prediction');
});

// Service Worker registration (for future PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Will register service worker in future updates
    });
}
