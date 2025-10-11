class VadProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        // Voice activity detection parameters
        const opts = options.processorOptions || {};
        this.voiceStopDelayMs = opts.voiceStopDelay || 2000; // 2 seconds default
        this.speakingThreshold = opts.speakingThreshold || 0.02;
        this.minSpeakingDuration = opts.minSpeakingDuration || 300; // Minimum 300ms of speech
        
        // State tracking
        this.isSpeaking = false;
        this.lastVoiceTime = 0;
        this.firstVoiceTime = 0;
        this.silenceStartTime = 0;
        this.frameCount = 0;
        
        // Smoothing parameters
        this.rmsHistory = [];
        this.historySize = 5;
        
        this.port.onmessage = (event) => {
            if (event.data.reset) {
                this.reset();
            }
        };
    }
    
    reset() {
        this.isSpeaking = false;
        this.lastVoiceTime = 0;
        this.firstVoiceTime = 0;
        this.silenceStartTime = 0;
        this.rmsHistory = [];
    }
    
    calculateRMS(samples) {
        let sum = 0;
        for (let i = 0; i < samples.length; i++) {
            sum += samples[i] * samples[i];
        }
        return Math.sqrt(sum / samples.length);
    }
    
    getSmoothedRMS(currentRMS) {
        this.rmsHistory.push(currentRMS);
        if (this.rmsHistory.length > this.historySize) {
            this.rmsHistory.shift();
        }
        
        // Calculate average
        const sum = this.rmsHistory.reduce((a, b) => a + b, 0);
        return sum / this.rmsHistory.length;
    }
    
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        if (!input || input.length === 0) {
            return true;
        }
        
        const samples = input[0];
        if (!samples || samples.length === 0) {
            return true;
        }
        
        this.frameCount++;
        const currentTimeMs = this.frameCount * (samples.length / sampleRate) * 1000;
        
        // Calculate RMS (volume level)
        const rms = this.calculateRMS(samples);
        const smoothedRMS = this.getSmoothedRMS(rms);
        
        // Detect voice activity
        const isVoiceDetected = smoothedRMS > this.speakingThreshold;
        
        if (isVoiceDetected) {
            // Voice detected
            if (!this.isSpeaking) {
                // First voice detection
                this.firstVoiceTime = currentTimeMs;
                this.lastVoiceTime = currentTimeMs;
                this.isSpeaking = true;
                this.silenceStartTime = 0;
                
                this.port.postMessage({ 
                    speaking: true,
                    level: smoothedRMS
                });
            } else {
                // Continuing to speak
                this.lastVoiceTime = currentTimeMs;
                this.silenceStartTime = 0;
            }
        } else {
            // No voice detected (silence)
            if (this.isSpeaking) {
                if (this.silenceStartTime === 0) {
                    this.silenceStartTime = currentTimeMs;
                }
                
                const silenceDuration = currentTimeMs - this.silenceStartTime;
                const speakingDuration = this.lastVoiceTime - this.firstVoiceTime;
                
                // Only stop if:
                // 1. Silence lasted long enough (voiceStopDelayMs)
                // 2. User spoke for minimum duration (minSpeakingDuration)
                if (silenceDuration >= this.voiceStopDelayMs && 
                    speakingDuration >= this.minSpeakingDuration) {
                    
                    this.isSpeaking = false;
                    this.port.postMessage({ 
                        speaking: false,
                        duration: speakingDuration
                    });
                    this.reset();
                }
            }
        }
        
        return true;
    }
}

registerProcessor('vad-processor', VadProcessor);