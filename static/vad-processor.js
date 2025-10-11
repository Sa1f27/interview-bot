class VadProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        // Voice activity detection parameters
        const opts = options.processorOptions || {};
        this.voiceStopDelayMs = opts.voiceStopDelay || 1500; // Reduced to 1.5 seconds
        this.speakingThreshold = opts.speakingThreshold || 0.04; // Increased threshold to ignore background noise
        this.minSpeakingDuration = opts.minSpeakingDuration || 400; // Reduced minimum duration
        this.maxRecordingDuration = opts.maxRecordingDuration || 8000; // Force stop after 8 seconds
        
        // State tracking
        this.isSpeaking = false;
        this.lastVoiceTime = 0;
        this.firstVoiceTime = 0;
        this.silenceStartTime = 0;
        this.recordingStartTime = 0;
        this.frameCount = 0;
        
        // Smoothing parameters
        this.rmsHistory = [];
        this.historySize = 3; // Reduced for faster response
        
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
        this.recordingStartTime = 0;
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
        
        // Initialize recording start time
        if (this.recordingStartTime === 0) {
            this.recordingStartTime = currentTimeMs;
        }
        
        // FORCE STOP after max duration (8 seconds)
        const recordingDuration = currentTimeMs - this.recordingStartTime;
        if (recordingDuration >= this.maxRecordingDuration) {
            if (this.isSpeaking) {
                this.isSpeaking = false;
                this.port.postMessage({ 
                    speaking: false,
                    reason: 'max_duration',
                    duration: recordingDuration
                });
                this.reset();
            }
            return true;
        }
        
        // Calculate RMS (volume level)
        const rms = this.calculateRMS(samples);
        const smoothedRMS = this.getSmoothedRMS(rms);
        
        // Detect voice activity with higher threshold (less sensitive to noise)
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
                
                // Stop if:
                // 1. Silence lasted long enough (voiceStopDelayMs)
                // 2. User spoke for minimum duration (minSpeakingDuration)
                if (silenceDuration >= this.voiceStopDelayMs && 
                    speakingDuration >= this.minSpeakingDuration) {
                    
                    this.isSpeaking = false;
                    this.port.postMessage({ 
                        speaking: false,
                        reason: 'silence',
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