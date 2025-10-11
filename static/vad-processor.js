class VadProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        // Convert delay from milliseconds (from options) to seconds for comparison with `currentTime`
        this.voice_stop_delay_s = (options.processorOptions.voiceStopDelay || 1000) / 1000.0;
        this.speaking_threshold = options.processorOptions.speakingThreshold || 0.05;
        this.speaking = false;
        this.last_voice_time = 0;
        this.port.onmessage = (event) => {
            if (event.data.speakingThreshold) {
                this.speaking_threshold = event.data.speakingThreshold;
            }
        };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const samples = input[0];
            let sum = 0;
            for (let i = 0; i < samples.length; i++) {
                sum += samples[i] * samples[i];
            }
            const rms = Math.sqrt(sum / samples.length);

            if (rms > this.speaking_threshold) {
                this.last_voice_time = currentTime;
                if (!this.speaking) {
                    this.speaking = true;
                    this.port.postMessage({ speaking: true });
                }
            } else if (this.speaking && currentTime - this.last_voice_time > this.voice_stop_delay_s) {
                this.speaking = false;
                this.port.postMessage({ speaking: false });
            }
        }
        return true;
    }
}

registerProcessor('vad-processor', VadProcessor);
