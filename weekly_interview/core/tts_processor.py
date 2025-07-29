# weekly_interview/core/tts_processor.py
"""
Simple TTS Processor - Fixed EdgeTTS with fallback
Separated from ai_services.py for modularity
"""

import asyncio
import edge_tts
import logging
import re
from typing import AsyncGenerator, List
from .config import config

logger = logging.getLogger(__name__)

class UltraFastTTSProcessor:
    """Simple TTS processor with EdgeTTS error fixes and fallback"""
    
    def __init__(self):
        self.voice = config.TTS_VOICE
        self.rate = config.TTS_SPEED
        self.available_voices = None
        self._voices_checked = False
    
    async def _check_voice_availability(self):
        """Check if configured voice is available"""
        if self._voices_checked:
            return
            
        try:
            voices = await edge_tts.list_voices()
            self.available_voices = [voice["Name"] for voice in voices]
            
            if self.voice not in self.available_voices:
                logger.warning(f"?? Voice '{self.voice}' not available. Switching to fallback.")
                # Find first available English voice
                english_voices = [v for v in self.available_voices if v.startswith("en-")]
                if english_voices:
                    self.voice = english_voices[0]
                    logger.info(f"? Using fallback voice: {self.voice}")
                else:
                    self.voice = self.available_voices[0] if self.available_voices else "en-US-JennyNeural"
                    logger.info(f"? Using default voice: {self.voice}")
            
            self._voices_checked = True
            
        except Exception as e:
            logger.warning(f"?? Could not verify voice availability: {e}")
            # Use a known working voice as fallback
            self.voice = "en-US-JennyNeural"
            self._voices_checked = True
    
    def split_text_optimized(self, text: str) -> List[str]:
        """Optimized text splitting for minimal latency"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > config.TTS_CHUNK_SIZE * 5:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    async def generate_ultra_fast_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate ultra-fast audio with error handling and fallback"""
        try:
            # Check voice availability first
            await self._check_voice_availability()
            
            chunks = self.split_text_optimized(text)
            logger.info(f"?? Generating TTS for {len(chunks)} chunks with voice: {self.voice}")
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                # Generate audio for this chunk with retries
                audio_data = await self._generate_chunk_audio_with_retry(chunk, max_retries=3)
                
                if audio_data:
                    yield audio_data
                else:
                    logger.warning(f"?? Chunk {i+1} failed, generating silent audio")
                    # Generate silent audio as fallback
                    silent_audio = await self._generate_silent_audio(len(chunk.split()) * 0.5)
                    if silent_audio:
                        yield silent_audio
                        
        except Exception as e:
            logger.error(f"? Ultra-fast TTS error: {e}")
            # Generate silent audio for entire text as final fallback
            try:
                words = len(text.split())
                duration = max(2.0, min(10.0, words * 0.5))
                silent_audio = await self._generate_silent_audio(duration)
                if silent_audio:
                    yield silent_audio
                    logger.info(f"?? Generated {duration:.1f}s silent audio as fallback")
            except Exception as fallback_error:
                logger.error(f"? Even fallback failed: {fallback_error}")
                # Return minimal audio to prevent complete failure
                yield b'\x00' * 1024
    
    async def _generate_chunk_audio_with_retry(self, chunk: str, max_retries: int = 3) -> bytes:
        """Generate audio for chunk with retry logic"""
        for attempt in range(max_retries):
            try:
                logger.debug(f"?? TTS attempt {attempt + 1} for: '{chunk[:50]}...'")
                
                # Create TTS communicator with current settings
                tts = edge_tts.Communicate(
                    text=chunk,
                    voice=self.voice,
                    rate=self.rate
                )
                
                audio_data = b""
                
                # Collect audio data with timeout
                async def collect_audio():
                    nonlocal audio_data
                    async for tts_chunk in tts.stream():
                        if tts_chunk["type"] == "audio" and tts_chunk["data"]:
                            audio_data += tts_chunk["data"]
                
                # Add timeout to prevent hanging
                await asyncio.wait_for(collect_audio(), timeout=10.0)
                
                if audio_data and len(audio_data) > 100:  # Ensure we got meaningful audio
                    logger.debug(f"? Generated {len(audio_data)} bytes of audio")
                    return audio_data
                else:
                    raise Exception(f"No audio data received (got {len(audio_data)} bytes)")
                    
            except asyncio.TimeoutError:
                logger.warning(f"?? TTS timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Progressive delay
            except Exception as e:
                logger.warning(f"?? TTS attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Progressive delay
                    
                    # Try different voice on last retry
                    if attempt == max_retries - 2 and self.available_voices:
                        old_voice = self.voice
                        # Try a different English voice
                        english_voices = [v for v in self.available_voices if v.startswith("en-") and v != self.voice]
                        if english_voices:
                            self.voice = english_voices[0]
                            logger.info(f"?? Switching voice from {old_voice} to {self.voice}")
        
        logger.error(f"? All TTS attempts failed for chunk: '{chunk[:50]}...'")
        return b""
    
    async def _generate_silent_audio(self, duration_seconds: float) -> bytes:
        """Generate silent WAV audio for fallback"""
        try:
            import struct
            
            sample_rate = 44100
            samples = int(sample_rate * duration_seconds)
            
            # Create WAV header
            data_size = samples * 2  # 16-bit samples
            file_size = data_size + 36
            
            wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                b'RIFF', file_size, b'WAVE', b'fmt ', 16,
                1, 1, sample_rate, sample_rate * 2, 2, 16,
                b'data', data_size
            )
            
            # Silent audio data
            silent_data = b'\x00\x00' * samples
            
            return wav_header + silent_data
            
        except Exception as e:
            logger.error(f"? Silent audio generation failed: {e}")
            return b'\x00' * 1024  # Minimal fallback
    
    async def health_check(self) -> dict:
        """Check TTS system health"""
        try:
            await self._check_voice_availability()
            
            # Quick test
            test_audio = await self._generate_chunk_audio_with_retry("Test", max_retries=1)
            
            return {
                "status": "healthy" if test_audio else "degraded",
                "provider": "EdgeTTS",
                "voice": self.voice,
                "available_voices": len(self.available_voices) if self.available_voices else 0
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "EdgeTTS",
                "error": str(e)
            }