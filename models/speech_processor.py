import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from TTS.api import TTS
import numpy as np
from typing import Optional, Dict, Union, List
import logging

class SpeechProcessor(nn.Module):
    """
    Handles speech-to-text and text-to-speech processing using Whisper and TTS models
    """
    def __init__(
        self,
        whisper_model: str = "openai/whisper-small",
        tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        supported_languages: List[str] = ["en", "es", "fr", "de", "it", "ja", "ko", "zh"]
    ):
        super().__init__()
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.supported_languages = supported_languages
        
        # Initialize Whisper for STT
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            whisper_model
        ).to(device)
        
        # Initialize TTS
        self.tts_model = TTS(tts_model).to(device)
        
        # Audio preprocessing
        self.sample_rate = 16000
        self.audio_transform = torchaudio.transforms.Resample(
            orig_freq=44100,
            new_freq=self.sample_rate
        )
    
    def transcribe_audio(
        self,
        audio_input: Union[str, torch.Tensor],
        source_lang: str = "en"
    ) -> Dict[str, str]:
        """
        Convert speech to text using Whisper
        Args:
            audio_input: Path to audio file or audio tensor
            source_lang: Source language code
        Returns:
            Dict containing transcription and detected language
        """
        try:
            if isinstance(audio_input, str):
                waveform, sample_rate = torchaudio.load(audio_input)
                if sample_rate != self.sample_rate:
                    waveform = self.audio_transform(waveform)
            else:
                waveform = audio_input
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Process audio with Whisper
            input_features = self.whisper_processor(
                waveform.squeeze().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
                language=source_lang,
                task="transcribe"
            )
            
            generated_ids = self.whisper_model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448
            )
            
            transcription = self.whisper_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            return {
                "text": transcription,
                "detected_language": source_lang
            }
            
        except Exception as e:
            self.logger.error(f"Error in speech transcription: {str(e)}")
            return {"error": str(e)}
    
    def synthesize_speech(
        self,
        text: str,
        output_path: Optional[str] = None,
        target_lang: str = "en",
        speaker: Optional[str] = None
    ) -> Union[np.ndarray, str]:
        """
        Convert text to speech using TTS
        Args:
            text: Input text to synthesize
            output_path: Optional path to save audio file
            target_lang: Target language code
            speaker: Optional speaker identifier for multi-speaker models
        Returns:
            Audio waveform array or path to saved audio file
        """
        try:
            if target_lang not in self.supported_languages:
                raise ValueError(f"Language {target_lang} not supported")
            
            # Generate speech
            wav = self.tts_model.tts(
                text=text,
                language=target_lang,
                speaker=speaker
            )
            
            if output_path:
                self.tts_model.save_wav(wav, output_path)
                return output_path
            
            return wav
            
        except Exception as e:
            self.logger.error(f"Error in speech synthesis: {str(e)}")
            return {"error": str(e)}
    
    def translate_speech(
        self,
        audio_input: Union[str, torch.Tensor],
        source_lang: str,
        target_lang: str
    ) -> Dict[str, str]:
        """
        Translate speech from one language to another
        Args:
            audio_input: Path to audio file or audio tensor
            source_lang: Source language code
            target_lang: Target language code
        Returns:
            Dict containing original and translated text
        """
        try:
            # First transcribe the audio
            transcription = self.transcribe_audio(audio_input, source_lang)
            
            if "error" in transcription:
                return transcription
            
            # Use the language model manager for translation
            # This requires integration with the language model manager
            # TODO: Implement translation using language model manager
            
            return {
                "original_text": transcription["text"],
                "translated_text": "Translation not implemented yet",
                "source_lang": source_lang,
                "target_lang": target_lang
            }
            
        except Exception as e:
            self.logger.error(f"Error in speech translation: {str(e)}")
            return {"error": str(e)}
    
    def process_continuous_audio(
        self,
        audio_stream,
        callback_fn=None,
        buffer_size: int = 4096
    ):
        """
        Process continuous audio stream for real-time transcription
        Args:
            audio_stream: Audio input stream
            callback_fn: Function to call with transcription results
            buffer_size: Size of audio buffer to process
        """
        try:
            buffer = []
            
            for chunk in audio_stream:
                buffer.append(chunk)
                
                if len(buffer) * buffer_size >= self.sample_rate * 2:  # Process every 2 seconds
                    audio_data = torch.cat(buffer, dim=1)
                    transcription = self.transcribe_audio(audio_data)
                    
                    if callback_fn and "text" in transcription:
                        callback_fn(transcription["text"])
                    
                    buffer = []
                    
        except Exception as e:
            self.logger.error(f"Error in continuous audio processing: {str(e)}")
            return {"error": str(e)}
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.supported_languages
