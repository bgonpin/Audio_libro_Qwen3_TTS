"""
Engine module for the Qwen3-TTS Audiobook application.
Handles background synthesis processing, text fragmentation, and audio saving.
"""
import os
import torch
import soundfile as sf
from PySide6.QtCore import QThread, Signal
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel = None

class TTSWorker(QThread):
    """
    Worker thread that handles the TTS generation process using Qwen3-TTS.
    It splits text into chunks, generates audio for each, and concatenates them.
    Signals:
        progress (int): Percentage of progress in text synthesis.
        log (str): Status messages for the UI logs.
        finished (str): Emitted when synthesis is complete, with the path to the resulting file.
        error (str): Emitted if an error occurs during processing.
    """
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, input_path, output_dir, mode, config, direct_text=None):
        """
        Initializes the TTS worker.
        Args:
            input_path (str): Path to the input file (if any).
            output_dir (str): Directory where the result will be saved.
            mode (str): "CustomVoice" or "VoiceDesign".
            config (dict): Model parameters (speaker, language, speed, instruct).
            direct_text (str, optional): Direct text input transcribed instead of a file.
        """
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.mode = mode  # "CustomVoice" or "VoiceDesign"
        self.config = config
        self.direct_text = direct_text
        self.is_running = True
        self.model = None

    def run(self):
        """
        Main execution loop for the worker thread.
        Loads the model, processes text chunks, generates audio, and saves the final MP3.
        """
        try:
            if Qwen3TTSModel is None:
                self.error.emit("Librería qwen-tts no encontrada.")
                return

            self.log.emit(f"Cargando modelo {self.mode}...")
            
            model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" if self.mode == "CustomVoice" else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            
            # Check for flash_attn availability
            has_flash_attn = False
            try:
                import flash_attn
                has_flash_attn = True
                self.log.emit("FlashAttention-2 detectado y habilitado.")
            except ImportError:
                self.log.emit("FlashAttention-2 no detectado. Usando SDPA (fallback).")
            
            attn_impl = "flash_attention_2" if (torch.cuda.is_available() and has_flash_attn) else "sdpa"

            # Load model
            self.model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl
            )
            self.log.emit("Modelo cargado correctamente.")

            # Read text
            if self.direct_text:
                text = self.direct_text
            else:
                with open(self.input_path, 'r', encoding='utf-8') as f:
                    text = f.read()

            if not text.strip():
                self.error.emit("El texto de entrada está vacío.")
                return

            # Simple chunking
            chunks = self.split_text(text, max_chars=500)
            self.log.emit(f"Texto dividido en {len(chunks)} fragmentos.")

            # Prepare speed instruction
            speed_map = {
                "Muy lento": "hablar muy lentamente, ",
                "Lento": "hablar lento, ",
                "Normal": "",
                "Rápido": "hablar rápido, ",
                "Muy rápido": "hablar muy rápido, "
            }
            speed_prefix = speed_map.get(self.config.get("speed", "Normal"), "")
            
            base_instruct = self.config.get("instruct", "")
            full_instruct = f"{speed_prefix}{base_instruct}"

            all_audio = []
            sample_rate = 24000

            for i, chunk in enumerate(chunks):
                if not self.is_running:
                    break
                
                self.log.emit(f"Procesando fragmento {i+1}/{len(chunks)}...")
                
                if self.mode == "CustomVoice":
                    audio, sr = self.model.generate_custom_voice(
                        text=chunk,
                        speaker=self.config.get("speaker", "Vivian"),
                        language=self.config.get("language", "auto"),
                        instruct=full_instruct
                    )
                else: # VoiceDesign
                    audio, sr = self.model.generate_voice_design(
                        text=chunk,
                        instruct=full_instruct,
                        language=self.config.get("language", "auto")
                    )
                
                sample_rate = sr
                
                # Ensure audio is a numpy array
                if hasattr(audio, 'detach'):
                    audio = audio.detach().cpu().numpy()
                elif isinstance(audio, list):
                    import numpy as np
                    audio = np.array(audio)
                
                # Flatten to 1D to avoid concatenation issues if shapes are (1, N)
                import numpy as np
                if isinstance(audio, np.ndarray) and audio.ndim > 1:
                    audio = audio.flatten()
                
                all_audio.append(audio)
                self.progress.emit(int(((i + 1) / len(chunks)) * 100))

            if all_audio:
                import numpy as np
                final_audio = np.concatenate(all_audio)
                
                # Save as MP3 using pydub
                self.log.emit("Convirtiendo a MP3...")
                try:
                    from pydub import AudioSegment
                    # Convert float32 numpy to int16 (standard for AudioSegment)
                    # Qwen-TTS generates float32 in range [-1, 1]
                    audio_int16 = (final_audio * 32767).astype(np.int16)
                    
                    audio_segment = AudioSegment(
                        audio_int16.tobytes(), 
                        frame_rate=sample_rate,
                        sample_width=2, # 16-bit
                        channels=1
                    )
                    
                    output_base = "direct_text_output"
                    if self.input_path:
                        output_base = os.path.splitext(os.path.basename(self.input_path))[0]
                    
                    output_filename = os.path.join(self.output_dir, f"{output_base}.mp3")
                    audio_segment.export(output_filename, format="mp3")
                    self.finished.emit(output_filename)
                except ImportError:
                    self.log.emit("Pydub no encontrado, guardando como WAV por defecto.")
                    output_base = "direct_text_output"
                    if self.input_path:
                        output_base = os.path.splitext(os.path.basename(self.input_path))[0]
                    
                    output_filename = os.path.join(self.output_dir, f"{output_base}.wav")
                    sf.write(output_filename, final_audio, sample_rate)
                    self.finished.emit(output_filename)

        except Exception as e:
            self.error.emit(str(e))

    def split_text(self, text, max_chars=500):
        """
        Splits text into chunks of maximum max_chars, trying to respect sentence boundaries.
        Uses regex to detect sentence ends (. ? ! \n).
        
        Args:
            text (str): The raw text to fragment.
            max_chars (int): The maximum character count per chunk.
            
        Returns:
            list: A list of text strings (chunks).
        """
        import re
        # Regex to split by . ? ! followed by space or newline, or just newlines
        # Keeps the delimiter with the preceding sentence
        sentences = re.split(r'(?<=[.?!])\s+|\n+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If a single sentence is longer than max_chars, we must break it down
            if len(sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Break long sentence by words
                words = sentence.split(' ')
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > max_chars:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                    else:
                        temp_chunk += word + " "
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                # Normal case: sentence fits or we add to current chunk
                if len(current_chunk) + len(sentence) + 1 > max_chars:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    current_chunk += sentence + " "
                    
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def stop(self):
        """
        Request the worker thread to stop its execution loop.
        """
        self.is_running = False
