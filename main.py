"""
Main entry point for the Qwen3-TTS Audiobook application.
Connects the UI with the TTS engine and manages the application lifecycle.
"""
import sys
import os
from PySide6.QtWidgets import QApplication, QMessageBox
from ui_main import MainWindow
from tts_engine import TTSWorker

class AudiobookApp(MainWindow):
    """
    Controller class that inherits from MainWindow.
    Handles the orchestration between user inputs and the transcription worker.
    """
    def __init__(self):
        """
        Initializes the application and connects UI signals to controller logic.
        """
        super().__init__()
        self.worker = None
        self.btn_start.clicked.connect(self.start_transcription)

    def start_transcription(self):
        """
        Gathers configuration from the UI and starts the TTSWorker thread.
        Validates input paths and text content before starting.
        """
        # Check active tab
        is_text_mode = self.tabs.currentIndex() == 1
        direct_text = None
        input_path = None
        
        if is_text_mode:
            direct_text = self.direct_text_input.toPlainText()
            if not direct_text.strip():
                QMessageBox.warning(self, "Error", "Por favor ingresa algún texto para transcribir.")
                return
        else:
            input_path = self.input_file_path.text()
            if not input_path or not os.path.exists(input_path):
                QMessageBox.warning(self, "Error", "Por favor selecciona un archivo de entrada válido.")
                return
        
        output_dir = self.output_dir_path.text()
        if not output_dir or not os.path.exists(output_dir):
            QMessageBox.warning(self, "Error", "Por favor selecciona una carpeta de salida válida.")
            return

        mode = self.mode_combo.currentText()
        size = "1.7B" if "1.7B" in self.size_combo.currentText() else "0.6B"
        
        clone_audio = None
        if mode == "VoiceClone":
            clone_audio = self.clone_audio_path.text()
            if not clone_audio or not os.path.exists(clone_audio):
                QMessageBox.warning(self, "Error", "Por favor selecciona un archivo de audio de referencia para la clonación.")
                return

        config = {
            "speaker": self.speaker_combo.currentText(),
            "language": self.lang_combo.currentText(),
            "speed": self.speed_combo.currentText(),
            "instruct": self.prompt_input.text() if mode == "VoiceDesign" else "",
            "size": size,
            "clone_audio": clone_audio
        }

        self.btn_start.setEnabled(False)
        self.log(f"--- Iniciando proceso ({size}, {mode}) ---")
        
        self.worker = TTSWorker(input_path, output_dir, mode, config, direct_text=direct_text)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def update_progress(self, value):
        """
        Updates the UI progress bar.
        Args:
            value (int): Progress percentage (0-100).
        """
        self.progress_bar.setValue(value)

    def on_finished(self, output_path):
        """
        Callback for when the synthesis worker successfully finishes.
        Updates the UI, logs the result, and loads the audio into the player.
        Args:
            output_path (str): The path to the generated MP3 file.
        """
        self.btn_start.setEnabled(True)
        self.progress_bar.setValue(100)
        self.log(f"--- Proceso finalizado ---")
        self.log(f"Archivo guardado en: {output_path}")
        QMessageBox.information(self, "Éxito", f"Transcripción completada con éxito.\nArchivo: {output_path}")
        # Load into the built-in player (inherited from MainWindow)
        self.load_audio(output_path)

    def on_error(self, message):
        """
        Callback for when an error occurs during synthesis.
        Displays an error dialog and re-enables the UI.
        Args:
            message (str): The error message.
        """
        self.log(f"ERROR: {message}")
        QMessageBox.critical(self, "Error", f"Ocurrió un error:\n{message}")
        self.btn_start.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudiobookApp()
    window.show()
    sys.exit(app.exec())
