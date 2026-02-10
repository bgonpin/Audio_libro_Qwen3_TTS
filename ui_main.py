"""
UI module for the Qwen3-TTS Audiobook application.
Defines the main window, layouts, and user interactions.
"""
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QComboBox, QProgressBar, QTextEdit, QLineEdit,
                             QGroupBox, QFormLayout, QTabWidget, QSlider)
from PySide6.QtCore import Qt, Signal, Slot, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import sys
import os

class MainWindow(QMainWindow):
    """
    The main window of the application. 
    Manages the PySide6 UI components, including tabs for input, 
    model configuration, progress logs, and the built-in audio player.
    """
    def __init__(self):
        """
        Initializes the main window, creates the layout, and sets up the multimedia player.
        """
        super().__init__()
        self.setWindowTitle("Audio Libro Qwen3-TTS")
        self.setMinimumSize(800, 700)
        
        # Audio Player Backend
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        # Connect player signals
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.playbackStateChanged.connect(self.update_play_pause_button)
        
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # --- Input Selection Tabs ---
        self.tabs = QTabWidget()
        
        # Tab 1: File Selection
        self.file_tab = QWidget()
        file_tab_layout = QFormLayout(self.file_tab)
        
        self.input_file_path = QLineEdit()
        self.input_file_path.setPlaceholderText("Selecciona un archivo de texto...")
        btn_browse_input = QPushButton("Explorar")
        btn_browse_input.clicked.connect(self.browse_input_file)
        
        input_file_layout = QHBoxLayout()
        input_file_layout.addWidget(self.input_file_path)
        input_file_layout.addWidget(btn_browse_input)
        file_tab_layout.addRow("Archivo de entrada:", input_file_layout)
        
        self.tabs.addTab(self.file_tab, "Desde Archivo")
        
        # Tab 2: Direct Text Input
        self.text_tab = QWidget()
        text_tab_layout = QVBoxLayout(self.text_tab)
        self.direct_text_input = QTextEdit()
        self.direct_text_input.setPlaceholderText("Escribe o pega aquí el texto que deseas transcribir...")
        text_tab_layout.addWidget(self.direct_text_input)
        
        self.tabs.addTab(self.text_tab, "Texto Directo")
        
        main_layout.addWidget(self.tabs)
        
        # --- Common Output Selection ---
        output_group = QGroupBox("Carpeta de Salida")
        output_layout = QFormLayout(output_group)
        self.output_dir_path = QLineEdit()
        self.output_dir_path.setPlaceholderText("Selecciona carpeta de salida...")
        btn_browse_output = QPushButton("Explorar")
        btn_browse_output.clicked.connect(self.browse_output_dir)
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_path)
        output_dir_layout.addWidget(btn_browse_output)
        output_layout.addRow("Carpeta de salida:", output_dir_layout)
        main_layout.addWidget(output_group)
        
        # --- Model Configuration Group ---
        config_group = QGroupBox("Configuración del Modelo")
        config_layout = QFormLayout(config_group)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["CustomVoice", "VoiceDesign"])
        self.mode_combo.currentIndexChanged.connect(self.toggle_mode_ui)
        config_layout.addRow("Modo:", self.mode_combo)
        
        # CustomVoice controls
        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems(["Vivian", "Ryan", "Aiden", "Eric", "Serena"])
        self.speaker_row_label = QLabel("Hablante:")
        config_layout.addRow(self.speaker_row_label, self.speaker_combo)
        
        # VoiceDesign controls
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Ej: A deep, resonant male voice, narrator style...")
        self.prompt_input.setVisible(False)
        self.prompt_row_label = QLabel("Descripción de voz:")
        self.prompt_row_label.setVisible(False)
        config_layout.addRow(self.prompt_row_label, self.prompt_input)
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["auto", "english", "chinese", "spanish", "japanese", "french", "german", "italian", "korean", "portuguese", "russian"])
        self.lang_combo.setCurrentText("spanish")
        config_layout.addRow("Idioma:", self.lang_combo)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Muy lento", "Lento", "Normal", "Rápido", "Muy rápido"])
        self.speed_combo.setCurrentText("Normal")
        config_layout.addRow("Velocidad:", self.speed_combo)
        
        main_layout.addWidget(config_group)
        
        # --- Progress and Logs ---
        logs_group = QGroupBox("Progreso y Registros")
        logs_layout = QVBoxLayout(logs_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        logs_layout.addWidget(self.progress_bar)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        logs_layout.addWidget(self.log_area)
        
        main_layout.addWidget(logs_group)
        
        # --- Audio Player Group ---
        player_group = QGroupBox("Reproductor de Audio")
        player_layout = QVBoxLayout(player_group)
        
        player_controls = QHBoxLayout()
        self.btn_play_pause = QPushButton("▶ Reproducir")
        self.btn_stop = QPushButton("■ Detener")
        
        self.time_label = QLabel("00:00 / 00:00")
        
        player_controls.addWidget(self.btn_play_pause)
        player_controls.addWidget(self.btn_stop)
        player_controls.addStretch()
        player_controls.addWidget(self.time_label)
        
        self.player_slider = QSlider(Qt.Horizontal)
        self.player_slider.setRange(0, 0)
        self.player_slider.sliderMoved.connect(self.set_position)
        
        player_layout.addLayout(player_controls)
        player_layout.addWidget(self.player_slider)
        
        main_layout.addWidget(player_group)
        
        # --- Actions ---
        actions_layout = QHBoxLayout()
        self.btn_start = QPushButton("Iniciar Transcripción")
        
        # --- Player Button Connections ---
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_stop.clicked.connect(self.stop_audio)
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        
        actions_layout.addStretch()
        actions_layout.addWidget(self.btn_start)
        main_layout.addLayout(actions_layout)

    def browse_input_file(self):
        """
        Opens a file dialog to select a text or markdown file for transcription.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo de texto", "", "Archivos de texto (*.txt *.md);;Todos los archivos (*)")
        if file_path:
            self.input_file_path.setText(file_path)

    def browse_output_dir(self):
        """
        Opens a directory dialog to select where the output MP3 will be saved.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de salida")
        if dir_path:
            self.output_dir_path.setText(dir_path)

    def toggle_mode_ui(self):
        """
        Updates the UI to show/hide controls based on the selected TTS mode (CustomVoice/VoiceDesign).
        """
        is_voice_design = self.mode_combo.currentText() == "VoiceDesign"
        self.speaker_combo.setVisible(not is_voice_design)
        self.speaker_row_label.setVisible(not is_voice_design)
        self.prompt_input.setVisible(is_voice_design)
        self.prompt_row_label.setVisible(is_voice_design)

    def log(self, message):
        """
        Appends a message to the log text area.
        Args:
            message (str): The message to display.
        """
        self.log_area.append(message)

    def toggle_play_pause(self):
        """
        Toggles playback of the built-in media player.
        """
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def stop_audio(self):
        """
        Stops the built-in media player.
        """
        self.media_player.stop()

    def update_play_pause_button(self, state):
        """
        Updates the Play/Pause button text based on the player state.
        """
        if state == QMediaPlayer.PlayingState:
            self.btn_play_pause.setText("⏸ Pausar")
        else:
            self.btn_play_pause.setText("▶ Reproducir")

    def update_position(self, position):
        """
        Updates the player slider and time label as the audio plays.
        """
        if not self.player_slider.isSliderDown():
            self.player_slider.setValue(position)
        self.update_time_label(position, self.media_player.duration())

    def update_duration(self, duration):
        """
        Updates the player slider range when a new audio file is loaded.
        """
        self.player_slider.setRange(0, duration)
        self.update_time_label(self.media_player.position(), duration)

    def set_position(self, position):
        """
        Seeks to a specific position in the audio when the slider is moved.
        """
        self.media_player.setPosition(position)

    def update_time_label(self, current, total):
        """
        Formats and updates the time indicator label (e.g., 01:23 / 05:00).
        """
        def format_ms(ms):
            s = ms // 1000
            m = s // 60
            s = s % 60
            return f"{m:02d}:{s:02d}"
        self.time_label.setText(f"{format_ms(current)} / {format_ms(total)}")

    def load_audio(self, file_path):
        """
        Loads an MP3 file into the built-in media player.
        Args:
            file_path (str): The full path to the audio file.
        """
        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        self.log(f"Audio cargado en el reproductor: {os.path.basename(file_path)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
