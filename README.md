# 🎙️ Qwen3-TTS Audiobook Creator

Una potente herramienta de escritorio para convertir libros y textos en audio de alta calidad utilizando el modelo de última generación **Qwen3-TTS-1.7B**. Diseñada para ofrecer una experiencia de lectura natural, fluida y totalmente personalizada.

![Interfaz de la Aplicación](https://img.shields.io/badge/UI-PySide6-blue)
![Modelo](https://img.shields.io/badge/Model-Qwen3--TTS--1.7B-green)
![Licencia](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Características Principales

- **📖 Entrada Flexible**: Procesa archivos de texto completo (`.txt`, `.md`) o simplemente pega texto directamente en la pestaña dedicada.
- **🔊 Voces de Alta Calidad**:
  - **CustomVoice**: Elige entre 9 voces predefinidas premium (Vivian, Ryan, Aiden, Eric, Serena, etc.).
  - **VoiceDesign**: Define tu propia voz mediante una descripción textual (ej: "A deep, resonant male voice, narrator style").
- **⚡ Velocidad Variable**: Ajusta el ritmo de la narración con 5 niveles (Muy lento, Lento, Normal, Rápido, Muy rápido).
- **🧩 Fragmentación Inteligente**: Sistema de división de texto avanzado que respeta oraciones y signos de puntuación para una entonación natural.
- **🎵 Reproductor Integrado**: Escucha tus audiolibros directamente en la aplicación sin necesidad de software externo.
- **💾 Exportación Directa**: Genera archivos `.mp3` optimizados con nombres dinámicos basados en la entrada.

---

## 🛠️ Tecnología Empleada

- **Lenguaje**: Python 3.12
- **IA/ML**:
  - [Qwen-TTS](https://github.com/Qwen-AI/Qwen-TTS): Modelo base de 1.7 mil millones de parámetros.
  - **PyTorch**: Motor de inferencia.
  - **FlashAttention-2**: Optimización de velocidad (opcional).
- **Interfaz Gráfica**: PySide6 (Qt para Python).
- **Procesamiento de Audio**:
  - **Pydub**: Para la gestión y exportación en formato MP3.
  - **Soundfile & PyAudio**: Para manejo de buffers de audio.
  - **FFmpeg**: Backend necesario para la conversión de formatos.

---

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/Audio_libro_Qwen3_TTS.git
cd Audio_libro_Qwen3_TTS
```

### 2. Dependencias del Sistema (Linux)
Asegúrate de tener instalados los códecs de audio y las herramientas de compilación:
```bash
sudo apt update
sudo apt install ffmpeg sox libsox-fmt-all nvidia-cuda-toolkit -y
```

### 3. Configuración del Entorno
Hemos automatizado la creación del entorno Conda y la instalación de dependencias en un solo script:
```bash
chmod +x run_app.sh
./run_app.sh
```
*Este script creará el entorno `qwen3-tts-audiobook` e instalará los ~3.5GB de modelos necesarios en la primera ejecución.*

---

## 📖 Manual de Usuario

### Paso 1: Selección de Entrada
- **Pestaña "Desde Archivo"**: Haz clic en "Explorar" y selecciona un libro o documento.
- **Pestaña "Texto Directo"**: Pega el fragmento que quieras escuchar inmediatamente.

### Paso 2: Configuración de Salida
- Selecciona la **Carpeta de Salida** donde se guardará tu archivo `.mp3`.

### Paso 3: Personalización de Voz
- En el modo **CustomVoice**, selecciona tu narrador favorito.
- En el modo **VoiceDesign**, escribe una descripción detallada de cómo quieres que suene la voz.
- Elige el **Idioma** (se recomienda "auto" para detección automática) y la **Velocidad**.

### Paso 4: Generación
- Haz clic en **"Iniciar Transcripción"**.
- Sigue el progreso visual en la barra y en los registros en tiempo real.
- Al finalizar, aparecerá un aviso de éxito y el audio se cargará en el reproductor.

### Paso 5: Reproducción
- Usa el reproductor integrado en la parte inferior para escuchar la obra. Puedes pausar, detener o saltar a cualquier minuto usando la barra deslizante.

---

## 💡 Consejos de Rendimiento

> [!IMPORTANT]
> **Aceleración GPU**: Para obtener la mejor velocidad, asegúrate de tener una tarjeta NVIDIA y los drivers instalados. La aplicación detectará automáticamente tu GPU.

> [!TIP]
> **FlashAttention-2**: Si tienes una GPU moderna (RTX serie 3000 o 4000) y el kit de CUDA instalado, la aplicación irá hasta 4 veces más rápido.

---

## 📄 Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---
*Desarrollado con ❤️ para amantes de la lectura y la tecnología.*
