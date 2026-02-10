#!/bin/bash
# Script para lanzar la aplicación Audio Libro Qwen3-TTS

ENV_NAME="qwen3-tts-audiobook"

# Función para comprobar si el entorno conda existe
check_env_exists() {
    conda env list | grep -q "^$ENV_NAME "
    return $?
}

# Inicializar conda para el script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! check_env_exists; then
    echo "El entorno conda '$ENV_NAME' no existe. Creándolo..."
    conda create -n "$ENV_NAME" python=3.12 -y
    
    echo "Instalando dependencias necesarias (esto puede tardar unos minutos)..."
    conda run -n "$ENV_NAME" pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    conda run -n "$ENV_NAME" pip install -U qwen-tts PySide6 soundfile pydub
    
    echo "Intentando instalar FlashAttention-2 para mayor velocidad (opcional)..."
    conda run -n "$ENV_NAME" pip install flash-attn --no-build-isolation || echo "Advertencia: No se pudo instalar FlashAttention-2 automáticamente. Se usará el modo estándar (SDPA)."
    
    echo "Entorno configurado correctamente."
fi

# Activar el entorno conda
conda activate "$ENV_NAME"

# Ejecutar la aplicación
echo "Iniciando la aplicación..."
python main.py
