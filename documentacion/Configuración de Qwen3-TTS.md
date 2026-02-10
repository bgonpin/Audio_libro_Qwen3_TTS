

# Guía de Instalación y Configuración de Qwen3-TTS en Ubuntu con GPU NVIDIA RTX 5060Ti

## 1. Requisitos del Sistema y Consideraciones de Hardware

### 1.1 Especificaciones de Hardware Recomendadas

La implementación exitosa de Qwen3-TTS en un entorno de producción o desarrollo local requiere una evaluación meticulosa de los recursos computacionales disponibles, particularmente cuando se trabaja con modelos de lenguaje de voz que procesan información acústica y lingüística de manera simultánea. La arquitectura de Qwen3-TTS, desarrollada por el equipo de Tongyi Qianwen de Alibaba Cloud, ha sido optimizada para ofrecer rendimiento excepcional incluso en hardware de gama media, lo que la distingue significativamente de otras soluciones de síntesis de voz que típicamente demandan recursos computacionales prohibitivos. La familia de modelos Qwen3-TTS presenta dos variantes principales en términos de tamaño de parámetros: la versión **1.7B**, que ofrece la máxima calidad de síntesis y expresividad prosódica, y la versión **0.6B**, diseñada para escenarios donde la velocidad de inferencia y el consumo de recursos son prioritarios sobre la fidelidad absoluta de la voz generada.

#### 1.1.1 GPU NVIDIA RTX 5060Ti: 16 GB VRAM (óptimo para modelos 1.7B)

La **NVIDIA RTX 5060Ti** representa una elección excepcional para el despliegue de Qwen3-TTS, particularmente la variante de **16 GB de memoria de video**, que proporciona un margen de seguridad considerable para operar los modelos 1.7B en precisión bfloat16 sin riesgo de agotamiento de memoria. Según las especificaciones técnicas documentadas, la RTX 5060Ti incorpora la arquitectura **Blackwell GB206** con **4,608 núcleos CUDA** de quinta generación, ofreciendo **759 TOPS** de rendimiento en operaciones INT8 y **23.7 TFLOPS** en precisión FP16, lo que representa un incremento del **57% en throughput de inferencia mixta** comparado con generaciones anteriores como la RTX 4060. Esta capacidad computacional es particularmente relevante para la síntesis de voz en tiempo real, donde la latencia de primer token y el throughput sostenido determinan la viabilidad de aplicaciones interactivas.

La memoria **GDDR7 de 16 GB** operando a **28 Gbps**, combinada con un bus de **128 bits** que proporciona **448 GB/s de ancho de banda**, garantiza que los pesos del modelo y los estados de activación intermedios puedan residir eficientemente en la memoria de video sin necesidad de paginación hacia la memoria del sistema, eliminando cuellos de botella críticos en la inferencia. El caché L2 de **32 MB** mejora significativamente la tasa de aciertos para el caché KV, elemento fundamental en la arquitectura de atención de los modelos transformer que subyacen a Qwen3-TTS. Para la variante 1.7B en precisión bfloat16, el consumo típico de VRAM se sitúa entre **5.0 y 5.7 GB** durante la generación de voz personalizada, dejando amplio margen para batch processing o múltiples instancias concurrentes del modelo.

La compatibilidad con **PCIe 5.0 x8** duplica el ancho de banda de transferencia entre CPU y GPU respecto a PCIe 4.0, acelerando la carga inicial de pesos de modelo y la transferencia de datos de audio de referencia en operaciones de clonación de voz. El soporte nativo para **FP4 y FP8** en los Tensor Cores de quinta generación permite futuras optimizaciones de cuantización que podrían reducir aún más el consumo de memoria sin degradación perceptible de calidad, aunque la implementación actual de Qwen3-TTS se beneficia primordialmente de la precisión bfloat16.

#### 1.1.2 RAM del Sistema: 16 GB mínimo recomendado

La memoria del sistema desempeña un papel crítico en el ecosistema de inferencia de Qwen3-TTS, extendiéndose más allá del mero almacenamiento de pesos de modelo. Los **16 GB de RAM recomendados** permiten gestionar eficientemente múltiples procesos concurrentes: el runtime de Python con sus dependencias, el sistema operativo y servicios de fondo, buffers de audio para preprocesamiento y postprocesamiento, y especialmente la memoria requerida durante la compilación de kernels CUDA para operaciones de atención optimizadas. La instalación de **FlashAttention 2**, altamente recomendada para maximizar el rendimiento, puede consumir temporalmente hasta **8 GB de RAM** durante el proceso de compilación, particularmente cuando se utiliza el flag `--no-build-isolation` que permite el uso de caché de compilación persistente.

En configuraciones con RAM limitada, la instalación de FlashAttention puede fallar con errores de agotamiento de memoria del sistema operativo, situación que se mitiga mediante la variable de entorno `MAX_JOBS=4` que limita el paralelismo de compilación. Adicionalmente, la carga de modelos desde almacenamiento local o descarga inicial desde repositorios remotos beneficia de RAM suficiente para caching de archivos y operaciones de descompresión. Para flujos de trabajo que involucran procesamiento de audio de referencia en formatos de alta resolución o duración extendida, la RAM adicional facilita operaciones de resampling, normalización y extracción de características acústicas sin presión de memoria.

#### 1.1.3 Almacenamiento: 3-5 GB para pesos de modelo, preferiblemente SSD NVMe

Los requisitos de almacenamiento para Qwen3-TTS, aunque modestos en comparación con modelos de lenguaje de gran escala, merecen atención cuidadosa para garantizar una experiencia de usuario fluida. Los pesos de modelo para la variante 1.7B ocupan aproximadamente **3.4 GB** en precisión bfloat16, mientras que la variante 0.6B requiere aproximadamente **1.2 GB**. Sin embargo, la estructura completa de archivos de modelo, incluyendo tokenizadores, configuraciones de arquitectura, embeddings de voces predefinidas y caché de compilación de kernels, puede extenderse hasta **5 GB** por variante completa.

El uso de almacenamiento **SSD NVMe** con latencias de acceso inferiores a **100 microsegundos** y throughput sostenido superior a **3 GB/s** elimina la latencia perceptible durante la carga inicial del modelo, operación que típicamente ocurre una vez por sesión de trabajo pero que en entornos de desarrollo iterativo puede repetirse frecuentemente. La organización de directorios recomendada separa los modelos por funcionalidad: `Qwen3-TTS-12Hz-1.7B-Base` para clonación de voz, `Qwen3-TTS-12Hz-1.7B-VoiceDesign` para diseño de voces mediante descripción textual, y `Qwen3-TTS-12Hz-1.7B-CustomVoice` para síntesis con voces predefinidas, facilitando la gestión de múltiples capacidades sin duplicación innecesaria de archivos compartidos como tokenizadores.

### 1.2 Requisitos de Software Base

La stack tecnológica de Qwen3-TTS se construye sobre fundamentos de software maduros y ampliamente adoptados en la comunidad de aprendizaje profundo, con dependencias específicas de versión que garantizan compatibilidad y reproducibilidad. La selección cuidadosa de versiones de componentes críticos —Python, CUDA, PyTorch y bibliotecas auxiliares— constituye un factor determinante para evitar frustrantes ciclos de depuración de incompatibilidades.

#### 1.2.1 Sistema Operativo: Ubuntu 22.04 LTS o superior

**Ubuntu 22.04 LTS (Jammy Jellyfish)** proporciona el equilibrio óptimo entre estabilidad de largo plazo, soporte de hardware moderno y disponibilidad de paquetes de software actualizados. El kernel Linux 5.15 incluido ofrece drivers de NVIDIA optimizados y soporte robusto para las capacidades de computación de la arquitectura Blackwell. Para despliegues en infraestructura cloud, imágenes preconfiguradas como "Ubuntu Server 22.04 LTS R535 CUDA 12.2 with Docker" aceleran significativamente la preparación del entorno al incluir drivers NVIDIA versión 535 y CUDA Toolkit 12.2 preinstalados y verificados. La compatibilidad con versiones posteriores de Ubuntu está garantizada dado el uso de estándares POSIX y APIs de Linux ampliamente soportadas.

#### 1.2.2 Python: Versión 3.12 (recomendado para compatibilidad)

La **versión 3.12 de Python** introduce optimizaciones significativas en el rendimiento del intérprete, reducciones en el consumo de memoria de objetos pequeños, y mejoras en la precisión de mensajes de error que facilitan el desarrollo y depuración. Críticamente, el paquete `qwen-tts` ha sido validado exhaustivamente contra **Python 3.12**, evitando problemas de compatibilidad que pueden surgir con versiones más recientes donde las dependencias de extensiones C pueden no estar completamente actualizadas. La creación de entornos virtuales mediante Conda o venv con Python 3.12 garantiza un aislamiento limpio de dependencias del sistema y reproducibilidad entre diferentes máquinas de desarrollo y producción.

#### 1.2.3 CUDA: Versión 12.2 o compatible con PyTorch cu128

La arquitectura de computación paralela de NVIDIA CUDA constituye el fundamento de aceleración GPU para Qwen3-TTS. La **versión 12.2 del CUDA Toolkit** proporciona el compilador nvcc, bibliotecas matemáticas optimizadas, y APIs de runtime necesarias para la ejecución de kernels PyTorch compilados para esta generación. La correspondencia entre CUDA Toolkit, drivers NVIDIA y versiones de PyTorch sigue una matriz de compatibilidad estricta: **PyTorch 2.5+ con soporte CUDA 12.8 (`cu128`)** requiere **drivers NVIDIA 535 o superior**, disponibles en la RTX 5060Ti. La instalación de PyTorch mediante el índice específico `https://download.pytorch.org/whl/cu128` garantiza la obtención de binarios precompilados con las optimizaciones de arquitectura Blackwell.

#### 1.2.4 Drivers NVIDIA: Versión 535 o superior

Los **drivers de dispositivo NVIDIA versión 535** introducen soporte completo para la arquitectura Ada Lovelace y Blackwell, incluyendo las RTX 40-series y 50-series, junto con optimizaciones en la gestión de memoria GPU y nuevas primitivas de sincronización relevantes para cargas de trabajo de inferencia. La verificación de instalación correcta mediante `nvidia-smi` debe mostrar la versión del driver, la versión de CUDA runtime disponible, y el estado de todos los dispositivos GPU detectados, confirmando que el subsistema de kernel de NVIDIA está operativo y accesible desde espacio de usuario.

## 2. Preparación del Entorno de Desarrollo

### 2.1 Configuración de Drivers y CUDA

La preparación meticulosa del entorno de software de bajo nivel —drivers de kernel, bibliotecas de runtime CUDA, y herramientas de desarrollo— establece las bases para una experiencia de desarrollo productiva y libre de obstáculos técnicos. Los problemas en esta etapa inicial se manifiestan típicamente como errores crípticos durante la importación de PyTorch o fallos silenciosos en la detección de GPU, consumiendo tiempo valioso de depuración que puede evitarse con verificación sistemática.

#### 2.1.1 Verificación de instalación de drivers NVIDIA: `nvidia-smi`

El comando `nvidia-smi` (NVIDIA System Management Interface) proporciona una ventana completa al estado del subsistema GPU, incluyendo información de versión de driver, versión de CUDA runtime, temperatura, utilización de GPU y memoria, y procesos activos. Una ejecución exitosa que muestra la RTX 5060Ti con sus 16 GB de memoria confirma que el driver de kernel está correctamente cargado, el módulo `nvidia` está presente en el kernel, y la interfaz de ioctl entre espacio de usuario y kernel está operativa. La salida típica incluye líneas como:

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.154.05             Driver Version: 535.154.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     Off | 00000000:01:00.0 Off |                  N/A |
|  0%   35C    P8              12W / 180W |    256MiB / 16384MiB |      0%      Default |
+-----------------------------------------+----------------------+----------------------+
```

La discrepancia entre la versión de CUDA reportada por `nvidia-smi` (versión de runtime del driver) y la versión del CUDA Toolkit instalado (`nvcc --version`) es normal y esperada; lo crítico es que la versión del toolkit no exceda la soportada por el driver.

#### 2.1.2 Instalación de CUDA Toolkit 12.2 si no está presente

Para sistemas donde CUDA Toolkit no está preinstalado, el proceso de instalación desde los repositorios oficiales de NVIDIA garantiza la obtención de componentes verificados y actualizaciones de seguridad. El procedimiento recomendado para Ubuntu 22.04 involucra:

```bash
# Actualización de repositorios e instalación de dependencias
sudo apt update && sudo apt install -y build-essential

# Descarga del instalador de red de CUDA 12.2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Instalación del toolkit completo
sudo apt install -y cuda-toolkit-12-2
```

La instalación completa incluye el compilador nvcc, las bibliotecas cuBLAS, cuDNN (si se selecciona), herramientas de profiling como Nsight, y ejemplos de código. Para entornos de producción mínimos, la selección de paquetes específicos (`cuda-compiler-12-2`, `cuda-libraries-12-2`) reduce la huella de instalación.

#### 2.1.3 Configuración de variables de entorno CUDA en `.bashrc`

La correcta configuración de variables de entorno asegura que herramientas de compilación y runtime localicen automáticamente los componentes de CUDA sin especificación manual de rutas. Las variables esenciales incluyen:

```bash
# Añadir al final de ~/.bashrc
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

La aplicación de cambios mediante `source ~/.bashrc` o reinicio de sesión activa estas configuraciones. La verificación mediante `which nvcc` y `nvcc --version` confirma la correcta exposición del compilador CUDA.

### 2.2 Creación de Entorno Virtual Aislado

El aislamiento de dependencias de Python mediante entornos virtuales constituye una práctica fundamental de ingeniería de software, preventiva de conflictos de versión entre proyectos y facilitadora de reproducibilidad. **Conda**, como gestor de paquetes y entornos, ofrece ventajas adicionales en la gestión de dependencias binarias no-Python y la creación de entornos exportables.

#### 2.2.1 Instalación de Conda o uso de venv nativo

**Miniconda** proporciona una instalación mínima del sistema Conda, descargable mediante:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

La aceptación de términos de servicio para los canales principal y R de Conda evita interrupciones durante la instalación de paquetes:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Alternativamente, el módulo `venv` de Python estándar proporciona aislamiento sin dependencias externas:

```bash
python3.12 -m venv ~/qwen3-tts-env
source ~/qwen3-tts-env/bin/activate
```

#### 2.2.2 Creación del entorno: `conda create -n qwen3-tts python=3.12 -y`

La creación del entorno dedicado `qwen3-tts` con **Python 3.12** establece un namespace aislado donde todas las operaciones posteriores de instalación de paquetes son independientes del sistema Python global y de otros proyectos. El flag `-y` automatiza la confirmación de instalación, facilitando scripting desatendido. La estructura de directorios creada en `~/miniconda3/envs/qwen3-tts/` contiene una instalación completa de Python 3.12, pip, y herramientas asociadas, lista para recibir el ecosistema de dependencias de Qwen3-TTS.

#### 2.2.3 Activación del entorno: `conda activate qwen3-tts`

La activación del entorno modifica las variables de entorno del shell actual, prependiendo la ruta del entorno a `PATH` y estableciendo `CONDA_DEFAULT_ENV`. El prompt del shell típicamente indica el entorno activo mediante prefijo `(qwen3-tts)`. La desactivación mediante `conda deactivate` restaura el entorno original, permitiendo conmutación rápida entre proyectos. Para automatización, la activación puede combinarse con comandos en una línea: `conda run -n qwen3-tts python script.py`.

## 3. Instalación de Dependencias Principales

### 3.1 Instalación de PyTorch con Soporte CUDA

PyTorch constituye el framework de aprendizaje profundo fundamental sobre el cual Qwen3-TTS construye sus capacidades de inferencia. La correcta instalación de PyTorch con soporte CUDA habilitado es crítica; una instalación CPU-only resultaría en ejecución extremadamente lenta en CPU, mientras que una instalación con versión CUDA incompatible generaría errores de carga de bibliotecas compartidas o fallos silenciosos en la detección de GPU.

#### 3.1.1 Comando de instalación específico para CUDA 12.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

El índice de paquetes `cu128` de PyTorch proporciona binarios precompilados específicamente para **CUDA 12.8**, optimizados para las arquitecturas de GPU más recientes incluyendo Blackwell. La especificación explícita del índice evita la instalación de versiones genéricas o CPU-only que podrían seleccionarse por defecto. El comando completo recomendado, incluyendo torchaudio para operaciones de procesamiento de audio, es:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

La instalación de `torchvision` es opcional para operaciones puramente de audio, pero se incluye típicamente por conveniencia en pipelines multimodales. La descarga inicial puede alcanzar **2-3 GB** de binarios comprimidos, expandiéndose a aproximadamente **5 GB** de bibliotecas compartidas y headers de desarrollo.

#### 3.1.2 Verificación de instalación CUDA disponible en PyTorch

La verificación post-instalación confirma que PyTorch puede detectar y comunicarse correctamente con la GPU:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

Una salida exitosa muestra `CUDA available: True`, la versión de CUDA compilada en PyTorch, y el nombre correcto de la RTX 5060Ti. Cualquier discrepancia indica problema en la instalación que debe resolverse antes de continuar.

### 3.2 Instalación del Paquete Qwen3-TTS

El paquete `qwen-tts` distribuido mediante PyPI encapsula la implementación de referencia de Qwen3-TTS, incluyendo definiciones de modelo, lógica de inferencia, utilidades de procesamiento de audio, y interfaces de línea de comandos. La instalación desde PyPI garantiza la obtención de la versión estable más reciente con dependencias resueltas automáticamente.

#### 3.2.1 Instalación desde PyPI: `pip install -U qwen-tts`

El flag `-U` o `--upgrade` asegura la obtención de la versión más reciente, incluso si una versión anterior está presente en el entorno. La resolución de dependencias de pip instalará automáticamente paquetes requeridos como `transformers`, `accelerate`, `soundfile`, y otras bibliotecas auxiliares. La versión específica de `transformers` requerida (**4.57.3**) será instalada o actualizada según sea necesario, aunque esto puede generar conflictos si otros paquetes en el entorno requieren versiones más recientes de transformers.

#### 3.2.2 Alternativa: Instalación desde fuente con `git clone` y `pip install -e .`

Para desarrolladores que requieren modificar el código fuente, acceder a características en desarrollo, o contribuir al proyecto, la instalación editable desde el repositorio Git oficial proporciona máxima flexibilidad:

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
pip install -e .
```

El modo editable (`-e`) crea enlaces simbólicos en lugar de copia de archivos, permitiendo que modificaciones al código fuente se reflejen inmediatamente sin reinstalación. Esta modalidad es esencial para debugging, profiling, y experimentación con modificaciones de arquitectura o lógica de inferencia.

#### 3.2.3 Verificación de instalación correcta del paquete

La importación exitosa del módulo principal y acceso a sus componentes públicos verifica la integridad de la instalación:

```python
from qwen_tts import Qwen3TTSModel
print(f"Qwen3TTSModel import successful: {Qwen3TTSModel}")
```

La ausencia de errores de `ModuleNotFoundError` o importación confirma que el paquete está correctamente registrado en el path de Python y sus dependencias transitivas son resolvibles.

### 3.3 Instalación Opcional de FlashAttention 2

**FlashAttention 2** representa una reimplementación algorítmica fundamental de la operación de atención en transformers, reduciendo la complejidad de memoria de cuadrática a lineal con respecto a la longitud de secuencia mediante técnicas de tiling y recomputación de activaciones. Para Qwen3-TTS, esto se traduce en reducción significativa de consumo de VRAM y aceleración de **2-3x** en la generación de secuencias de audio largas.

#### 3.3.1 Beneficios: Reducción de uso de VRAM y mejora de velocidad 2-3x

La operación de atención estándar materializa explícitamente la matriz de atención completa, consumiendo memoria proporcional al cuadrado de la longitud de secuencia. Para secuencias de tokens de audio que pueden extenderse a miles de elementos, esto rápidamente agota la VRAM disponible. FlashAttention 2 elimina esta materialización mediante computación de bloques en SRAM de alta velocidad, reduciendo el requerimiento de memoria HBM (memoria de video principal) y permitiendo secuencias más largas o batch sizes mayores. Los benchmarks reportados muestran **speedups de 2-4x** en GPUs NVIDIA recientes, con mayor beneficio en secuencias más largas.

#### 3.3.2 Comando de instalación: `pip install -U flash-attn --no-build-isolation`

La instalación desde fuentes es necesaria dado que FlashAttention 2 contiene kernels CUDA personalizados que deben compilados para la arquitectura GPU específica. El flag `--no-build-isolation` permite que el proceso de compilación acceda a caché de compilación existente y reutilice objetos previamente compilados cuando sea posible, acelerando reinstalaciones:

```bash
pip install -U flash-attn --no-build-isolation
```

El proceso de compilación puede extenderse **10-30 minutos** dependiendo de la velocidad de CPU y disco, generando múltiples archivos de extensión `.so` para diferentes configuraciones de precisión y tamaño de cabeza de atención.

#### 3.3.3 Configuración para sistemas con RAM limitada: `MAX_JOBS=4 pip install -U flash-attn --no-build-isolation`

La compilación paralela de múltiples archivos fuente simultáneamente, aunque acelera el proceso total, puede agotar la RAM del sistema en máquinas con configuraciones modestas. La variable de entorno `MAX_JOBS` limita el grado de paralelismo del sistema de compilación:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Un valor de **4 jobs** típicamente mantiene el consumo de RAM por debajo de **8 GB**, adecuado para sistemas con 16 GB totales considerando la sobrecarga del sistema operativo y otros procesos.

#### 3.3.4 Requisito de compatibilidad de hardware para FlashAttention 2

FlashAttention 2 requiere GPUs con **capacidad de computación (compute capability) 8.0 o superior**, correspondiente a arquitecturas Ampere (RTX 30-series), Ada Lovelace (RTX 40-series), y Blackwell (RTX 50-series). La RTX 5060Ti con compute capability **8.9 o 9.0** (dependiendo de la clasificación específica de Blackwell) cumple ampliamente este requisito. Adicionalmente, se requiere **CUDA 11.6 o superior**, condición satisfecha por la configuración CUDA 12.2/12.8 recomendada.

## 4. Descarga y Gestión de Modelos

### 4.1 Modelos Disponibles y sus Funciones

La familia Qwen3-TTS se organiza en una matriz de modelos diferenciados por tamaño de parámetros (0.6B vs 1.7B), frecuencia de tokenizador (12Hz vs 25Hz), y funcionalidad especializada (Base, CustomVoice, VoiceDesign). La comprensión de estas distinciones permite seleccionar el modelo óptimo para cada aplicación.

| Modelo | Parámetros | Función Principal | VRAM Requerida (BF16) | Latencia |
|--------|-----------|-------------------|----------------------|----------|
| Qwen3-TTS-12Hz-1.7B-Base | 1.7B | Clonación de voz con referencia de audio | ~5.0-5.7 GB | ~97 ms |
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | Clonación de voz rápida, menor calidad | ~2.5-3.0 GB | ~60 ms |
| **Qwen3-TTS-12Hz-1.7B-CustomVoice** | **1.7B** | **Voces predefinidas (9 presets)** | **~5.0-5.7 GB** | **~97 ms** |
| **Qwen3-TTS-12Hz-1.7B-VoiceDesign** | **1.7B** | **Diseño de voces por descripción textual** | **~5.0-5.7 GB** | **~97 ms** |

*Tabla 1: Comparativa de modelos Qwen3-TTS. Las variantes en negrita son el foco de esta guía para prosificación de audio.*

#### 4.1.1 Modelos Base: Qwen3-TTS-12Hz-1.7B-Base y Qwen3-TTS-12Hz-0.6B-Base (clonación de voz)

Los modelos **Base** proporcionan la funcionalidad fundamental de **clonación de voz (voice cloning)** mediante referencia de audio: dado un segmento de audio de referencia de **3-30 segundos** y su transcripción opcional, el modelo sintetiza nueva voz que preserva las características timbricas del hablante de referencia. La variante **1.7B** ofrece mayor fidelidad de clonación y mejor manejo de casos desafiantes (acentos fuertes, habla rápida, ruido de fondo), mientras que la **0.6B** prioriza la velocidad de inferencia. Estos modelos son la base sobre la cual se construyen las capacidades de CustomVoice y VoiceDesign mediante fine-tuning especializado.

#### 4.1.2 Modelos CustomVoice: Qwen3-TTS-12Hz-1.7B-CustomVoice y Qwen3-TTS-12Hz-0.6B-CustomVoice (voces predefinidas)

Los modelos **CustomVoice** encapsulan **nueve voces predefinidas de alta calidad** —**Vivian, Ryan, Aiden, Eric, Serena,** y otras— entrenadas mediante fine-tuning del modelo Base con datos de hablantes profesionales curados. Cada voz posee características timbricas, rangos de pitch, y patrones de prosodia consistentes, permitiendo selección determinística de personalidad vocal sin necesidad de audio de referencia. Esta modalidad es ideal para aplicaciones donde se requiere **consistencia de voz across múltiples generaciones y sesiones**, como asistentes virtuales, sistemas de navegación, o contenido educativo.

#### 4.1.3 Modelos VoiceDesign: Qwen3-TTS-12Hz-1.7B-VoiceDesign (diseño de voces por descripción)

El modelo **VoiceDesign** representa la capacidad más innovadora de Qwen3-TTS: la **generación de voces completamente nuevas a partir de descripciones textuales en lenguaje natural**. Mediante entrenamiento con pares de descripción-voz, el modelo aprende a mapear atributos lingüísticos ("voz masculina profunda, tono calmado, estilo de narrador documental") a parámetros acústicos en el espacio latente del modelo. Esta capacidad elimina la necesidad de grabaciones de referencia o selección de voces predefinidas, permitiendo la **creación ilimitada de personajes vocales únicos** para aplicaciones creativas, accesibilidad, y localización.

### 4.2 Métodos de Descarga de Pesos

La gestión eficiente de los archivos de modelo —que pueden alcanzar varios gigabytes— impacta directamente en el tiempo de preparación del entorno y la experiencia de primera ejecución.

#### 4.2.1 Descarga automática durante la primera ejecución del modelo

La API `from_pretrained()` de Qwen3-TTS integra descarga automática desde **Hugging Face Hub** o **ModelScope** mediante el protocolo git-lfs. En primera invocación con un identificador de modelo no cacheado localmente, la biblioteca descarga automáticamente los archivos necesarios al directorio de caché del usuario (`~/.cache/huggingface/hub/` o `~/.cache/modelscope/hub/`), mostrando progreso de descarga y verificación de integridad mediante hashes. Este método es el más conveniente para usuarios finales, aunque depende de conectividad estable y puede ser lento para usuarios en regiones con latencia elevada a servidores de Hugging Face.

#### 4.2.2 Descarga manual mediante ModelScope (recomendado para China continental)

Para usuarios en China continental, **ModelScope (魔搭社区)** proporciona mirrors de alta velocidad con réplicas de modelos Qwen3-TTS. La descarga manual mediante la interfaz web o CLI de ModelScope, seguida de especificación de ruta local en `from_pretrained()`, evita problemas de conectividad intermitente:

```bash
# Instalación de CLI de ModelScope
pip install modelscope

# Descarga de modelo específico
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./models/CustomVoice
```

La especificación de `local_files_only=True` o ruta explícita en la carga del modelo fuerza el uso de archivos locales descargados.

#### 4.2.3 Descarga manual mediante Hugging Face Hub

La herramienta `huggingface-cli` proporciona descarga programática con reanudación de descargas interrumpidas y verificación de integridad:

```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./models/VoiceDesign --local-dir-use-symlinks False
```

El flag `--local-dir-use-symlinks False` copia archivos en lugar de crear enlaces simbólicos, facilitando la portabilidad del directorio de modelos entre sistemas.

#### 4.2.4 Estructura de directorios locales para almacenamiento de modelos

La organización recomendada de modelos locales facilita la gestión de múltiples variantes y versiones:

```
~/models/qwen3-tts/
├── Qwen3-TTS-12Hz-1.7B-Base/
│   ├── config.json
│   ├── model.safetensors
│   ├── generation_config.json
│   └── ...
├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
│   └── ...
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
│   └── ...
└── Qwen3-TTS-Tokenizer-12Hz/
    └── ...
```

La variable de entorno `HF_HOME` o `MODELSCOPE_CACHE` puede redirigir el caché predeterminado a esta estructura personalizada.

## 5. Configuración para Prosificación de Audio

### 5.1 Fundamentos de la Prosificación en Qwen3-TTS

La **prosificación** —el control fino de entonación, ritmo, énfasis, y expresividad en la síntesis de voz— constituye el diferenciador clave entre sistemas TTS mecánicos y voces que comunican genuinamente significado emocional y pragmático. Qwen3-TTS incorpora mecanismos explícitos de control prosódico que elevan significativamente la calidad percibida y la utilidad en aplicaciones donde la expresividad es crítica.

#### 5.1.1 Definición de prosificación: control de entonación, ritmo y expresividad

En el contexto de Qwen3-TTS, la prosificación abarca tres dimensiones interrelacionadas: la **entonación** (variaciones de pitch fundamental que codifican estructura de oración, foco informativo, y actitud del hablante), el **ritmo** (patrones de duración de sílabas y pausas que agrupan información en unidades prosódicas), y la **expresividad** (modulación dinámica de energía y timbre que comunica estado emocional y engagement). El modelo **12Hz**, con su tokenizador de menor latencia, está particularmente optimizado para capturar y reproducir matices prosódicos finos que requieren resolución temporal elevada.

#### 5.1.2 Parámetros de generación que afectan la prosodia: `instruct`, `language`, `speaker`

El sistema de control prosódico de Qwen3-TTS se expresa principalmente mediante tres parámetros de API. El parámetro **`instruct`** acepta descripciones textuales de estilo de habla que el modelo interpreta para modificar su generación: instrucciones como **"用特别愤怒的语气说"** (hablar con tono particularmente enojado), **"温柔但略带疲惫的语气"** (tono gentil pero ligeramente cansado), o **"充满磁性的新闻主播男声，语速适中"** (voz masculina magnética de presentador de noticias, velocidad moderada) guían al modelo hacia realizaciones prosódicas específicas. El parámetro **`language`** establece expectativas de patrones prosódicos propios del idioma objetivo, mientras que **`speaker`** en modo CustomVoice selecciona perfiles prosódicos predefinidos asociados a cada voz.

#### 5.1.3 Relación entre tokenizador 12Hz y latencia ultra-baja en síntesis

La elección del tokenizador **12Hz** (12.5 Hz de frecuencia de muestreo en el espacio latente) versus 25Hz representa un compromiso fundamental entre latencia y fidelidad de modelado acústico. El tokenizador 12Hz, con su arquitectura de **múltiples codebooks (16 capas)** y decodificación CNN ligera, logra una **latencia de primer paquete de 97 milisegundos** —suficiente para interacción en tiempo real percibida como instantánea— mientras mantiene calidad de síntesis comparable a sistemas de mayor complejidad. Esta **ultra-baja latencia** es crítica para aplicaciones de diálogo en tiempo real, donde delays superiores a 200ms degradan significativamente la naturalidad de la interacción.

### 5.2 Optimización para RTX 5060Ti

La configuración óptima de parámetros de inferencia maximiza la utilización de las capacidades específicas de la RTX 5060Ti, equilibrando calidad de salida, velocidad de generación, y capacidad de throughput.

#### 5.2.1 Selección de precisión: `torch.bfloat16` para equilibrio calidad/rendimiento

El formato **bfloat16 (Brain Floating Point 16)** ofrece el rango dinámico de float32 con la mitad de bits de almacenamiento, siendo particularmente adecuado para redes neuronales donde la magnitud de activaciones es más crítica que la precisión de mantisa. Para Qwen3-TTS, **bfloat16 proporciona calidad de síntesis indistinguible de float32** en evaluación perceptual, mientras reduce el consumo de VRAM a aproximadamente **5 GB para el modelo 1.7B** —dejando margen sustancial para secuencias largas o batch processing en la RTX 5060Ti de 16 GB. La conversión automática de formatos en las Tensor Cores de la arquitectura Blackwell acelera adicionalmente las operaciones matriciales en bfloat16.

#### 5.2.2 Configuración de `device_map="auto"` o `"cuda:0"` explícito

El parámetro `device_map` controla la colocación de componentes del modelo en dispositivos de computación. Para sistemas de GPU única como la configuración RTX 5060Ti, **`"cuda:0"`** especifica explícitamente el dispositivo, mientras **`"auto"`** delega la decisión a la biblioteca `accelerate` de Hugging Face, que típicamente selecciona GPU cuando disponible. La especificación explícita elimina ambigüedad y facilita debugging de problemas de dispositivo.

#### 5.2.3 Implementación de `attn_implementation="flash_attention_2"` cuando sea posible

La selección del backend de atención impacta directamente en memoria y velocidad. La configuración **`"flash_attention_2"`** habilita el uso de FlashAttention 2 si instalado, con fallback automático a `"sdpa"` (Scaled Dot Product Attention de PyTorch 2.0+) si no disponible. Para secuencias de audio típicas en TTS, **FlashAttention 2 reduce el consumo de memoria de atención en 30-50% y acelera la generación en 20-40%**.

## 6. Uso de la Variante CustomVoice

### 6.1 Carga del Modelo CustomVoice

La inicialización del modelo CustomVoice establece el contexto de inferencia para generación con voces predefinidas, cargando pesos de modelo, tokenizador, y embeddings de voces en memoria GPU.

#### 6.1.1 Inicialización con `Qwen3TTSModel.from_pretrained()`

El método de clase `from_pretrained()` es el punto de entrada estándar para cargar modelos Qwen3-TTS, proporcionando abstracción sobre la complejidad de inicialización de componentes:

```python
from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

La primera ejecución con un identificador de modelo no cacheado dispara la descarga automática de archivos desde Hugging Face Hub, mostrando progreso y estimaciones de tiempo restante.

#### 6.1.2 Especificación de repositorio: `"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"`

El identificador de modelo sigue la convención de Hugging Face: organización (`Qwen`), nombre de modelo (`Qwen3-TTS-12Hz-1.7B-CustomVoice`). La variante `12Hz` indica el tokenizador de baja latencia, `1.7B` el tamaño de parámetros, y `CustomVoice` la funcionalidad especializada. La variante `0.6B` está disponible para escenarios de recursos limitados.

#### 6.1.3 Configuración de parámetros de dispositivo y precisión

Los parámetros `device_map` y `torch_dtype` deben coordinarse: `device_map="cuda:0"` con `torch_dtype=torch.bfloat16` para ejecución GPU optimizada, o `device_map="cpu"` con `torch_dtype=torch.float32` para fallback CPU (notablemente más lento). El parámetro `attn_implementation` selecciona el backend de atención, con `"flash_attention_2"` preferido cuando disponible.

### 6.2 Generación de Audio con Voces Predefinidas

La generación con CustomVoice aprovecha las nueve voces predefinidas, cada una con características prosódicas distintivas que pueden moduladas mediante parámetros adicionales.

#### 6.2.1 Listado de voces disponibles: Vivian, Ryan, Aiden, Eric, Serena, entre otras (9 presets)

Las **nueve voces predefinidas** de CustomVoice ofrecen diversidad de género, edad percibida, y registro vocal:

| Voz | Caracterización Típica | Aplicaciones Recomendadas |
|-----|------------------------|---------------------------|
| **Vivian** | Femenina, joven, clara y profesional | Asistentes virtuales, narración educativa |
| **Ryan** | Masculino, adulto, cálido y cercano | Podcasts, contenido de marca personal |
| **Aiden** | Masculino, joven, energético y dinámico | Gaming, marketing digital, entretenimiento |
| **Eric** | Masculino, maduro, autoritario y calmado | Documentales, noticias, contenido institucional |
| **Serena** | Femenina, adulta, suave y empática | Atención al cliente, meditación, salud mental |
| *(4 voces adicionales)* | Variedad de registros y estilos | Casos de uso especializados |

*Tabla 2: Voces predefinidas de CustomVoice con caracterización y aplicaciones sugeridas.*

#### 6.2.2 Parámetro `speaker`: selección de voz predefinida

El parámetro **`speaker`** especifica la voz a utilizar mediante su identificador string exacto. La selección determina el embedding de voz inicial que condiciona toda la generación, estableciendo características timbricas fundamentales que persisten independientemente de modificaciones prosódicas posteriores mediante `instruct`.

#### 6.2.3 Parámetro `language`: especificación de idioma objetivo o "Auto"

El parámetro **`language`** acepta códigos de idioma ISO (ej. `"en"`, `"zh"`, `"es"`, `"ja"`) o el valor especial **`"Auto"`** que delega la detección al modelo. La especificación explícita mejora la coherencia prosódica para idiomas con patrones de entonación distintivos, como el tono en mandarín o el acento en español.

#### 6.2.4 Parámetro `instruct`: instrucciones de estilo emocional ("Happy", "Whispering", "愤怒的语气")

El parámetro **`instruct`** habilita el control prosódico fino mediante descripciones textuales de estilo. Ejemplos efectivos incluyen:

- **"Happy"** o **"Excited"**: tono elevado, ritmo acelerado, mayor variabilidad de pitch
- **"Whispering"** o **"Soft"**: volumen reducido, fricativas aspiradas, intimidad percibida
- **"愤怒的语气"** (tono enojado): pitch elevado, ataque fuerte de consonantes, ritmo irregular
- **"温柔但略带疲惫的语气"** (gentil pero cansado): dinámica comprimida, declinación de pitch al final de frases, pausas extendidas

### 6.3 Ejemplo de Implementación CustomVoice

#### 6.3.1 Importación de librerías: `torch`, `soundfile`, `qwen_tts`

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
```

#### 6.3.2 Generación con `model.generate_custom_voice()`

```python
# Carga del modelo
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# Texto a sintetizar
text = "Welcome to the future of voice synthesis. This is a demonstration of Qwen3-TTS CustomVoice."

# Generación con voz Vivian, estilo profesional entusiasta
audio, sample_rate = model.generate_custom_voice(
    text=text,
    speaker="Vivian",
    language="en",
    instruct="Professional and enthusiastic tone, clear articulation"
)

print(f"Audio generado: {len(audio)/sample_rate:.2f} segundos a {sample_rate} Hz")
```

#### 6.3.3 Guardado de audio con `soundfile.write()`

```python
# Guardado en formato WAV de alta calidad
sf.write("output_vivian_professional.wav", audio, sample_rate)
print("Audio guardado: output_vivian_professional.wav")
```

## 7. Uso de la Variante VoiceDesign

### 7.1 Carga del Modelo VoiceDesign

La variante VoiceDesign requiere el modelo específico entrenado para mapeo de descripciones textuales a características vocales, con consideraciones de memoria idénticas a CustomVoice dado que comparten arquitectura base.

#### 7.1.1 Especificación de repositorio: `"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"`

```python
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

#### 7.1.2 Consideraciones de memoria: modelo único disponible en 1.7B

A diferencia de CustomVoice, **VoiceDesign únicamente está disponible en variante 1.7B**, dado que la complejidad del mapeo descripción-voz requiere la capacidad representacional del modelo completo. El consumo de VRAM se mantiene en **~5.0-5.7 GB** en BF16, compatible con la RTX 5060Ti.

### 7.2 Diseño de Voces mediante Descripciones Textuales

La capacidad distintiva de VoiceDesign es la **síntesis de voz sin referencia de audio ni selección de preset**, generando voces únicas a partir de especificaciones lingüísticas.

#### 7.2.1 Parámetro `instruct`: descripción detallada de características vocales deseadas

En VoiceDesign, el parámetro **`instruct`** adquiere mayor protagonismo, ya que define completamente la identidad vocal a generar. La descripción debe ser **específica, dimensional, y evocativa**, cubriendo: género percibido, rango de edad, registro (grave/agudo), cualidades timbricas (raspado, nasal, resonante), estilo de habla (conversacional, declamatorio, íntimo), y contexto emocional.

#### 7.2.2 Ejemplos de prompts efectivos: "A deep, resonant male voice, narrator style, calm and professional"

| Tipo de Voz | Prompt Efectivo | Aplicación |
|-------------|-----------------|------------|
| Narrador documental | "A deep, resonant male voice, narrator style, calm and professional, measured pace with strategic pauses" | Documentales, audiolibros de no-ficción |
| Asistente de lujo | "A sophisticated female voice, mid-30s, warm and reassuring, subtle British accent, premium customer service tone" | Concierge virtual, atención VIP |
| Personaje animado | "A youthful, energetic female voice, slightly nasal, rapid speech with playful inflections, cartoon character energy" | Animación, contenido infantil |
| Vocero institucional | "A mature male voice, baritone range, authoritative yet approachable, clear enunciation, news anchor gravitas" | Comunicados corporativos, noticias |
| Voz terapéutica | "A soft, breathy female voice, slow and deliberate, minimal pitch variation, ASMR-like intimacy without whisper" | Meditación, terapia digital, sueño |

*Tabla 3: Ejemplos de prompts de VoiceDesign para diferentes aplicaciones.*

#### 7.2.3 Parámetro `text`: contenido a sintetizar con la voz diseñada

El parámetro **`text`** contiene el contenido lingüístico a sintetizar, que debe estar **alineado con el estilo vocal especificado** para máxima coherencia perceptual. Un texto técnico con voz diseñada como "playful cartoon character" generará disonancia; la iteración conjunta de descripción vocal y contenido es recomendable.

### 7.3 Ejemplo de Implementación VoiceDesign

#### 7.3.1 Estructura de llamada a `model.generate_voice_design()`

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Carga del modelo VoiceDesign
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# Diseño de voz mediante descripción textual
voice_description = (
    "A deep, resonant male voice with subtle gravel texture, "
    "mature and authoritative, documentary narrator style, "
    "measured pace with strategic pauses for emphasis, "
    "warm undertones suggesting wisdom and trustworthiness"
)

# Contenido a sintetizar
text = "In the vast expanse of human knowledge, few discoveries have transformed our understanding as profoundly as the mapping of the human genome."

# Generación con voz diseñada
audio, sample_rate = model.generate_voice_design(
    text=text,
    instruct=voice_description,
    language="en"
)

# Guardado
sf.write("output_documentary_narrator.wav", audio, sample_rate)
print(f"Audio generado: {len(audio)/sample_rate:.2f} segundos")
```

#### 7.3.2 Iteración y refinamiento de descripciones para resultados óptimos

La generación de voces mediante descripción es inherentemente **estocástica y exploratoria**. Se recomienda:

1. **Documentar descripciones y resultados** para establecer biblioteca de voces reproducibles
2. **Variar un dimensión a la vez** (ej. solo pitch, solo velocidad) para aislar efectos
3. **Combinar descripciones exitosas** mediante concatenación de atributos verificados
4. **Generar múltiples muestras** con misma descripción para evaluar consistencia
5. **Ajustar granularidad**: descripciones demasiado vagas producen resultados impredecibles; demasiado específicas pueden generar conflictos internos

## 8. Interfaz Web Local (Opcional)

### 8.1 Lanzamiento del Demo Gradio

El paquete `qwen-tts` incluye una **interfaz web de demostración basada en Gradio** que permite exploración interactiva de capacidades sin escritura de código.

#### 8.1.1 Comando para CustomVoice: `qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000`

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000
```

Los parámetros `--ip 0.0.0.0` exponen el servicio en todas las interfaces de red (accesible remotamente), mientras `--port 8000` especifica el puerto de escucha.

#### 8.1.2 Comando para VoiceDesign: `qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --ip 0.0.0.0 --port 8000`

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --ip 0.0.0.0 --port 8000
```

#### 8.1.3 Acceso mediante navegador: `http://localhost:8000`

La interfaz web proporciona: selección de voz/descripción, entrada de texto, controles de parámetros de generación, reproducción inmediata de resultados, y descarga de audio generado. Es particularmente útil para **demostraciones a stakeholders** y **exploración rápida del espacio de parámetros**.

### 8.2 Configuración HTTPS para Modelo Base

Para el modelo Base que requiere **carga de audio de referencia**, la transmisión segura de datos de voz biométricos justifica la configuración HTTPS incluso en despliegues locales.

#### 8.2.1 Generación de certificado autofirmado con OpenSSL

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

#### 8.2.2 Lanzamiento con parámetros `--ssl-certfile` y `--ssl-keyfile`

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000 --ssl-certfile cert.pem --ssl-keyfile key.pem
```

#### 8.2.3 Consideraciones de seguridad para despliegues de producción

- Los **certificados autofirmados** generan advertencias de seguridad en navegadores; para producción, utilizar certificados de autoridad de confianza (Let's Encrypt, etc.)
- La **exposición en `0.0.0.0`** debe combinarse con firewall restrictivo o autenticación de aplicación
- Los **audios de referencia** para clonación contienen información biométrica sensible; aplicar políticas de retención y acceso acordes con regulaciones de privacidad aplicables

## 9. Solución de Problemas y Optimización

### 9.1 Problemas Comunes de Instalación

#### 9.1.1 Conflictos de versión con `transformers==4.57.3`

El paquete `qwen-tts` especifica **transformers 4.57.3** de manera estricta. Si otros proyectos requieren versiones más recientes, considerar:

- **Entornos completamente aislados** (contenedores Docker, máquinas virtuales)
- **Instalación de qwen-tts en entorno dedicado** sin otras dependencias de NLP
- **Uso de herramientas de resolución de conflictos** como `pip-compile` o `poetry` para análisis de grafo de dependencias

#### 9.1.2 Errores de compilación en FlashAttention: verificación de compatibilidad de hardware

Los errores de compilación típicamente indican: **compute capability insuficiente** (requerido ≥8.0, RTX 5060Ti cumple), **CUDA toolkit no encontrado** (verificar `CUDA_HOME`), o **RAM insuficiente durante compilación** (usar `MAX_JOBS=4` o menor). La verificación de requisitos previos mediante `python -m flash_attn.info` (si disponible) o consulta de issues en repositorio oficial orienta la resolución.

#### 9.1.3 Descargas fallidas de modelos: configuración de mirrors alternativos

Para fallos de conectividad a Hugging Face Hub:

```python
# Configuración de endpoint alternativo
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Mirror para China continental

# O uso de ModelScope como alternativa
from modelscope import snapshot_download
model_dir = snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
```

### 9.2 Optimización de Rendimiento en RTX 5060Ti

#### 9.2.1 Monitoreo de uso de VRAM con `nvidia-smi` durante inferencia

La ejecución concurrente de `nvidia-smi dmon` durante generación revela patrones de consumo de memoria y utilización de GPU. Picos inesperados de VRAM pueden indicar: **secuencias de texto excesivamente largas** (aumentar chunking), **batch size inadecuado**, o **memory leaks** en iteraciones prolongadas (liberar caché de CUDA periódicamente con `torch.cuda.empty_cache()`).

#### 9.2.2 Ajuste de `max_new_tokens` para secuencias largas

El parámetro **`max_new_tokens`** controla la longitud máxima de generación. Para texto de entrada largo, el valor por defecto puede truncar salida; incrementar proporcionalmente a la longitud de entrada, considerando que **cada token de audio representa ~80ms de voz sintetizada** en el tokenizador 12Hz.

#### 9.2.3 Uso de cuantización GPTQ-Int8 para reducción de memoria 50-70%

Para escenarios de memoria extremadamente limitada, la cuantización **GPTQ-Int8** reduce pesos de modelo a 8 bits con degradación mínima perceptible. La implementación requiere: instalación de `auto-gptq`, descarga de variantes cuantizadas del modelo (cuando disponibles), o cuantización manual con calibración representativa. La RTX 5060Ti con 16 GB típicamente **no requiere cuantización** para operación single-model, pero esta opción habilita **múltiples instancias concurrentes** o **modelos adicionales en memoria**.

### 9.3 Consideraciones de Producción

#### 9.3.1 Despliegue con vLLM para inferencia optimizada

Para **throughput máximo en servicio de múltiples usuarios**, **vLLM** proporciona: scheduling continuo de requests, paging de atención eficiente, y batching dinámico. La integración de Qwen3-TTS con vLLM requiere adaptación de la clase de modelo a interfaz compatible; el repositorio oficial de Qwen3-TTS incluye ejemplos de configuración.

#### 9.3.2 Configuración de batch processing para cargas de trabajo múltiples

El procesamiento batch de múltiples textos simultáneamente mejora throughput mediante mejor utilización de GPU. Consideraciones: **textos de longitud similar** en cada batch para minimizar padding, **tamaño de batch limitado por VRAM disponible** (experimentar con 4-16 dependiendo de longitud), y **acumulación de gradientes deshabilitada** en inferencia pura.

#### 9.3.3 Limpieza de memoria GPU entre generaciones cuando sea necesario

Para **servicios de larga duración** o **generación iterativa extensa**, la fragmentación de memoria CUDA puede degradar rendimiento. La llamada periódica a:

```python
import torch
torch.cuda.empty_cache()  # Libera caché de memoria no utilizada
torch.cuda.ipc_collect()   # Recolecta memoria inter-proceso si aplica
```

restaura estado de memoria óptimo. La frecuencia de limpieza debe balancear **overhead de operación** versus **degradación por fragmentación**; típicamente cada 100-1000 generaciones o cuando `nvidia-smi` muestra crecimiento monotónico de memoria reservada no explicada por tamaño de modelo.

