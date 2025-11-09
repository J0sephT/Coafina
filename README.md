# ⚽ Soccer Analyzer - Análisis de Fútbol con IA

Sistema de análisis de video de fútbol usando YOLO y Streamlit.

## 🚀 Características

- 🎯 **Detección de jugadores** con YOLO
- 👥 **Clasificación automática** de equipos (Azul vs Rojo)
- ⚽ **Tracking del balón** con trail visual
- 🥅 **Identificación de porteros**
- 🧑‍⚖️ **Detección de árbitros**
- 📊 **Estadísticas detalladas**

## 📁 Estructura del Proyecto
```
soccer-analyzer/
├── app.py                              # Aplicación Streamlit
├── requirements.txt                    # Dependencias
├── README.md                          # Este archivo
├── best_jugadores_chiquito.pt         # Modelo jugadores (REQUERIDO)
├── best_campo_chiquito.pt             # Modelo campo (REQUERIDO)
├── ball_little.pt                     # Modelo balón (OPCIONAL)
└── utils/
    ├── __init__.py
    ├── detectors.py
    ├── trackers.py
    └── processor.py
```

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone <tu-repo>
cd soccer-analyzer
```

### 2. Crear entorno virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Colocar modelos

Coloca los archivos `.pt` en la **raíz del proyecto**:
- `best_jugadores_chiquito.pt`
- `best_campo_chiquito.pt`
- `ball_little.pt` (opcional)

## ▶️ Uso

### Ejecutar localmente
```bash
streamlit run app.py
```

Abre el navegador en: http://localhost:8501

### Uso básico

1. **Sube un video** (.mp4, .avi, .mov)
2. **Configura opciones** (video completo o segmento)
3. **Inicia análisis** con el botón 🚀
4. **Descarga resultado** cuando termine

## 🌐 Despliegue en Streamlit Cloud

### 1. Preparar repositorio
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Subir modelos a Google Drive / Dropbox

Los modelos `.pt` son muy grandes para GitHub. Súbelos a:
- Google Drive (público)
- Dropbox
- Hugging Face Hub

### 3. Modificar `app.py`

Descarga automática de modelos:
```python
import gdown

@st.cache_resource
def download_models():
    # IDs de tus archivos en Drive
    files = {
        "best_jugadores_chiquito.pt": "FILE_ID_1",
        "best_campo_chiquito.pt": "FILE_ID_2",
        "ball_little.pt": "FILE_ID_3"
    }
    
    for filename, file_id in files.items():
        if not os.path.exists(filename):
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                filename,
                quiet=False
            )
```

### 4. Desplegar

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu repositorio
3. Selecciona `app.py`
4. Deploy 🚀

## ⚙️ Requisitos

- Python 3.8+
- CUDA (opcional, para GPU)
- 8GB RAM mínimo
- GPU recomendada

## 📝 Notas

- **GPU:** Acelera el procesamiento ~10x
- **Videos largos:** Procesa segmentos para evitar timeouts
- **Modelos:** Deben estar en la raíz del proyecto

## 👨‍💻 Autor

**J0sephT**

## 📄 Licencia

MIT License