# app.py
"""
Aplicación Streamlit para análisis de fútbol con IA
"""

import streamlit as st
import os
import tempfile
import time
from pathlib import Path
import torch
import warnings

# 🔧 SOLUCIÓN: Configurar PyTorch para permitir cargar modelos de ultralytics
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
warnings.filterwarnings('ignore', category=SyntaxWarning)

from ultralytics import YOLO
from utils.detectors import PlayerDetector, FieldDetector
from utils.processor import create_team_classifier, process_video

# Configuración de la página
st.set_page_config(
    page_title="⚽ Análisis de Fútbol con IA",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">⚽ Análisis de Fútbol con IA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detección de jugadores, clasificación de equipos y tracking del balón</div>', unsafe_allow_html=True)

# Rutas de modelos (en la raíz del proyecto)
PLAYER_MODEL_PATH = "best_jugadores_chiquito.pt"
FIELD_MODEL_PATH = "best_campo_chiquito.pt"
BALL_MODEL_PATH = "ball_little.pt"


@st.cache_resource
def load_models():
    """Cargar modelos YOLO (se cachea para no recargar)"""
    try:
        with st.spinner("🔄 Cargando modelos de IA..."):
            # Verificar que existan los archivos
            if not os.path.exists(PLAYER_MODEL_PATH):
                st.error(f"❌ No se encuentra el modelo: {PLAYER_MODEL_PATH}")
                return None, None, None
            
            if not os.path.exists(FIELD_MODEL_PATH):
                st.error(f"❌ No se encuentra el modelo: {FIELD_MODEL_PATH}")
                return None, None, None
            
            # Cargar modelos
            player_yolo = YOLO(PLAYER_MODEL_PATH)
            field_yolo = YOLO(FIELD_MODEL_PATH)
            
            player_detector = PlayerDetector(player_yolo)
            field_detector = FieldDetector(field_yolo)
            
            # Modelo de balón (opcional)
            ball_yolo = None
            if os.path.exists(BALL_MODEL_PATH):
                try:
                    ball_yolo = YOLO(BALL_MODEL_PATH)
                except:
                    st.warning("⚠️ No se pudo cargar el modelo de balón")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.success(f"✅ Modelos cargados correctamente en: {device.upper()}")
            
            return player_detector, field_detector, ball_yolo
    
    except Exception as e:
        st.error(f"❌ Error al cargar modelos: {str(e)}")
        return None, None, None


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Información del sistema
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"💻 Dispositivo: **{device}**")
    
    st.markdown("---")
    
    # Configuración de procesamiento
    st.subheader("🎬 Opciones de Video")
    
    process_full = st.checkbox("Procesar video completo", value=False)
    
    if not process_full:
        col1, col2 = st.columns(2)
        with col1:
            start_sec = st.number_input(
                "Inicio (seg)", 
                min_value=0, 
                value=0,
                help="Segundo donde empieza el procesamiento"
            )
        with col2:
            duration_sec = st.number_input(
                "Duración (seg)", 
                min_value=1, 
                value=10,
                help="Cuántos segundos procesar"
            )
    
    st.markdown("---")
    
    # Información del proyecto
    st.subheader("ℹ️ Información")
    st.markdown("""
    **Características:**
    - 🎯 Detección de jugadores
    - 👥 Clasificación automática de equipos
    - ⚽ Tracking del balón con trail
    - 🥅 Identificación de porteros
    - 🧑‍⚖️ Detección de árbitros
    
    **Desarrollado por:** J0sephT
    """)


# Contenido principal
def main():
    # Cargar modelos
    player_detector, field_detector, ball_yolo = load_models()
    
    if player_detector is None or field_detector is None:
        st.error("❌ No se pudieron cargar los modelos. Verifica que los archivos .pt estén en la raíz del proyecto.")
        return
    
    # Upload de video
    st.header("📤 Cargar Video")
    uploaded_file = st.file_uploader(
        "Sube un video de fútbol (.mp4, .avi, .mov)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Formatos soportados: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Mostrar video original
        st.subheader("🎥 Video Original")
        st.video(video_path)
        
        # Obtener info del video
        import supervision as sv
        video_info = sv.VideoInfo.from_video_path(video_path)
        duration = video_info.total_frames / video_info.fps
        
        # Mostrar información
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Duración", f"{duration:.1f}s")
        with col2:
            st.metric("📐 Resolución", f"{video_info.width}x{video_info.height}")
        with col3:
            st.metric("🎬 FPS", f"{video_info.fps}")
        with col4:
            st.metric("🎞️ Frames", f"{video_info.total_frames}")
        
        # Botón de procesamiento
        if st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True):
            
            # Calcular frames a procesar
            if process_full:
                start_frame = 0
                max_frames = None
                # 🔧 MEJORA: Limitar frames en Streamlit Cloud
                if video_info.total_frames > 600:  # ~20 segundos a 30fps
                    st.warning("⚠️ Video muy largo. Se procesarán solo los primeros 20 segundos para optimizar memoria.")
                    max_frames = 600
                st.info(f"📊 Procesando video completo: {max_frames or video_info.total_frames} frames")
            else:
                start_frame = int(start_sec * video_info.fps)
                max_frames = int(duration_sec * video_info.fps)
                # 🔧 MEJORA: Limitar duración máxima
                if max_frames > 600:
                    st.warning("⚠️ Duración muy larga. Se limitará a 20 segundos para optimizar memoria.")
                    max_frames = 600
                st.info(f"📊 Procesando desde {start_sec}s por {min(duration_sec, 20)}s ({max_frames} frames)")
            
            # Verificar límites
            if start_frame >= video_info.total_frames:
                st.error("❌ El tiempo de inicio excede la duración del video")
                return
            
            # Crear team classifier
            with st.spinner("🤖 Entrenando clasificador de equipos..."):
                team_classifier = create_team_classifier(
                    video_path, 
                    player_detector,
                    stride=30,
                    max_crops=500
                )
            
            st.success("✅ Clasificador entrenado")
            
            # Archivo de salida
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            # Procesamiento con barra de progreso
            st.subheader("⚙️ Procesando Video")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Procesando... {int(progress * 100)}%")
            
            start_time = time.time()
            
            try:
                # Procesar video
                stats = process_video(
                    video_path=video_path,
                    output_path=output_path,
                    player_detector=player_detector,
                    field_detector=field_detector,
                    ball_yolo=ball_yolo,
                    team_classifier=team_classifier,
                    start_frame=start_frame,
                    max_frames=max_frames,
                    progress_callback=update_progress
                )
                
                processing_time = time.time() - start_time
                
                # Completar barra de progreso
                progress_bar.progress(1.0)
                status_text.text("✅ Procesamiento completado")
                
                # Mostrar estadísticas
                st.success(f"✅ Video procesado en {processing_time:.1f} segundos")
                
                st.subheader("📊 Estadísticas del Análisis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎬 Frames Procesados", stats['frames_procesados'])
                    st.metric("👥 Equipo Azul", stats['jugadores_equipo_0'])
                with col2:
                    st.metric("🔍 Detecciones Totales", stats['total_detecciones'])
                    st.metric("👥 Equipo Rojo", stats['jugadores_equipo_1'])
                with col3:
                    ball_percent = (stats['detecciones_balon'] / stats['frames_procesados'] * 100) if stats['frames_procesados'] > 0 else 0
                    st.metric("⚽ Detección Balón", f"{ball_percent:.1f}%")
                    st.metric("🥅 Porteros", stats['porteros_detectados'])
                
                # Mostrar video procesado
                st.subheader("🎬 Video Procesado")
                st.video(output_path)
                
                # Botón de descarga
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Descargar Video Analizado",
                        data=f,
                        file_name=f"analisis_{uploaded_file.name}",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
                # Limpiar archivos temporales
                try:
                    os.unlink(video_path)
                except:
                    pass
            
            except Exception as e:
                st.error(f"❌ Error durante el procesamiento: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
