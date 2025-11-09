"""
⚽ Soccer Analyzer - Análisis de Fútbol con IA
Todo en un solo archivo - Simplificado
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
from collections import deque
from ultralytics import YOLO
import torch
import supervision as sv
from sports.common.team import TeamClassifier

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="⚽ Análisis de Fútbol con IA",
    page_icon="⚽",
    layout="wide"
)

# IDs de clases
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# Rutas de modelos
PLAYER_MODEL = "best_jugadores_chiquito.pt"
FIELD_MODEL = "best_campo_chiquito.pt"
BALL_MODEL = "ball_little.pt"

# ==================== CLASES AUXILIARES ====================

class BallTracker:
    def __init__(self, buffer_size=30):
        self.buffer = deque(maxlen=buffer_size)
        self.last_position = None
    
    def update(self, detections):
        if len(detections) == 0:
            return None
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        if self.last_position is None:
            self.last_position = xy[0]
        else:
            distances = np.linalg.norm(xy - self.last_position, axis=1)
            self.last_position = xy[np.argmin(distances)]
        self.buffer.append(self.last_position)
        return self.last_position
    
    def reset(self):
        self.buffer.clear()
        self.last_position = None

class BallAnnotator:
    def __init__(self, radius=12):
        self.radius = radius
        self.ball_color = (0, 255, 0)
        self.trail_color = (0, 255, 255)
    
    def annotate(self, frame, position, trail_buffer):
        annotated = frame.copy()
        
        # Trail
        for i in range(1, len(trail_buffer)):
            if trail_buffer[i-1] is None or trail_buffer[i] is None:
                continue
            x1, y1 = map(int, trail_buffer[i-1])
            x2, y2 = map(int, trail_buffer[i])
            alpha = i / len(trail_buffer)
            thickness = max(2, int(4 * alpha))
            cv2.line(annotated, (x1, y1), (x2, y2), self.trail_color, thickness, cv2.LINE_AA)
        
        # Balón
        if position is not None:
            x, y = map(int, position)
            cv2.circle(annotated, (x, y), self.radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(annotated, (x, y), self.radius, self.ball_color, -1, cv2.LINE_AA)
            cv2.circle(annotated, (x, y), self.radius, (0, 0, 0), 1, cv2.LINE_AA)
        
        return annotated

# ==================== FUNCIONES ====================

@st.cache_resource
def load_models():
    """Cargar modelos YOLO"""
    try:
        if not os.path.exists(PLAYER_MODEL):
            st.error(f"❌ Modelo no encontrado: {PLAYER_MODEL}")
            return None, None, None
        
        player_yolo = YOLO(PLAYER_MODEL)
        field_yolo = YOLO(FIELD_MODEL)
        ball_yolo = YOLO(BALL_MODEL) if os.path.exists(BALL_MODEL) else None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.success(f"✅ Modelos cargados en: {device.upper()}")
        
        return player_yolo, field_yolo, ball_yolo
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None, None

def get_detections(model, frame, confidence=0.3):
    results = model.predict(frame, conf=confidence, verbose=False)
    return sv.Detections.from_ultralytics(results[0])

def resolve_goalkeepers(players, goalkeepers):
    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    p_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    team_0_centroid = p_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = p_xy[players.class_id == 1].mean(axis=0)
    
    gk_teams = []
    for gk in gk_xy:
        dist_0 = np.linalg.norm(gk - team_0_centroid)
        dist_1 = np.linalg.norm(gk - team_1_centroid)
        gk_teams.append(0 if dist_0 < dist_1 else 1)
    
    return np.array(gk_teams)

def create_classifier(video_path, player_model):
    """Crear clasificador de equipos"""
    frame_gen = sv.get_video_frames_generator(video_path, stride=30)
    crops = []
    
    for frame in frame_gen:
        detections = get_detections(player_model, frame, 0.3)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        players = detections[detections.class_id == PLAYER_ID]
        crops += [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
        if len(crops) >= 500:
            break
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = TeamClassifier(device=device)
    classifier.fit(crops)
    return classifier

def process_video(video_path, output_path, player_model, ball_model, classifier, 
                 start_frame=0, max_frames=None, progress_callback=None):
    """Procesar video"""
    
    video_info = sv.VideoInfo.from_video_path(video_path)
    
    if max_frames:
        output_info = sv.VideoInfo(
            width=video_info.width, height=video_info.height,
            fps=video_info.fps, total_frames=max_frames
        )
    else:
        output_info = sv.VideoInfo(
            width=video_info.width, height=video_info.height,
            fps=video_info.fps, total_frames=video_info.total_frames - start_frame
        )
    
    # Trackers
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=30)
    
    # Anotadores
    ellipse_team0 = sv.EllipseAnnotator(color=sv.Color.from_hex('#00BFFF'), thickness=2)
    ellipse_team1 = sv.EllipseAnnotator(color=sv.Color.from_hex('#FF1493'), thickness=2)
    ellipse_referee = sv.EllipseAnnotator(color=sv.Color.from_hex('#FFD700'), thickness=2)
    
    label_team0 = sv.LabelAnnotator(
        color=sv.Color.from_hex('#00BFFF'), text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER, text_scale=0.5
    )
    label_team1 = sv.LabelAnnotator(
        color=sv.Color.from_hex('#FF1493'), text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER, text_scale=0.5
    )
    label_referee = sv.LabelAnnotator(
        color=sv.Color.from_hex('#FFD700'), text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER, text_scale=0.5
    )
    
    ball_annotator = BallAnnotator()
    
    # Stats
    stats = {
        'frames_procesados': 0, 'jugadores_equipo_0': 0, 'jugadores_equipo_1': 0,
        'porteros_detectados': 0, 'arbitros_detectados': 0, 'detecciones_balon': 0,
        'total_detecciones': 0
    }
    
    # Procesamiento
    frame_gen = sv.get_video_frames_generator(
        source_path=video_path,
        start=start_frame,
        end=start_frame + max_frames if max_frames else None
    )
    
    with sv.VideoSink(output_path, output_info) as sink:
        for frame_idx, frame in enumerate(frame_gen):
            stats['frames_procesados'] += 1
            
            if progress_callback and frame_idx % 10 == 0:
                progress_callback((frame_idx + 1) / output_info.total_frames)
            
            # Detecciones
            detections = get_detections(player_model, frame, 0.3)
            
            if len(detections) == 0:
                sink.write_frame(frame)
                continue
            
            stats['total_detecciones'] += len(detections)
            detections = tracker.update_with_detections(detections)
            
            # Balón
            if ball_model:
                ball_results = ball_model.predict(frame, conf=0.3, verbose=False)
                ball_detections = sv.Detections.from_ultralytics(ball_results[0])
                if len(ball_detections) > 0:
                    ball_detections = ball_detections.with_nms(threshold=0.1)
            else:
                ball_detections = detections[detections.class_id == BALL_ID]
            
            ball_pos = ball_tracker.update(ball_detections)
            if ball_pos is not None:
                stats['detecciones_balon'] += 1
            
            # Separar por clase
            goalkeepers = detections[detections.class_id == GOALKEEPER_ID]
            players = detections[detections.class_id == PLAYER_ID]
            referees = detections[detections.class_id == REFEREE_ID]
            
            stats['porteros_detectados'] += len(goalkeepers)
            stats['arbitros_detectados'] += len(referees)
            
            # Clasificar equipos
            players_team0 = sv.Detections.empty()
            players_team1 = sv.Detections.empty()
            
            if len(players) > 0:
                crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
                team_ids = classifier.predict(crops)
                
                stats['jugadores_equipo_0'] += (team_ids == 0).sum()
                stats['jugadores_equipo_1'] += (team_ids == 1).sum()
                
                team0_mask = team_ids == 0
                team1_mask = team_ids == 1
                
                if team0_mask.any():
                    players_team0 = players[team0_mask]
                if team1_mask.any():
                    players_team1 = players[team1_mask]
            
            # Porteros
            if len(goalkeepers) > 0 and len(players) > 0:
                players_team0.class_id = np.zeros(len(players_team0), dtype=int)
                players_team1.class_id = np.ones(len(players_team1), dtype=int)
                
                merged = sv.Detections.merge([players_team0, players_team1])
                gk_teams = resolve_goalkeepers(merged, goalkeepers)
                
                if (gk_teams == 0).any():
                    players_team0 = sv.Detections.merge([players_team0, goalkeepers[gk_teams == 0]])
                if (gk_teams == 1).any():
                    players_team1 = sv.Detections.merge([players_team1, goalkeepers[gk_teams == 1]])
            
            # Anotar
            annotated = frame.copy()
            
            if len(players_team0) > 0:
                labels = [str(int(tid)) for tid in players_team0.tracker_id]
                annotated = ellipse_team0.annotate(annotated, players_team0)
                annotated = label_team0.annotate(annotated, players_team0, labels=labels)
            
            if len(players_team1) > 0:
                labels = [str(int(tid)) for tid in players_team1.tracker_id]
                annotated = ellipse_team1.annotate(annotated, players_team1)
                annotated = label_team1.annotate(annotated, players_team1, labels=labels)
            
            if len(referees) > 0:
                labels = [str(int(tid)) for tid in referees.tracker_id]
                annotated = ellipse_referee.annotate(annotated, referees)
                annotated = label_referee.annotate(annotated, referees, labels=labels)
            
            # Balón
            annotated = ball_annotator.annotate(annotated, ball_pos, ball_tracker.buffer)
            
            # Info
            cv2.putText(annotated, f"Frame: {start_frame + frame_idx}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Azul: {len(players_team0)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 191, 0), 2)
            cv2.putText(annotated, f"Rojo: {len(players_team1)}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 20, 147), 2)
            
            ball_status = "SI" if ball_pos is not None else "NO"
            ball_color = (0, 255, 0) if ball_pos is not None else (0, 0, 255)
            cv2.putText(annotated, f"Balon: {ball_status}",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)
            
            sink.write_frame(annotated)
    
    return stats

# ==================== UI ====================

st.markdown('<h1 style="text-align: center; color: #1E88E5;">⚽ Análisis de Fútbol con IA</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Detección de jugadores, clasificación de equipos y tracking del balón</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"💻 Dispositivo: **{device}**")
    st.markdown("---")
    
    process_full = st.checkbox("Procesar video completo", value=False)
    
    if not process_full:
        col1, col2 = st.columns(2)
        with col1:
            start_sec = st.number_input("Inicio (seg)", min_value=0, value=0)
        with col2:
            duration_sec = st.number_input("Duración (seg)", min_value=1, value=10)
    
    st.markdown("---")
    st.markdown("**Desarrollado por:** J0sephT")

# Main
def main():
    player_model, field_model, ball_model = load_models()
    
    if player_model is None:
        st.error("❌ No se pudieron cargar los modelos")
        return
    
    st.header("📤 Cargar Video")
    uploaded_file = st.file_uploader(
        "Sube un video de fútbol",
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.subheader("🎥 Video Original")
        st.video(video_path)
        
        video_info = sv.VideoInfo.from_video_path(video_path)
        duration = video_info.total_frames / video_info.fps
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Duración", f"{duration:.1f}s")
        with col2:
            st.metric("📐 Resolución", f"{video_info.width}x{video_info.height}")
        with col3:
            st.metric("🎬 FPS", f"{video_info.fps}")
        with col4:
            st.metric("🎞️ Frames", f"{video_info.total_frames}")
        
        if st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True):
            
            if process_full:
                start_frame = 0
                max_frames = min(600, video_info.total_frames)  # Límite 20s
                if video_info.total_frames > 600:
                    st.warning("⚠️ Video largo. Se procesarán solo 20 segundos.")
            else:
                start_frame = int(start_sec * video_info.fps)
                max_frames = min(int(duration_sec * video_info.fps), 600)
                if max_frames >= 600:
                    st.warning("⚠️ Se limitará a 20 segundos.")
            
            if start_frame >= video_info.total_frames:
                st.error("❌ Tiempo de inicio inválido")
                return
            
            with st.spinner("🤖 Entrenando clasificador..."):
                classifier = create_classifier(video_path, player_model)
            
            st.success("✅ Clasificador entrenado")
            
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            st.subheader("⚙️ Procesando Video")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Procesando... {int(progress * 100)}%")
            
            start_time = time.time()
            
            try:
                stats = process_video(
                    video_path, output_path,
                    player_model, ball_model, classifier,
                    start_frame, max_frames, update_progress
                )
                
                processing_time = time.time() - start_time
                
                progress_bar.progress(1.0)
                status_text.text("✅ Completado")
                
                st.success(f"✅ Procesado en {processing_time:.1f}s")
                
                st.subheader("📊 Estadísticas")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎬 Frames", stats['frames_procesados'])
                    st.metric("👥 Equipo Azul", stats['jugadores_equipo_0'])
                with col2:
                    st.metric("🔍 Detecciones", stats['total_detecciones'])
                    st.metric("👥 Equipo Rojo", stats['jugadores_equipo_1'])
                with col3:
                    ball_pct = (stats['detecciones_balon'] / stats['frames_procesados'] * 100) if stats['frames_procesados'] > 0 else 0
                    st.metric("⚽ Balón", f"{ball_pct:.1f}%")
                    st.metric("🥅 Porteros", stats['porteros_detectados'])
                
                st.subheader("🎬 Video Procesado")
                st.video(output_path)
                
                with open(output_path, 'rb') as f:
                    st.download_button(
                        "⬇️ Descargar Video",
                        f,
                        file_name=f"analisis_{uploaded_file.name}",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
                try:
                    os.unlink(video_path)
                except:
                    pass
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
