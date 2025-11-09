# utils/processor.py
"""
Pipeline principal de procesamiento de video
"""

import cv2
import numpy as np
import supervision as sv
import torch
from sports.common.team import TeamClassifier
from .detectors import get_detections, BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID
from .trackers import BallTracker, BallAnnotator


def resolve_goalkeepers_team_id(players, goalkeepers):
    """
    Asignar equipo a porteros basándose en cercanía al centroide
    
    Args:
        players: Detecciones de jugadores con class_id (0 o 1)
        goalkeepers: Detecciones de porteros
    
    Returns:
        np.array: IDs de equipo para porteros
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    
    return np.array(goalkeepers_team_id)


def create_team_classifier(video_path, player_detector, stride=30, max_crops=500):
    """
    Crear y entrenar clasificador de equipos
    
    Args:
        video_path: Ruta del video
        player_detector: Detector de jugadores
        stride: Frames entre muestras
        max_crops: Máximo de crops a recolectar
    
    Returns:
        TeamClassifier entrenado
    """
    frame_generator = sv.get_video_frames_generator(video_path, stride=stride)
    
    crops = []
    for frame in frame_generator:
        detections = get_detections(player_detector, frame, confidence=0.3)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        players = detections[detections.class_id == PLAYER_ID]
        crops += [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
        
        if len(crops) >= max_crops:
            break
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    
    return team_classifier


def process_video(
    video_path,
    output_path,
    player_detector,
    field_detector,
    ball_yolo,
    team_classifier,
    start_frame=0,
    max_frames=None,
    progress_callback=None
):
    """
    Pipeline principal de procesamiento
    
    Args:
        video_path: Ruta del video de entrada
        output_path: Ruta del video de salida
        player_detector: Detector de jugadores
        field_detector: Detector de campo
        ball_yolo: Modelo YOLO del balón
        team_classifier: Clasificador de equipos
        start_frame: Frame inicial
        max_frames: Máximo de frames a procesar
        progress_callback: Función para actualizar progreso
    
    Returns:
        dict: Estadísticas del procesamiento
    """
    
    # Obtener info del video
    video_info = sv.VideoInfo.from_video_path(video_path)
    
    if max_frames:
        output_info = sv.VideoInfo(
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps,
            total_frames=max_frames
        )
    else:
        output_info = sv.VideoInfo(
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps,
            total_frames=video_info.total_frames - start_frame
        )
    
    # Inicializar trackers
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    tracker.reset()
    
    ball_tracker = BallTracker(buffer_size=30)
    ball_tracker.reset()
    
    # Anotadores por equipo
    ellipse_team0 = sv.EllipseAnnotator(
        color=sv.Color.from_hex('#00BFFF'), 
        thickness=2
    )
    ellipse_team1 = sv.EllipseAnnotator(
        color=sv.Color.from_hex('#FF1493'), 
        thickness=2
    )
    ellipse_referee = sv.EllipseAnnotator(
        color=sv.Color.from_hex('#FFD700'), 
        thickness=2
    )
    
    label_team0 = sv.LabelAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER,
        text_scale=0.5,
        text_thickness=1,
        text_padding=5
    )
    label_team1 = sv.LabelAnnotator(
        color=sv.Color.from_hex('#FF1493'),
        text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER,
        text_scale=0.5,
        text_thickness=1,
        text_padding=5
    )
    label_referee = sv.LabelAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER,
        text_scale=0.5,
        text_thickness=1,
        text_padding=5
    )
    
    ball_annotator = BallAnnotator(
        radius=12,
        ball_color=(0, 255, 0),
        trail_color=(0, 255, 255)
    )
    
    # Estadísticas
    stats = {
        'frames_procesados': 0,
        'jugadores_equipo_0': 0,
        'jugadores_equipo_1': 0,
        'porteros_detectados': 0,
        'arbitros_detectados': 0,
        'detecciones_balon': 0,
        'total_detecciones': 0
    }
    
    # Procesamiento frame por frame
    frame_generator = sv.get_video_frames_generator(
        source_path=video_path,
        start=start_frame,
        end=start_frame + max_frames if max_frames else None
    )
    
    with sv.VideoSink(output_path, output_info) as sink:
        for frame_idx, frame in enumerate(frame_generator):
            stats['frames_procesados'] += 1
            
            # Actualizar progreso
            if progress_callback and frame_idx % 10 == 0:
                progress = (frame_idx + 1) / output_info.total_frames
                progress_callback(progress)
            
            # Detección de jugadores
            detections = get_detections(player_detector, frame, confidence=0.3)
            
            if len(detections) == 0:
                sink.write_frame(frame)
                continue
            
            stats['total_detecciones'] += len(detections)
            detections = tracker.update_with_detections(detections)
            
            # Detección del balón
            if ball_yolo:
                try:
                    ball_results = ball_yolo.predict(frame, conf=0.3, verbose=False)[0]
                    ball_detections = sv.Detections.from_ultralytics(ball_results)
                    if len(ball_detections) > 0:
                        ball_detections = ball_detections.with_nms(threshold=0.1)
                except:
                    ball_detections = detections[detections.class_id == BALL_ID]
            else:
                ball_detections = detections[detections.class_id == BALL_ID]
                if len(ball_detections) > 0:
                    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            
            ball_position = ball_tracker.update(ball_detections)
            if ball_position is not None:
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
                players_team_id = team_classifier.predict(crops)
                
                stats['jugadores_equipo_0'] += (players_team_id == 0).sum()
                stats['jugadores_equipo_1'] += (players_team_id == 1).sum()
                
                team0_mask = players_team_id == 0
                team1_mask = players_team_id == 1
                
                if team0_mask.any():
                    players_team0 = players[team0_mask]
                if team1_mask.any():
                    players_team1 = players[team1_mask]
            
            # Asignar porteros a equipos
            if len(goalkeepers) > 0 and len(players) > 0:
                merged_players = sv.Detections.merge([players_team0, players_team1])
                
                # Asignar class_id temporal para resolve_goalkeepers
                if len(players_team0) > 0:
                    players_team0.class_id = np.zeros(len(players_team0), dtype=int)
                if len(players_team1) > 0:
                    players_team1.class_id = np.ones(len(players_team1), dtype=int)
                
                merged_players = sv.Detections.merge([players_team0, players_team1])
                goalkeepers_team_id = resolve_goalkeepers_team_id(merged_players, goalkeepers)
                
                team0_gk_mask = goalkeepers_team_id == 0
                team1_gk_mask = goalkeepers_team_id == 1
                
                if team0_gk_mask.any():
                    players_team0 = sv.Detections.merge([players_team0, goalkeepers[team0_gk_mask]])
                if team1_gk_mask.any():
                    players_team1 = sv.Detections.merge([players_team1, goalkeepers[team1_gk_mask]])
            
            # Anotar frame
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
            
            # Dibujar balón con trail
            annotated = ball_annotator.annotate(annotated, ball_position, ball_tracker.buffer)
            
            # Información en pantalla
            info_y = 30
            cv2.putText(annotated, f"Frame: {start_frame + frame_idx}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
            cv2.putText(annotated, f"Azul: {len(players_team0)}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 191, 0), 2)
            info_y += 25
            cv2.putText(annotated, f"Rojo: {len(players_team1)}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 20, 147), 2)
            info_y += 25
            
            ball_status = "SI" if ball_position is not None else "NO"
            ball_color = (0, 255, 0) if ball_position is not None else (0, 0, 255)
            cv2.putText(annotated, f"Balon: {ball_status}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)
            
            sink.write_frame(annotated)
    
    return stats