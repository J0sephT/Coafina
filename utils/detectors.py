# utils/detectors.py
"""
Detectores YOLO para jugadores, campo y balón
"""

import supervision as sv
import numpy as np

# IDs de clases
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3


class PlayerDetector:
    """Detector de jugadores usando YOLO"""
    
    def __init__(self, model):
        self.model = model
    
    def infer(self, frame, confidence=0.3):
        """Inferencia en un frame"""
        return self.model.predict(frame, conf=confidence, verbose=False)


class FieldDetector:
    """Detector de campo usando YOLO"""
    
    def __init__(self, model):
        self.model = model
    
    def infer(self, frame, confidence=0.3):
        """Inferencia en un frame"""
        return self.model.predict(frame, conf=confidence, verbose=False)


def get_detections(model, frame, confidence=0.3):
    """
    Obtener detecciones de un frame
    
    Args:
        model: Detector (PlayerDetector o FieldDetector)
        frame: Frame del video
        confidence: Umbral de confianza
    
    Returns:
        supervision.Detections
    """
    results = model.infer(frame, confidence=confidence)
    result = results[0] if isinstance(results, list) else results
    return sv.Detections.from_ultralytics(result)


def get_keypoints(model, frame, confidence=0.3):
    """
    Obtener keypoints del campo
    
    Args:
        model: FieldDetector
        frame: Frame del video
        confidence: Umbral de confianza
    
    Returns:
        supervision.KeyPoints o None
    """
    results = model.infer(frame, confidence=confidence)
    result = results[0] if isinstance(results, list) else results
    
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        if result.keypoints.xy.numel() > 0:
            return sv.KeyPoints.from_ultralytics(result)
    
    return None