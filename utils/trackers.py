# utils/trackers.py
"""
Trackers para el balón y anotadores visuales
"""

import cv2
import numpy as np
from collections import deque
import supervision as sv


class BallTracker:
    """Tracker simple del balón basado en distancia"""
    
    def __init__(self, buffer_size: int = 30):
        self.buffer = deque(maxlen=buffer_size)
        self.last_position = None
    
    def update(self, detections):
        """
        Actualiza la posición del balón
        
        Args:
            detections: supervision.Detections del balón
        
        Returns:
            tuple o None: (x, y) del balón
        """
        if len(detections) == 0:
            return None
        
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        
        if self.last_position is None:
            self.last_position = xy[0]
        else:
            distances = np.linalg.norm(xy - self.last_position, axis=1)
            index = np.argmin(distances)
            self.last_position = xy[index]
        
        self.buffer.append(self.last_position)
        return self.last_position
    
    def reset(self):
        """Resetear el tracker"""
        self.buffer.clear()
        self.last_position = None


class BallAnnotator:
    """Anotador para dibujar el balón y su trayectoria"""
    
    def __init__(self, 
                 radius: int = 12,
                 ball_color=(0, 255, 0),      # Verde en BGR
                 trail_color=(0, 255, 255)):  # Amarillo en BGR
        self.radius = radius
        self.ball_color = ball_color
        self.trail_color = trail_color
    
    def annotate(self, frame: np.ndarray, position, trail_buffer) -> np.ndarray:
        """
        Dibuja el balón y su trayectoria
        
        Args:
            frame: Frame del video
            position: Posición actual del balón
            trail_buffer: Buffer de posiciones previas
        
        Returns:
            Frame anotado
        """
        annotated = frame.copy()
        
        # Dibujar trayectoria
        for i in range(1, len(trail_buffer)):
            if trail_buffer[i - 1] is None or trail_buffer[i] is None:
                continue
            
            x1, y1 = map(int, trail_buffer[i - 1])
            x2, y2 = map(int, trail_buffer[i])
            
            alpha = i / len(trail_buffer)
            thickness = max(2, int(4 * alpha))
            
            cv2.line(annotated, (x1, y1), (x2, y2), 
                    self.trail_color, thickness, cv2.LINE_AA)
        
        # Dibujar balón actual
        if position is not None:
            x, y = map(int, position)
            
            # Halo blanco
            cv2.circle(annotated, (x, y), self.radius + 2, 
                      (255, 255, 255), -1, cv2.LINE_AA)
            
            # Círculo principal
            cv2.circle(annotated, (x, y), self.radius, 
                      self.ball_color, -1, cv2.LINE_AA)
            
            # Borde negro
            cv2.circle(annotated, (x, y), self.radius, 
                      (0, 0, 0), 1, cv2.LINE_AA)
        
        return annotated