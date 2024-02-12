from abc import ABC
import numpy as np


class Box(ABC):
    
    def __init__(self, trackid: int, classid: int, score: float, x1: int, y1: int, x2: int, y2: int, cx: int, cy: int, is_used: bool):
        self.trackid: int = trackid
        self.classid: int = classid
        self.score: float = score
        self.x1: int = x1
        self.y1: int = y1
        self.x2: int = x2
        self.y2: int = y2
        self.cx: int = cx
        self.cy: int = cy
        self.is_used: bool = is_used
        

class Body(Box):
    def __init__(self, trackid: int, classid: int, score: float, x1: int, y1: int, x2: int, y2: int, cx: int, cy: int, is_used: bool, head: Box, hand1: Box, hand2: Box):
        super().__init__(trackid=trackid, classid=classid, score=score, x1=x1, y1=y1, x2=x2, y2=y2, cx=cx, cy=cy, is_used=is_used)
        self.head: Head = head
        

class Head(Box):
    def __init__(self, trackid: int, classid: int, score: float, x1: int, y1: int, x2: int, y2: int, cx: int, cy: int, is_used: bool, face: Box, face_landmarks: np.ndarray):
        super().__init__(trackid=trackid, classid=classid, score=score, x1=x1, y1=y1, x2=x2, y2=y2, cx=cx, cy=cy, is_used=is_used)
        self.face: Box = face
        self.face_landmarks: np.ndarray = face_landmarks
        
