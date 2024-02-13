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
        
