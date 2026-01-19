
import math

class DistanceEstimator:
    def __init__(self, focal_length_px=1000, known_width_m=None, known_height_m=None):
        """
        Initialize estimator with camera focal length and object dimensions.
        """
        self.focal_length = focal_length_px
        
        # Approximate real-world object dimensions (meters)
        self.known_widths = known_width_m or {
            'car': 1.8,
            'bus': 2.5,
            'truck': 2.5,
            'person': 0.5,
            'bike': 0.6,
            'traffic sign': 0.6,
            'stop sign': 0.75,
            'cone': 0.3
        }
        self.known_heights = known_height_m or {
            'car': 1.5,
            'bus': 3.2,
            'truck': 3.5,
            'person': 1.7,
            'bike': 1.0,
            'traffic sign': 0.8, # Pole + sign often taller, but sign itself is small
            'stop sign': 0.75,
            'cone': 0.5
        }

    def estimate_distance(self, box, class_name):
        """
        Estimate distance based on bounding box height/width.
        :param box: [x1, y1, x2, y2]
        :param class_name: Detected class name
        :return: Distance in meters (float) or None if class unknown
        """
        if class_name not in self.known_heights:
            # Fallback or return None
            return None

        # Box dimensions
        x1, y1, x2, y2 = box
        h_px = y2 - y1
        
        # Distance = (Focal_Length * Real_Height) / Image_Height
        real_h = self.known_heights[class_name]
        
        if h_px == 0:
            return 0.0
            
        distance = (self.focal_length * real_h) / h_px
        return round(distance, 2)
