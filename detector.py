# detector.py
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, config):
        self.model_path = config.get('model.path')
        self.tracker_config = config.get('model.tracker_config_file', "bytetrack.yaml") # Default si no está en config
        self.min_global_conf = config.get('model.min_global_confidence_for_tracker', 0.1)
        self.debug_mode = config.get('processing.debug_mode', False)
        
        try:
            self.model = YOLO(self.model_path)
            if self.debug_mode: print(f"[DETECTOR] Modelo YOLO cargado desde: {self.model_path}")
        except Exception as e:
            print(f"Error Crítico: No se pudo cargar el modelo YOLO desde '{self.model_path}'.")
            raise RuntimeError(f"Fallo al cargar modelo YOLO: {e}")

    def track_objects(self, frame):
        if frame is None:
            if self.debug_mode: print("[DETECTOR] Error: Frame de entrada es None para track_objects.")
            return None
        try:
            results = self.model.track(source=frame, persist=True, 
                                       tracker=self.tracker_config, 
                                       conf=self.min_global_conf, 
                                       verbose=False) 
            return results[0] 
        except Exception as e:
            print(f"Error durante model.track(): {e}")
            return None