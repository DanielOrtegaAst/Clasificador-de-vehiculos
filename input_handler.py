# input_handler.py
import cv2
from pathlib import Path
import glob
import os
import shutil
import time

class JobInputController:
    def __init__(self, job_source_type_param, job_source_path_param, app_config_instance):
        self.config_app = app_config_instance
        self.job_source_type = job_source_type_param
        self.job_source_path = job_source_path_param
        self.debug_mode = self.config_app.get('processing.debug_mode', False)

        # ... (inicialización de watch_folder_path_str, etc., como antes) ...
        self.watch_folder_path_str = self.config_app.get('source.watch_folder_path')
        self.image_glob_pattern = self.config_app.get('source.image_folder_glob', "*.jpg")
        self.processed_suffix = self.config_app.get('source.processed_folder_suffix', '_procesado')
        self.move_to_processed_path_str = self.config_app.get('source.move_to_processed_path')


        self.cap = None
        self.current_job_image_files = []
        self.current_job_image_idx = 0
        self.current_processing_folder_path = None
        
        # --- INICIO: Almacenamiento de Múltiples Frames para Payload ---
        self.first_frame_of_job = None
        self.middle_frame_of_job = None
        self.last_frame_of_job = None
        self.job_total_frames_read = 0 # Para determinar el frame intermedio
        # --- FIN: Almacenamiento de Múltiples Frames ---

        if self.debug_mode:
            print(f"[JOB_INPUT] Inicializando. Job_Type='{self.job_source_type}', Job_Path='{self.job_source_path}'")
            # ... (resto de los prints de debug del init)

        if self.job_source_path is None and self.job_source_type not in ["watch_folder"]:
             raise ValueError(f"Error: job_source_path es None para el tipo '{self.job_source_type}'.")

        if self.job_source_type in ["rtsp", "video_file"]:
            # ... (inicialización de self.cap como antes) ...
            if not self.job_source_path: raise ValueError(f"Ruta vacía para {self.job_source_type}")
            self.cap = cv2.VideoCapture(self.job_source_path)
            if not self.cap.isOpened(): raise ConnectionError(f"No se pudo abrir video: {self.job_source_path}")
            if self.debug_mode: print(f"  Fuente de video del trabajo '{self.job_source_path}' abierta.")
        elif self.job_source_type == "image_file":
            # ... (inicialización como antes) ...
            if not self.job_source_path or not Path(self.job_source_path).is_file():
                raise FileNotFoundError(f"Archivo de imagen no encontrado: {self.job_source_path}")
            self.current_job_image_files = [self.job_source_path]
            if self.debug_mode: print(f"  Fuente de imagen individual: {self.job_source_path}")
        elif self.job_source_type == "image_folder":
            # ... (inicialización como antes) ...
            folder_path = Path(self.job_source_path)
            if not folder_path.is_dir(): raise NotADirectoryError(f"Carpeta no encontrada: {self.job_source_path}")
            self.current_job_image_files = sorted([str(p) for p in folder_path.glob(self.image_glob_pattern)])
            if not self.current_job_image_files: raise FileNotFoundError(f"No imágenes en carpeta: {self.job_source_path}")
            if self.debug_mode: print(f"  Cargadas {len(self.current_job_image_files)} imágenes para trabajo desde: {self.job_source_path}")
            self.current_processing_folder_path = folder_path
        elif self.job_source_type == "watch_folder":
            # ... (inicialización como antes) ...
            if not self.watch_folder_path_str or not Path(self.watch_folder_path_str).is_dir():
                raise NotADirectoryError(f"Carpeta a monitorear no encontrada: {self.watch_folder_path_str}")
        else:
            raise ValueError(f"Tipo de fuente de trabajo no soportado: {self.job_source_type}")

    def find_and_load_new_sequence(self):
        # ... (como antes) ...
        if self.job_source_type != "watch_folder": return False
        watch_path = Path(self.watch_folder_path_str)
        for item in sorted(watch_path.iterdir()):
            if item.is_dir() and not item.name.endswith(self.processed_suffix):
                if self.move_to_processed_path_str and Path(self.move_to_processed_path_str, item.name).exists():
                    continue
                image_files = sorted(list(item.glob(self.image_glob_pattern)))
                if image_files:
                    if self.debug_mode: print(f"[JOB_INPUT] Nueva secuencia encontrada: {item.name}")
                    self.current_processing_folder_path = item
                    self.current_job_image_files = [str(p) for p in image_files]
                    self.current_job_image_idx = 0
                    self.reset_payload_frames() # <--- Resetear frames para la nueva secuencia
                    return True
        return False

    def read_frame(self):
        frame = None
        ret = False
        current_file_name_for_api = self.get_current_processing_source_name()

        if self.job_source_type in ["rtsp", "video_file"]:
            if self.cap: ret, frame = self.cap.read()
        elif self.job_source_type in ["image_file", "image_folder", "watch_folder"]:
            if self.current_job_image_idx < len(self.current_job_image_files):
                image_path_str = self.current_job_image_files[self.current_job_image_idx]
                current_file_name_for_api = str(Path(image_path_str).name)
                frame = cv2.imread(image_path_str)
                self.current_job_image_idx += 1
                if frame is not None:
                    ret = True
                elif self.debug_mode: print(f"Advertencia [JOB_INPUT]: No se pudo leer imagen: {image_path_str}")
        
        if ret and frame is not None:
            self.job_total_frames_read += 1 # Incrementar contador de frames leídos para este job
            if self.job_total_frames_read == 1: # Primer frame del job
                self.first_frame_of_job = frame.copy()
                if self.debug_mode: print(f"[JOB_INPUT] Primer frame del job capturado (Fuente: {current_file_name_for_api})")
            
            # Siempre actualizar el último frame
            self.last_frame_of_job = frame.copy()

            # Lógica para el frame intermedio (aproximado)
            # Esto se podría hacer de forma más precisa si se conoce el total de frames de antemano
            # o actualizándolo si el total de frames es pequeño.
            # Para una secuencia de imágenes, podemos calcular el medio una vez que las hemos leído todas.
            # Para un stream, es más complicado.
            # Por ahora, actualizaremos el "middle" si estamos a mitad de una secuencia de imágenes conocida.
            if self.job_source_type in ["image_file", "image_folder", "watch_folder"]:
                total_files = len(self.current_job_image_files)
                if total_files > 2 and self.job_total_frames_read == (total_files // 2) + (total_files % 2): # Frame central
                    self.middle_frame_of_job = frame.copy()
                    if self.debug_mode: print(f"[JOB_INPUT] Frame intermedio del job capturado (Fuente: {current_file_name_for_api})")
            # Para RTSP/video, capturar un frame intermedio es más difícil sin saber la duración.
            # Se podría tomar uno después de X segundos o Y frames, pero no sería el "verdadero" medio.
            # Lo dejamos así por ahora, enfocándonos en secuencias de imágenes.

        return ret, frame, current_file_name_for_api

    def get_payload_images(self): # <--- NUEVO MÉTODO
        """Devuelve un diccionario con los frames seleccionados para el payload."""
        images = {}
        if self.config_app.get('processing.payload_images.include_first', False) and self.first_frame_of_job is not None:
            images['first'] = self.first_frame_of_job
        
        # Para secuencias de imágenes, si no se capturó un frame intermedio exacto,
        # y si el total de frames es > 2, podemos tomar el del medio de la lista de archivos procesados.
        if self.config_app.get('processing.payload_images.include_middle', False):
            if self.middle_frame_of_job is not None:
                 images['middle'] = self.middle_frame_of_job
            elif self.job_source_type in ["image_file", "image_folder", "watch_folder"] and len(self.current_job_image_files) > 2:
                # Si no se capturó dinámicamente, tomar el del medio de la lista de archivos si ya se procesaron todos
                # Esto es más para el final del job. Durante el read_frame, ya intentamos capturarlo.
                # Aquí podríamos recalcular si es necesario, pero es mejor la captura dinámica.
                # Por ahora, solo usamos el capturado dinámicamente.
                pass


        if self.config_app.get('processing.payload_images.include_last', False) and self.last_frame_of_job is not None:
            images['last'] = self.last_frame_of_job
        
        if self.debug_mode: print(f"[JOB_INPUT] Imágenes para payload recuperadas: {list(images.keys())}")
        return images
    
    def reset_payload_frames(self): # <--- NUEVO MÉTODO (o renombrado y expandido)
        if self.debug_mode: print("[JOB_INPUT] Reseteando frames para payload.")
        self.first_frame_of_job = None
        self.middle_frame_of_job = None
        self.last_frame_of_job = None
        self.job_total_frames_read = 0

    def mark_current_sequence_as_processed(self):
        # ... (como antes) ...
        if not self.current_processing_folder_path or not self.current_processing_folder_path.exists():
            if self.debug_mode and self.current_processing_folder_path : print(f"[JOB_INPUT] Carpeta {self.current_processing_folder_path} no existe para marcarla.")
            return
        # ... (resto de la lógica de renombrar/mover)
        self.reset_last_frame_for_payload() # Asegurarse de resetearlo

    def release(self):
        if self.cap: self.cap.release()
        if self.debug_mode: print(f"[JOB_INPUT] Recurso de captura liberado para: {self.job_source_path}")

    def get_current_processing_source_name(self):
        # ... (como antes) ...
        if self.current_processing_folder_path: return self.current_processing_folder_path.name
        if self.job_source_type == "image_file" and self.current_job_image_files:
             return str(Path(self.current_job_image_files[0]).name)
        return str(Path(self.job_source_path).name if self.job_source_path and Path(self.job_source_path).is_file() else self.job_source_path)