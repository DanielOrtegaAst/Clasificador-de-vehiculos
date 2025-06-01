import cv2
from pathlib import Path
import glob
import os
import shutil
import time

class JobInputController:
    """
    Gestiona la carga de frames para un "trabajo" de procesamiento específico.
    Un trabajo puede ser una sola imagen, un archivo de video, una carpeta de imágenes (secuencia),
    o puede operar en modo "watch_folder" para encontrar y procesar secuencias en subcarpetas.
    """
    def __init__(self, job_source_type_param, job_source_path_param, app_config_instance):
        """
        Inicializa el controlador de entrada para un trabajo o para el modo watch_folder.

        Args:
            job_source_type_param (str): Tipo de fuente para este trabajo/operación 
                                         (ej. "image_folder", "video_file", "watch_folder").
            job_source_path_param (str or None): Ruta a la fuente. Para "watch_folder", esta es la
                                                 carpeta raíz a monitorear, no una secuencia específica.
            app_config_instance (AppConfig): Instancia de la configuración global de la aplicación.
        """
        self.config_app = app_config_instance
        self.job_source_type = job_source_type_param
        self.job_source_path = job_source_path_param # Tipo de operación para esta instancia
        
        self.debug_mode = self.config_app.get('processing.debug_mode', False)

        # Parámetros para modo watch_folder (leídos de la config global)
        self.watch_folder_path_str = self.config_app.get('source.watch_folder_path')
        self.image_glob_pattern = self.config_app.get('source.image_folder_glob', "*.jpg")
        self.processed_suffix = self.config_app.get('source.processed_folder_suffix', '_procesado')
        self.move_to_processed_path_str = self.config_app.get('source.move_to_processed_path')


        self.cap = None # Para VideoCapture de rtsp/video_file
        self.current_job_image_files = [] # Lista de archivos para image_file/image_folder/secuencia de watch_folder
        self.current_job_image_idx = 0 # Índice para iterar sobre current_job_image_files

        # Path de la subcarpeta de secuencia actual (en modo watch_folder o si el job es image_folder)
        self.current_processing_folder_path = None
        
        # Almacenamiento de Múltiples Frames para Payload ---
        self.first_frame_of_job = None # Primer frame leído del job actual
        self.middle_frame_of_job = None # Frame intermedio leído del job actual
        self.last_frame_of_job = None # Último frame leído del job actual
        self.job_total_frames_read = 0 # Contador de frames leídos para el job actual

        if self.debug_mode:
            print(f"[JOB_INPUT] Inicializando. Job_Type='{self.job_source_type}', Job_Path='{self.job_source_path}'")
            # ... (resto de los prints de debug del init)

        # Inicializar fuente si no es watch_folder (watch_folder carga secuencias dinámicamente)
        if self.job_source_path is None and self.job_source_type not in ["watch_folder"]:
             raise ValueError(f"Error: job_source_path es None para el tipo '{self.job_source_type}'.")

        if self.job_source_type in ["rtsp", "video_file"]:
            if not self.job_source_path: raise ValueError(f"Ruta vacía para {self.job_source_type}")
            self.cap = cv2.VideoCapture(self.job_source_path)
            if not self.cap.isOpened(): raise ConnectionError(f"No se pudo abrir video: {self.job_source_path}")
            if self.debug_mode: print(f"  Fuente de video del trabajo '{self.job_source_path}' abierta.")
        elif self.job_source_type == "image_file":
            if not self.job_source_path or not Path(self.job_source_path).is_file():
                raise FileNotFoundError(f"Archivo de imagen no encontrado: {self.job_source_path}")
            self.current_job_image_files = [self.job_source_path]
            if self.debug_mode: print(f"  Fuente de imagen individual: {self.job_source_path}")
        elif self.job_source_type == "image_folder":
            folder_path = Path(self.job_source_path)
            if not folder_path.is_dir(): raise NotADirectoryError(f"Carpeta no encontrada: {self.job_source_path}")
            self.current_job_image_files = sorted([str(p) for p in folder_path.glob(self.image_glob_pattern)])
            if not self.current_job_image_files: raise FileNotFoundError(f"No imágenes en carpeta: {self.job_source_path}")
            if self.debug_mode: print(f"  Cargadas {len(self.current_job_image_files)} imágenes para trabajo desde: {self.job_source_path}")
            self.current_processing_folder_path = folder_path
        elif self.job_source_type == "watch_folder":
            if not self.watch_folder_path_str or not Path(self.watch_folder_path_str).is_dir():
                raise NotADirectoryError(f"Carpeta a monitorear no encontrada: {self.watch_folder_path_str}")
        else:
            raise ValueError(f"Tipo de fuente de trabajo no soportado: {self.job_source_type}")

    def find_and_load_new_sequence(self):
        """
        Para modo 'watch_folder', busca y carga la siguiente secuencia de imágenes no procesada
        desde una subcarpeta de `watch_folder_root_path_str`.
        Actualiza el estado interno para procesar esta nueva secuencia.

        Returns:
            bool: True si una nueva secuencia fue encontrada y cargada, False en caso contrario.
        """
        if self.job_source_type != "watch_folder": return False
        watch_path = Path(self.watch_folder_path_str)
        for item in sorted(watch_path.iterdir()): # Procesar en orden alfabético para consistencia
            if item.is_dir() and not item.name.endswith(self.processed_suffix):
                # Chequeo adicional si se usa move_to_processed_path
                if self.move_to_processed_path_str and Path(self.move_to_processed_path_str, item.name).exists():
                    continue
                image_files = sorted(list(item.glob(self.image_glob_pattern)))
                if image_files:
                    if self.debug_mode: print(f"[JOB_INPUT] Nueva secuencia encontrada: {item.name}")
                    self.current_processing_folder_path = item
                    self.current_job_image_files = [str(p) for p in image_files]
                    self.current_job_image_idx = 0
                    self.reset_payload_frames() # <--- Resetear frames para la nueva secuencia
                    return True # Nueva secuencia lista
        return False # No hay nuevas secuencias

    def read_frame(self):
        """
        Lee el siguiente frame para el trabajo/secuencia actual.
        Actualiza `last_frame_for_payload_in_job` y `first_frame_of_job` (si es el primero del job).

        Returns:
            tuple: (bool ret, numpy.ndarray frame, str current_file_name_for_api)
                   ret es True si el frame se leyó correctamente.
                   frame es el frame leído (o None).
                   current_file_name_for_api es el nombre del archivo procesado (para image_folder/image_file)
                   o la ruta general de la fuente (para rtsp/video).
        """
        frame = None
        ret = False
        # Nombre por defecto para API, se sobreescribe para archivos de imagen individuales
        current_file_name_for_api = self.get_current_processing_source_name()

        # Determinar el tipo de fuente para la lectura actual
        # Si es watch_folder, pero ya se cargó una secuencia, opera como image_folder
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
            
            self.last_frame_of_job = frame.copy()

        return ret, frame, current_file_name_for_api
    
    def reset_payload_frames(self):
        """Resetea los frames guardados para el payload y el contador de frames del job."""
        if self.debug_mode: print("[JOB_INPUT] Reseteando frames para payload.")
        self.first_frame_of_job = None
        self.middle_frame_of_job = None
        self.last_frame_of_job = None
        self.job_total_frames_read = 0

    def release(self):
        """Libera el recurso de VideoCapture si se estaba usando."""
        if self.cap: self.cap.release()
        if self.debug_mode: print(f"[JOB_INPUT] Recurso de captura liberado para: {self.job_source_path}")

    def get_current_processing_source_name(self):
        """
        Devuelve un nombre descriptivo de la fuente que se está procesando actualmente.
        Para 'watch_folder', es el nombre de la subcarpeta.
        Para 'image_file' o 'image_folder' como job, es el nombre base.
        Para 'rtsp' o 'video_file', es la ruta/URL.
        """
        if self.current_processing_folder_path: return self.current_processing_folder_path.name
         # Para jobs directos de image_file o image_folder
        if self.job_source_type == "image_file" and self.current_job_image_files:
             return str(Path(self.current_job_image_files[0]).name)
        return str(Path(self.job_source_path).name if self.job_source_path and Path(self.job_source_path).is_file() else self.job_source_path)
    