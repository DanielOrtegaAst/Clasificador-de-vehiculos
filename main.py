import cv2
import time
from pathlib import Path
from flask import Flask, request, jsonify
import threading
import argparse
import datetime
import json
import base64 # Para codificar el video
import os

from config_loader import AppConfig
from utils import draw_vehicle_tire_counts # Ya no necesitamos encode_image_to_base64 aquí directamente
from input_handler import JobInputController
from detector import ObjectDetector
from tracker_logic import TireCounterLogic
from api_client import APIClient

# --- Variables Globales para Componentes Compartidos ---
cfg_global = None
detector_global = None
api_client_global = None

# --- Servidor Flask ---
flask_app = Flask(__name__)
processing_lock = threading.Lock()
job_queue = []

def initialize_global_components():
    global detector_global, api_client_global, cfg_global
    print("[MAIN] Inicializando componentes globales (Config, Detector YOLO, API Client)...")
    try:
        cfg_global = AppConfig(config_path_str="config.yaml")
        detector_global = ObjectDetector(cfg_global)
        if cfg_global.get('external_server.enabled'):
            api_client_global = APIClient(cfg_global)
        print("[MAIN] Componentes globales inicializados.")
        return True
    except Exception as e:
        print(f"Error crítico durante la inicialización global de componentes: {e}")
        import traceback
        traceback.print_exc()
        return False

@flask_app.route('/process_vehicle_data', methods=['POST'])
def queue_process_request():
    global cfg_global
    timestamp_recepcion = datetime.datetime.now().isoformat()
    if cfg_global and cfg_global.get('processing.debug_mode'):
        print(f"[{timestamp_recepcion}] Petición POST recibida en /process_vehicle_data")

    if not request.is_json:
        return jsonify({"status": "error", "message": "Payload debe ser JSON"}), 400
    
    data = request.get_json()
    job_source_type = data.get('source_type')
    job_source_path = data.get('source_path')

    if not job_source_type or not job_source_path:
        return jsonify({"status": "error", "message": "Faltan 'source_type' o 'source_path'"}), 400

    with processing_lock:
        job_queue.append({'type': job_source_type, 'path': job_source_path, 'received_at': timestamp_recepcion})
    
    if cfg_global and cfg_global.get('processing.debug_mode'):
        print(f"  Trabajo para '{job_source_path}' (Tipo: {job_source_type}) añadido a la cola. Trabajos pendientes: {len(job_queue)}")
    return jsonify({"status": "success", "message": f"Trabajo para '{job_source_path}' encolado."}), 202

def job_processor_worker():
    global cfg_global, detector_global, api_client_global
    tire_counter_worker = TireCounterLogic(cfg_global, api_client_instance=api_client_global)
    print("[JOB_WORKER] Hilo procesador de trabajos iniciado.")

    while True:
        current_job = None
        with processing_lock:
            if job_queue: current_job = job_queue.pop(0)
        
        if current_job:
            job_type, job_path = current_job['type'], current_job['path']
            job_name = str(Path(job_path).name)
            display_window_title = f"Procesando Job: {job_name}"
            visualization_active_for_this_job = False
            
            print(f"\n[JOB_WORKER] Iniciando procesado para: '{job_name}' (Tipo: {job_type})")

            # --- INICIO: Preparación para Video de Salida ---
            processed_frames_for_video = [] # Lista para guardar los frames anotados
            video_payload_config = cfg_global.get('processing.payload_video', {})
            create_video_output = video_payload_config.get('include_processed_video', False)
            # --- FIN: Preparación para Video de Salida ---

            try:
                job_input_ctrl = JobInputController(job_type, job_path, cfg_global)
                tire_counter_worker.reset_state_for_new_job()
                frame_idx_job = 0
                # first_frame_of_job = None # Ya no se usa para enviar una sola imagen estática
                processed_successfully = True

                while True:
                    ret, frame, current_file_name_api = job_input_ctrl.read_frame()
                    if not ret: break
                    
                    frame_idx_job += 1
                    yolo_results = detector_global.track_objects(frame.copy())
                    current_vehicle_detections_this_frame = tire_counter_worker.process_job_detections(
                        yolo_results, frame_idx_job, frame.shape
                    )

                    output_frame_for_display_and_video = frame.copy()
                    if yolo_results: # Dibujar siempre para el video si está habilitado, y para display si está habilitado
                        plot_options = cfg_global.get('processing.visualization_plot_options', {})
                        plot_args = { "conf": plot_options.get('show_conf',True), "line_width": plot_options.get('line_width',1), "font_size": plot_options.get('font_size',0.4), "labels": plot_options.get('show_labels',True) }
                        output_frame_for_display_and_video = yolo_results.plot(**plot_args)
                    
                    output_frame_for_display_and_video = draw_vehicle_tire_counts(
                        output_frame_for_display_and_video, 
                        current_vehicle_detections_this_frame, 
                        tire_counter_worker.vehicle_physical_tires_current_job,
                        cfg_global
                    )

                    if create_video_output: # Guardar frame para el video
                        # Redimensionar frame para el video de salida si es necesario
                        vid_w = video_payload_config.get('output_video_frame_max_width', 0)
                        vid_h = video_payload_config.get('output_video_frame_max_height', 0)
                        frame_to_write = output_frame_for_display_and_video
                        if vid_w > 0 and vid_h > 0:
                            current_h, current_w = frame_to_write.shape[:2]
                            if current_w != vid_w or current_h != vid_h: # Solo redimensionar si es diferente
                                frame_to_write = cv2.resize(frame_to_write, (vid_w, vid_h), interpolation=cv2.INTER_AREA)
                        processed_frames_for_video.append(frame_to_write)


                    if cfg_global.get('processing.show_visualization_per_job'):
                        visualization_active_for_this_job = True
                        cv2.imshow(display_window_title, output_frame_for_display_and_video)
                        key_press = cv2.waitKey(cfg_global.get('processing.visualization_wait_key',1)) & 0xFF
                        if key_press == ord('q'):
                            processed_successfully = False; break
                
                if visualization_active_for_this_job:
                    try: cv2.destroyWindow(display_window_title)
                    except: pass
                
                video_base64 = None
                if processed_successfully and create_video_output and processed_frames_for_video:
                    if cfg_global.get('processing.debug_mode'): print(f"  [JOB_WORKER] Creando video de salida para '{job_name}' con {len(processed_frames_for_video)} frames.")

                    first_processed_frame = processed_frames_for_video[0]
                    height, width, _ = first_processed_frame.shape # Usar dimensiones del frame (ya redimensionado para el video)

                    # Leer codec y extensión de la configuración
                    codec_str = video_payload_config.get('output_video_codec', 'mp4v') # Leerá 'avc1'
                    video_ext = video_payload_config.get('output_video_extension', '.mp4') # Leerá '.mp4'

                    temp_video_filename = f"temp_output_{job_name.replace(' ', '_').replace('.', '_')}{video_ext}" # Asegurar que el nombre sea seguro

                    # Aplicar el FourCC leído de la config
                    fourcc = cv2.VideoWriter_fourcc(*codec_str) # Ej: cv2.VideoWriter_fourcc(*'avc1')

                    out_video_fps = video_payload_config.get('output_video_fps', 10)

                    if cfg_global.get('processing.debug_mode'):
                        print(f"    Creando video con: filename='{temp_video_filename}', fourcc='{codec_str}', fps={out_video_fps}, size=({width}x{height})")

                    video_writer = cv2.VideoWriter(temp_video_filename, fourcc, out_video_fps, (width, height))

                    if not video_writer.isOpened():
                        print(f"    [JOB_WORKER] ERROR: No se pudo abrir VideoWriter para '{temp_video_filename}' con codec '{codec_str}'. ¿Está soportado?")
                    else:
                        for proc_frame in processed_frames_for_video:
                            video_writer.write(proc_frame)
                        video_writer.release()
                        if cfg_global.get('processing.debug_mode'): print(f"    [JOB_WORKER] Video temporal '{temp_video_filename}' CREADO y cerrado.")

                        # ... (resto de la lógica para codificar a Base64 y eliminar el temporal)
                        try:
                            with open(temp_video_filename, "rb") as video_file:
                                video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
                            if cfg_global.get('processing.debug_mode'): print(f"    [JOB_WORKER] Video codificado a Base64 (longitud: {len(video_base64)}).")
                        except Exception as e_b64:
                            print(f"    [JOB_WORKER] Error codificando video a Base64: {e_b64}")
                        finally:
                            try:
                                if os.path.exists(temp_video_filename): # Verificar antes de borrar
                                    os.remove(temp_video_filename)
                                    if cfg_global.get('processing.debug_mode'): print(f"    [JOB_WORKER] Video temporal '{temp_video_filename}' eliminado.")
                            except Exception as e_del:
                                if cfg_global.get('processing.debug_mode'):print(f"    [JOB_WORKER] No se pudo eliminar el video temporal '{temp_video_filename}': {e_del}")


                if processed_successfully:
                    final_payload = tire_counter_worker.finalize_job_and_prepare_payload(job_source_name=job_name)
                    if final_payload and api_client_global:
                        if cfg_global.get('processing.debug_mode'):print(f"  [JOB_WORKER] Enviando resultado final para '{job_name}'...")
                        api_client_global.send_vehicle_data(
                            final_payload, 
                            video_base64_to_send=video_base64, # Nuevo argumento
                            job_source_name=job_name
                        )
                    # ... (resto del log si el envío API está deshabilitado) ...
                # ... (resto del bloque try como estaba) ...
            # ... (except y finally del job como estaban) ...
            except Exception as e_job: # Mover job_input_ctrl.release() al finally del job
                print(f"  [JOB_WORKER] ERROR CRÍTICO procesando el trabajo para '{job_name}': {e_job}")
                import traceback
                traceback.print_exc()
            finally:
                if 'job_input_ctrl' in locals() and job_input_ctrl: job_input_ctrl.release()
                if visualization_active_for_this_job:
                    try: cv2.destroyWindow(display_window_title)
                    except: pass
            if cfg_global.get('processing.debug_mode'): print(f"[JOB_WORKER] Procesamiento de trabajo '{job_name}' finalizado.")
        else:
            time.sleep(0.05) # Espera corta si no hay jobs


if __name__ == '__main__':
    if not initialize_global_components():
        exit()

    processor_thread = threading.Thread(target=job_processor_worker, daemon=True)
    processor_thread.start()

    flask_server_enabled = cfg_global.get('command_server.enabled', False)
    
    parser = argparse.ArgumentParser(description="Aplicación de conteo de vehículos y llantas.")
    parser.add_argument("--process_folder",type=str,default=None,help="Ruta a carpeta para procesar (ejecución única).")
    args = parser.parse_args()

    # Si se pasa --process_folder, se añade a la cola y el worker lo tomará.
    if args.process_folder:
        print(f"Modo de ejecución única: Añadiendo carpeta '{args.process_folder}' a la cola de trabajos.")
        with processing_lock: # Asegurar acceso seguro a la cola
            job_queue.append({'type': "image_folder", 'path': args.process_folder, 'received_at': datetime.datetime.now().isoformat()})
        
        # Si Flask no va a correr, necesitamos una forma de que el programa principal espere
        # a que este job único termine antes de salir.
        if not flask_server_enabled:
            print("[MAIN] Servidor Flask DESHABILITADO. Esperando que el job de --process_folder termine...")
            while True:
                time.sleep(1)
                with processing_lock:
                    if not job_queue: # Suponiendo que el worker lo toma y lo quita rápidamente
                        # Podríamos necesitar un chequeo más robusto o una señal del worker
                        print("[MAIN] Cola de trabajos vacía. Asumiendo que el job de --process_folder ha terminado.")
                        break
            time.sleep(2) # Dar un par de segundos extra para que el worker imprima mensajes finales


    if flask_server_enabled:
        server_host = cfg_global.get('command_server.host', '0.0.0.0')
        server_port = cfg_global.get('command_server.port', 5001)
        print(f"[MAIN] Iniciando servidor Flask de comandos en http://{server_host}:{server_port}")
        try:
            flask_app.run(host=server_host, port=server_port, debug=cfg_global.get('processing.debug_mode', False), use_reloader=False)
        except KeyboardInterrupt: print("\n[MAIN] Servidor Flask detenido por el usuario.")
        except Exception as e_flask: print(f"[MAIN] Error en servidor Flask: {e_flask}")
    elif not args.process_folder: # Si Flask está deshabilitado Y no se pasó --process_folder
        print("[MAIN] Servidor de comandos Flask DESHABILITADO y no se especificó --process_folder.")
        print("La aplicación se cerrará. Para mantenerla activa, habilita el servidor de comandos o procesa una carpeta.")

    print("[MAIN] Señalando al hilo procesador que podría ser momento de terminar (si está inactivo)...")
    # El hilo worker es daemon, así que terminará cuando el hilo principal (Flask o el bucle de espera) termine.
    # Si queremos un cierre más "gracioso" del worker, se necesitaría un evento de parada.

    print("[MAIN] Aplicación finalizada.")
    # Ya no llamamos a cv2.destroyAllWindows() aquí, se maneja por job o al final de la app si es necesario.
    # Si se usó visualización, es probable que las ventanas ya estén cerradas o el usuario las cierre.
    # Si queremos un cierre forzado de CUALQUIER ventana de OpenCV restante al final de la app:
    if cfg_global and cfg_global.get('processing.show_visualization_per_job'):
         cv2.destroyAllWindows()