# server_receptor.py
from flask import Flask, request, jsonify, render_template_string, send_from_directory, url_for
import datetime
import json
import os
import base64
from pathlib import Path # Para manejo de rutas

# --- Configuración e Inicialización ---
SERVER_DEBUG_MODE = True # Default, se intentará sobreescribir con config
VIDEO_CODEC_CONFIG = 'mp4v' # Default
VIDEO_EXTENSION_CONFIG = '.mp4' # Default

try:
    # Asumimos que config_loader.py y config.yaml están en el mismo directorio o en PYTHONPATH
    from config_loader import AppConfig
    # Crear una instancia de configuración específica para el servidor receptor
    # Esto asume que config.yaml tiene las claves que el servidor necesita,
    # como 'processing.debug_mode' y 'processing.payload_video.output_video_codec'
    cfg_receptor_app = AppConfig(config_path_str="config.yaml") # Carga el config.yaml general
    SERVER_DEBUG_MODE = cfg_receptor_app.get('processing.debug_mode', True)
    # Obtener el codec de video de la configuración para pasarlo a la plantilla
    payload_video_config = cfg_receptor_app.get('processing.payload_video', {})
    VIDEO_CODEC_CONFIG = payload_video_config.get('output_video_codec', 'mp4v')
    VIDEO_EXTENSION_CONFIG = payload_video_config.get('output_video_extension', '.mp4')
    print(f"[SERVER_RECEPTOR] debug_mode: {SERVER_DEBUG_MODE}, video_codec_config: {VIDEO_CODEC_CONFIG}, video_ext_config: {VIDEO_EXTENSION_CONFIG}")
except ImportError:
    print("[SERVER_RECEPTOR] ADVERTENCIA: No se pudo importar AppConfig de config_loader. Usando SERVER_DEBUG_MODE=True y codecs/extensión por defecto.")
except FileNotFoundError:
    print("[SERVER_RECEPTOR] ADVERTENCIA: config.yaml no encontrado. Usando SERVER_DEBUG_MODE=True y codecs/extensión por defecto.")
except Exception as e_cfg_load:
    print(f"[SERVER_RECEPTOR] ADVERTENCIA: Error cargando config: {e_cfg_load}. Usando SERVER_DEBUG_MODE=True y codecs/extensión por defecto.")

app = Flask(__name__)
app.config['SERVER_DEBUG_MODE'] = SERVER_DEBUG_MODE # Guardar en la config de Flask
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024 # Límite de 64MB para el payload

LOG_FILE = "received_vehicle_data_detailed.log"
SUMMARY_LOG_FILE = "received_vehicle_summary.csv"

# Carpeta para guardar y servir los videos procesados
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_VIDEOS_DIR_NAME = "processed_videos"
PROCESSED_VIDEOS_ABSOLUTE_PATH = BASE_DIR / PROCESSED_VIDEOS_DIR_NAME
PROCESSED_VIDEOS_ABSOLUTE_PATH.mkdir(parents=True, exist_ok=True)

# Crear encabezado del CSV si no existe
if not os.path.exists(SUMMARY_LOG_FILE):
    try:
        with open(SUMMARY_LOG_FILE, "w", encoding="utf-8", newline='') as f_csv:
             import csv
             csv_writer = csv.writer(f_csv)
             csv_writer.writerow([
                 "timestamp_recepcion_servidor", "job_source_name", "vehicle_unique_id",
                 "vehicle_class", "tire_count", "source_id_cliente",
                 "timestamp_evento_cliente", "video_filename_o_status"
             ])
    except Exception as e:
        print(f"[SERVER_RECEPTOR] Error creando archivo CSV de resumen: {e}")

MAX_LOG_ENTRIES_IN_MEMORY = 100
recent_log_entries = [] # Lista de strings (cada entrada es un JSON del payload original)

IMAGE_DISPLAY_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Visor de Vehículo Procesado</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; color: #212529; }
        .container { max-width: 900px; margin: auto; background-color: #fff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
        h3 { color: #0056b3; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        th, td { text-align: left; padding: 10px; border: 1px solid #dee2e6; }
        th { background-color: #e9ecef; color: #495057; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        video { max-width: 100%; height: auto; border: 1px solid #ced4da; border-radius: 4px; margin-top: 10px; background-color: #f0f0f0; } /* Color de fondo si el video no carga */
        pre { background-color: #e9ecef; padding: 15px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; border: 1px solid #ced4da; font-size:0.9em; max-height: 300px; overflow-y: auto;}
        .log-link { margin-top: 25px; display: inline-block; margin-right:15px; text-decoration:none; background-color:#007bff; color:white; padding:8px 12px; border-radius:4px;}
        .log-link:hover{background-color:#0056b3;}
        .video-container p {font-style:italic; color:#6c757d; font-size:0.9em;}
    </style>
</head>
<body>
    <div class="container">
        <h1>Resultado del Procesamiento Vehicular</h1>
        {% if data_for_template %} {# Cambiado de 'data' a 'data_for_template' para coincidir con la llamada a render_template_string #}
            <h3>Información Principal del Vehículo</h3>
            <table>
                <tr><th>ID Único del Job (Vehículo)</th><td>{{ data_for_template.get('vehicle_unique_id', 'N/A') }}</td></tr>
                <tr><th>Clase de Vehículo</th><td>{{ data_for_template.get('vehicle_class', 'N/A') }}</td></tr>
                <tr><th>Conteo de Llantas</th><td>{{ data_for_template.get('tire_count', 'N/A') }}</td></tr>
                <tr><th>Fuente del Job (Cliente)</th><td>{{ data_for_template.get('job_source_name', 'N/A') }}</td></tr>
                <tr><th>ID de Fuente (Cliente)</th><td>{{ data_for_template.get('source_id', 'N/A') }}</td></tr>
                <tr><th>Timestamp Evento (Cliente)</th><td>{{ data_for_template.get('timestamp_event', 'N/A') }}</td></tr>
                <tr><th>Timestamp Recepción (Servidor)</th><td>{{ reception_time_for_template }}</td></tr>
                <tr><th>Video Guardado</th><td>{{ data_for_template.get('processed_video_filename', 'No procesado/guardado') }}</td></tr>
            </table>

            <div class="video-container">
                <h3>Video del Procesamiento:</h3>
                {% if data_for_template.get('processed_video_url_path') %}
                    <video controls width="720" preload="metadata" loop autoplay muted>
                        <source src="{{ data_for_template.processed_video_url_path }}" type="video/mp4">
                        Tu navegador no soporta la etiqueta de video o el formato MP4 (codec esperado: {{ video_codec_info_for_template }}).
                    </video>
                {% elif data_for_template.get('video_sent_status') == 'included_but_failed_to_save' %}
                    <p>Se recibió el video, pero hubo un error al guardarlo en el servidor.</p>
                {% elif data_for_template.get('video_sent_status') == 'configured_but_not_provided' %}
                    <p>Video configurado para envío por el cliente, pero no se incluyó en el payload.</p>
                {% elif data_for_template.get('video_sent_status') == 'not_included' %}
                     <p>El cliente no incluyó video en este payload.</p>
                {% else %}
                    <p>No hay video disponible para este evento.</p>
                {% endif %}
            </div>

            <hr style="margin-top:30px; margin-bottom:30px;">
            <h3>Payload JSON Completo Recibido (sin datos binarios de video):</h3>
            <pre>{{ raw_json_str_for_template }}</pre>
        {% else %}
            <p>Aún no se han recibido datos de vehículos. Envía un POST a <code>/api/vehicle_processed_data</code>.</p>
        {% endif %}
        <a href="{{ url_for('show_log_page') }}" class="log-link">Ver Log Detallado</a>
        <a href="{{ url_for('show_summary_log_page') }}" class="log-link">Ver Resumen CSV</a>
    </div>
</body>
</html>
"""

# Asegúrate que en tu función `show_last_vehicle_page` en server_receptor.py,
# estés pasando las variables a render_template_string con los nombres correctos:
# data_for_template=g_last_vehicle_data_for_template, 
# reception_time_for_template=g_last_reception_time_for_template,
# raw_json_str_for_template=g_last_raw_json_str_for_template,
# video_codec_info_for_template=VIDEO_CODEC_CONFIG

# Variables globales para almacenar los datos del último vehículo para la página principal
g_last_vehicle_data_for_template = None
g_last_reception_time_for_template = None
g_last_raw_json_str_for_template = None

@app.route('/api/vehicle_processed_data', methods=['POST'])
def receive_vehicle_data():
    global g_last_vehicle_data_for_template, g_last_reception_time_for_template 
    global g_last_raw_json_str_for_template, recent_log_entries
    
    current_server_debug_mode = app.config.get('SERVER_DEBUG_MODE', True)
    timestamp_recepcion_servidor = datetime.datetime.now().isoformat()
    
    if current_server_debug_mode: print(f"[{timestamp_recepcion_servidor}] Petición POST recibida")

    if not request.is_json:
        if current_server_debug_mode: print("  Error: Payload no es JSON.")
        return jsonify({"status": "error", "message": "Payload debe ser JSON"}), 400

    try:
        data_recibida_original = request.get_json()
        data_para_template_y_log_preview = data_recibida_original.copy()

        video_filename_saved_for_csv = data_recibida_original.get('video_sent_status', 'not_included')

        if 'processed_video_base64' in data_recibida_original and data_recibida_original['processed_video_base64']:
            video_b64_data = data_recibida_original['processed_video_base64']
            
            # Crear nombre de archivo sin espacios y más seguro
            job_id_part = data_recibida_original.get('vehicle_unique_id', 'video')
            job_id_part_safe = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in job_id_part) # Limpiar nombre
            
            video_filename = f"{job_id_part_safe}_{int(datetime.datetime.now().timestamp())}{VIDEO_EXTENSION_CONFIG}"
            video_save_path = PROCESSED_VIDEOS_ABSOLUTE_PATH / video_filename
            
            try:
                video_binary_data = base64.b64decode(video_b64_data)
                with open(video_save_path, "wb") as vf:
                    vf.write(video_binary_data)
                
                video_url = url_for('serve_processed_video', filename=video_filename, _external=False)
                data_para_template_y_log_preview['processed_video_url_path'] = video_url
                data_para_template_y_log_preview['processed_video_filename'] = video_filename
                video_filename_saved_for_csv = video_filename # Para el CSV
                if current_server_debug_mode: print(f"  Video decodificado y guardado como: {video_save_path}. URL: {video_url}")
            except Exception as e_vid_save:
                print(f"  ERROR al guardar video decodificado: {e_vid_save}")
                data_para_template_y_log_preview['video_sent_status'] = 'included_but_failed_to_save'
                video_filename_saved_for_csv = 'error_al_guardar'
        
        if 'processed_video_base64' in data_para_template_y_log_preview:
            data_para_template_y_log_preview['processed_video_base64'] = f"Presente (longitud: {len(data_recibida_original['processed_video_base64'])})"
        
        # Actualizar variables globales para la página principal
        g_last_vehicle_data_for_template = data_para_template_y_log_preview
        g_last_reception_time_for_template = timestamp_recepcion_servidor
        g_last_raw_json_str_for_template = json.dumps(data_para_template_y_log_preview, indent=2, ensure_ascii=False)

        if current_server_debug_mode:
            print(f"  Datos para Vehículo Job ID: {data_recibida_original.get('vehicle_unique_id', 'N/A')}")
            print(f"  Payload procesado para display/log: {g_last_raw_json_str_for_template}")

        # Log detallado en archivo (con el payload original completo, incluyendo B64 si venía)
        log_entry_file = {"timestamp_recepcion_servidor": timestamp_recepcion_servidor, "datos_payload_original": data_recibida_original}
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(json.dumps(log_entry_file, ensure_ascii=False) + "\n")
            if current_server_debug_mode: print(f"  Datos detallados guardados en {LOG_FILE}")
        except Exception as e: print(f"  Error guardando en log detallado: {e}")

        # Log resumido en CSV
        try:
            with open(SUMMARY_LOG_FILE, "a", encoding="utf-8", newline='') as f_csv:
                import csv; writer = csv.writer(f_csv)
                writer.writerow([
                    timestamp_recepcion_servidor,
                    data_recibida_original.get('job_source_name', ''),
                    data_recibida_original.get('vehicle_unique_id', ''),
                    data_recibida_original.get('vehicle_class', ''),
                    data_recibida_original.get('tire_count', 0),
                    data_recibida_original.get('source_id', ''), # Este es el 'job_source_name' del cliente
                    data_recibida_original.get('timestamp_event', ''),
                    video_filename_saved_for_csv 
                ])
            if current_server_debug_mode: print(f"  Resumen guardado en {SUMMARY_LOG_FILE}")
        except Exception as e: print(f"  Error guardando en CSV resumen: {e}")
        
        recent_log_entries.append(json.dumps(log_entry_file, indent=2, ensure_ascii=False))
        if len(recent_log_entries) > MAX_LOG_ENTRIES_IN_MEMORY: recent_log_entries.pop(0)

        return jsonify({"status": "success", "message": "Datos recibidos y video procesado (si aplica)"}), 200

    except Exception as e:
        print(f"  Error crítico al procesar JSON o datos: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": f"Error interno del servidor: {e}"}), 500

@app.route('/', methods=['GET'])
def show_last_vehicle_page():
    return render_template_string(IMAGE_DISPLAY_PAGE_TEMPLATE, 
                                  data_for_template=g_last_vehicle_data_for_template, 
                                  reception_time_for_template=g_last_reception_time_for_template,
                                  raw_json_str_for_template=g_last_raw_json_str_for_template,
                                  video_codec_info_for_template=VIDEO_CODEC_CONFIG)

@app.route('/videos/<path:filename>') # Ruta para servir videos
def serve_processed_video(filename):
    if app.config.get('SERVER_DEBUG_MODE', True):
        print(f"[SERVER_RECEPTOR] Solicitud para servir video: {filename} desde {PROCESSED_VIDEOS_ABSOLUTE_PATH}")
    try:
        return send_from_directory(str(PROCESSED_VIDEOS_ABSOLUTE_PATH), filename, as_attachment=False)
    except FileNotFoundError:
        if app.config.get('SERVER_DEBUG_MODE', True): print(f"  Video no encontrado: {PROCESSED_VIDEOS_ABSOLUTE_PATH / filename}")
        return "Video no encontrado", 404
    except Exception as e:
        if app.config.get('SERVER_DEBUG_MODE', True): print(f"  Error sirviendo video {filename}: {e}")
        return "Error al servir video", 500

@app.route('/log', methods=['GET'])
def show_log_page():
    log_content_html = "<pre>" + "\n<hr>\n".join(reversed(recent_log_entries)) + "</pre>"
    return f"<h1>Log de Recepciones Detalladas (JSON, Últimas {MAX_LOG_ENTRIES_IN_MEMORY})</h1><a href=\"{url_for('show_last_vehicle_page')}\">Volver</a>{log_content_html}"

@app.route('/summary_log', methods=['GET'])
def show_summary_log_page():
    try:
        header_row = "<tr><th>Timestamp Recepción</th><th>Job Source</th><th>Veh Unique ID</th><th>Clase</th><th>Llantas</th><th>Source ID Cliente</th><th>Timestamp Evento Cliente</th><th>Video/Status</th></tr>"
        table_rows = ""
        with open(SUMMARY_LOG_FILE, "r", encoding="utf-8") as f_csv:
            import csv
            reader = csv.reader(f_csv)
            header = next(reader, None) # Leer y descartar encabezado del archivo CSV
            for row in reader:
                table_rows += "<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>"
        content = f"<table>{header_row}{table_rows}</table>"
        return f"<h1>Resumen CSV de Recepciones</h1><a href=\"{url_for('show_last_vehicle_page')}\">Volver</a>{content}"
    except FileNotFoundError: return "Archivo de resumen CSV no encontrado.", 404
    except Exception as e_csv_read: return f"Error leyendo archivo CSV: {e_csv_read}", 500

if __name__ == '__main__':
    # El puerto se podría sacar de config.yaml si se define una sección para el servidor receptor
    server_port = 5005 
    cfg_server_receptor_port = cfg_receptor_app.get('external_server.receptor_port') # Ejemplo de clave en config
    if cfg_server_receptor_port:
        server_port = int(cfg_server_receptor_port)

    print(f"Iniciando servidor Flask RECEPTOR en http://0.0.0.0:{server_port}")
    # Para producción, debug=False. use_reloader=False es bueno si debug=True en entornos con hilos.
    app.run(host='0.0.0.0', port=server_port, debug=SERVER_DEBUG_MODE, use_reloader=SERVER_DEBUG_MODE)