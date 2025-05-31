# api_client.py
import requests
import json
import datetime
# Ya no se necesita encode_image_to_base64 aquí, se hace para video en main.py

class APIClient:
    def __init__(self, config):
        self.config = config
        self.server_enabled = config.get('external_server.enabled', False)
        self.server_url = config.get('external_server.url')
        self.timeout = config.get('external_server.timeout_seconds', 5)
        self.debug_mode = config.get('processing.debug_mode', False)
        
        # La config de 'payload_video' ahora se usa en main.py para generar el video
        self.include_video = config.get('processing.payload_video.include_processed_video', False)

        if self.server_enabled and not self.server_url:
            print("[API_CLIENT] ADVERTENCIA: Envío habilitado pero 'external_server.url' no configurada.")
        # ... (resto del init)

    # Modificado para aceptar video_base64_to_send
    def send_vehicle_data(self, vehicle_data_payload, video_base64_to_send=None, job_source_name="unknown_job_source"):
        if not self.server_enabled or not self.server_url:
            # ... (manejo de no habilitado) ...
            return False

        payload_to_send = vehicle_data_payload.copy()
        payload_to_send['timestamp_event'] = datetime.datetime.now().isoformat()
        payload_to_send['source_id'] = job_source_name # job_source_name es más descriptivo que self.base_source_path
        payload_to_send['video_sent_status'] = "not_included"

        if self.include_video and video_base64_to_send: # Usar video_base64_to_send
            payload_to_send['processed_video_base64'] = video_base64_to_send
            payload_to_send['video_sent_status'] = "included"
            if self.debug_mode: print(f"[API_CLIENT] Incluyendo video Base64 en payload para Job: {job_source_name}.")
        elif self.include_video and not video_base64_to_send:
            if self.debug_mode: print(f"[API_CLIENT] Configurado para incluir video, pero no se proporcionó video Base64 para Job: {job_source_name}.")
            payload_to_send['video_sent_status'] = "configured_but_not_provided"

        
        headers = {'Content-Type': 'application/json'}
        # Log del payload sin el video completo
        payload_for_log = {k:v for k,v in payload_to_send.items() if k != 'processed_video_base64'}
        if 'processed_video_base64' in payload_to_send : 
            payload_for_log['processed_video_base64_status'] = "Present (Length: {})".format(len(payload_to_send['processed_video_base64']))
        
        if self.debug_mode: print(f"[API_CLIENT] Intentando enviar datos a {self.server_url}: {json.dumps(payload_for_log, indent=2)}")

        try:
            response = requests.post(self.server_url, data=json.dumps(payload_to_send), headers=headers, timeout=self.timeout)
            response.raise_for_status()
            if self.debug_mode: print(f"[API_CLIENT] Datos para Vehículo (Job: {job_source_name}) enviados. Respuesta: {response.status_code}")
            return True
        except Exception as e:
            print(f"[API_CLIENT] ERROR general enviando para Vehículo (Job: {job_source_name}): {e}")
            return False