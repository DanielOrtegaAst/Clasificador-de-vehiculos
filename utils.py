import numpy as np
import cv2 
import base64

def compute_iou(box1, box2):
    """
    Calcula la Intersección sobre Unión (IoU) entre dos bounding boxes.
    Las cajas se esperan en formato (x1, y1, x2, y2).

    Args:
        box1 (list or np.array): Coordenadas de la primera caja [x1, y1, x2, y2].
        box2 (list or np.array): Coordenadas de la segunda caja [x1, y1, x2, y2].

    Returns:
        float: Valor de IoU entre 0.0 y 1.0. Retorna 0.0 si no hay superposición
               o si el área de unión es cero.
    """
    # Determinar las coordenadas (x, y) de la intersección del rectángulo
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])

    # Calcular el área de la intersección
    # Asegurarse que la intersección tenga ancho y alto positivos
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    
    # Calcular el área de cada bounding box
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    
    # Calcular el área de la unión
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def encode_image_to_base64(image_frame, max_width, max_height, quality=75, debug_mode=False):
    """
    Redimensiona una imagen si es necesario para que quepa dentro de max_width y max_height
    manteniendo la relación de aspecto, y luego la codifica a formato JPEG Base64.

    Args:
        image_frame (numpy.ndarray): El frame de la imagen a codificar.
        max_width (int): Ancho máximo para la imagen redimensionada.
                         Si es 0 o negativo, no se aplica límite de ancho.
        max_height (int): Alto máximo para la imagen redimensionada.
                          Si es 0 o negativo, no se aplica límite de alto.
        quality (int, optional): Calidad de la compresión JPEG (0-100). Defaults to 75.
        debug_mode (bool, optional): Si es True, imprime mensajes de depuración. Defaults to False.

    Returns:
        str or None: El string Base64 de la imagen JPEG, o None si ocurre un error.
    """
    if image_frame is None:
        if debug_mode: print("[UTILS_ENCODE] Error: image_frame de entrada es None.")
        return None
    try:
        h, w = image_frame.shape[:2]
        scale_w = 1.0
        scale_h = 1.0
        
        # Calcular escala basada en el ancho
        if max_width > 0 and w > max_width:
            scale_w = max_width / w
        
        # Calcular escala basada en el alto
        if max_height > 0 and h > max_height:
            scale_h = max_height / h
        
        # Usar la escala más restrictiva (la menor) para asegurar que quepa en ambas dimensiones
        final_scale = min(scale_w, scale_h)
        
        resized_image = image_frame # Por defecto, usar la imagen original
        if 0 < final_scale < 1.0: # Solo redimensionar si es necesario y la escala es válida
            new_w, new_h = int(w * final_scale), int(h * final_scale)
            # Asegurar que las nuevas dimensiones sean positivas antes de redimensionar
            if new_w > 0 and new_h > 0:
                resized_image = cv2.resize(image_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if debug_mode: print(f"[UTILS_ENCODE] Imagen redimensionada de {w}x{h} a {new_w}x{new_h}")
            elif debug_mode:
                print(f"[UTILS_ENCODE] Advertencia: Nuevas dimensiones inválidas ({new_w}, {new_h}) para redimensionar. Usando imagen original.")
        elif debug_mode and final_scale >= 1.0 : # No se redimensiona porque ya es más pequeña o igual
             print(f"[UTILS_ENCODE] No se requiere redimensionamiento (escala calculada: {final_scale:.2f}). Original: {w}x{h}")

        # Codificar la imagen (original o redimensionada) a JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', resized_image, encode_param)
        if not success:
            if debug_mode: print("[UTILS_ENCODE] Error: cv2.imencode falló al convertir a JPEG.")
            return None
            
        # Convertir el buffer JPEG a string Base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        if debug_mode: print(f"[UTILS_ENCODE] Imagen codificada a Base64 (longitud del string: {len(jpg_as_text)}).")
        return jpg_as_text
    except Exception as e:
        print(f"[UTILS_ENCODE] Error excepcional al codificar imagen a Base64: {e}")
        import traceback
        traceback.print_exc() # Imprimir el traceback completo para una mejor depuración
        return None
    
def draw_vehicle_tire_counts(frame, current_frame_vehicle_detections, vehicle_physical_tires_data, config_obj):
    """
    Dibuja las etiquetas de conteo de llantas sobre el frame para los vehículos detectados.
    Esta función modifica `frame_to_draw_on` directamente.

    Args:
        frame_to_draw_on (numpy.ndarray): El frame de video sobre el cual dibujar.
                                          Se asume que es una copia para no modificar el original si no se desea.
        current_frame_vehicle_detections (dict): Detecciones de vehículos en el frame actual.
                                                 Formato: {track_id: {'box': ..., 'class_id': ...}}
        vehicle_physical_tires_data (dict): Estado de las llantas físicas por vehículo.
                                            Formato: {v_id: {anchor_id: {...}}}
        config_obj (AppConfig): La instancia de configuración para acceder a parámetros
                                 como class_names y opciones de visualización.

    Returns:
        numpy.ndarray: El frame con las etiquetas de conteo dibujadas.
    """
    # Determinar si se debe mostrar la visualización según la configuración
    if not config_obj.get('processing.show_visualization_per_job', False): # Default a False si no está
        return frame

    class_names_list = config_obj.get('classes.names', [])
    plot_opts = config_obj.get('processing.visualization_plot_options', {}) # Obtener el sub-diccionario
    
    # Parámetros de dibujo, configurables o con valores por defecto
    font_scale = plot_opts.get('custom_label_font_scale', 0.6)
    font_thickness = plot_opts.get('custom_label_thickness', 1)
    y_offset_config = plot_opts.get('custom_label_y_offset', 20)
    text_color = (0, 255, 0) # Verde

    for v_id, v_data in current_frame_vehicle_detections.items():
        v_box = v_data['box']
        v_class_id = v_data['class_id']
        
        try:
            v_class_name = class_names_list[v_class_id]
        except IndexError:
            v_class_name = f"ClaseID {int(v_class_id)}" # Asegurar int para el f-string
        
        # Obtener el conteo de llantas para este vehículo
        num_tires = len(vehicle_physical_tires_data.get(v_id, {}))
        label = f"{v_class_name} ID:{v_id} - Llantas: {num_tires}"
        
        text_x = int(v_box[0])
        # Posición Y: Debajo de la caja por defecto
        text_y = int(v_box[3]) + y_offset_config 

        # Calcular tamaño del texto para ajustes
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        frame_height = frame.shape[0]
        
        # Ajustar posición Y si se sale por abajo o por arriba
        if text_y + baseline > frame_height: # Se sale por abajo
            text_y = int(v_box[1]) - y_offset_config # Intentar arriba de la caja (y_offset va hacia arriba desde y1)
        if text_y - text_height < 0: # Todavía se sale por arriba (o y_offset es muy grande)
                 text_y = int(v_box[3]) - baseline - 5 # Dentro de la caja, en la parte inferior

        cv2.putText(frame, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return frame
