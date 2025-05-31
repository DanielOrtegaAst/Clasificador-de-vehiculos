# utils.py
import numpy as np
import cv2 
import base64

# ... (compute_iou y encode_image_to_base64 sin cambios) ...
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0


def draw_vehicle_tire_counts(frame, current_frame_vehicle_detections, vehicle_physical_tires_data, config_obj):
    """
    Dibuja las etiquetas de conteo de llantas sobre el frame para los vehículos detectados.
    """
    # Usa 'show_visualization_per_job' para controlar si se dibuja esta información específica
    if not config_obj.get('processing.show_visualization_per_job', False): # Default a False si no está
        return frame

    class_names_list = config_obj.get('classes.names', [])
    plot_opts = config_obj.get('processing.visualization_plot_options', {}) # Obtener el sub-diccionario
    
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
        
        num_tires = len(vehicle_physical_tires_data.get(v_id, {}))
        label = f"{v_class_name} ID:{v_id} - Llantas: {num_tires}"
        
        text_x = int(v_box[0])
        text_y = int(v_box[3]) + y_offset_config 

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