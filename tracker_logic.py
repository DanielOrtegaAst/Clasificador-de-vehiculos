# tracker_logic.py
from utils import compute_iou 
import numpy as np
from collections import Counter

class TireCounterLogic:
    def __init__(self, config, api_client_instance=None):
        self.config = config
        self.api_client = api_client_instance
        self.debug_mode = config.get('processing.debug_mode', False)

        self.tire_class_id = config.tire_class_id
        self.vehicle_class_ids = config.vehicle_class_ids
        self.class_names = config.get('classes.names', [])
        
        self.per_class_thresholds = config.numeric_per_class_conf_thresholds
        self.default_conf_threshold = config.get('confidence_thresholds.default_post_filter')
        
        self.iou_thresh_same_tire = config.get('tire_logic.iou_threshold_same_physical_tire')
        self.veh_box_exp_x_perc = config.get('tire_logic.vehicle_box_expansion_x_percent', 0.0)
        self.veh_box_exp_y_perc = config.get('tire_logic.vehicle_box_expansion_y_percent', 0.0)
        self.y_min_frac = config.get('tire_logic.min_y_fraction_from_veh_top')
        self.y_max_ext = config.get('tire_logic.max_y_extension_below_veh_bottom_fraction')
        self.area_ratio_tol = config.get('tire_logic.accepted_tire_area_ratio_tolerance', 3.0)
        self.min_abs_tire_area = config.get('tire_logic.min_absolute_tire_pixel_area', 50)
        self.frames_to_keep_data = config.get('processing.frames_to_keep_data_for_lost_tracks', 300)

        self.vehicle_physical_tires_current_job = {}
        self.tracked_vehicles_info_current_job = {} # Ahora almacenará una lista de class_ids

        if self.debug_mode: print("TireCounterLogic inicializado (Modo Servicio).")
        if self.tire_class_id == -1 or not self.vehicle_class_ids:
            print("ADVERTENCIA: IDs de clase para llantas o vehículos no configurados.")

    def reset_state_for_new_job(self):
        if self.debug_mode: print("[TRACKER_LOGIC] Reseteando estado para nuevo trabajo.")
        self.vehicle_physical_tires_current_job.clear()
        self.tracked_vehicles_info_current_job.clear()

    def _get_main_vehicle_from_job_detections(self):
        # ... (lógica para obtener main_vehicle_track_id como antes, basada en frames_seen_count o área) ...
        if not self.tracked_vehicles_info_current_job:
            return None, None

        main_vehicle_track_id = None
        max_frames_seen = -1
        
        for v_id, data in self.tracked_vehicles_info_current_job.items():
            frames_seen = data.get('frames_seen_count', 0)
            if frames_seen > max_frames_seen:
                max_frames_seen = frames_seen
                main_vehicle_track_id = v_id
            elif frames_seen == max_frames_seen and main_vehicle_track_id is None: # Fallback simple
                 main_vehicle_track_id = v_id

        if main_vehicle_track_id is None and self.tracked_vehicles_info_current_job:
            largest_area = -1
            for v_id, data in self.tracked_vehicles_info_current_job.items():
                box = data.get('box') # Última caja conocida
                if box is not None:
                    area = (box[2]-box[0]) * (box[3]-box[1])
                    if area > largest_area:
                        largest_area = area
                        main_vehicle_track_id = v_id
        
        if main_vehicle_track_id:
            return main_vehicle_track_id, self.tracked_vehicles_info_current_job[main_vehicle_track_id]
        return None, None


    def process_job_detections(self, yolo_results, current_frame_idx_in_job, frame_shape):
        current_job_vehicle_detections_this_frame = {} # Vehículos detectados en *este* frame

        if yolo_results is None or yolo_results.boxes is None or yolo_results.boxes.id is None:
            if self.debug_mode: print(f"  FRAME_JOB {current_frame_idx_in_job}: No objetos con IDs.")
            return current_job_vehicle_detections_this_frame

        track_ids = yolo_results.boxes.id.cpu().numpy().astype(int)
        detected_classes = yolo_results.boxes.cls.cpu().numpy().astype(int)
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        confidences = yolo_results.boxes.conf.cpu().numpy()

        frame_all_tire_detections = []
        for i in range(len(track_ids)):
            class_id = detected_classes[i]
            conf = confidences[i]
            box = boxes[i]
            track_id = track_ids[i]
            
            threshold = self.per_class_thresholds.get(class_id, self.default_conf_threshold)
            if conf < threshold:
                continue

            if class_id in self.vehicle_class_ids:
                # Guardar la detección actual para la visualización de este frame
                current_job_vehicle_detections_this_frame[track_id] = {'box': box, 'class_id': class_id}
                
                if track_id not in self.tracked_vehicles_info_current_job:
                    self.tracked_vehicles_info_current_job[track_id] = {
                        'box': box, # Última caja vista
                        'detected_class_ids_history': [class_id], # <--- NUEVO: Iniciar historial de clases
                        'first_seen_frame_in_job': current_frame_idx_in_job,
                        'last_seen_frame_in_job': current_frame_idx_in_job,
                        'frames_seen_count': 1
                    }
                    self.vehicle_physical_tires_current_job[track_id] = {}
                else:
                    # Añadir la clase actual al historial
                    self.tracked_vehicles_info_current_job[track_id]['detected_class_ids_history'].append(class_id)
                    self.tracked_vehicles_info_current_job[track_id].update({
                        'box': box, 
                        'last_seen_frame_in_job': current_frame_idx_in_job,
                        'frames_seen_count': self.tracked_vehicles_info_current_job[track_id].get('frames_seen_count', 0) + 1
                    })
                    # Actualizar la 'class_id' en current_job_vehicle_detections_this_frame con la clase de este frame
                    # para la visualización, pero la clase final se decidirá por moda.
                    current_job_vehicle_detections_this_frame[track_id]['class_id'] = class_id


            elif class_id == self.tire_class_id:
                frame_all_tire_detections.append({'track_id': track_id, 'box': box, 'conf': conf})


        for v_track_id, v_data_in_frame in current_job_vehicle_detections_this_frame.items():
            v_box_orig = v_data_in_frame['box'] # Caja del vehículo de este frame
            # ... (lógica de expansión de caja como la tenías, usando v_box_orig y frame_shape) ...
            v_box_for_tire_assoc, v_height_for_tire_assoc = v_box_orig, (v_box_orig[3] - v_box_orig[1])
            if self.veh_box_exp_x_perc > 0 or self.veh_box_exp_y_perc > 0:
                orig_v_w, orig_v_h = (v_box_orig[2]-v_box_orig[0]), (v_box_orig[3]-v_box_orig[1])
                exp_px_x, exp_px_y = orig_v_w*self.veh_box_exp_x_perc, orig_v_h*self.veh_box_exp_y_perc
                x1,y1,x2,y2 = v_box_orig[0]-exp_px_x, v_box_orig[1]-exp_px_y, v_box_orig[2]+exp_px_x, v_box_orig[3]+exp_px_y
                if frame_shape:
                    fh,fw = frame_shape[:2]
                    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(fw,x2),min(fh,y2)
                v_box_for_tire_assoc = np.array([x1,y1,x2,y2])
                v_height_for_tire_assoc = v_box_for_tire_assoc[3]-v_box_for_tire_assoc[1]
            if v_height_for_tire_assoc <=0: v_height_for_tire_assoc = 1


            if self.debug_mode and frame_all_tire_detections:
                v_cname = self.class_names[v_data_in_frame['class_id']]
                print(f"  FRAME_JOB {current_frame_idx_in_job}, Veh {v_track_id} ({v_cname}): Caja Asoc: {v_box_for_tire_assoc.astype(int)}")

            current_vehicle_tire_slots = self.vehicle_physical_tires_current_job.get(v_track_id,{}) # Obtener/crear
            for anchor_key in current_vehicle_tire_slots: current_vehicle_tire_slots[anchor_key]['updated_this_frame'] = False

            for tire_det_data in frame_all_tire_detections:
                t_id_current_frame, t_box = tire_det_data['track_id'], tire_det_data['box']
                # ... (Filtros X, Y, Fusión IoU, Filtro Tamaño, Creación Nueva Ranura como antes) ...
                # La clave es que 'vehicle_slots_for_this_vehicle' ahora es 'current_vehicle_tire_slots'
                # y el 'anchor_id' para una nueva ranura de llanta será 't_id_current_frame'.
                tcx, tcy = (t_box[0]+t_box[2])/2, (t_box[1]+t_box[3])/2
                x_ok = (v_box_for_tire_assoc[0] <= tcx <= v_box_for_tire_assoc[2])
                min_y, max_y = (v_box_for_tire_assoc[1] + v_height_for_tire_assoc * self.y_min_frac), \
                               (v_box_for_tire_assoc[3] + v_height_for_tire_assoc * self.y_max_ext)
                y_ok = (min_y <= tcy <= max_y)
                if not (x_ok and y_ok): continue

                matched = False
                for anchor, slot in current_vehicle_tire_slots.items():
                    iou = compute_iou(t_box, slot['box'])
                    if iou > self.iou_thresh_same_tire:
                        slot.update({'box':t_box, 'latest_track_id':t_id_current_frame, 'last_seen_frame_in_job':current_frame_idx_in_job, 'updated_this_frame':True})
                        matched = True; break
                if matched: continue
                
                t_area = (t_box[2]-t_box[0])*(t_box[3]-t_box[1])
                if t_area < self.min_abs_tire_area: continue
                size_ok = True
                if current_vehicle_tire_slots:
                    s_areas, n_exist = sum(s.get('area',t_area) for s in current_vehicle_tire_slots.values()), len(current_vehicle_tire_slots)
                    if n_exist > 0:
                        avg_a = s_areas/n_exist
                        if not (avg_a/self.area_ratio_tol <= t_area <= avg_a*self.area_ratio_tol): size_ok=False
                if not size_ok:
                    if self.debug_mode: print(f"    Llanta TrackID {t_id_current_frame} RECHAZADA (TAMAÑO REL) para Veh {v_track_id}.")
                    continue
                
                if t_id_current_frame not in current_vehicle_tire_slots:
                    current_vehicle_tire_slots[t_id_current_frame] = {'latest_track_id':t_id_current_frame, 'box':t_box, 'area':t_area, 'last_seen_frame_in_job':current_frame_idx_in_job, 'updated_this_frame':True, 'first_seen_frame_in_job': current_frame_idx_in_job}
                    if self.debug_mode: print(f"    NUEVA LLANTA FÍSICA (Anchor {t_id_current_frame}) para Veh {v_track_id}. Área: {t_area:.0f}")
                elif self.debug_mode:
                     current_vehicle_tire_slots[t_id_current_frame].update({'box':t_box, 'latest_track_id':t_id_current_frame, 'area':t_area, 'last_seen_frame_in_job':current_frame_idx_in_job, 'updated_this_frame':True})


        return current_job_vehicle_detections_this_frame


    def finalize_job_and_prepare_payload(self, job_source_name):
        main_v_track_id, main_v_data_from_job_info = self._get_main_vehicle_from_job_detections()

        if main_v_track_id is None or main_v_data_from_job_info is None:
            if self.debug_mode: print(f"[FINALIZE_JOB] No se pudo determinar un vehículo principal para '{job_source_name}'.")
            return None

        # --- INICIO: Determinar la clase más frecuente ---
        detected_class_ids_history = main_v_data_from_job_info.get('detected_class_ids_history', [])
        if detected_class_ids_history:
            # Ahora Counter estará definido
            most_common_class_id = Counter(detected_class_ids_history).most_common(1)[0][0]
            final_vehicle_class_id = most_common_class_id
            if self.debug_mode: print(f"  [FINALIZE_JOB] Historial de clases para Veh {main_v_track_id}: {detected_class_ids_history}. Clase más común: {final_vehicle_class_id} ({self.class_names[final_vehicle_class_id] if 0 <= final_vehicle_class_id < len(self.class_names) else 'Desconocida'})")
        else:
            # ... (lógica de fallback como la tenías) ...
            final_vehicle_class_id = self.vehicle_class_ids[0] if self.vehicle_class_ids else -1 
            if self.debug_mode: print(f"  [FINALIZE_JOB] ADVERTENCIA: Historial de clases vacío para Veh {main_v_track_id}. Usando fallback a clase ID: {final_vehicle_class_id}")
            if main_v_data_from_job_info.get('detected_class_ids_history'): # Doble check, si está vacío pero la clave existe
                 if main_v_data_from_job_info['detected_class_ids_history']: # Y si la lista no está vacía
                    final_vehicle_class_id = main_v_data_from_job_info['detected_class_ids_history'][0]


        final_vehicle_class_name = self.class_names[final_vehicle_class_id] if 0 <= final_vehicle_class_id < len(self.class_names) else f"ClaseID_Desconocida_{int(final_vehicle_class_id)}"
        # --- FIN: Determinar la clase más frecuente ---

        tire_slots = self.vehicle_physical_tires_current_job.get(main_v_track_id, {})
        num_tires = len(tire_slots)

        payload = {
            "vehicle_unique_id": f"{job_source_name}_{int(main_v_track_id)}",
            "vehicle_class": final_vehicle_class_name,
            "tire_count": int(num_tires),
            "vehicle_box_xyxy": [int(c) for c in main_v_data_from_job_info.get('box', [])],
            "first_seen_frame_in_job": main_v_data_from_job_info.get('first_seen_frame_in_job'),
            "last_seen_frame_in_job": main_v_data_from_job_info.get('last_seen_frame_in_job'),
            "total_frames_vehicle_seen_in_job": main_v_data_from_job_info.get('frames_seen_count'),
            "job_source_name": job_source_name,
            "status": "job_completed"
        }
        if self.debug_mode: print(f"  [FINALIZE_JOB] Payload preparado para '{job_source_name}', Vehículo Principal ID {main_v_track_id} (Clase Final: {final_vehicle_class_name}): {num_tires} llantas.")
        return payload
    
    # cleanup_old_tracks (para streams, necesita adaptarse para usar el historial de clases también si se quiere)
    # Por ahora, cleanup_old_tracks usaría la última clase conocida del vehicle_data.
    # Si necesitas que cleanup_old_tracks también use la clase más frecuente, esa lógica
    # tendría que aplicarse allí también, o el payload de cleanup debe indicar que es una clase "tentativa".
