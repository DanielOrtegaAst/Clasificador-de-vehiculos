# config_loader.py
import yaml
from pathlib import Path

class AppConfig:
    """
    Clase para cargar y gestionar la configuración de la aplicación desde un archivo YAML.
    """
    def __init__(self, config_path_str="config.yaml"):
        # Construir la ruta al archivo de configuración relativa a este archivo
        base_dir = Path(__file__).resolve().parent
        config_path = base_dir / config_path_str
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            if self.config_data is None: # Archivo vacío o YAML inválido que resulta en None
                print(f"Error Crítico: El archivo de configuración '{config_path}' está vacío o es inválido.")
                self.config_data = {} # Evitar errores NoneType más adelante
                # exit() # O manejar más drásticamente
            else:
                print(f"Configuración cargada exitosamente desde: {config_path}")
            self._resolve_class_ids_and_thresholds()
        except FileNotFoundError:
            print(f"Error Crítico: Archivo de configuración no encontrado en {config_path}")
            self.config_data = {}
            # exit()
        except yaml.YAMLError as e:
            print(f"Error Crítico: Error al parsear el archivo de configuración YAML: {e}")
            self.config_data = {}
            # exit()
        except Exception as e:
            print(f"Error Crítico: Ocurrió un error inesperado al cargar la configuración: {e}")
            self.config_data = {}
            # exit()

    def _resolve_class_ids_and_thresholds(self):
        """
        Resuelve los IDs numéricos de las clases y umbrales a partir de los nombres.
        """
        all_class_names = self.get('classes.names', [])
        if not isinstance(all_class_names, list) or not all_class_names: # Chequeo adicional
            print("Advertencia: 'classes.names' no está definida o no es una lista en la configuración.")
            self.tire_class_id = -1
            self.vehicle_class_ids = []
            self.numeric_per_class_conf_thresholds = {}
            return

        tire_name = self.get('classes.tire_class_name')
        try:
            self.tire_class_id = all_class_names.index(tire_name) if tire_name else -1
        except (ValueError, TypeError):
            print(f"Advertencia: 'classes.tire_class_name' ('{tire_name}') no encontrado o no definido.")
            self.tire_class_id = -1

        self.vehicle_class_ids = []
        vehicle_names = self.get('classes.vehicle_class_names', [])
        if isinstance(vehicle_names, list):
            for name in vehicle_names:
                try:
                    self.vehicle_class_ids.append(all_class_names.index(name))
                except ValueError:
                    print(f"Advertencia: Nombre de vehículo '{name}' no encontrado en 'classes.names'.")
        
        self.numeric_per_class_conf_thresholds = {}
        conf_thresholds_by_name = self.get('confidence_thresholds.per_class', {})
        if isinstance(conf_thresholds_by_name, dict):
            for name, threshold in conf_thresholds_by_name.items():
                try:
                    class_id = all_class_names.index(name)
                    self.numeric_per_class_conf_thresholds[class_id] = float(threshold) # Asegurar float
                except ValueError:
                    print(f"Advertencia: Nombre de clase '{name}' en 'confidence_thresholds.per_class' no encontrado.")
                except TypeError:
                     print(f"Advertencia: Umbral para '{name}' no es un número válido.")


    def get(self, key_path, default=None):
        """
        Obtiene un valor de la configuración usando una ruta de claves separadas por puntos.
        """
        if self.config_data is None: return default
            
        keys = key_path.split('.')
        value = self.config_data
        try:
            for key in keys:
                if not isinstance(value, dict): return default # Si un path intermedio no es un dict
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

# No instanciar cfg globalmente aquí, se hará en main.py