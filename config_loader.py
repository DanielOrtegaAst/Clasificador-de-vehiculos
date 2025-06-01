import yaml
from pathlib import Path

class AppConfig:
    """
    Clase para cargar, gestionar y proporcionar acceso a la configuración 
    de la aplicación desde un archivo YAML.

    Resuelve automáticamente los IDs numéricos de las clases y los umbrales de confianza
    a partir de los nombres de clase definidos en el archivo de configuración.
    """
    def __init__(self, config_path_str="config.yaml"):
        """
        Inicializa AppConfig cargando el archivo de configuración.
        Args:
            config_path_str (str, optional): Nombre del archivo de configuración YAML.
                                             Se espera que esté en el mismo directorio que este script.
                                             Defaults to "config.yaml".
        """
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
        Método privado para resolver los IDs numéricos de las clases y los umbrales
        de confianza por clase a partir de los nombres proporcionados en la configuración.
        Estos valores resueltos se almacenan como atributos de la instancia para fácil acceso.
        """
        # Asegurar que all_class_names sea una lista y no esté vacía para proceder
        all_class_names = self.get('classes.names', [])
        if not isinstance(all_class_names, list) or not all_class_names: # Chequeo adicional
            print("Advertencia: 'classes.names' no está definida o no es una lista en la configuración.")
            self.tire_class_id = -1
            self.vehicle_class_ids = []
            self.numeric_per_class_conf_thresholds = {}
            return

        # Resolver TIRE_CLASS_ID
        tire_name = self.get('classes.tire_class_name')
        try:
            self.tire_class_id = all_class_names.index(tire_name) if tire_name else -1
        except (ValueError, TypeError):
            print(f"Advertencia: 'classes.tire_class_name' ('{tire_name}') no encontrado o no definido.")
            self.tire_class_id = -1

        # Resolver VEHICLE_CLASS_IDS
        self.vehicle_class_ids = []
        vehicle_names = self.get('classes.vehicle_class_names', [])
        if isinstance(vehicle_names, list):
            for name in vehicle_names:
                try:
                    self.vehicle_class_ids.append(all_class_names.index(name))
                except ValueError:
                    print(f"Advertencia: Nombre de vehículo '{name}' no encontrado en 'classes.names'.")
        
        # Resolver PER_CLASS_CONF_THRESHOLDS para usar IDs numéricos
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
        Ejemplo: config.get('source.type', "default_value")

        Args:
            key_path (str): Ruta de la clave, ej. "seccion.subseccion.parametro".
            default (any, optional): Valor a devolver si la clave no se encuentra. Defaults to None.

        Returns:
            any: El valor de la configuración o el valor por defecto.
        """
        if self.config_data is None: return default
            
        keys = key_path.split('.')
        value = self.config_data
        try:
            for key in keys:
                if not isinstance(value, dict): return default # Si un path intermedio no es un dict
                value = value[key]  # Acceder al siguiente nivel del diccionario
            return value
        except KeyError: # La clave específica no se encontró en el nivel actual del diccionario
            # Es normal que algunas claves no existan si son opcionales, por eso se devuelve default.
            # print(f"Advertencia en config_loader: Clave de configuración '{key_path}' no encontrada, usando valor por defecto: {default}")
            return default
        except TypeError: # Ocurre si 'value' se vuelve None en algún punto intermedio y se intenta indexar
            return default
