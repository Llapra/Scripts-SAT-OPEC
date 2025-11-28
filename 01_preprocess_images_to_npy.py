import os
import glob
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from datetime import datetime

# --- CONFIGURACIÓN ---
INPUT_FOLDER = "Recortes-satelitales"
OUTPUT_FOLDER = "Datos_Temperatura_NPY"
LUT_FILE = "calibracion_color_temp.npz"
TARGET_SIZE = (128, 128) # Redimensionar a 128x128 para la CNN

# LÍMITE DE CORTE (Hold-Out para validación visual)
# Todo lo que sea igual o posterior a esta fecha NO se procesará.
FECHA_LIMITE = datetime(2025, 10, 1) 

def cargar_lut(ruta_lut):
    """
    Carga la LUT y prepara un árbol KDTree para búsquedas rápidas de color.
    """
    data = np.load(ruta_lut)
    lut_rgb = data['rgb']
    lut_temp = data['temp']
    kdtree = cKDTree(lut_rgb)
    return kdtree, lut_temp

def obtener_fecha_desde_nombre(nombre_archivo):
    """
    Parsea '2025-08-01-0021_band-13.png' -> datetime object
    """
    try:
        parte_fecha = nombre_archivo.split('_')[0]
        # Formato esperado: Año-Mes-Dia-HoraMinuto
        return datetime.strptime(parte_fecha, "%Y-%m-%d-%H%M")
    except ValueError:
        return None

def procesar_y_convertir():
    # Crear carpeta de salida
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Cargar herramientas de calibración
    if not os.path.exists(LUT_FILE):
        print(f"Error: No se encontró '{LUT_FILE}'. Ejecuta primero el script de calibración.")
        return
        
    print("Cargando tabla de calibración...")
    tree, temps = cargar_lut(LUT_FILE)
    
    # Buscar imágenes
    patron = os.path.join(INPUT_FOLDER, "*.png")
    archivos = glob.glob(patron)
    archivos.sort()
    
    total = len(archivos)
    print(f"Total de imágenes encontradas: {total}")
    print(f"Aplicando filtro de fecha: Solo procesar antes de {FECHA_LIMITE}")
    
    procesados = 0
    omitidos_fecha = 0
    
    for i, ruta_img in enumerate(archivos):
        try:
            nombre_base = os.path.basename(ruta_img)
            
            # 1. Verificar Fecha
            fecha_img = obtener_fecha_desde_nombre(nombre_base)
            if fecha_img and fecha_img >= FECHA_LIMITE:
                omitidos_fecha += 1
                continue # Saltar esta imagen, es para el futuro test visual
            
            nombre_npy = nombre_base.replace('.png', '.npy')
            ruta_salida = os.path.join(OUTPUT_FOLDER, nombre_npy)
            
            # Si ya existe, saltar (optimización)
            if os.path.exists(ruta_salida):
                procesados += 1
                continue

            # 2. Cargar y Procesar
            img = Image.open(ruta_img).convert('RGB')
            
            # Redimensionar (Resize) ANTES de convertir para velocidad
            img_resized = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
            
            # Convertir a Array
            img_array = np.array(img_resized)
            shape_orig = img_array.shape
            
            # Aplanar para búsqueda masiva
            pixels_flat = img_array.reshape(-1, 3)
            
            # Mapeo de Color -> Temperatura
            _, indices = tree.query(pixels_flat)
            temps_flat = temps[indices]
            
            # Reconstruir 2D
            matriz_temp = temps_flat.reshape((shape_orig[0], shape_orig[1]))
            
            # Guardar como float16
            matriz_temp = matriz_temp.astype(np.float16)
            np.save(ruta_salida, matriz_temp)
            
            procesados += 1
            if procesados % 100 == 0:
                print(f"Procesando... {procesados} listos. (Último: {nombre_npy})")
                
        except Exception as e:
            print(f"Error en {ruta_img}: {e}")

    print("-" * 30)
    print(f"¡Proceso Finalizado!")
    print(f"Imágenes convertidas a NPY: {procesados}")
    print(f"Imágenes reservadas (Omitidas): {omitidos_fecha}")
    print(f"Datos guardados en '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    procesar_y_convertir()