"""
Script para procesar videos y extraer variables de interés, generando un CSV final.
"""

import os
import numpy as np
import pickle as pk
import pandas as pd
import cv2
from pathlib import Path
import re

from scipy.stats import skew, entropy
from skimage.measure import label, regionprops_table
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


############################################################
#                    CONSTANTES GLOBALES                   #
############################################################

THRESHOLD = 0.6

############################################################
#                    FUNCIONES AUXILIARES                  #
############################################################

def extraer_numero_frame(nombre_archivo):
    """
    Extrae el número del string en el formato "frame_X.pk".
    Ejemplo:
      extraer_numero_frame("frame_0.pk") retorna 0
      extraer_numero_frame("frame_2.pk") retorna 2
    """
    match = re.search(r'frame_(\d+)\.pk', nombre_archivo)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("El formato del archivo no es el esperado 'frame_<numero>.pk'.")

def convertirIndiceVideo(cadena):
    """
    Convierte una cadena con ceros a la izquierda a su valor numérico.
    Ejemplo:
      convertirIndiceVideo("001") retorna 1
      convertirIndiceVideo("010") retorna 10
    """
    try:
        return int(cadena)
    except ValueError:
        raise ValueError("La cadena debe contener solo dígitos.")

def calculate_basic_variables(matrix):
    """
    Calcula valores básicos seleccionados para una matriz de valores.

    Métricas calculadas:
      - mean: Promedio de los valores.
      - median: Valor mediano.
      - std_dev: Desviación estándar.
      - max: Valor máximo.
      - min: Valor mínimo.
      - range: Rango (diferencia entre el máximo y el mínimo).
      - skewness: Asimetría de la distribución.
      - entropy: Entropía basada en el histograma de 256 bins.
    """
    flat = matrix.ravel()
    min_val = flat.min()
    max_val = flat.max()
    hist, _ = np.histogram(flat, bins=256)

    return {
        'mean': flat.mean(),
        'median': np.median(flat),
        'std_dev': flat.std(),
        'max': max_val,
        'min': min_val,
        'range': max_val - min_val,
        'skewness': skew(flat),
        'entropy': entropy(hist)
    }

def calculate_selected_threshold_variables(matrix, threshold=THRESHOLD):
    """
    Calcula valores seleccionados para una matriz de valores filtrados por un umbral (> threshold).

    Métricas calculadas:
      - threshold_mean: Promedio de los valores filtrados.
      - threshold_median: Mediana de los valores filtrados.
      - threshold_std_dev: Desviación estándar.
      - threshold_max: Valor máximo.
      - threshold_min: Valor mínimo.
      - threshold_range: Rango (diferencia entre el máximo y el mínimo).
      - threshold_skewness: Asimetría de la distribución.
      - threshold_entropy: Entropía basada en el histograma de 256 bins.
    """
    filtered_matrix = matrix[matrix > threshold]

    if filtered_matrix.size == 0:
        return {
            'threshold_mean': None,
            'threshold_median': None,
            'threshold_std_dev': None,
            'threshold_max': None,
            'threshold_min': None,
            'threshold_range': None,
            'threshold_skewness': None,
            'threshold_entropy': None,
        }

    flat = filtered_matrix.ravel()
    min_val = flat.min()
    max_val = flat.max()
    num_bins = min(256, max(2, int(np.ptp(flat))))
    hist, _ = np.histogram(flat, bins=num_bins)

    return {
        'threshold_mean': flat.mean(),
        'threshold_median': np.median(flat),
        'threshold_std_dev': flat.std(),
        'threshold_max': max_val,
        'threshold_min': min_val,
        'threshold_range': max_val - min_val,
        'threshold_skewness': skew(flat),
        'threshold_entropy': entropy(hist),
    }

def compute_geometric_properties(matrix, threshold=THRESHOLD):
    """
    Devuelve un diccionario con:
      - 'area': número de píxeles en la máscara binaria (matrix >= threshold).
      - 'bbox_ymin', 'bbox_xmin', 'bbox_ymax', 'bbox_xmax': límites de la caja delimitadora
        o None si no hay píxeles sobre el umbral.
      - 'center_y', 'center_x': coordenadas del centro o None si no hay píxeles sobre el umbral.
    """
    mask = (matrix >= threshold)
    area = np.sum(mask)

    if area == 0:
        return {
            'area': 0,
            'bbox_ymin': None,
            'bbox_xmin': None,
            'bbox_ymax': None,
            'bbox_xmax': None,
            'center_y': None,
            'center_x': None
        }

    rows, cols = np.where(mask)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    return {
        'area': area,
        'bbox_ymin': min_row,
        'bbox_xmin': min_col,
        'bbox_ymax': max_row,
        'bbox_xmax': max_col,
        'center_y': rows.mean(),
        'center_x': cols.mean()
    }

def compute_regionprops(matrix, threshold=THRESHOLD):
    """
    Aplica un umbral a 'matrix' para crear una máscara binaria.
    Etiqueta las componentes conectadas y obtiene propiedades específicas
    de cada región (sin calcular 'label' ni 'centroid').
    Retorna las propiedades de la región con mayor perímetro, o None si no hay regiones.

    Devuelve un diccionario con:
      - 'orientation'
      - 'eccentricity'
      - 'major_axis_length'
      - 'minor_axis_length'
      - 'perimeter'
      - 'solidity'
    """
    mask = (matrix >= threshold)
    labeled = label(mask)

    props = regionprops_table(
        labeled,
        properties=[
            'orientation',
            'eccentricity',
            'major_axis_length',
            'minor_axis_length',
            'perimeter',
            'solidity'
        ]
    )

    df_props = pd.DataFrame(props)
    if df_props.empty:
        return None

    idx_max = df_props['perimeter'].idxmax()
    return df_props.loc[idx_max].to_dict()

def calculate_glcm_variables(matrix, distances=[1], angles=[0]):
    """
    Calcula valores de textura usando la Matriz de Co-ocurrencia de Niveles de Gris (GLCM).
    """
    if matrix.size == 0:
        return {
            'contrast': None,
            'homogeneity': None,
            'GLCM_energy': None,
            'GLCM_correlation': None
        }

    matrix_8u = (matrix * 255).astype(np.uint8)

    glcm = graycomatrix(matrix_8u, distances=distances, angles=angles,
                        symmetric=True, normed=True)

    return {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'GLCM_energy': graycoprops(glcm, 'energy').mean(),
        'GLCM_correlation': graycoprops(glcm, 'correlation').mean()
    }

def calculate_contours_variables(matrix, threshold=THRESHOLD):
    """
    Calcula el número de contornos detectados en una imagen binaria basada en un umbral.
    """
    if matrix.size == 0:
        return {'num_contours': None}

    binary_matrix = (matrix >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return {'num_contours': len(contours)}

def compute_distance(center1, center2):
    """
    Calcula la distancia euclidiana entre dos centros dados (row_center, col_center).
    Devuelve np.nan si alguno de los centros o sus coordenadas son None.
    """
    if (
        center1 is None or center2 is None or
        center1[0] is None or center1[1] is None or
        center2[0] is None or center2[1] is None
    ):
        return np.nan

    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


############################################################
#   FUNCIONES PARA PROCESAR UN VIDEO Y VARIOS VIDEOS       #
############################################################

def process_video_full_variables(video_path, organs, threshold=THRESHOLD):
    """
    Procesa todos los frames (pickle) de un video y calcula distintos valores para cada órgano.
    Retorna un DataFrame con las variables calculadas.
    """
    rows = []
    frame_files = sorted(os.listdir(video_path))

    for frame_file in frame_files:
        frame_path = os.path.join(video_path, frame_file)
        with open(frame_path, 'rb') as f:
            frame_data = pk.load(f)

        organ_variables = {}
        for organ_idx, organ_name in enumerate(organs):
            organ_matrix = frame_data[:, :, organ_idx]

            basic_vars = calculate_basic_variables(organ_matrix)
            threshold_vars = calculate_selected_threshold_variables(organ_matrix, threshold)
            geom_props = compute_geometric_properties(organ_matrix, threshold)
            region_props = compute_regionprops(organ_matrix, threshold)
            glcm_vars = calculate_glcm_variables(organ_matrix)
            contours_vars = calculate_contours_variables(organ_matrix, threshold)

            organ_variables[organ_name] = {
                'basic_variables': basic_vars,
                'threshold_variables': threshold_vars,
                'geom_props': geom_props,
                'regionprops': region_props,
                'glcm_variables': glcm_vars,
                'contours_variables': contours_vars
            }

        for organ_name in organs:
            row_data = {
                'frame': extraer_numero_frame(frame_file),
                'organ': organs.index(organ_name) 
            }

            row_data.update(organ_variables[organ_name]['geom_props'])
            row_data.update(organ_variables[organ_name]['basic_variables'])
            row_data.update(organ_variables[organ_name]['threshold_variables'] or {})
            regprops = organ_variables[organ_name]['regionprops']
            if regprops is not None:
                row_data.update(regprops)
            row_data.update(organ_variables[organ_name]['glcm_variables'] or {})
            row_data.update(organ_variables[organ_name]['contours_variables'] or {})

            center1 = (
                organ_variables[organ_name]['geom_props']['center_y'],
                organ_variables[organ_name]['geom_props']['center_x']
            )
            for other_name in organs:
                if organ_name == other_name:
                    dist = 0
                else:
                    center2 = (
                        organ_variables[other_name]['geom_props']['center_y'],
                        organ_variables[other_name]['geom_props']['center_x']
                    )
                    dist = compute_distance(center1, center2)
                row_data[f'distance_to_{other_name}'] = dist

            rows.append(row_data)

    return pd.DataFrame(rows)

def process_all_videos_full_variables(base_path, organs, threshold=THRESHOLD):
    """
    Procesa todos los videos disponibles en base_path usando la función process_video_full_variables.
    Para cada video, se asume que los frames se encuentran en la subcarpeta 'seg'.
    Se añade una columna 'video' para identificar el video de origen.

    Returns:
      - pd.DataFrame con todas las variables calculadas para todos los videos.
    """
    all_dfs = []
    video_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    for video_dir in video_dirs:
        video_path = os.path.join(base_path, video_dir, 'seg')
        if not os.path.exists(video_path):
            print(f"⚠️ La ruta {video_path} no existe. Saltando este video.")
            continue

        df_video = process_video_full_variables(video_path, organs, threshold)
        print(f"procesando video {video_dir}")
        df_video.insert(0, 'video', convertirIndiceVideo(video_dir))
        all_dfs.append(df_video)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    return df_all

############################################################
#   FUNCIONES PARA LIMPIEZA DE COLUMNAS Y RELLENAR NaN     #
############################################################

def columnas_constantes(df, threshold_similitud=0.05):
    """
    Identifica columnas en un DataFrame cuyos valores se mantienen prácticamente constantes.
    
    Parámetros:
    - df: DataFrame a analizar.
    - threshold_similitud: Umbral por debajo del cual la columna se considera prácticamente constante.
    
    Retorna:
    - Lista de nombres de columnas que son prácticamente constantes.
    """
    similitudes = {}
    columnas_const = []

    for col in df.columns:
        std_dev = df[col].std()
        mean_value = df[col].mean()
        similitud = std_dev / (abs(mean_value) + 1e-10) 
        similitudes[col] = similitud
        if similitud < threshold_similitud:
            columnas_const.append(col)


    return columnas_const

def eliminar_columnas_constantes(df, threshold_similitud=0.05):
    """
    Elimina columnas en un DataFrame cuyos valores se mantienen prácticamente constantes.

    Parámetros:
    - df: DataFrame a analizar.
    - threshold_similitud: Umbral por debajo del cual la columna se considera prácticamente constante.

    Retorna:
    - DataFrame sin las columnas prácticamente constantes.
    """
    columnas_a_eliminar = columnas_constantes(df, threshold_similitud=threshold_similitud)
    
    print("\nColumnas eliminadas por ser prácticamente constantes:")
    print(columnas_a_eliminar)

    return df.drop(columns=columnas_a_eliminar)

def fill_nan_by_group(df):
    """
    Rellena valores NaN agrupando por (video, organ) y ordenando por frame.
    Primero hace ffill, luego bfill, y si un grupo permanece con todos NaN,
    sustituye a 0 esos valores. Se aplica a todas las columnas excepto
    ['video', 'organ', 'frame'].
    """
    df = df.sort_values(["video", "organ", "frame"])
    cols = df.columns.difference(["video", "organ", "frame"])

    df[cols] = (
        df.groupby(["video", "organ"])[cols]
          .transform(lambda g: g.ffill().bfill().fillna(0))
    )

    return df


############################################################
#                    EJECUCIÓN PRINCIPAL                   #
############################################################

if __name__ == "__main__":
    BASE_DIR = Path.cwd()
    videos_path = BASE_DIR / 'raw_data' / 'segments'
    ORGANS = ['Anus', 'Bladder', 'Levator ani muscle', 'Pubis',
              'Rectum', 'Urethra', 'Uterus', 'Vagina']

    df_videos_with_NaN = process_all_videos_full_variables(videos_path, ORGANS)

    df_limpiado = eliminar_columnas_constantes(df_videos_with_NaN, threshold_similitud=0.05)

    df_complete = fill_nan_by_group(df_limpiado)

    output_csv_path = BASE_DIR / 'data_storage' / 'csv_by_frame' / "Frame-data-60.csv"
    df_complete.to_csv(output_csv_path, index=False)
    print(f"\nArchivo CSV generado en: {output_csv_path}")
