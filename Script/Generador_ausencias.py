import os
import numpy as np
import rasterio
import geopandas as gpd

from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm

# =====================================================
# RUTAS DE ENTRADA Y SALIDA
# =====================================================
ruta_raster = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Superficies\DEM\red_raster_Antioquia.tif"
ruta_puntos_inundacion = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Puntos\Base_datos_Completa.shp"
ruta_salida = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Puntos\Ausencias_V1.shp"

# =====================================================
# PARÁMETROS
# =====================================================
numero_puntos = 1270          # cantidad final de puntos aleatorios deseados
distancia_minima = 0.009050573      # distancia mínima a puntos de inundación, en unidades del CRS
semilla = 1234                 # semilla para reproducibilidad

# =====================================================
# LEER RASTER Y OBTENER CENTROS DE PÍXELES VÁLIDOS
# =====================================================
print("Leyendo raster y detectando píxeles válidos...")

with rasterio.open(ruta_raster) as src:
    raster = src.read(1)
    nodata = src.nodata
    transform = src.transform
    crs_raster = src.crs

    if nodata is not None:
        mascara_valida = raster != nodata
    else:
        mascara_valida = np.isfinite(raster)

    mascara_valida &= np.isfinite(raster)

    filas, cols = np.where(mascara_valida)

    if len(filas) == 0:
        raise ValueError("No se encontraron píxeles válidos en el raster.")

    xs, ys = rasterio.transform.xy(
        transform,
        filas,
        cols,
        offset="center"
    )

    coords_validas = np.column_stack([xs, ys])

print(f"Píxeles válidos encontrados: {len(coords_validas)}")

# =====================================================
# LEER PUNTOS DE INUNDACIÓN
# =====================================================
print("Leyendo puntos de inundación...")

puntos_inundacion = gpd.read_file(ruta_puntos_inundacion)

if puntos_inundacion.crs != crs_raster:
    puntos_inundacion = puntos_inundacion.to_crs(crs_raster)

puntos_inundacion = puntos_inundacion[
    puntos_inundacion.geometry.notnull() &
    (~puntos_inundacion.geometry.is_empty)
].copy()

if len(puntos_inundacion) == 0:
    raise ValueError("El shapefile de inundación no contiene puntos válidos.")

coords_inundacion = np.array([
    (geom.x, geom.y) for geom in puntos_inundacion.geometry
])

# =====================================================
# FILTRAR CANDIDATOS POR DISTANCIA A INUNDACIONES
# =====================================================
print("Filtrando píxeles cercanos a puntos de inundación...")

tree = cKDTree(coords_inundacion)

distancias, _ = tree.query(
    coords_validas,
    k=1,
    workers=-1
)

mascara_filtrada = distancias >= distancia_minima
coords_filtradas = coords_validas[mascara_filtrada]

print(f"Candidatos después del filtro de distancia: {len(coords_filtradas)}")

if len(coords_filtradas) == 0:
    raise ValueError("No quedaron puntos candidatos después del filtro de distancia.")

if len(coords_filtradas) < numero_puntos:
    raise ValueError(
        f"Solo hay {len(coords_filtradas)} candidatos disponibles, "
        f"pero solicitaste {numero_puntos} puntos."
    )

# =====================================================
# SELECCIÓN ALEATORIA
# =====================================================
print("Seleccionando puntos aleatorios...")

rng = np.random.default_rng(semilla)

indices = rng.choice(
    len(coords_filtradas),
    size=numero_puntos,
    replace=False
)

coords_seleccionadas = coords_filtradas[indices]

# =====================================================
# CREAR SHAPEFILE DE SALIDA
# =====================================================
print("Creando shapefile de salida...")

geometria = [
    Point(x, y) for x, y in coords_seleccionadas
]

gdf_salida = gpd.GeoDataFrame(
    {
        "ID": np.arange(1, numero_puntos + 1),
        "Tipo": "No_Inund",
        "Dist_Min": distancias[mascara_filtrada][indices]
    },
    geometry=geometria,
    crs=crs_raster
)

os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

gdf_salida.to_file(ruta_salida)

print("Proceso finalizado.")
print(f"Puntos generados: {len(gdf_salida)}")
print("Archivo generado:")
print(ruta_salida)
