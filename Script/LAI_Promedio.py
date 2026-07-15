import os
import numpy as np
import rasterio
import geopandas as gpd

from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling
from tqdm import tqdm
from numba import njit

# =====================================================
# RUTAS
# =====================================================
ruta_flowdir = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Superficies\DEM\FlowDir.tif"
ruta_lai = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Superficies\LAI\LAI.tif"
ruta_puntos = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Otros\Shape\Puntos_Snap.shp"
ruta_salida = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Otros\Shape\Puntos_LAI.shp"
ruta_tif_lai_ajustado = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Superficies\LAI\LAI_Acumulado.tif"
ruta_tif_lai_flowacc = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Superficies\LAI\LAI_FlowAccum_Mean.tif"

# =====================================================
# CONFIGURACIÓN
# =====================================================
NODATA_SALIDA = -9999.0

# =====================================================
# FUNCIONES NUMBA
# =====================================================
@njit
def vecino_drena_hacia_celda(flowdir_vecino, di, dj):

    if di == 0 and dj == -1 and flowdir_vecino == 1:
        return True
    if di == -1 and dj == -1 and flowdir_vecino == 2:
        return True
    if di == -1 and dj == 0 and flowdir_vecino == 4:
        return True
    if di == -1 and dj == 1 and flowdir_vecino == 8:
        return True
    if di == 0 and dj == 1 and flowdir_vecino == 16:
        return True
    if di == 1 and dj == 1 and flowdir_vecino == 32:
        return True
    if di == 1 and dj == 0 and flowdir_vecino == 64:
        return True
    if di == 1 and dj == -1 and flowdir_vecino == 128:
        return True

    return False


@njit
def calcular_lai_medio_cuenca_y_raster(
    flowdir,
    lai,
    fila_ini,
    col_ini,
    visitado,
    marca,
    raster_lai_acc
):
    filas, cols = flowdir.shape
    max_celdas = filas * cols

    cola_f = np.empty(max_celdas, dtype=np.int32)
    cola_c = np.empty(max_celdas, dtype=np.int32)

    inicio = 0
    fin = 0

    suma = 0.0
    conteo = 0

    cola_f[fin] = fila_ini
    cola_c[fin] = col_ini
    fin += 1

    visitado[fila_ini, col_ini] = marca

    while inicio < fin:

        f = cola_f[inicio]
        c = cola_c[inicio]
        inicio += 1

        valor_lai = lai[f, c]

        if np.isfinite(valor_lai):
            suma += valor_lai
            conteo += 1

        for di in range(-1, 2):
            for dj in range(-1, 2):

                if di == 0 and dj == 0:
                    continue

                nf = f + di
                nc = c + dj

                if nf < 0 or nf >= filas or nc < 0 or nc >= cols:
                    continue

                if visitado[nf, nc] == marca:
                    continue

                fd_vecino = flowdir[nf, nc]

                if vecino_drena_hacia_celda(fd_vecino, di, dj):
                    visitado[nf, nc] = marca
                    cola_f[fin] = nf
                    cola_c[fin] = nc
                    fin += 1

    if conteo > 0:
        promedio = suma / conteo
    else:
        promedio = np.nan

    for k in range(fin):
        f = cola_f[k]
        c = cola_c[k]
        raster_lai_acc[f, c] = promedio

    return promedio


# =====================================================
# PROGRAMA PRINCIPAL
# =====================================================
print("Leyendo Flow Direction y ajustando LAI a su grilla...")

with rasterio.open(ruta_flowdir) as src_fd, rasterio.open(ruta_lai) as src_lai:

    flowdir = src_fd.read(1).astype(np.uint8)

    transform = src_fd.transform
    crs_raster = src_fd.crs
    nodata_fd = src_fd.nodata

    filas = src_fd.height
    cols = src_fd.width

    lai = np.full((filas, cols), np.nan, dtype=np.float32)

    nodata_lai = src_lai.nodata

    reproject(
        source=rasterio.band(src_lai, 1),
        destination=lai,
        src_transform=src_lai.transform,
        src_crs=src_lai.crs,
        src_nodata=nodata_lai,
        dst_transform=src_fd.transform,
        dst_crs=src_fd.crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear
    )

if nodata_fd is not None:
    flowdir[flowdir == nodata_fd] = 0

if nodata_lai is not None:
    lai[lai == nodata_lai] = np.nan

print("Dimensiones Flow Direction:", flowdir.shape)
print("Dimensiones LAI ajustado:", lai.shape)

# =====================================================
# LEER PUNTOS
# =====================================================
print("Leyendo puntos de cierre...")

puntos = gpd.read_file(ruta_puntos)

if puntos.crs != crs_raster:
    puntos = puntos.to_crs(crs_raster)

# =====================================================
# MATRICES EN MEMORIA
# =====================================================
visitado = np.zeros((filas, cols), dtype=np.int32)
raster_lai_acc = np.full((filas, cols), np.nan, dtype=np.float32)

# =====================================================
# CALCULAR LAI MEDIO POR CUENCA
# =====================================================
print("Calculando LAI medio por cuenca...")

lai_mean = []

for idx, geom in tqdm(enumerate(puntos.geometry), total=len(puntos)):

    if geom is None or geom.is_empty:
        lai_mean.append(NODATA_SALIDA)
        continue

    x = geom.x
    y = geom.y

    try:
        fila, col = rowcol(transform, x, y)
    except Exception:
        lai_mean.append(NODATA_SALIDA)
        continue

    if fila < 0 or fila >= filas or col < 0 or col >= cols:
        lai_mean.append(NODATA_SALIDA)
        continue

    if flowdir[fila, col] == 0:
        lai_mean.append(NODATA_SALIDA)
        continue

    marca = idx + 1

    promedio = calcular_lai_medio_cuenca_y_raster(
        flowdir,
        lai,
        fila,
        col,
        visitado,
        marca,
        raster_lai_acc
    )

    if np.isfinite(promedio):
        lai_mean.append(float(promedio))
    else:
        lai_mean.append(NODATA_SALIDA)

puntos["LAI_MEAN"] = lai_mean

# =====================================================
# EXPORTAR RASTER LAI MEDIO FLOW ACCUMULATION
# =====================================================
print("Exportando raster LAI medio del Flow Accumulation...")

raster_lai_acc_export = np.where(
    np.isfinite(raster_lai_acc),
    raster_lai_acc,
    NODATA_SALIDA
).astype(np.float32)

perfil_lai_acc = {
    "driver": "GTiff",
    "height": filas,
    "width": cols,
    "count": 1,
    "dtype": "float32",
    "crs": crs_raster,
    "transform": transform,
    "nodata": NODATA_SALIDA,
    "compress": "lzw"
}

os.makedirs(os.path.dirname(ruta_tif_lai_flowacc), exist_ok=True)

with rasterio.open(ruta_tif_lai_flowacc, "w", **perfil_lai_acc) as dst:
    dst.write(raster_lai_acc_export, 1)

print("Raster exportado en:")
print(ruta_tif_lai_flowacc)

# =====================================================
# GUARDAR SHAPEFILE
# =====================================================
print("Guardando shapefile de salida...")

os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
puntos.to_file(ruta_salida)

print("Proceso finalizado.")
print("Shapefile generado:")
print(ruta_salida)