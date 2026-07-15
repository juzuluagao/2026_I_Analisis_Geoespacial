import os
import glob
import re
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pyhdf.SD import SD, SDC

# ==========================
# Rutas
# ==========================
ruta_entrada = r"C:\LAI"
ruta_salida = r"C:\LAI\LAI_PROMEDIO\LAI.tif"

nombre_ds = "Lai_500m"
nodata_val = -9999

# ==========================
# Buscar archivos HDF
# ==========================
archivos_hdf = glob.glob(os.path.join(ruta_entrada, "*.hdf"))

if len(archivos_hdf) == 0:
    raise FileNotFoundError("No se encontraron archivos .hdf")

print(f"Archivos encontrados: {len(archivos_hdf)}")

# ==========================
# Función para extraer georreferencia MODIS
# ==========================
def obtener_georef_modis(hdf):
    metadata = hdf.attributes()
    struct_metadata = metadata["StructMetadata.0"]

    ul = re.search(r"UpperLeftPointMtrs=\(([-\d\.]+),([-\d\.]+)\)", struct_metadata)
    lr = re.search(r"LowerRightMtrs=\(([-\d\.]+),([-\d\.]+)\)", struct_metadata)

    if ul is None or lr is None:
        raise ValueError("No se pudo extraer la georreferencia del HDF")

    xmin = float(ul.group(1))
    ymax = float(ul.group(2))
    xmax = float(lr.group(1))
    ymin = float(lr.group(2))

    return xmin, ymin, xmax, ymax

# ==========================
# Acumuladores
# ==========================
suma_lai = None
contador = None
transform = None
height = None
width = None

# ==========================
# Procesamiento
# ==========================
for archivo in archivos_hdf:
    print("Procesando:", os.path.basename(archivo))

    hdf = SD(archivo, SDC.READ)

    ds = hdf.select(nombre_ds)
    lai = ds[:].astype(np.float32)

    # Extraer dimensiones
    if height is None:
        height, width = lai.shape

        xmin, ymin, xmax, ymax = obtener_georef_modis(hdf)
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    # ==========================
    # Limpieza
    # Valores inválidos MODIS LAI: 249-255
    # ==========================
    lai[(lai >= 249) & (lai <= 255)] = 0

    # Factor de escala LAI
    lai *= 0.1

    if suma_lai is None:
        suma_lai = np.zeros_like(lai, dtype=np.float32)
        contador = np.zeros_like(lai, dtype=np.float32)

    mascara = ~np.isnan(lai)

    suma_lai[mascara] += lai[mascara]
    contador[mascara] += 1

    hdf.end()

# ==========================
# Promedio
# ==========================
lai_prom = np.full_like(suma_lai, np.nan, dtype=np.float32)

mascara_prom = contador > 0
lai_prom[mascara_prom] = suma_lai[mascara_prom] / contador[mascara_prom]

lai_salida = np.where(np.isnan(lai_prom), nodata_val, lai_prom).astype(np.float32)

# ==========================
# Estadísticas
# ==========================
print("Dimensiones:", lai_prom.shape)
print("Mínimo:", np.nanmin(lai_prom))
print("Máximo:", np.nanmax(lai_prom))
print("Promedio:", np.nanmean(lai_prom))


# ==========================
# Exportar GeoTIFF
# ==========================
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

with rasterio.open(
    ruta_salida,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="float32",
    # Definición PROJ exacta para la Proyección Sinusoidal de MODIS
    crs="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs",
    transform=transform,
    nodata=nodata_val,
    compress="lzw"
) as dst:
    dst.write(lai_salida, 1)
    
print("Raster promedio exportado en:")
print(ruta_salida)

# ==========================
# Gráfica del raster promedio
# ==========================
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))

# Enmascarar NoData para que no aparezca en la figura
lai_plot = np.ma.masked_where(lai_salida == nodata_val, lai_salida)

im = plt.imshow(
    lai_plot,
    cmap='YlGn',
    origin='upper',
    vmin=np.nanmin(lai_prom),
    vmax=np.nanmax(lai_prom)
)

plt.title(f"LAI promedio ({len(archivos_hdf)} imágenes MODIS)")
plt.xlabel("Columna")
plt.ylabel("Fila")

cbar = plt.colorbar(im)
cbar.set_label("LAI")

plt.tight_layout()
plt.show()