import os
import sys
from pathlib import Path

# Limpiar posibles rutas incorrectas heredadas
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

# Buscar proj.db dentro del entorno virtual activo
entorno = Path(sys.prefix)

resultados = list(entorno.rglob("proj.db"))

print("Entorno activo:")
print(entorno)

print("\nArchivos proj.db encontrados:")
for ruta in resultados:
    print(ruta)
    
    
import os
import sys
from pathlib import Path

entorno = Path(sys.prefix)

rutas_rasterio = [
    ruta.parent
    for ruta in entorno.rglob("proj.db")
    if "rasterio" in str(ruta).lower()
]

if not rutas_rasterio:
    raise FileNotFoundError(
        "No se encontró proj.db dentro de la instalación de rasterio."
    )

ruta_proj = rutas_rasterio[0]

os.environ["PROJ_DATA"] = str(ruta_proj)
os.environ["PROJ_LIB"] = str(ruta_proj)

print("PROJ configurado en:")
print(ruta_proj)
print("proj.db existe:", (ruta_proj / "proj.db").exists()) 

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from affine import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

ruta_raster = r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Final\Susceptibilidad.tif"
ruta_norte = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Mapas\Flecha_norte.png"

with rasterio.open(ruta_raster) as src:
    dst_crs = "EPSG:9377"
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )

    factor = max(1, int(np.ceil(max(width, height) / 2000)))
    width_red = max(1, width // factor)
    height_red = max(1, height // factor)

    transform_red = transform * Affine.scale(
        width / width_red,
        height / height_red
    )

    prob = np.full((height_red, width_red), np.nan, dtype="float32")
    prob_cat = np.full((height_red, width_red), np.nan, dtype="float32")

    reproject(
        source=rasterio.band(src, 1),
        destination=prob,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=transform_red,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
        num_threads=2,
        warp_mem_limit=128
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=prob_cat,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=transform_red,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
        num_threads=2,
        warp_mem_limit=128
    )

left, bottom, right, top = array_bounds(
    height_red,
    width_red,
    transform_red
)

valores_validos = prob[np.isfinite(prob)]

print("Mínimo:", np.min(valores_validos))
print("Máximo:", np.max(valores_validos))
print("Media:", np.mean(valores_validos))

if np.max(valores_validos) > 1 and np.max(valores_validos) <= 100:
    prob = prob / 100

prob[(prob < 0) | (prob > 1)] = np.nan
prob_cat = prob.copy()

prob_cat[(prob_cat < 0) | (prob_cat > 1)] = np.nan

categorias = np.full(prob_cat.shape, np.nan, dtype="float32")
categorias[(prob_cat >= 0.0) & (prob_cat <= 0.2)] = 1
categorias[(prob_cat > 0.2) & (prob_cat <= 0.4)] = 2
categorias[(prob_cat > 0.4) & (prob_cat <= 0.6)] = 3
categorias[(prob_cat > 0.6) & (prob_cat <= 0.8)] = 4
categorias[(prob_cat > 0.8) & (prob_cat <= 1.0)] = 5

colores = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
cmap_cat = ListedColormap(colores)
norm_cat = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5)
flecha = mpimg.imread(ruta_norte)

def barra_escala(ax, longitud=100000, segmentos=4):
    x0 = left + 0.04 * (right - left)
    y0 = bottom + 0.04 * (top - bottom)
    alto = 0.01 * (top - bottom)
    tramo = longitud / segmentos

    for i in range(segmentos):
        ax.add_patch(Rectangle(
            (x0 + i * tramo, y0),
            tramo,
            alto,
            facecolor="black" if i % 2 == 0 else "white",
            edgecolor="black",
            linewidth=0.8,
            zorder=10
        ))
        ax.text(
            x0 + i * tramo,
            y0 + alto * 1.5,
            f"{int(i * tramo / 1000)}",
            ha="center",
            va="bottom",
            fontsize=8,
            zorder=10
        )

    ax.text(
        x0 + longitud,
        y0 + alto * 1.5,
        f"{int(longitud / 1000)} km",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=10
    )

def formato_mapa(ax, titulo):
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100000))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.tick_params(
        top=True, bottom=True, left=True, right=True,
        labeltop=True, labelbottom=True,
        labelleft=True, labelright=True, labelsize=8
    )

    for etiqueta in ax.get_yticklabels():
        etiqueta.set_rotation(90)
        etiqueta.set_va("center")

    ax.set_xlabel("Coordenada Este (m)", fontweight="bold")
    ax.set_ylabel("Coordenada Norte (m)", fontweight="bold")
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.25)
    ax.set_aspect("equal")

    ax.add_artist(AnnotationBbox(
        OffsetImage(flecha, zoom=0.018),
        (0.90, 0.88),
        xycoords="axes fraction",
        frameon=False,
        zorder=10
    ))

    barra_escala(ax)

fig, ax = plt.subplots(figsize=(9, 9), dpi=120)

im = ax.imshow(
    prob,
    extent=[left, right, bottom, top],
    origin="upper",
    cmap="RdYlBu_r",
    vmin=0,
    vmax=1
)

formato_mapa(ax, "Probabilidad de inundación")

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.03
)

cbar.set_label(
    "Probabilidad de inundación",
    fontweight="bold"
)

plt.subplots_adjust(
    left=0.12,
    right=0.88,
    bottom=0.10,
    top=0.90
)

plt.show()

fig, ax = plt.subplots(figsize=(9, 9), dpi=120)

ax.imshow(
    categorias,
    extent=[left, right, bottom, top],
    origin="upper",
    cmap=cmap_cat,
    norm=norm_cat
)

formato_mapa(ax, "Categorías de susceptibilidad")

leyenda = ax.legend(
    handles=[
        Patch(facecolor=colores[0], edgecolor="gray", label="Muy baja: 0.0 ≤ p ≤ 0.2"),
        Patch(facecolor=colores[1], edgecolor="gray", label="Baja: 0.2 < p ≤ 0.4"),
        Patch(facecolor=colores[2], edgecolor="gray", label="Media: 0.4 < p ≤ 0.6"),
        Patch(facecolor=colores[3], edgecolor="gray", label="Alta: 0.6 < p ≤ 0.8"),
        Patch(facecolor=colores[4], edgecolor="gray", label="Muy alta: 0.8 < p ≤ 1.0")
    ],
    title="Susceptibilidad",
    loc="lower right",
    frameon=True,
    framealpha=0.9,
    facecolor="white",
    edgecolor="gray",
    fontsize=8
)

leyenda.get_title().set_fontweight("bold")

plt.subplots_adjust(
    left=0.12,
    right=0.92,
    bottom=0.10,
    top=0.90
)

plt.show()








# =============================================================================
# Incertidumbre
# =============================================================================

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from affine import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

ruta_raster = r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Final\Incertidumbre_GP.tif"
ruta_norte = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Mapas\Flecha_norte.png"

with rasterio.open(ruta_raster) as src:
    dst_crs = "EPSG:9377"

    transform, width, height = calculate_default_transform(
        src.crs,
        dst_crs,
        src.width,
        src.height,
        *src.bounds
    )

    factor = max(1, int(np.ceil(max(width, height) / 2000)))
    width_red = max(1, width // factor)
    height_red = max(1, height // factor)

    transform_red = transform * Affine.scale(
        width / width_red,
        height / height_red
    )

    incertidumbre = np.full(
        (height_red, width_red),
        np.nan,
        dtype="float32"
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=incertidumbre,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=transform_red,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
        num_threads=2,
        warp_mem_limit=128
    )

left, bottom, right, top = array_bounds(
    height_red,
    width_red,
    transform_red
)

valores_validos = incertidumbre[np.isfinite(incertidumbre)]

vmin = np.min(valores_validos)
vmax = np.max(valores_validos)
media = np.mean(valores_validos)

print("Mínimo:", vmin)
print("Máximo:", vmax)
print("Media:", media)

flecha = mpimg.imread(ruta_norte)

def barra_escala(ax, longitud=100000, segmentos=4):
    x0 = left + 0.04 * (right - left)
    y0 = bottom + 0.04 * (top - bottom)
    alto = 0.01 * (top - bottom)
    tramo = longitud / segmentos

    for i in range(segmentos):
        ax.add_patch(Rectangle(
            (x0 + i * tramo, y0),
            tramo,
            alto,
            facecolor="black" if i % 2 == 0 else "white",
            edgecolor="black",
            linewidth=0.8,
            zorder=10
        ))

        ax.text(
            x0 + i * tramo,
            y0 + alto * 1.5,
            f"{int(i * tramo / 1000)}",
            ha="center",
            va="bottom",
            fontsize=8,
            zorder=10
        )

    ax.text(
        x0 + longitud,
        y0 + alto * 1.5,
        f"{int(longitud / 1000)} km",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=10
    )

def formato_mapa(ax, titulo):
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

    ax.xaxis.set_major_locator(
        ticker.MultipleLocator(100000)
    )

    ax.yaxis.set_major_locator(
        ticker.MultipleLocator(100000)
    )

    ax.xaxis.set_major_formatter(
        ticker.StrMethodFormatter("{x:.0f}")
    )

    ax.yaxis.set_major_formatter(
        ticker.StrMethodFormatter("{x:.0f}")
    )

    ax.tick_params(
        top=True,
        bottom=True,
        left=True,
        right=True,
        labeltop=True,
        labelbottom=True,
        labelleft=True,
        labelright=True,
        labelsize=8
    )

    for etiqueta in ax.get_yticklabels():
        etiqueta.set_rotation(90)
        etiqueta.set_va("center")

    ax.set_xlabel(
        "Coordenada Este (m)",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Coordenada Norte (m)",
        fontweight="bold"
    )

    ax.set_title(
        titulo,
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(
        linestyle="--",
        linewidth=0.5,
        alpha=0.25
    )

    ax.set_aspect("equal")

    ax.add_artist(
        AnnotationBbox(
            OffsetImage(
                flecha,
                zoom=0.018
            ),
            (0.90, 0.88),
            xycoords="axes fraction",
            frameon=False,
            zorder=10
        )
    )

    barra_escala(ax)

fig, ax = plt.subplots(
    figsize=(9, 9),
    dpi=120
)

im = ax.imshow(
    incertidumbre,
    extent=[
        left,
        right,
        bottom,
        top
    ],
    origin="upper",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax
)

formato_mapa(
    ax,
    "Incertidumbre del proceso gaussiano"
)

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.03
)

cbar.set_label(
    "Desviación estándar posterior",
    fontweight="bold"
)

plt.subplots_adjust(
    left=0.12,
    right=0.88,
    bottom=0.10,
    top=0.90
)

plt.show()