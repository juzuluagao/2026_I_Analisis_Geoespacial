import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, log_loss, brier_score_loss
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
from sklearn.gaussian_process.kernels import Matern

ruta_shape = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Puntos\Eventos_y_ausencias_V1.shp"
ruta_salida = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Puntos\Resultados_GP_RationalQuadratic.gpkg"

gdf = gpd.read_file(ruta_shape)
gdf["log_A"] = np.log(gdf["Area"].where(gdf["Area"] > 0))
var = ["log_A", "Pend", "Elev", "LAI", "TPI", "Curv_Mean", "TRI_Mean"]

df = gdf[["Inund"] + var].replace([-9999, np.inf, -np.inf], np.nan).dropna()
X = df[var]
y = df["Inund"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
X_all_sc = scaler.transform(X)

kernel = ConstantKernel(
    constant_value=6.43**2,
    constant_value_bounds="fixed"
) * Matern(
    length_scale=3.84,
    nu=1.5,
    length_scale_bounds="fixed"
)

modelo_gp = GaussianProcessClassifier(kernel=kernel, optimizer=None, max_iter_predict=100, warm_start=True, random_state=42)
modelo_gp.fit(X_train_sc, y_train)

print("\nKernel utilizado:")
print(modelo_gp.kernel_)

prob_train = modelo_gp.predict_proba(X_train_sc)[:, 1]
prob_test = modelo_gp.predict_proba(X_test_sc)[:, 1]
pred_train = (prob_train >= 0.5).astype(int)
pred_test = (prob_test >= 0.5).astype(int)

import rasterio
import numpy as np
import matplotlib.pyplot as plt

import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

rutas = {
    "Area": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Area_ajustada.tif",
    "Pend": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Pendiente.tif",
    "Elev": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\DEM.tif",
    "LAI": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\LAI.tif",
    "TPI": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\TPI.tif",
    "Curv_Mean": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Curvatura_ajustada.tif",
    "TRI_Mean": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\TRI.tif"
}

ruta_salida = r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Susceptibilidad_V2.tif"


with rasterio.open(rutas["Area"]) as src:
    perfil = src.profile


with rasterio.open(rutas["Area"]) as src_area, \
     rasterio.open(rutas["Pend"]) as src_pend, \
     rasterio.open(rutas["Elev"]) as src_elev, \
     rasterio.open(rutas["LAI"]) as src_lai, \
     rasterio.open(rutas["TPI"]) as src_tpi, \
     rasterio.open(rutas["Curv_Mean"]) as src_curv, \
     rasterio.open(rutas["TRI_Mean"]) as src_tri:


    perfil.update(
        dtype="float32",
        count=1,
        nodata=np.nan
    )


    with rasterio.open(
        ruta_salida,
        "w",
        **perfil
    ) as dst:


        ventanas = list(src_area.block_windows(1))
        
        for _, ventana in tqdm(
            ventanas,
            desc="Generando mapa de susceptibilidad",
            total=len(ventanas)
        ):
            area = src_area.read(
                1,
                window=ventana
            ).astype("float32")

            pend = src_pend.read(
                1,
                window=ventana
            ).astype("float32")

            elev = src_elev.read(
                1,
                window=ventana
            ).astype("float32")

            lai = src_lai.read(
                1,
                window=ventana
            ).astype("float32")

            tpi = src_tpi.read(
                1,
                window=ventana
            ).astype("float32")

            curv = src_curv.read(
                1,
                window=ventana
            ).astype("float32")

            tri = src_tri.read(
                1,
                window=ventana
            ).astype("float32")


            area[area < -9990] = np.nan
            pend[pend < -9990] = np.nan
            elev[elev < -9990] = np.nan
            lai[lai < -9990] = np.nan
            tpi[tpi < -9990] = np.nan
            curv[curv < -9990] = np.nan
            tri[tri < -9990] = np.nan


            log_A = np.full(
                area.shape,
                np.nan,
                dtype="float32"
            )

            mask_area = area > 0

            log_A[mask_area] = np.log(
                area[mask_area]
            )


            X_pixel = np.column_stack([
                log_A.ravel(),
                pend.ravel(),
                elev.ravel(),
                lai.ravel(),
                tpi.ravel(),
                curv.ravel(),
                tri.ravel()
            ])


            prob = np.full(
                X_pixel.shape[0],
                np.nan,
                dtype="float32"
            )


            mask = np.all(
                np.isfinite(X_pixel),
                axis=1
            )


            if np.any(mask):

                X_sc = scaler.transform(
                    X_pixel[mask]
                )

                mask_sc = np.all(
                    np.isfinite(X_sc),
                    axis=1
                )


                if np.any(mask_sc):

                    prob[mask] = modelo_gp.predict_proba(
                        X_sc
                    )[:,1]


            dst.write(
                prob.reshape(area.shape),
                1,
                window=ventana
            )


print("Mapa generado")


with rasterio.open(ruta_salida) as src:

    mapa = src.read(1)


plt.figure(figsize=(8,6))

plt.imshow(
    mapa,
    cmap="RdYlBu_r",
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Probabilidad de inundación"
)

plt.axis("off")
plt.tight_layout()
plt.show()














import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, log_loss, brier_score_loss
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
from sklearn.gaussian_process.kernels import Matern
from scipy.linalg import solve_triangular

ruta_shape = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Puntos\Eventos_y_ausencias_V1.shp"
ruta_salida = r"C:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\Data\Puntos\Resultados_GP_RationalQuadratic.gpkg"

gdf = gpd.read_file(ruta_shape)
gdf["log_A"] = np.log(gdf["Area"].where(gdf["Area"] > 0))
var = ["log_A", "Pend", "Elev", "LAI", "TPI", "Curv_Mean", "TRI_Mean"]

df = gdf[["Inund"] + var].replace([-9999, np.inf, -np.inf], np.nan).dropna()
X = df[var]
y = df["Inund"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
X_all_sc = scaler.transform(X)

kernel = ConstantKernel(
    constant_value=6.43**2,
    constant_value_bounds="fixed"
) * Matern(
    length_scale=3.84,
    nu=1.5,
    length_scale_bounds="fixed"
)

modelo_gp = GaussianProcessClassifier(kernel=kernel, optimizer=None, max_iter_predict=100, warm_start=True, random_state=42)
modelo_gp.fit(X_train_sc, y_train)

print("\nKernel utilizado:")
print(modelo_gp.kernel_)

prob_train = modelo_gp.predict_proba(X_train_sc)[:, 1]
prob_test = modelo_gp.predict_proba(X_test_sc)[:, 1]
pred_train = (prob_train >= 0.5).astype(int)
pred_test = (prob_test >= 0.5).astype(int)

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

rutas = {
    "Area": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Area_ajustada.tif",
    "Pend": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Pendiente.tif",
    "Elev": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\DEM.tif",
    "LAI": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\LAI.tif",
    "TPI": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\TPI.tif",
    "Curv_Mean": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Curvatura_ajustada.tif",
    "TRI_Mean": r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\TRI.tif"
}

ruta_salida = r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Final\Susceptibilidad.tif"
ruta_incertidumbre = r"C:\Analisis_Geoespacial\Archivos pesados\DEM\Antioquia\Final\Incertidumbre_GP.tif"

estimador = modelo_gp.base_estimator_

with rasterio.open(rutas["Area"]) as src:
    perfil = src.profile

with rasterio.open(rutas["Area"]) as src_area, \
     rasterio.open(rutas["Pend"]) as src_pend, \
     rasterio.open(rutas["Elev"]) as src_elev, \
     rasterio.open(rutas["LAI"]) as src_lai, \
     rasterio.open(rutas["TPI"]) as src_tpi, \
     rasterio.open(rutas["Curv_Mean"]) as src_curv, \
     rasterio.open(rutas["TRI_Mean"]) as src_tri:

    perfil.update(
        dtype="float32",
        count=1,
        nodata=np.nan
    )

    with rasterio.open(
        ruta_salida,
        "w",
        **perfil
    ) as dst, rasterio.open(
        ruta_incertidumbre,
        "w",
        **perfil
    ) as dst_inc:

        ventanas = list(src_area.block_windows(1))
        
        for _, ventana in tqdm(
            ventanas,
            desc="Generando susceptibilidad e incertidumbre",
            total=len(ventanas)
        ):
            area = src_area.read(
                1,
                window=ventana
            ).astype("float32")

            pend = src_pend.read(
                1,
                window=ventana
            ).astype("float32")

            elev = src_elev.read(
                1,
                window=ventana
            ).astype("float32")

            lai = src_lai.read(
                1,
                window=ventana
            ).astype("float32")

            tpi = src_tpi.read(
                1,
                window=ventana
            ).astype("float32")

            curv = src_curv.read(
                1,
                window=ventana
            ).astype("float32")

            tri = src_tri.read(
                1,
                window=ventana
            ).astype("float32")

            area[area < -9990] = np.nan
            pend[pend < -9990] = np.nan
            elev[elev < -9990] = np.nan
            lai[lai < -9990] = np.nan
            tpi[tpi < -9990] = np.nan
            curv[curv < -9990] = np.nan
            tri[tri < -9990] = np.nan

            log_A = np.full(
                area.shape,
                np.nan,
                dtype="float32"
            )

            mask_area = area > 0

            log_A[mask_area] = np.log(
                area[mask_area]
            )

            X_pixel = np.column_stack([
                log_A.ravel(),
                pend.ravel(),
                elev.ravel(),
                lai.ravel(),
                tpi.ravel(),
                curv.ravel(),
                tri.ravel()
            ])

            prob = np.full(
                X_pixel.shape[0],
                np.nan,
                dtype="float32"
            )

            incertidumbre = np.full(
                X_pixel.shape[0],
                np.nan,
                dtype="float32"
            )

            mask = np.all(
                np.isfinite(X_pixel),
                axis=1
            )

            if np.any(mask):

                X_sc = scaler.transform(
                    X_pixel[mask]
                )

                mask_sc = np.all(
                    np.isfinite(X_sc),
                    axis=1
                )

                if np.any(mask_sc):

                    prob_valid = modelo_gp.predict_proba(
                        X_sc
                    )[:,1]

                    prob[mask] = prob_valid

                    K_star = estimador.kernel_(
                        estimador.X_train_,
                        X_sc
                    )

                    v = solve_triangular(
                        estimador.L_,
                        estimador.W_sr_[:, np.newaxis] * K_star,
                        lower=True
                    )

                    var_f = estimador.kernel_.diag(
                        X_sc
                    ) - np.einsum(
                        "ij,ij->j",
                        v,
                        v
                    )

                    var_f = np.maximum(
                        var_f,
                        0
                    )

                    std_f = np.sqrt(
                        var_f
                    )

                    incertidumbre[mask] = std_f

            dst.write(
                prob.reshape(area.shape),
                1,
                window=ventana
            )

            dst_inc.write(
                incertidumbre.reshape(area.shape),
                1,
                window=ventana
            )

print("Mapa generado")
print("Mapa de incertidumbre generado")

with rasterio.open(ruta_salida) as src:
    mapa = src.read(1)

plt.figure(figsize=(8,6))

plt.imshow(
    mapa,
    cmap="RdYlBu_r",
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Probabilidad de inundación"
)

plt.axis("off")
plt.tight_layout()
plt.show()

with rasterio.open(ruta_incertidumbre) as src:
    mapa_inc = src.read(1)

plt.figure(figsize=(8,6))

plt.imshow(
    mapa_inc,
    cmap="magma"
)

plt.colorbar(
    label="Desviación estándar posterior del GP"
)

plt.axis("off")
plt.tight_layout()
plt.show()