# ============================================================
# 1. CARGAR LIBRERÍAS
# ============================================================

library(sf)
library(spdep)
library(Matrix)
library(INLA)
library(ggplot2)
library(pROC)


# ============================================================
# 2. LEER EL SHAPEFILE
# ============================================================

ruta <- paste0(
  "C:/Analisis_Geoespacial/2026_I_Analisis_Geoespacial/",
  "Data/Poligonos/Poligonos/R/Muncipios_ON.shp"
)

aoi <- st_read(
  ruta,
  quiet = TRUE
)

print(aoi)
print(st_crs(aoi))
print(names(aoi))


# ============================================================
# 3. CORREGIR Y REVISAR LAS GEOMETRÍAS
# ============================================================

aoi <- st_make_valid(aoi)

# Eliminar geometrías vacías
aoi <- aoi[
  !st_is_empty(aoi),
]

# Revisar validez
print(
  table(st_is_valid(aoi))
)


# ============================================================
# 4. REVISAR LAS VARIABLES DEL MODELO
# ============================================================

variables_modelo <- c(
  "Tot_event",
  "Area",
  "Pend_mean",
  "TRI_mean"
)

variables_faltantes <- setdiff(
  variables_modelo,
  names(aoi)
)

if (length(variables_faltantes) > 0) {
  
  stop(
    paste(
      "Faltan las siguientes columnas:",
      paste(
        variables_faltantes,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 5. CONVERTIR VARIABLES A NUMÉRICO
# ============================================================

aoi$Tot_event <- as.numeric(aoi$Tot_event)
aoi$Area      <- as.numeric(aoi$Area)
aoi$Pend_mean <- as.numeric(aoi$Pend_mean)
aoi$TRI_mean  <- as.numeric(aoi$TRI_mean)


# ============================================================
# 6. FILTRAR DATOS VÁLIDOS
# ============================================================

aoi <- aoi[
  complete.cases(
    aoi$Tot_event,
    aoi$Area,
    aoi$Pend_mean,
    aoi$TRI_mean
  ) &
    aoi$Area > 0 &
    aoi$Tot_event >= 0,
]

# Reordenar las filas
aoi <- aoi[
  seq_len(nrow(aoi)),
]

# Identificador consecutivo para INLA
aoi$id_area <- seq_len(nrow(aoi))

rownames(aoi) <- as.character(
  aoi$id_area
)

cat(
  "\nNúmero de municipios usados:",
  nrow(aoi),
  "\n"
)


# ============================================================
# 7. CREAR LA VECINDAD QUEEN
# ============================================================

aoi.nb <- poly2nb(
  aoi,
  queen = TRUE,
  row.names = as.character(aoi$id_area)
)

print(aoi.nb)
summary(aoi.nb)

# Número de vecinos por municipio
aoi$n_vecinos <- card(aoi.nb)

# Identificar municipios sin vecinos
sin_vecinos <- which(
  aoi$n_vecinos == 0
)

if (length(sin_vecinos) > 0) {
  
  warning(
    paste(
      "Hay municipios sin vecinos:",
      paste(
        sin_vecinos,
        collapse = ", "
      )
    )
  )
  
} else {
  
  message(
    "Todos los municipios tienen al menos un vecino."
  )
}


# ============================================================
# 8. MATRIZ BINARIA DE ADYACENCIA
# ============================================================

aoi.mat <- nb2mat(
  aoi.nb,
  style = "B",
  zero.policy = TRUE
)

# Convertir a matriz dispersa
aoi.mat <- Matrix(
  aoi.mat,
  sparse = TRUE
)

rownames(aoi.mat) <- as.character(
  aoi$id_area
)

colnames(aoi.mat) <- as.character(
  aoi$id_area
)

# Matriz convencional para inspección
mat <- as.matrix(aoi.mat)

print(
  aoi.mat[
    1:min(10, nrow(aoi.mat)),
    1:min(10, ncol(aoi.mat))
  ]
)


# ============================================================
# 9. OBJETO DE PESOS PARA MORAN
# ============================================================

aoi.listw <- nb2listw(
  aoi.nb,
  style = "W",
  zero.policy = TRUE
)


# ============================================================
# 10. PREPARAR LOS DATOS PARA INLA
# ============================================================

datos_inla <- st_drop_geometry(aoi)

str(datos_inla)


# ============================================================
# 11. FÓRMULA DEL MODELO LEROUX
# ============================================================

formula_leroux <- Tot_event ~
  1 +
  Pend_mean +
  TRI_mean +
  f(
    id_area,
    
    # Modelo Leroux en R-INLA
    model = "besagproper2",
    
    # Matriz de vecindad
    graph = aoi.mat,
    
    # El modelo Leroux es propio y no necesita
    # restricción de suma cero
    constr = FALSE
  )


# ============================================================
# 12. MODELO BINOMIAL NEGATIVO LEROUX
# ============================================================

m1_leroux <- inla(
  formula = formula_leroux,
  
  family = "nbinomial",
  
  data = datos_inla,
  
  # Modela la tasa de eventos por unidad de área
  offset = log(Area),
  
  control.predictor = list(
    compute = TRUE
  ),
  
  control.compute = list(
    dic = TRUE,
    waic = TRUE,
    cpo = TRUE,
    config = TRUE,
    return.marginals.predictor = TRUE
  )
)

summary(m1_leroux)


# ============================================================
# 13. VALORES AJUSTADOS DEL MODELO LEROUX
# ============================================================

valores_ajustados <- m1_leroux$summary.fitted.values$mean[
  seq_len(nrow(aoi))
]

aoi$conteo_ajustado <- valores_ajustados

# Guardar también la incertidumbre de las predicciones
aoi$prediccion_sd <-
  m1_leroux$summary.fitted.values$sd[
    seq_len(nrow(aoi))
  ]

aoi$prediccion_q025 <-
  m1_leroux$summary.fitted.values$`0.025quant`[
    seq_len(nrow(aoi))
  ]

aoi$prediccion_q975 <-
  m1_leroux$summary.fitted.values$`0.975quant`[
    seq_len(nrow(aoi))
  ]


# ============================================================
# 14. EXTRAER LOS HIPERPARÁMETROS
# ============================================================

hiperparametros <- m1_leroux$summary.hyperpar

print(hiperparametros)

nombres_hiperparametros <- rownames(
  hiperparametros
)


# ------------------------------------------------------------
# 14.1 PARÁMETRO SIZE DE LA BINOMIAL NEGATIVA
# ------------------------------------------------------------

fila_size <- grep(
  pattern = "size|overdispersion",
  x = nombres_hiperparametros,
  ignore.case = TRUE
)

if (length(fila_size) == 0) {
  
  stop(
    paste(
      "No se encontró el parámetro size.",
      "Verifica que family = 'nbinomial'."
    )
  )
}

size_nb <- hiperparametros[
  fila_size[1],
  "mean"
]


# ------------------------------------------------------------
# 14.2 PARÁMETRO LAMBDA DEL MODELO LEROUX
# ------------------------------------------------------------

fila_lambda <- grep(
  pattern = "lambda",
  x = nombres_hiperparametros,
  ignore.case = TRUE
)

if (length(fila_lambda) > 0) {
  
  lambda_leroux <- hiperparametros[
    fila_lambda[1],
    "mean"
  ]
  
  lambda_q025 <- hiperparametros[
    fila_lambda[1],
    "0.025quant"
  ]
  
  lambda_q975 <- hiperparametros[
    fila_lambda[1],
    "0.975quant"
  ]
  
} else {
  
  lambda_leroux <- NA_real_
  lambda_q025 <- NA_real_
  lambda_q975 <- NA_real_
  
  warning(
    "No fue posible identificar el parámetro lambda."
  )
}


# ------------------------------------------------------------
# 14.3 PRECISIÓN DEL EFECTO ESPACIAL
# ------------------------------------------------------------

fila_precision <- grep(
  pattern = "precision",
  x = nombres_hiperparametros,
  ignore.case = TRUE
)

if (length(fila_precision) > 0) {
  
  precision_leroux <- hiperparametros[
    fila_precision[1],
    "mean"
  ]
  
} else {
  
  precision_leroux <- NA_real_
}


cat(
  "\nParámetro size de la binomial negativa:",
  size_nb,
  "\n"
)

cat(
  "Lambda del modelo Leroux:",
  lambda_leroux,
  "\n"
)

cat(
  "Intervalo creíble 95 % de lambda:",
  lambda_q025,
  "-",
  lambda_q975,
  "\n"
)

cat(
  "Precisión del efecto espacial:",
  precision_leroux,
  "\n"
)


# ============================================================
# 15. RESIDUOS DE PEARSON BINOMIAL NEGATIVA
# ============================================================

# Para una binomial negativa:
#
# Var(Y) = mu + mu² / size

varianza_nb <- valores_ajustados +
  valores_ajustados^2 / size_nb

res_pearson_leroux <- (
  aoi$Tot_event - valores_ajustados
) / sqrt(
  pmax(
    varianza_nb,
    .Machine$double.eps
  )
)

aoi$res_pearson_leroux <-
  res_pearson_leroux


# ============================================================
# 16. DIFERENCIAS OBSERVADO - AJUSTADO
# ============================================================

aoi$diferencia <- (
  aoi$Tot_event -
    aoi$conteo_ajustado
)

aoi$diferencia_absoluta <- abs(
  aoi$diferencia
)


# ============================================================
# 17. R2 DE EFRON
# ============================================================

sse <- sum(
  (
    aoi$Tot_event -
      valores_ajustados
  )^2,
  na.rm = TRUE
)

sst <- sum(
  (
    aoi$Tot_event -
      mean(
        aoi$Tot_event,
        na.rm = TRUE
      )
  )^2,
  na.rm = TRUE
)

R2_efron <- if (sst > 0) {
  
  1 - sse / sst
  
} else {
  
  NA_real_
}


# R2 basado en correlación como referencia
R2_correlacion <- if (
  sd(aoi$Tot_event) > 0 &&
  sd(valores_ajustados) > 0
) {
  
  cor(
    aoi$Tot_event,
    valores_ajustados,
    use = "complete.obs"
  )^2
  
} else {
  
  NA_real_
}


# ============================================================
# 18. AIC CONDICIONAL APROXIMADO
# ============================================================

logLik_condicional <- sum(
  dnbinom(
    x = aoi$Tot_event,
    size = size_nb,
    mu = valores_ajustados,
    log = TRUE
  ),
  na.rm = TRUE
)

p_efectivo <- as.numeric(
  m1_leroux$waic$p.eff
)

if (
  length(p_efectivo) == 0 ||
  is.na(p_efectivo)
) {
  
  p_efectivo <-
    nrow(m1_leroux$summary.fixed) +
    nrow(m1_leroux$summary.hyperpar)
}

AIC_aproximado <- (
  -2 * logLik_condicional +
    2 * p_efectivo
)


# ============================================================
# 19. ROC-AUC PARA PRESENCIA O AUSENCIA
# ============================================================

# 0 = ningún evento
# 1 = uno o más eventos
aoi$presencia_evento <- as.integer(
  aoi$Tot_event > 0
)

# P(Y > 0) para una binomial negativa
aoi$prob_presencia <- 1 - dnbinom(
  x = 0,
  size = size_nb,
  mu = valores_ajustados
)

roc_obj <- NULL
AUC_ROC <- NA_real_

if (
  length(
    unique(aoi$presencia_evento)
  ) == 2
) {
  
  roc_obj <- pROC::roc(
    response = aoi$presencia_evento,
    predictor = aoi$prob_presencia,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  AUC_ROC <- as.numeric(
    pROC::auc(roc_obj)
  )
  
} else {
  
  warning(
    paste(
      "No se puede calcular ROC-AUC:",
      "la variable presencia_evento solamente tiene una clase."
    )
  )
}


# ============================================================
# 20. MORAN GLOBAL DE LOS RESIDUOS
# ============================================================

set.seed(123)

moran_leroux <- moran.mc(
  x = aoi$res_pearson_leroux,
  listw = aoi.listw,
  nsim = 999,
  alternative = "greater",
  zero.policy = TRUE
)

print(moran_leroux)

I_moran <- unname(
  moran_leroux$statistic
)

p_moran <- moran_leroux$p.value


# ============================================================
# 21. PREPARAR EL MORAN SCATTERPLOT
# ============================================================

residuos_estandarizados <- as.numeric(
  scale(
    aoi$res_pearson_leroux
  )
)

rezago_espacial <- lag.listw(
  aoi.listw,
  residuos_estandarizados,
  zero.policy = TRUE
)

datos_moran <- data.frame(
  residuos = residuos_estandarizados,
  rezago = as.numeric(rezago_espacial)
)


# ============================================================
# 22. GRÁFICO DE DISPERSIÓN DE MORAN
# ============================================================

grafico_moran <- ggplot(
  datos_moran,
  aes(
    x = residuos,
    y = rezago
  )
) +
  
  geom_point(
    color = "#4C8DC4",
    size = 2.3,
    alpha = 0.85
  ) +
  
  geom_hline(
    yintercept = 0,
    linetype = "dotted",
    linewidth = 0.45,
    color = "grey40"
  ) +
  
  geom_vline(
    xintercept = 0,
    linetype = "dotted",
    linewidth = 0.45,
    color = "grey40"
  ) +
  
  geom_abline(
    intercept = 0,
    slope = I_moran,
    color = "#C51B1D",
    linewidth = 1.1,
    linetype = "dashed"
  ) +
  
  labs(
    title = paste0(
      "Gráfico de Dispersión de Moran ",
      "(Moran Scatterplot)"
    ),
    
    subtitle = paste0(
      "Modelo Leroux | Moran I = ",
      round(I_moran, 4),
      " | p = ",
      format.pval(
        p_moran,
        digits = 4,
        eps = 0.0001
      )
    ),
    
    x = paste0(
      "Residuos de Pearson del municipio ",
      "(estandarizados)"
    ),
    
    y = paste0(
      "Promedio de los residuos de los vecinos ",
      "(Spatial Lag)"
    )
  ) +
  
  theme_minimal(
    base_size = 14
  ) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 17
    ),
    
    plot.subtitle = element_text(
      size = 13
    ),
    
    panel.grid.minor = element_blank()
  )

print(grafico_moran)


# ============================================================
# 23. MAPA DE RESIDUOS DE PEARSON
# ============================================================

limite_residuos <- max(
  abs(aoi$res_pearson_leroux),
  na.rm = TRUE
)

if (
  !is.finite(limite_residuos) ||
  limite_residuos == 0
) {
  
  limite_residuos <- 1
}

mapa_residuos <- ggplot(aoi) +
  
  geom_sf(
    aes(
      fill = res_pearson_leroux
    ),
    color = "grey35",
    linewidth = 0.15
  ) +
  
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    limits = c(
      -limite_residuos,
      limite_residuos
    ),
    oob = scales::squish,
    name = "Residuo\nde Pearson"
  ) +
  
  labs(
    title = "Residuos de Pearson del modelo Leroux",
    
    subtitle = paste0(
      "Binomial negativa | Moran I = ",
      round(I_moran, 4),
      " | p = ",
      format.pval(
        p_moran,
        digits = 4,
        eps = 0.0001
      )
    ),
    
    caption = paste0(
      "Azul: conteos menores que los estimados. ",
      "Rojo: conteos mayores que los estimados."
    )
  ) +
  
  coord_sf(
    datum = NA
  ) +
  
  theme_void(
    base_size = 13
  ) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 17
    ),
    
    plot.subtitle = element_text(
      size = 12
    ),
    
    plot.caption = element_text(
      hjust = 0
    ),
    
    legend.position = "right"
  )

print(mapa_residuos)


# ============================================================
# 24. GRÁFICO ROC
# ============================================================

if (!is.null(roc_obj)) {
  
  grafico_roc <- pROC::ggroc(
    roc_obj,
    legacy.axes = TRUE,
    linewidth = 1.1
  ) +
    
    geom_abline(
      intercept = 0,
      slope = 1,
      linetype = "dashed",
      color = "grey50"
    ) +
    
    coord_equal() +
    
    labs(
      title = "Curva ROC: presencia de eventos",
      
      subtitle = paste0(
        "Modelo Leroux | AUC = ",
        round(AUC_ROC, 4)
      ),
      
      x = "1 - Especificidad",
      y = "Sensibilidad"
    ) +
    
    theme_minimal(
      base_size = 14
    ) +
    
    theme(
      plot.title = element_text(
        face = "bold"
      )
    )
  
  print(grafico_roc)
}


# ============================================================
# 25. EXTRAER EL EFECTO ESPACIAL LEROUX
# ============================================================

efecto_espacial <- m1_leroux$summary.random$id_area

# Hacer corresponder el ID del efecto con el ID de cada área
efecto_espacial <- efecto_espacial[
  match(
    aoi$id_area,
    efecto_espacial$ID
  ),
]

aoi$efecto_leroux_media <-
  efecto_espacial$mean

aoi$efecto_leroux_sd <-
  efecto_espacial$sd

aoi$efecto_leroux_q025 <-
  efecto_espacial$`0.025quant`

aoi$efecto_leroux_q975 <-
  efecto_espacial$`0.975quant`


# ============================================================
# 26. CPO Y LPML
# ============================================================

aoi$CPO <- m1_leroux$cpo$cpo
aoi$PIT <- m1_leroux$cpo$pit

LPML <- sum(
  log(
    pmax(
      m1_leroux$cpo$cpo,
      .Machine$double.xmin
    )
  ),
  na.rm = TRUE
)


# ============================================================
# 27. TABLA DE MÉTRICAS
# ============================================================

metricas_modelo <- data.frame(
  Modelo = "Binomial negativa Leroux",
  
  R2_Efron = R2_efron,
  
  R2_correlacion = R2_correlacion,
  
  AIC_condicional_aprox =
    AIC_aproximado,
  
  DIC = m1_leroux$dic$dic,
  
  WAIC = m1_leroux$waic$waic,
  
  LPML = LPML,
  
  ROC_AUC_presencia = AUC_ROC,
  
  Moran_I_residuos = I_moran,
  
  Moran_p_valor = p_moran,
  
  Size_NB = size_nb,
  
  Lambda_Leroux = lambda_leroux,
  
  Lambda_q025 = lambda_q025,
  
  Lambda_q975 = lambda_q975,
  
  Precision_Leroux = precision_leroux
)

cat("\n")
cat("===============================================\n")
cat("MÉTRICAS DEL MODELO BINOMIAL NEGATIVO LEROUX\n")
cat("===============================================\n")

print(
  metricas_modelo,
  digits = 5,
  row.names = FALSE
)

cat(
  "\nR2 de Efron:",
  round(R2_efron, 4),
  "\n"
)

cat(
  "R2 de correlación:",
  round(R2_correlacion, 4),
  "\n"
)

cat(
  "AIC condicional aproximado:",
  round(AIC_aproximado, 2),
  "\n"
)

cat(
  "DIC:",
  round(m1_leroux$dic$dic, 2),
  "\n"
)

cat(
  "WAIC:",
  round(m1_leroux$waic$waic, 2),
  "\n"
)

cat(
  "ROC-AUC:",
  round(AUC_ROC, 4),
  "\n"
)

cat(
  "Moran I:",
  round(I_moran, 4),
  "\n"
)

cat(
  "P-valor Moran:",
  p_moran,
  "\n"
)

cat(
  "Lambda Leroux:",
  round(lambda_leroux, 4),
  "\n"
)

cat(
  "IC 95 % de lambda:",
  round(lambda_q025, 4),
  "-",
  round(lambda_q975, 4),
  "\n"
)


# ============================================================
# 28. GUARDAR RESULTADOS
# ============================================================

carpeta_salida <- paste0(
  "C:/Analisis_Geoespacial/2026_I_Analisis_Geoespacial/",
  "Data/Poligonos/Poligonos/Resultados_Leroux"
)

dir.create(
  carpeta_salida,
  recursive = TRUE,
  showWarnings = FALSE
)


# ------------------------------------------------------------
# 28.1 GUARDAR GEOPACKAGE
# ------------------------------------------------------------

ruta_gpkg <- file.path(
  carpeta_salida,
  "Resultados_Leroux.gpkg"
)

st_write(
  aoi,
  ruta_gpkg,
  layer = "modelo_leroux",
  delete_layer = TRUE,
  quiet = TRUE
)


# ------------------------------------------------------------
# 28.2 GUARDAR TABLA DE MÉTRICAS
# ------------------------------------------------------------

write.csv(
  metricas_modelo,
  file.path(
    carpeta_salida,
    "metricas_modelo_Leroux.csv"
  ),
  row.names = FALSE
)


# ------------------------------------------------------------
# 28.3 GUARDAR MORAN SCATTERPLOT
# ------------------------------------------------------------

ggsave(
  filename = file.path(
    carpeta_salida,
    "Moran_scatterplot_Leroux.png"
  ),
  plot = grafico_moran,
  width = 11,
  height = 7,
  dpi = 300,
  bg = "white"
)


# ------------------------------------------------------------
# 28.4 GUARDAR MAPA DE RESIDUOS
# ------------------------------------------------------------

ggsave(
  filename = file.path(
    carpeta_salida,
    "Mapa_residuos_Leroux.png"
  ),
  plot = mapa_residuos,
  width = 9,
  height = 9,
  dpi = 300,
  bg = "white"
)


# ------------------------------------------------------------
# 28.5 GUARDAR CURVA ROC
# ------------------------------------------------------------

if (!is.null(roc_obj)) {
  
  ggsave(
    filename = file.path(
      carpeta_salida,
      "Curva_ROC_Leroux.png"
    ),
    plot = grafico_roc,
    width = 7,
    height = 7,
    dpi = 300,
    bg = "white"
  )
}


cat(
  "\nResultados guardados en:\n",
  carpeta_salida,
  "\n"
)

tabla_hiper <- m1_leroux$summary.hyperpar

tabla_hiper[
  grep(
    "lambda",
    rownames(tabla_hiper),
    ignore.case = TRUE
  ),
]