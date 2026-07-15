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

aoi <- st_read(ruta, quiet = TRUE)

print(aoi)
print(st_crs(aoi))
names(aoi)


# ============================================================
# 3. CORREGIR Y REVISAR LAS GEOMETRÍAS
# ============================================================

aoi <- st_make_valid(aoi)

# Eliminar geometrías vacías
aoi <- aoi[!st_is_empty(aoi), ]

# Revisar validez
print(table(st_is_valid(aoi)))


# ============================================================
# 4. REVISAR LAS VARIABLES DEL MODELO
# ============================================================

# Solo se incluyen las variables que realmente entrarán
# en la fórmula ICAR
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
      paste(variables_faltantes, collapse = ", ")
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
aoi <- aoi[seq_len(nrow(aoi)), ]

# Identificador consecutivo para INLA
aoi$id_area <- seq_len(nrow(aoi))

rownames(aoi) <- as.character(aoi$id_area)

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

# Número de vecinos
aoi$n_vecinos <- card(aoi.nb)

# Municipios sin vecinos
sin_vecinos <- which(aoi$n_vecinos == 0)

if (length(sin_vecinos) > 0) {
  
  warning(
    paste(
      "Hay municipios sin vecinos:",
      paste(sin_vecinos, collapse = ", ")
    )
  )
  
} else {
  
  message("Todos los municipios tienen al menos un vecino.")
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

rownames(aoi.mat) <- as.character(aoi$id_area)
colnames(aoi.mat) <- as.character(aoi$id_area)

# Matriz convencional, por si se requiere inspeccionarla
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
# 11. FÓRMULA DEL MODELO ICAR
# ============================================================

formula_icar <- Tot_event ~
  1 +
  Pend_mean +
  TRI_mean +
  f(
    id_area,
    model = "besag",
    graph = aoi.mat,
    constr = TRUE,
    scale.model = TRUE,
    adjust.for.con.comp = TRUE
  )


# ============================================================
# 12. MODELO BINOMIAL NEGATIVO ICAR
# ============================================================

m1_inla <- inla(
  formula = formula_icar,
  
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

summary(m1_inla)


# ============================================================
# 13. VALORES AJUSTADOS
# ============================================================

valores_ajustados <- m1_inla$summary.fitted.values$mean[
  seq_len(nrow(aoi))
]

aoi$conteo_ajustado <- valores_ajustados


# ============================================================
# 14. EXTRAER EL PARÁMETRO SIZE
# ============================================================

nombres_hiperparametros <- rownames(
  m1_inla$summary.hyperpar
)

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

size_nb <- m1_inla$summary.hyperpar[
  fila_size[1],
  "mean"
]

cat(
  "\nParámetro size de la binomial negativa:",
  size_nb,
  "\n"
)


# ============================================================
# 15. RESIDUOS DE PEARSON BINOMIAL NEGATIVA
# ============================================================

# Var(Y) = mu + mu² / size
varianza_nb <- valores_ajustados +
  valores_ajustados^2 / size_nb

res_pearson_m1 <- (
  aoi$Tot_event - valores_ajustados
) / sqrt(
  pmax(
    varianza_nb,
    .Machine$double.eps
  )
)

aoi$res_pearson_m1 <- res_pearson_m1


# ============================================================
# 16. R2 DE EFRON
# ============================================================

sse <- sum(
  (aoi$Tot_event - valores_ajustados)^2,
  na.rm = TRUE
)

sst <- sum(
  (aoi$Tot_event - mean(aoi$Tot_event))^2,
  na.rm = TRUE
)

R2_efron <- if (sst > 0) {
  
  1 - sse / sst
  
} else {
  
  NA_real_
}

# R2 basado en la correlación, presentado como referencia
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
# 17. AIC CONDICIONAL APROXIMADO
# ============================================================

# Log-verosimilitud binomial negativa evaluada en las
# medias posteriores de mu y size
logLik_condicional <- sum(
  dnbinom(
    x = aoi$Tot_event,
    size = size_nb,
    mu = valores_ajustados,
    log = TRUE
  ),
  na.rm = TRUE
)

# Número efectivo de parámetros estimado por WAIC
p_efectivo <- as.numeric(
  m1_inla$waic$p.eff
)

if (
  length(p_efectivo) == 0 ||
  is.na(p_efectivo)
) {
  
  # Alternativa en caso de que p.eff no esté disponible
  p_efectivo <- nrow(m1_inla$summary.fixed) +
    nrow(m1_inla$summary.hyperpar)
}

AIC_aproximado <- -2 * logLik_condicional +
  2 * p_efectivo


# ============================================================
# 18. ROC-AUC PARA PRESENCIA O AUSENCIA
# ============================================================

# La ROC necesita una respuesta binaria:
# 0 = ningún evento
# 1 = uno o más eventos
aoi$presencia_evento <- as.integer(
  aoi$Tot_event > 0
)

# Para una binomial negativa:
# P(Y > 0) = 1 - P(Y = 0)
aoi$prob_presencia <- 1 - dnbinom(
  x = 0,
  size = size_nb,
  mu = valores_ajustados
)

roc_obj <- NULL
AUC_ROC <- NA_real_

if (
  length(unique(aoi$presencia_evento)) == 2
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
      "la variable presencia_evento solo tiene una clase."
    )
  )
}


# ============================================================
# 19. MORAN GLOBAL DE LOS RESIDUOS
# ============================================================

set.seed(123)

moran_m1 <- moran.mc(
  x = aoi$res_pearson_m1,
  listw = aoi.listw,
  nsim = 999,
  alternative = "greater",
  zero.policy = TRUE
)

print(moran_m1)

I_moran <- unname(
  moran_m1$statistic
)

p_moran <- moran_m1$p.value


# ============================================================
# 20. PREPARAR EL MORAN SCATTERPLOT
# ============================================================

# Estandarizar residuos de Pearson
residuos_estandarizados <- as.numeric(
  scale(aoi$res_pearson_m1)
)

# Promedio ponderado de los residuos de los vecinos
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
# 21. GRÁFICO DE DISPERSIÓN DE MORAN
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
  
  # La pendiente de esta recta corresponde al Moran I
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
      "Índice de Moran (I) = ",
      round(I_moran, 4),
      "   |   p = ",
      format.pval(
        p_moran,
        digits = 4,
        eps = 0.0001
      )
    ),
    x = paste0(
      "Residuos de Pearson del Municipio ",
      "(estandarizados)"
    ),
    y = paste0(
      "Promedio de los Residuos de los Vecinos ",
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
# 22. MAPA DE RESIDUOS DE PEARSON
# ============================================================

limite_residuos <- max(
  abs(aoi$res_pearson_m1),
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
      fill = res_pearson_m1
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
    title = "Residuos de Pearson del modelo ICAR",
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
# 23. GRÁFICO ROC
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
        "AUC = ",
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
# 22. MAPA DE RESIDUOS DE PEARSON
# ============================================================

limite_residuos <- max(
  abs(aoi$res_pearson_m1),
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
      fill = res_pearson_m1
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
    title = "Residuos de Pearson del modelo ICAR",
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
# 23. GRÁFICO ROC
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
        "AUC = ",
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
# 24. EXTRAER EL EFECTO ESPACIAL ICAR
# ============================================================

efecto_espacial <- m1_inla$summary.random$id_area

efecto_espacial <- efecto_espacial[
  order(efecto_espacial$ID),
]

aoi$efecto_espacial_media <-
  efecto_espacial$mean

aoi$efecto_espacial_sd <-
  efecto_espacial$sd

aoi$efecto_espacial_q025 <-
  efecto_espacial$`0.025quant`

aoi$efecto_espacial_q975 <-
  efecto_espacial$`0.975quant`


# ============================================================
# 25. CPO Y LPML
# ============================================================

aoi$CPO <- m1_inla$cpo$cpo
aoi$PIT <- m1_inla$cpo$pit

LPML <- sum(
  log(m1_inla$cpo$cpo),
  na.rm = TRUE
)


# ============================================================
# 26. TABLA DE MÉTRICAS
# ============================================================

metricas_modelo <- data.frame(
  Modelo = "Binomial negativa ICAR",
  R2_Efron = R2_efron,
  R2_correlacion = R2_correlacion,
  AIC_condicional_aprox = AIC_aproximado,
  DIC = m1_inla$dic$dic,
  WAIC = m1_inla$waic$waic,
  LPML = LPML,
  ROC_AUC_presencia = AUC_ROC,
  Moran_I_residuos = I_moran,
  Moran_p_valor = p_moran,
  Size_NB = size_nb
)

cat("\n")
cat("=============================================\n")
cat("MÉTRICAS DEL MODELO BINOMIAL NEGATIVO ICAR\n")
cat("=============================================\n")

print(
  metricas_modelo,
  digits = 5,
  row.names = FALSE
)

cat("\nR2 de Efron:", round(R2_efron, 4), "\n")
cat(
  "AIC condicional aproximado:",
  round(AIC_aproximado, 2),
  "\n"
)
cat("DIC:", round(m1_inla$dic$dic, 2), "\n")
cat("WAIC:", round(m1_inla$waic$waic, 2), "\n")
cat("ROC-AUC:", round(AUC_ROC, 4), "\n")
cat("Moran I:", round(I_moran, 4), "\n")
cat("P-valor Moran:", p_moran, "\n")


# ============================================================
# 27. GUARDAR RESULTADOS
# ============================================================

carpeta_salida <- paste0(
  "C:/Analisis_Geoespacial/2026_I_Analisis_Geoespacial/",
  "Data/Poligonos/Poligonos/Resultados_ICAR"
)

dir.create(
  carpeta_salida,
  recursive = TRUE,
  showWarnings = FALSE
)


# ------------------------------------------------------------
# Guardar GeoPackage
# ------------------------------------------------------------

ruta_gpkg <- file.path(
  carpeta_salida,
  "Resultados_ICAR.gpkg"
)

st_write(
  aoi,
  ruta_gpkg,
  layer = "modelo_icar",
  delete_layer = TRUE,
  quiet = TRUE
)


# ------------------------------------------------------------
# Guardar tabla de métricas
# ------------------------------------------------------------

write.csv(
  metricas_modelo,
  file.path(
    carpeta_salida,
    "metricas_modelo_ICAR.csv"
  ),
  row.names = FALSE
)


# ------------------------------------------------------------
# Guardar gráfico de Moran
# ------------------------------------------------------------

ggsave(
  filename = file.path(
    carpeta_salida,
    "Moran_scatterplot_residuos.png"
  ),
  plot = grafico_moran,
  width = 11,
  height = 7,
  dpi = 300,
  bg = "white"
)


# ------------------------------------------------------------
# Guardar mapa de residuos
# ------------------------------------------------------------

ggsave(
  filename = file.path(
    carpeta_salida,
    "Mapa_residuos_ICAR.png"
  ),
  plot = mapa_residuos,
  width = 9,
  height = 9,
  dpi = 300,
  bg = "white"
)


# ------------------------------------------------------------
# Guardar curva ROC si pudo calcularse
# ------------------------------------------------------------

if (!is.null(roc_obj)) {
  
  ggsave(
    filename = file.path(
      carpeta_salida,
      "Curva_ROC_presencia_eventos.png"
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