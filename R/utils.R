# Funciones compartidas por los scripts del proyecto.

CLASES <- 0:4
NOMBRES_CLASE <- c("0" = "A", "1" = "B", "2" = "C", "3" = "D", "4" = "F")

cargar_datos <- function(ruta = file.path("data", "Student_performance_data.csv")) {
  df <- read.csv(ruta)
  # StudentID es solo un identificador.
  # GPA se elimina porque GradeClass se calcula directamente a partir de él:
  # dejarlo sería "darle la respuesta" a la red (fuga de información).
  df$StudentID <- NULL
  df$GPA <- NULL
  df
}

# División 70/30 con la misma semilla que el script original.
dividir <- function(df, prop = 0.70, semilla = 123) {
  set.seed(semilla)
  index <- sample(1:nrow(df), round(prop * nrow(df)))
  list(train = df[index, ], test = df[-index, ])
}

# Escalado min-max usando SOLO los mínimos/máximos del conjunto de entrenamiento,
# para que el conjunto de prueba no influya en el preprocesamiento.
escalar_minmax <- function(train, test, columnas) {
  mins <- sapply(train[columnas], min)
  maxs <- sapply(train[columnas], max)
  rango <- ifelse(maxs - mins == 0, 1, maxs - mins)
  escala <- function(d) as.data.frame(scale(d[columnas], center = mins, scale = rango))
  list(train = escala(train), test = escala(test))
}

# Codificación one-hot del objetivo: Class0 ... Class4.
one_hot <- function(y) {
  m <- sapply(CLASES, function(k) as.numeric(y == k))
  colnames(m) <- paste0("Class", CLASES)
  as.data.frame(m)
}

# Precisión, recall y F1 por clase a partir de la matriz de confusión.
metricas_por_clase <- function(real, predicho) {
  real <- factor(real, levels = CLASES)
  predicho <- factor(predicho, levels = CLASES)
  tabla <- table(Real = real, Predicho = predicho)
  vp <- diag(tabla)
  precision <- vp / pmax(colSums(tabla), 1)
  recall <- vp / pmax(rowSums(tabla), 1)
  f1 <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  list(
    tabla = tabla,
    accuracy = sum(vp) / sum(tabla),
    por_clase = data.frame(
      Clase = paste0(CLASES, " (", NOMBRES_CLASE, ")"),
      Soporte = as.vector(rowSums(tabla)),
      Precision = round(as.vector(precision), 3),
      Recall = round(as.vector(recall), 3),
      F1 = round(as.vector(f1), 3)
    )
  )
}

grafica_confusion <- function(tabla, titulo) {
  d <- as.data.frame(tabla)
  ggplot2::ggplot(d, ggplot2::aes(x = Predicho, y = Real, fill = Freq)) +
    ggplot2::geom_tile(color = "white") +
    ggplot2::geom_text(ggplot2::aes(label = Freq), size = 5) +
    ggplot2::scale_fill_gradient(low = "#f0f4ff", high = "#2b5797") +
    ggplot2::scale_y_discrete(limits = rev) +
    ggplot2::labs(title = titulo, x = "Clase predicha", y = "Clase real", fill = "Casos") +
    ggplot2::theme_minimal(base_size = 13)
}
