# Proyecto final — Técnicas de Inteligencia Artificial
# Predicción del rendimiento académico (GradeClass) con una red neuronal (neuralnet).
#
# Ejecutar desde la raíz del repositorio:
#   Rscript PROYECTO.R
# El entrenamiento es lento (varios minutos). El modelo se guarda en
# resultados/modelo_nn.rds y se reutiliza en ejecuciones posteriores;
# usa REENTRENAR <- TRUE para forzar un nuevo entrenamiento.

library(neuralnet)
library(ggplot2)
source(file.path("R", "utils.R"))

REENTRENAR <- FALSE
dir.create("resultados", showWarnings = FALSE)

# 1. Datos ------------------------------------------------------------------
df_notas <- cargar_datos()
entradas <- setdiff(names(df_notas), "GradeClass")
salidas <- paste0("Class", CLASES)

partes <- dividir(df_notas)
escalado <- escalar_minmax(partes$train, partes$test, entradas)
train_data <- cbind(escalado$train, one_hot(partes$train$GradeClass))
test_data <- cbind(escalado$test, one_hot(partes$test$GradeClass))

cat("Entrenamiento:", nrow(train_data), "| Prueba:", nrow(test_data), "\n")

# 2. Red neuronal -----------------------------------------------------------
# 12 entradas -> 5 neuronas ocultas (logística) -> 5 salidas, entropía cruzada.
formula_red <- as.formula(paste(paste(salidas, collapse = " + "), "~",
                                paste(entradas, collapse = " + ")))
ruta_modelo <- file.path("resultados", "modelo_nn.rds")

if (!REENTRENAR && file.exists(ruta_modelo)) {
  cat("Cargando modelo guardado:", ruta_modelo, "\n")
  nn_model <- readRDS(ruta_modelo)
} else {
  cat("Entrenando...\n")
  set.seed(123)
  inicio <- Sys.time()
  nn_model <- neuralnet(formula_red,
                        data = train_data,
                        hidden = c(5),
                        linear.output = FALSE,
                        err.fct = "ce",
                        act.fct = "logistic",
                        stepmax = 1e6,
                        lifesign = "minimal")
  cat("Tiempo de entrenamiento:", format(round(Sys.time() - inicio, 1)), "\n")
  saveRDS(nn_model, ruta_modelo)
}

png(file.path("resultados", "red_neuronal.png"), width = 1400, height = 1000, res = 110)
plot(nn_model, rep = "best", show.weights = FALSE, information = TRUE, fill = "#cfe0ff")
dev.off()

# 3. Evaluación -------------------------------------------------------------
pr.nn <- compute(nn_model, test_data[, entradas])
predicciones <- max.col(pr.nn$net.result) - 1
clases_reales <- partes$test$GradeClass

m <- metricas_por_clase(clases_reales, predicciones)
base <- max(prop.table(table(clases_reales)))  # predecir siempre la clase más frecuente

print(m$tabla)
print(m$por_clase, row.names = FALSE)
cat(sprintf("Precisión (accuracy): %.2f %%\n", 100 * m$accuracy))
cat(sprintf("Línea base (clase mayoritaria): %.2f %%\n", 100 * base))

ggsave(file.path("resultados", "matriz_confusion_nn.png"),
       grafica_confusion(m$tabla, "Red neuronal (neuralnet, 5 ocultas)"),
       width = 6.5, height = 5, dpi = 120)

sink(file.path("resultados", "metricas_nn.txt"))
cat("Red neuronal neuralnet: 12-5-5, logística, entropía cruzada\n\n")
print(m$tabla)
cat("\n")
print(m$por_clase, row.names = FALSE)
cat(sprintf("\nPrecisión (accuracy): %.2f %%\n", 100 * m$accuracy))
cat(sprintf("Línea base (clase mayoritaria): %.2f %%\n", 100 * base))
cat(sprintf("Error de entrenamiento (CE): %.3f | Pasos: %d\n",
            nn_model$result.matrix["error", 1], as.integer(nn_model$result.matrix["steps", 1])))
sink()
