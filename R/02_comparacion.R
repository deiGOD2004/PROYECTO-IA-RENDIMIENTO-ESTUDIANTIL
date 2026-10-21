# Comparación rápida con nnet: regresión logística multinomial (sin capa oculta)
# frente a redes de una capa oculta con distinto número de neuronas.
# Usa la misma división 70/30 y el mismo escalado que PROYECTO.R.
# Ejecutar desde la raíz del repositorio: Rscript R/02_comparacion.R

library(nnet)
library(ggplot2)
source(file.path("R", "utils.R"))

dir.create("resultados", showWarnings = FALSE)
df <- cargar_datos()
entradas <- setdiff(names(df), "GradeClass")
partes <- dividir(df)
escalado <- escalar_minmax(partes$train, partes$test, entradas)
y_train <- factor(partes$train$GradeClass, levels = CLASES)
y_test <- partes$test$GradeClass

evaluar <- function(modelo) {
  pred <- as.integer(as.character(predict(modelo, escalado$test, type = "class")))
  mean(pred == y_test)
}

resultados <- data.frame()

set.seed(123)
logit <- multinom(y_train ~ ., data = escalado$train, trace = FALSE, maxit = 1000)
resultados <- rbind(resultados, data.frame(Modelo = "Logística multinomial", Ocultas = 0,
                                           Accuracy = evaluar(logit)))

for (h in c(1, 2, 3, 5, 8, 10, 15)) {
  # Mejor de 5 inicializaciones aleatorias según el error de entrenamiento.
  set.seed(123)
  candidatos <- lapply(1:5, function(i)
    nnet(escalado$train, class.ind(y_train), size = h, softmax = TRUE,
         decay = 0.01, maxit = 2000, trace = FALSE))
  mejor <- candidatos[[which.min(sapply(candidatos, function(m) m$value))]]
  resultados <- rbind(resultados, data.frame(Modelo = "Red neuronal (nnet)", Ocultas = h,
                                             Accuracy = evaluar(mejor)))
}

resultados$Accuracy <- round(resultados$Accuracy, 4)
print(resultados, row.names = FALSE)
write.csv(resultados, file.path("resultados", "comparacion_modelos.csv"), row.names = FALSE)

base <- max(prop.table(table(y_test)))
g <- ggplot(subset(resultados, Ocultas > 0), aes(Ocultas, 100 * Accuracy)) +
  geom_line(color = "#2b5797") + geom_point(size = 3, color = "#2b5797") +
  geom_hline(yintercept = 100 * resultados$Accuracy[1], linetype = "dashed", color = "#c0504d") +
  geom_hline(yintercept = 100 * base, linetype = "dotted", color = "grey40") +
  annotate("text", x = 12, y = 100 * resultados$Accuracy[1] + 1, label = "Logística multinomial",
           color = "#c0504d") +
  annotate("text", x = 12, y = 100 * base + 1, label = "Clase mayoritaria", color = "grey40") +
  labs(title = "Precisión en prueba según neuronas ocultas", x = "Neuronas en la capa oculta",
       y = "Accuracy (%)") +
  theme_minimal(base_size = 13)
ggsave(file.path("resultados", "comparacion_modelos.png"), g, width = 7, height = 4.5, dpi = 120)
