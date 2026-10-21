library(neuralnet)
library(ggplot2)

df_notas <- read.csv("Student_performance_data.csv")

df_notas$StudentID <- NULL
df_notas$GPA <- NULL

maxs <- apply(df_notas, 2, max)
mins <- apply(df_notas, 2, min)
data_scaled <- as.data.frame(scale(df_notas, center = mins, scale = maxs - mins))

data_scaled$Class0 <- ifelse(df_notas$GradeClass == 0, 1, 0)
data_scaled$Class1 <- ifelse(df_notas$GradeClass == 1, 1, 0)
data_scaled$Class2 <- ifelse(df_notas$GradeClass == 2, 1, 0)
data_scaled$Class3 <- ifelse(df_notas$GradeClass == 3, 1, 0)
data_scaled$Class4 <- ifelse(df_notas$GradeClass == 4, 1, 0)

data_scaled$GradeClass <- NULL

set.seed(123)
index <- sample(1:nrow(data_scaled), round(0.70 * nrow(data_scaled)))
train_data <- data_scaled[index, ]
test_data <- data_scaled[-index, ]

nombres <- names(train_data)
entradas <- paste(nombres[!nombres %in% c("Class0","Class1","Class2","Class3","Class4")], collapse = " + ")
formula_red <- as.formula(paste("Class0 + Class1 + Class2 + Class3 + Class4 ~", entradas))

print("Entrenando...")

nn_model <- neuralnet(formula_red,
                      data = train_data,
                      hidden = c(5),
                      linear.output = FALSE,
                      err.fct = "ce",
                      act.fct = "logistic",
                      stepmax = 1e6)

plot(nn_model, rep = "best")

pr.nn <- compute(nn_model, test_data[, 1:12])

predicciones <- max.col(pr.nn$net.result) - 1
clases_reales <- max.col(test_data[, 13:17]) - 1

tabla_confusion <- table(Real = clases_reales, Predicho = predicciones)
print(tabla_confusion)

accuracy <- sum(diag(tabla_confusion)) / sum(tabla_confusion)
print(paste("Precision:", round(accuracy * 100, 2), "%"))