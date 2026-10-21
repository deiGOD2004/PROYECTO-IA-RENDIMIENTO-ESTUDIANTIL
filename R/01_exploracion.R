# Análisis exploratorio del conjunto de datos.
# Ejecutar desde la raíz del repositorio: Rscript R/01_exploracion.R

library(ggplot2)
source(file.path("R", "utils.R"))

dir.create("resultados", showWarnings = FALSE)
df <- cargar_datos()
df$Clase <- factor(NOMBRES_CLASE[as.character(df$GradeClass)], levels = NOMBRES_CLASE)

cat("Registros:", nrow(df), "| Atributos de entrada:", ncol(df) - 2,
    "| Valores faltantes:", sum(is.na(df)), "\n\n")
print(round(prop.table(table(df$Clase)), 3))

g1 <- ggplot(df, aes(Clase, fill = Clase)) +
  geom_bar(show.legend = FALSE) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.4) +
  labs(title = "Distribución de GradeClass", x = "Clase", y = "Estudiantes") +
  theme_minimal(base_size = 13)
ggsave(file.path("resultados", "distribucion_clases.png"), g1, width = 6, height = 4.5, dpi = 120)

g2 <- ggplot(df, aes(Clase, Absences, fill = Clase)) +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "Ausencias por clase", x = "Clase", y = "Ausencias") +
  theme_minimal(base_size = 13)
ggsave(file.path("resultados", "ausencias_por_clase.png"), g2, width = 6, height = 4.5, dpi = 120)

g3 <- ggplot(df, aes(Clase, StudyTimeWeekly, fill = Clase)) +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "Horas de estudio semanales por clase", x = "Clase", y = "Horas") +
  theme_minimal(base_size = 13)
ggsave(file.path("resultados", "estudio_por_clase.png"), g3, width = 6, height = 4.5, dpi = 120)

# Correlación de cada atributo con GradeClass
cors <- sort(sapply(df[setdiff(names(df), c("GradeClass", "Clase"))],
                    function(x) cor(x, df$GradeClass)))
d_cor <- data.frame(Atributo = factor(names(cors), levels = names(cors)), r = cors)
print(round(cors, 3))
g4 <- ggplot(d_cor, aes(r, Atributo, fill = r > 0)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Correlación con GradeClass", x = "r de Pearson", y = NULL) +
  theme_minimal(base_size = 13)
ggsave(file.path("resultados", "correlaciones.png"), g4, width = 6.5, height = 4.5, dpi = 120)
