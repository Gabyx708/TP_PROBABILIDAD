import pandas as pd
import matplotlib.pyplot as plt

dataset = "dataset/datos_red_ferroviaria.csv"

df = pd.read_csv(dataset, encoding="utf-8", sep=";")

roca = df[df["Línea"].str.upper() == "ROCA"].copy()

# Recuperar servicio vía Temperley
servicio = roca[
    roca["Servicio"] ==
    "ROC - Plaza Constitución-Temperley-Bosques (vía Temperley) (e)"
].copy()

# Convertir datos a numéricos
servicio["Trenes Programados"] = pd.to_numeric(
    servicio["Trenes Programados"]
        .astype(str)
        .str.replace(",", "", regex=False),
    errors="coerce"
)

servicio["Trenes Puntuales"] = pd.to_numeric(
    servicio["Trenes Puntuales"]
        .astype(str)
        .str.replace(",", "", regex=False),
    errors="coerce"
)

# Eliminar registros inválidos
servicio = servicio.dropna(
    subset=["Trenes Programados", "Trenes Puntuales"]
)

# ==================================================
# PROBABILIDAD DE PUNTUALIDAD POR AÑO
# ==================================================

resumen_anual = servicio.groupby("Año")[
    ["Trenes Programados", "Trenes Puntuales"]
].sum()

resumen_anual["Probabilidad Puntualidad"] = (
    resumen_anual["Trenes Puntuales"]
    / resumen_anual["Trenes Programados"]
)

# ==================================================
# MEJOR Y PEOR AÑO
# ==================================================

mejor_anio = resumen_anual["Probabilidad Puntualidad"].idxmax()
peor_anio = resumen_anual["Probabilidad Puntualidad"].idxmin()

prob_mejor = resumen_anual.loc[
    mejor_anio,
    "Probabilidad Puntualidad"
]

prob_peor = resumen_anual.loc[
    peor_anio,
    "Probabilidad Puntualidad"
]

print(
    f"Mejor año: {mejor_anio} "
    f"({prob_mejor * 100:.2f}% de puntualidad)"
)

print(
    f"Peor año: {peor_anio} "
    f"({prob_peor * 100:.2f}% de puntualidad)"
)

# ==================================================
# DETALLE POR AÑO
# ==================================================

for anio, fila in resumen_anual.iterrows():
    print(
        f"{anio}: "
        f"{fila['Probabilidad Puntualidad'] * 100:.2f}%"
    )

# ==================================================
# GRÁFICO
# ==================================================

plt.figure(figsize=(16, 6))

# Línea principal
plt.plot(
    resumen_anual.index,
    resumen_anual["Probabilidad Puntualidad"] * 100,
    marker="o",
    linewidth=2,
    label="Probabilidad de puntualidad"
)

# Promedio histórico
promedio = resumen_anual["Probabilidad Puntualidad"].mean() * 100

plt.axhline(
    promedio,
    linestyle="--",
    linewidth=1.5,
    label=f"Promedio histórico ({promedio:.2f}%)"
)

# Mostrar todos los años
plt.xticks(resumen_anual.index, rotation=45)

# Mejor año
plt.scatter(
    mejor_anio,
    prob_mejor * 100,
    s=250,
    marker="^",
    zorder=5,
    label=f"Mejor año ({mejor_anio})"
)

# Peor año
plt.scatter(
    peor_anio,
    prob_peor * 100,
    s=250,
    marker="v",
    zorder=5,
    label=f"Peor año ({peor_anio})"
)

# Etiqueta mejor año
plt.annotate(
    f"Mejor\n{mejor_anio}\n{prob_mejor*100:.2f}%",
    xy=(mejor_anio, prob_mejor * 100),
    xytext=(10, 15),
    textcoords="offset points"
)

# Etiqueta peor año
plt.annotate(
    f"Peor\n{peor_anio}\n{prob_peor*100:.2f}%",
    xy=(peor_anio, prob_peor * 100),
    xytext=(10, -35),
    textcoords="offset points"
)

plt.title(
    "Probabilidad de puntualidad por año\n"
    "ROC - Plaza Constitución - Temperley - Bosques (vía Temperley)"
)

plt.xlabel("Año")
plt.ylabel("Probabilidad de puntualidad (%)")

plt.grid(True)

plt.legend()

plt.tight_layout()
plt.show()