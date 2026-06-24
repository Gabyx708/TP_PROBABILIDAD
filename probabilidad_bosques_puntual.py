import pandas as pd
import matplotlib.pyplot as plt

dataset = "dataset/datos_red_ferroviaria.csv"

df = pd.read_csv(dataset,encoding="utf-8", sep=";")

roca = df[df["Línea"].str.upper() == "ROCA"]

## recuperar trenes via temperley
servicio = roca[
    roca["Servicio"] ==
    "ROC - Plaza Constitución-Temperley-Bosques (vía Temperley) (e)"
]

## convertir datos en numericos
servicio["Trenes Programados"] = pd.to_numeric(
    servicio["Trenes Programados"]
        .astype(str)
        .str.replace(",", ""),
    errors="coerce"
)

servicio["Trenes Puntuales"] = pd.to_numeric(
    servicio["Trenes Puntuales"]
        .astype(str)
        .str.replace(",", ""),
    errors="coerce"
)


### probabilidad de que un trenes programado llegue a tiempo
programados = servicio["Trenes Programados"].sum()
puntuales = servicio["Trenes Puntuales"].sum()

probabilidad_tren_puntual = puntuales / programados

print(
    f"Probabilidad de que un tren del servicio "
    f"'Plaza Constitución-Temperley-Bosques (vía Temperley)' "
    f"sea puntual: {probabilidad_tren_puntual:.4f}"
)


### probabilidad de que un tren programado en el anio 2025 llegue a tiempo segun el mes
servicio_2025 = servicio[servicio["Año"] == 2025]

resumen_mensual = servicio_2025.groupby("Mes")[
    ["Trenes Programados", "Trenes Puntuales"]
].sum()

resumen_mensual["Probabilidad Puntualidad"] = (
    resumen_mensual["Trenes Puntuales"]
    / resumen_mensual["Trenes Programados"]
)

for mes, fila in resumen_mensual.iterrows():
    print(
        f"{mes}: "
        f"{fila['Probabilidad Puntualidad']*100:.2f}%"
    )

# Orden cronológico de los meses
orden_meses = [
    "Enero", "Febrero", "Marzo", "Abril",
    "Mayo", "Junio", "Julio", "Agosto",
    "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

resumen_mensual = resumen_mensual.reindex(orden_meses)

# ==============================
# GRÁFICO DE BARRAS
# ==============================

plt.figure(figsize=(10, 5))

barras = plt.bar(
    resumen_mensual.index,
    resumen_mensual["Probabilidad Puntualidad"] * 100
)

for barra in barras:
    altura = barra.get_height()
    plt.text(
        barra.get_x() + barra.get_width()/2,
        altura + 0.5,
        f"{altura:.1f}%",
        ha="center"
    )

plt.title(
    "Probabilidad de puntualidad por mes - Año 2025\nServicio Plaza Constitución - Temperley - Bosques"
)

plt.xlabel("Mes")
plt.ylabel("Probabilidad (%)")

plt.xticks(rotation=45)
plt.ylim(0, 100)

plt.tight_layout()
plt.show()