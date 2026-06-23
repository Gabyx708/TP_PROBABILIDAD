import pandas as pd
import matplotlib.pyplot as plt

dataset = "dataset/datos_red_ferroviaria.csv"

df = pd.read_csv(dataset,encoding="utf-8", sep=";")


## TRENES PROGRAMADOS ===================================

roca = df[df["Línea"].str.upper() == "ROCA"]

roca["Trenes Programados"] = (
    roca["Trenes Programados"]
    .astype(str)
    .str.replace(",", "")
)

roca["Trenes Programados"] = pd.to_numeric(
    roca["Trenes Programados"],
    errors="coerce"
)

roca = roca.dropna(subset=["Trenes Programados"])

por_anio = roca.groupby("Año")["Trenes Programados"].sum().sort_index()

for anio, valor in por_anio.items():
    print(anio, valor)

### TRENES CANCELADOS =============================================

roca["Trenes Cancelados"] = (
    roca["Trenes Cancelados"]
    .astype(str)
    .str.replace(",", "")
)

roca["Trenes Cancelados"] = pd.to_numeric(
    roca["Trenes Cancelados"],
    errors="coerce"
)

# agrupar por año
cancelados_por_anio = roca.groupby("Año")["Trenes Cancelados"].sum().sort_index()

for anio, valor in cancelados_por_anio.items():
    print(anio, valor)

### GRAFICO DE BARRAS ====================================================

# unimos ambos resultados en una sola tabla
resumen = pd.DataFrame({
    "Programados": por_anio,
    "Cancelados": cancelados_por_anio
})

# por si hay años que faltan en alguna serie
resumen = resumen.fillna(0)

print(resumen)

resumen.plot(kind="bar", figsize=(10,5))

plt.title("Trenes Programados vs Cancelados - Línea ROCA")
plt.xlabel("Año")
plt.ylabel("Cantidad de trenes")
plt.legend(["Programados", "Cancelados"])
plt.xticks(rotation=45)

plt.show()