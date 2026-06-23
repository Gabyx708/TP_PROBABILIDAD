import pandas as pd
import matplotlib.pyplot as plt

dataset = "dataset/datos_red_ferroviaria.csv"

df = pd.read_csv(dataset,encoding="utf-8", sep=";")

roca = df[df["Línea"].str.upper() == "ROCA"]

# ==========================================
# TRENES PUNTUALES
# ==========================================

roca["Trenes Puntuales"] = (
    roca["Trenes Puntuales"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

roca["Trenes Puntuales"] = pd.to_numeric(
    roca["Trenes Puntuales"],
    errors="coerce"
)

puntuales_por_anio = (
    roca.groupby("Año")["Trenes Puntuales"]
    .sum()
    .sort_index()
)

for anio, valor in puntuales_por_anio.items():
    print(anio, valor)


# ==========================================
# TRENES ATRASADOS
# ==========================================

roca["Trenes Atrasados"] = (
    roca["Trenes Atrasados"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

roca["Trenes Atrasados"] = pd.to_numeric(
    roca["Trenes Atrasados"],
    errors="coerce"
)

atrasados_por_anio = (
    roca.groupby("Año")["Trenes Atrasados"]
    .sum()
    .sort_index()
)

for anio, valor in atrasados_por_anio.items():
    print(anio, valor)

resumen_puntualidad = pd.DataFrame({
    "Puntuales": puntuales_por_anio,
    "Atrasados": atrasados_por_anio
}).fillna(0)

print(resumen_puntualidad)

tasa_puntualidad = (
    puntuales_por_anio /
    (puntuales_por_anio + atrasados_por_anio)
) * 100

for anio, tasa in tasa_puntualidad.items():
    print(f"> {anio}: {tasa:.2f}%")

    
resumen_puntualidad.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Trenes Puntuales vs Atrasados - Línea ROCA")
plt.xlabel("Año")
plt.ylabel("Cantidad de trenes")
plt.xticks(rotation=45)
plt.legend(["Puntuales", "Atrasados"])

plt.tight_layout()
plt.show()

