import pandas as pd
import matplotlib.pyplot as plt
from math import comb


# ==============================
# Carga del dataset
# ==============================

archivo = "dataset/datos_red_ferroviaria.csv"

df = pd.read_csv(archivo, sep=";")


# ==============================
# Filtro de datos
# ==============================

# Se filtra la Línea Roca (robusto a mayúsculas/espacios).
df_roca = df[df["Línea"].astype(str).str.upper().str.strip() == "ROCA"].copy()

# Se filtra el servicio de Bosques usando contains para capturar variantes
# (acentos, mayúsculas/minúsculas, variantes pequeñas en el texto).
mask_servicio = (
    df_roca["Servicio"].astype(str).str.contains("Plaza Constituci", case=False, na=False)
    & df_roca["Servicio"].astype(str).str.contains("Bosques", case=False, na=False)
)
df_bosques = df_roca[mask_servicio].copy()

if df_bosques.empty:
    print("Advertencia: no se encontraron filas para el servicio Bosques con el filtro aplicado.")
    # Mostrar algunas variantes disponibles para ayudar al diagnóstico
    variantes = df_roca["Servicio"].dropna().unique()[:20]
    print("Algunas variantes de 'Servicio' en la línea ROCA (muestras):")
    for v in variantes:
        print(" -", v)
    raise SystemExit(1)

# Nombre(s) del servicio encontrado para mostrar en resultados
servicios_unicos = df_bosques["Servicio"].dropna().unique()
if len(servicios_unicos) > 1:
    servicio_bosques = "; ".join([str(s) for s in servicios_unicos])
else:
    servicio_bosques = str(servicios_unicos[0])


# ==============================
# Limpieza y conversión de datos
# ==============================

columnas_numericas = [
    "Trenes Corridos (TC)",
    "Trenes Atrasados"
]

# Limpiar valores no numéricos (comas, espacios, símbolos) antes de convertir
for columna in columnas_numericas:
    df_bosques[columna] = pd.to_numeric(
        df_bosques[columna]
            .astype(str)
            .str.replace(r'[^0-9]', '', regex=True),
        errors="coerce"
    )

# Se eliminan filas donde falten datos necesarios.
df_bosques = df_bosques.dropna(subset=columnas_numericas)

# Se eliminan meses donde Trenes Corridos sea 0,
# porque no se puede dividir por cero.
df_bosques = df_bosques[df_bosques["Trenes Corridos (TC)"] > 0]


# ==============================
# Definición de mes crítico
# ==============================

# Tasa de atraso mensual.
df_bosques["Tasa Atraso"] = (
    df_bosques["Trenes Atrasados"] /
    df_bosques["Trenes Corridos (TC)"]
)

# Umbral elegido: 10%.
umbral_critico = 0.10

# Un mes es crítico si más del 10% de los trenes corridos llegaron atrasados.
df_bosques["Mes Critico"] = df_bosques["Tasa Atraso"] > umbral_critico


# ==============================
# Parámetros de la distribución hipergeométrica
# ==============================

# N: cantidad total de meses observados.
N = len(df_bosques)

# k: cantidad de meses críticos.
k = df_bosques["Mes Critico"].sum()

# n: cantidad de meses que se seleccionan al azar sin reposición.
n = 6

# x: cantidad de meses críticos que se desea obtener dentro de la muestra.
x = 2


# ==============================
# Función hipergeométrica
# ==============================

def probabilidad_hipergeometrica(N, k, n, x):
    """
    Calcula la probabilidad hipergeométrica:

        P(X = x) = [C(k, x) * C(N-k, n-x)] / C(N, n)

    Donde:
    N = tamaño de la población
    k = cantidad de éxitos en la población
    n = tamaño de la muestra
    x = cantidad de éxitos deseados en la muestra
    """

    # Validación de valores posibles.
    if x < 0 or x > k or n - x > N - k or n > N:
        return 0

    return (comb(k, x) * comb(N - k, n - x)) / comb(N, n)


probabilidad = probabilidad_hipergeometrica(N, k, n, x)


# ==============================
# Resultados
# ==============================

print("ANÁLISIS HIPERGEOMÉTRICO - MESES CRÍTICOS POR ATRASO")
print("=" * 60)
print(f"Línea analizada: ROCA")
print(f"Servicio analizado: {servicio_bosques}")
print(f"Umbral de criticidad: {umbral_critico * 100:.0f}%")
print()

print("Definición:")
print(
    "Un mes se considera crítico si la tasa de atraso "
    "supera el 10% de los trenes corridos."
)
print()

print("Parámetros de la distribución hipergeométrica:")
print(f"N = Total de meses observados: {N}")
print(f"k = Meses críticos observados: {k}")
print(f"N-k = Meses no críticos observados: {N - k}")
print(f"n = Meses seleccionados sin reposición: {n}")
print(f"x = Meses críticos deseados en la muestra: {x}")
print()

print("Resultado:")
print(f"P(X = {x}) = {probabilidad:.6f}")
print(f"P(X = {x}) = {probabilidad * 100:.2f}%")
print()

print("Interpretación:")
print(
    f"Si se seleccionan {n} meses al azar del historial del servicio Bosques, "
    f"sin reposición, la probabilidad de que exactamente {x} sean meses críticos "
    f"es de {probabilidad * 100:.2f}%."
)


# ==============================
# Tabla de distribución completa
# ==============================

valores_x = range(0, n + 1)

probabilidades = [
    probabilidad_hipergeometrica(N, k, n, valor)
    for valor in valores_x
]

tabla = pd.DataFrame({
    "x": valores_x,
    "P(X=x)": probabilidades,
    "P(X=x) %": [p * 100 for p in probabilidades]
})

print()
print("Distribución completa:")
print(tabla.to_string(index=False))


# ==============================
# Gráfico
# ==============================

plt.figure(figsize=(8, 5))
plt.bar(tabla["x"], tabla["P(X=x)"], color="steelblue", edgecolor="black")

plt.title("Distribución hipergeométrica\nMeses críticos por atraso - Servicio Bosques")
plt.xlabel("Cantidad de meses críticos en la muestra")
plt.ylabel("Probabilidad")
plt.xticks(tabla["x"])
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()