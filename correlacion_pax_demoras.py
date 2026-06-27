import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================

dataset = "dataset/datos_red_ferroviaria.csv"
df = pd.read_csv(dataset, encoding="utf-8", sep=";")

roca = df[df["Línea"].str.upper() == "ROCA"].copy()

servicio = roca[
    roca["Servicio"] ==
    "ROC - Plaza Constitución-Temperley-Bosques (vía Temperley) (e)"
].copy()

# Convertir a numérico
for col in ["Trenes Programados", "Trenes Puntuales", "Trenes Atrasados", "Pasajeros Pagos (PAX)", "PAX x Tren"]:
    servicio[col] = pd.to_numeric(
        servicio[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

servicio = servicio.dropna(
    subset=["Trenes Programados", "Trenes Atrasados", "Pasajeros Pagos (PAX)"]
)

# Variable adicional: tasa de atraso (proporción de trenes atrasados)
servicio["Tasa de Atraso"] = servicio["Trenes Atrasados"] / servicio["Trenes Programados"]
servicio["PAX (millones)"] = servicio["Pasajeros Pagos (PAX)"] / 1_000_000

# ============================================================
# CORRELACIÓN ESTADÍSTICA
# ============================================================

r_pax_atrasados, p_pax_atrasados = stats.pearsonr(
    servicio["Pasajeros Pagos (PAX)"],
    servicio["Trenes Atrasados"]
)

r_pax_tasa, p_pax_tasa = stats.pearsonr(
    servicio["PAX (millones)"],
    servicio["Tasa de Atraso"]
)

print("=" * 60)
print("CORRELACIÓN: PASAJEROS vs DEMORAS")
print("Servicio: Constitución - Bosques (vía Temperley)")
print("=" * 60)
print(f"\n• Correlación PAX vs Trenes Atrasados (Pearson):")
print(f"  r = {r_pax_atrasados:.4f}  |  p-valor = {p_pax_atrasados:.4f}")
print(f"\n• Correlación PAX vs Tasa de Atraso (Pearson):")
print(f"  r = {r_pax_tasa:.4f}  |  p-valor = {p_pax_tasa:.4f}")

if abs(r_pax_tasa) < 0.3:
    interpretacion = "correlación débil (prácticamente independientes)"
elif abs(r_pax_tasa) < 0.6:
    interpretacion = "correlación moderada"
else:
    interpretacion = "correlación fuerte"

print(f"\n→ Interpretación: {interpretacion}")

# ============================================================
# GRÁFICOS
# ============================================================

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

colores_meses = plt.cm.tab10(np.linspace(0, 1, len(servicio)))

# ── Gráfico 1: Dispersión PAX vs Trenes Atrasados ──
ax1 = fig.add_subplot(gs[0, 0])

ax1.scatter(
    servicio["PAX (millones)"],
    servicio["Trenes Atrasados"],
    alpha=0.65, s=60, color="steelblue", edgecolors="white", linewidths=0.5
)

# Línea de tendencia
m1, b1, _, _, _ = stats.linregress(
    servicio["PAX (millones)"], servicio["Trenes Atrasados"]
)
x_line = np.linspace(servicio["PAX (millones)"].min(), servicio["PAX (millones)"].max(), 100)
ax1.plot(x_line, m1 * x_line + b1, color="tomato", linewidth=2, label=f"Tendencia (r={r_pax_atrasados:.3f})")

ax1.set_title("Pasajeros vs Trenes Atrasados", fontsize=11, fontweight="bold")
ax1.set_xlabel("Pasajeros (millones)")
ax1.set_ylabel("Trenes Atrasados")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Gráfico 2: Dispersión PAX vs Tasa de Atraso ──
ax2 = fig.add_subplot(gs[0, 1])

scatter = ax2.scatter(
    servicio["PAX (millones)"],
    servicio["Tasa de Atraso"] * 100,
    alpha=0.65, s=60, color="darkorange", edgecolors="white", linewidths=0.5
)

m2, b2, _, _, _ = stats.linregress(
    servicio["PAX (millones)"], servicio["Tasa de Atraso"] * 100
)
ax2.plot(x_line, m2 * x_line + b2, color="darkred", linewidth=2, label=f"Tendencia (r={r_pax_tasa:.3f})")

ax2.set_title("Pasajeros vs Tasa de Atraso (%)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Pasajeros (millones)")
ax2.set_ylabel("Tasa de Atraso (%)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Gráfico 3: Evolución temporal de PAX y Tasa de Atraso ──
ax3 = fig.add_subplot(gs[1, :])

# Ordenar por año y mes
orden_meses = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}
servicio["mes_num"] = servicio["Mes"].map(orden_meses)
servicio_ord = servicio.sort_values(["Año", "mes_num"]).reset_index(drop=True)
servicio_ord["periodo"] = (
    servicio_ord["Mes"].str[:3] + " " + servicio_ord["Año"].astype(str)
)

ax3_twin = ax3.twinx()

line1, = ax3.plot(
    range(len(servicio_ord)),
    servicio_ord["PAX (millones)"],
    color="steelblue", linewidth=2, marker="o", markersize=4, label="Pasajeros (millones)"
)

line2, = ax3_twin.plot(
    range(len(servicio_ord)),
    servicio_ord["Tasa de Atraso"] * 100,
    color="tomato", linewidth=2, linestyle="--", marker="s", markersize=4, label="Tasa de Atraso (%)"
)

# Etiquetas en eje x: mostrar solo cada 6 meses
tick_indices = list(range(0, len(servicio_ord), 6))
ax3.set_xticks(tick_indices)
ax3.set_xticklabels(
    [servicio_ord["periodo"].iloc[i] for i in tick_indices],
    rotation=45, ha="right", fontsize=8
)

ax3.set_xlabel("Período")
ax3.set_ylabel("Pasajeros (millones)", color="steelblue")
ax3_twin.set_ylabel("Tasa de Atraso (%)", color="tomato")
ax3.set_title(
    "Evolución temporal: Pasajeros vs Tasa de Atraso\n"
    "Servicio Constitución - Bosques (vía Temperley)",
    fontsize=11, fontweight="bold"
)

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc="upper left", fontsize=9)
ax3.grid(True, alpha=0.3)

# Título general
fig.suptitle(
    f"Análisis de Correlación: Pasajeros Pagos vs Demoras\n"
    f"(r = {r_pax_tasa:.4f} — {interpretacion})",
    fontsize=13, fontweight="bold", y=1.01
)

plt.savefig("correlacion_pax_demoras.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Gráfico guardado como 'correlacion_pax_demoras.png'")
