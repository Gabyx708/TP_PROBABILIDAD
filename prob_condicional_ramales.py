import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================

dataset = "dataset/datos_red_ferroviaria.csv"
df = pd.read_csv(dataset, encoding="utf-8", sep=";")

roca = df[df["Línea"].str.upper() == "ROCA"].copy()

# Servicios eléctricos principales de la Línea Roca
# Cada ramal tiene una estación terminal distinta → sirve como proxy de "zona"
SERVICIOS = {
    "ROC - Plaza Constitución-Temperley-Bosques (vía Temperley) (e)": "Bosques\n(vía Temperley)",
    "ROC - Plaza Constitución-Bosques (Vía Quilmes) (e)":            "Bosques\n(vía Quilmes)",
    "ROC - Plaza Constitución-La Plata (e)":                         "La Plata",
    "ROC - Plaza Constitución-Glew-Korn (e)":                        "Glew - A. Korn",
    "ROC - Plaza Constitución-Ezeiza (e)":                           "Ezeiza",
    "ROC - Plaza Constitución-Glew (e.)":                            "Glew",
}

servicios_roca = roca[roca["Servicio"].isin(SERVICIOS.keys())].copy()

for col in ["Trenes Programados", "Trenes Puntuales", "Trenes Atrasados"]:
    servicios_roca[col] = pd.to_numeric(
        servicios_roca[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

servicios_roca = servicios_roca.dropna(
    subset=["Trenes Programados", "Trenes Atrasados"]
)

# ============================================================
# CÁLCULOS DE PROBABILIDAD CONDICIONAL
# ============================================================

# Agrupamos por servicio (ramal)
resumen = servicios_roca.groupby("Servicio")[
    ["Trenes Programados", "Trenes Puntuales", "Trenes Atrasados"]
].sum().reset_index()

resumen["Ramal"] = resumen["Servicio"].map(SERVICIOS)

# Totales globales de la línea
total_programados = resumen["Trenes Programados"].sum()
total_atrasados   = resumen["Trenes Atrasados"].sum()

# P(Ramal_i): probabilidad de que un tren pertenezca al ramal i
resumen["P(Ramal)"] = resumen["Trenes Programados"] / total_programados

# P(Atrasado | Ramal_i): prob. de atraso dado que es del ramal i
resumen["P(Atrasado|Ramal)"] = resumen["Trenes Atrasados"] / resumen["Trenes Programados"]

# P(Puntual | Ramal_i)
resumen["P(Puntual|Ramal)"] = resumen["Trenes Puntuales"] / resumen["Trenes Programados"]

# P(Ramal_i ∩ Atrasado) = P(Ramal_i) * P(Atrasado | Ramal_i)  → probabilidad conjunta
resumen["P(Ramal ∩ Atrasado)"] = resumen["P(Ramal)"] * resumen["P(Atrasado|Ramal)"]

# Verificación por Teorema de Probabilidad Total
# P(Atrasado) = Σ P(Ramal_i) * P(Atrasado | Ramal_i)
P_atrasado_total = resumen["P(Ramal ∩ Atrasado)"].sum()

# P(Ramal_i | Atrasado) = Bayes: P(Ramal_i) * P(Atrasado | Ramal_i) / P(Atrasado)
resumen["P(Ramal|Atrasado)"] = resumen["P(Ramal ∩ Atrasado)"] / P_atrasado_total

# ============================================================
# IMPRESIÓN DE RESULTADOS
# ============================================================

print("=" * 70)
print("PROBABILIDAD CONDICIONAL DE DEMORA POR RAMAL — LÍNEA ROCA")
print("=" * 70)
print(f"\nTotal de trenes programados (todos los ramales): {total_programados:,.0f}")
print(f"Total de trenes atrasados:                      {total_atrasados:,.0f}")
print(f"\nP(Atrasado) por Teorema de Probabilidad Total:  {P_atrasado_total:.4f} = {P_atrasado_total*100:.2f}%")
print()

print(f"{'Ramal':<35} {'P(Ramal)':>10} {'P(Atr|Ramal)':>14} {'P(Ramal|Atr)':>14}")
print("-" * 75)

for _, row in resumen.sort_values("P(Atrasado|Ramal)", ascending=False).iterrows():
    ramal_label = row["Ramal"].replace("\n", " ")
    print(
        f"{ramal_label:<35} "
        f"{row['P(Ramal)']:>10.4f} "
        f"{row['P(Atrasado|Ramal)']:>13.4f}  "
        f"{row['P(Ramal|Atrasado)']:>13.4f}"
    )

# Ramal con mayor probabilidad de atraso
peor_ramal_idx = resumen["P(Atrasado|Ramal)"].idxmax()
peor_ramal = resumen.loc[peor_ramal_idx]

print()
print("=" * 70)
print(f"→ RAMAL CON MAYOR PROBABILIDAD DE ATRASO:")
print(f"  {peor_ramal['Ramal'].replace(chr(10), ' ')}")
print(f"  P(Atrasado | Ramal) = {peor_ramal['P(Atrasado|Ramal)']:.4f} = {peor_ramal['P(Atrasado|Ramal)']*100:.2f}%")

# Verificación de independencia
# Si el ramal y el atraso fueran independientes, se cumpliría:
# P(Atrasado | Ramal_i) = P(Atrasado)  para todo i
print()
print("=" * 70)
print("VERIFICACIÓN DE INDEPENDENCIA")
print(f"P(Atrasado) global = {P_atrasado_total:.4f}")
print("Si P(Atrasado|Ramal_i) ≈ P(Atrasado) para todo ramal, son independientes.")
print()
resumen_ord = resumen.sort_values("P(Atrasado|Ramal)", ascending=False)
for _, row in resumen_ord.iterrows():
    diff = row["P(Atrasado|Ramal)"] - P_atrasado_total
    signo = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "≈")
    print(f"  {row['Ramal'].replace(chr(10), ' '):<35}: {row['P(Atrasado|Ramal)']:.4f}  {signo}")

print()
print("→ Los ramales presentan tasas de atraso distintas a la global,")
print("  lo que indica DEPENDENCIA entre el ramal y la probabilidad de atraso.")

# ============================================================
# GRÁFICOS
# ============================================================

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

resumen_plot = resumen.sort_values("P(Atrasado|Ramal)", ascending=True)
labels_plot  = [r.replace("\n", "\n") for r in resumen_plot["Ramal"]]
colores      = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(resumen_plot)))

# ── Gráfico 1: P(Atrasado | Ramal) ──
ax1 = fig.add_subplot(gs[0, 0])
barras = ax1.barh(labels_plot, resumen_plot["P(Atrasado|Ramal)"] * 100, color=colores)

# Línea global
ax1.axvline(P_atrasado_total * 100, color="black", linewidth=1.5,
            linestyle="--", label=f"Global: {P_atrasado_total*100:.1f}%")

for bar, val in zip(barras, resumen_plot["P(Atrasado|Ramal)"]):
    ax1.text(val * 100 + 0.1, bar.get_y() + bar.get_height() / 2,
             f"{val*100:.1f}%", va="center", fontsize=8.5)

ax1.set_title("P(Atrasado | Ramal)\nProbabilidad condicional de atraso", fontsize=10, fontweight="bold")
ax1.set_xlabel("Probabilidad (%)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="x")

# ── Gráfico 2: P(Ramal | Atrasado) — Bayes ──
ax2 = fig.add_subplot(gs[0, 1])
resumen_bayes = resumen.sort_values("P(Ramal|Atrasado)", ascending=True)
colores2 = plt.cm.Blues(np.linspace(0.3, 0.9, len(resumen_bayes)))
barras2  = ax2.barh(
    [r.replace("\n", "\n") for r in resumen_bayes["Ramal"]],
    resumen_bayes["P(Ramal|Atrasado)"] * 100,
    color=colores2
)
for bar, val in zip(barras2, resumen_bayes["P(Ramal|Atrasado)"]):
    ax2.text(val * 100 + 0.1, bar.get_y() + bar.get_height() / 2,
             f"{val*100:.1f}%", va="center", fontsize=8.5)

ax2.set_title("P(Ramal | Atrasado) — Teorema de Bayes\n¿Qué ramal es más probable si hubo atraso?",
              fontsize=10, fontweight="bold")
ax2.set_xlabel("Probabilidad (%)")
ax2.grid(True, alpha=0.3, axis="x")

# ── Gráfico 3: P(Ramal) — distribución del parque de trenes ──
ax3 = fig.add_subplot(gs[1, 0])
resumen_pram = resumen.sort_values("P(Ramal)", ascending=False)
wedges, texts, autotexts = ax3.pie(
    resumen_pram["P(Ramal)"],
    labels=[r.replace("\n", "\n") for r in resumen_pram["Ramal"]],
    autopct="%1.1f%%",
    colors=plt.cm.tab10(np.linspace(0, 1, len(resumen_pram))),
    startangle=140,
    textprops={"fontsize": 8}
)
ax3.set_title("P(Ramal)\nDistribución de trenes por ramal", fontsize=10, fontweight="bold")

# ── Gráfico 4: Comparación puntual vs atrasado por ramal ──
ax4 = fig.add_subplot(gs[1, 1])
resumen_stk = resumen.sort_values("P(Atrasado|Ramal)", ascending=False)
x = np.arange(len(resumen_stk))
w = 0.35

ax4.bar(x - w/2, resumen_stk["P(Puntual|Ramal)"] * 100, w,
        label="P(Puntual|Ramal)", color="steelblue", alpha=0.85)
ax4.bar(x + w/2, resumen_stk["P(Atrasado|Ramal)"] * 100, w,
        label="P(Atrasado|Ramal)", color="tomato", alpha=0.85)

ax4.set_xticks(x)
ax4.set_xticklabels(
    [r.replace("\n", "\n") for r in resumen_stk["Ramal"]],
    rotation=30, ha="right", fontsize=8
)
ax4.set_ylabel("Probabilidad (%)")
ax4.set_title("P(Puntual|Ramal) vs P(Atrasado|Ramal)\nComparación por ramal", fontsize=10, fontweight="bold")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    "Probabilidad Condicional de Demoras por Ramal — Línea Roca\n"
    "(Aplicación del Teorema de Probabilidad Total y Bayes)",
    fontsize=13, fontweight="bold"
)

plt.savefig("prob_condicional_ramales.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Gráfico guardado como 'prob_condicional_ramales.png'")
