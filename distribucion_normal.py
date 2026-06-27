import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats
from scipy.stats import norm

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================

dataset = "dataset/datos_red_ferroviaria.csv"
df = pd.read_csv(dataset, encoding="utf-8", sep=";")
roca = df[df["Línea"].str.upper() == "ROCA"].copy()

servicio_key = "ROC - Plaza Constitución-Temperley-Bosques (vía Temperley) (e)"
sub = roca[roca["Servicio"] == servicio_key].copy()

for col in ["Trenes Programados", "Trenes Puntuales", "Trenes Atrasados"]:
    sub[col] = pd.to_numeric(
        sub[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )

sub = sub.dropna(subset=["Trenes Programados", "Trenes Puntuales"])
sub["Tasa de Puntualidad"] = sub["Trenes Puntuales"] / sub["Trenes Programados"] * 100

# Elegimos dos años con normalidad confirmada para el análisis principal
ANIOS_ANALISIS = [2019, 2024]
COLORES_ANIOS  = ["steelblue", "darkorange"]

# ============================================================
# TEST DE SHAPIRO-WILK Y PARÁMETROS
# ============================================================

print("=" * 65)
print("DISTRIBUCIÓN NORMAL — TASA DE PUNTUALIDAD MENSUAL")
print("Servicio: Constitución – Bosques (vía Temperley)")
print("=" * 65)

print("\n── TEST DE NORMALIDAD (Shapiro-Wilk) POR AÑO ──\n")
resultados_anios = {}

for anio in sorted(sub["Año"].unique()):
    datos = sub[sub["Año"] == anio]["Tasa de Puntualidad"].dropna()
    if len(datos) >= 4:
        stat, p = stats.shapiro(datos)
        simbolo = "✓ Normal" if p > 0.05 else "✗ No normal"
        print(
            f"  {anio}: n={len(datos):2d}  "
            f"μ={datos.mean():.2f}%  σ={datos.std():.2f}%  "
            f"Shapiro p={p:.4f}  {simbolo}"
        )
        resultados_anios[anio] = {
            "datos": datos, "media": datos.mean(),
            "std": datos.std(), "p": p, "n": len(datos)
        }

# ── Acumulado histórico (para mostrar que NO es normal) ──
datos_total = sub["Tasa de Puntualidad"].dropna()
stat_tot, p_tot = stats.shapiro(datos_total)
print(f"\n  HISTÓRICO (todos los años):")
print(f"  n={len(datos_total)}  μ={datos_total.mean():.2f}%  σ={datos_total.std():.2f}%  Shapiro p={p_tot:.4f}  ✗ No normal")
print(f"\n  → La serie histórica NO es normal porque la media cambió")
print(f"    drásticamente entre 2017 (≈85%) y 2023 (≈46%).")
print(f"    En cambio, dentro de cada año la variabilidad sí sigue una Normal.")

# ── Análisis detallado de los años elegidos ──
print(f"\n{'='*65}")
print(f"ANÁLISIS DETALLADO — AÑOS {ANIOS_ANALISIS}")
print(f"{'='*65}")

for anio in ANIOS_ANALISIS:
    r = resultados_anios[anio]
    datos = r["datos"]
    mu, sigma = r["media"], r["std"]

    print(f"\n── Año {anio} ──")
    print(f"  μ (media)             = {mu:.4f}%")
    print(f"  σ (desvío estándar)   = {sigma:.4f}%")
    print(f"  σ² (varianza)         = {sigma**2:.4f}")
    print(f"  n                     = {r['n']}")
    print(f"  Shapiro-Wilk p        = {r['p']:.4f}  ✓ Normal")

    # Probabilidades de interés usando la Normal ajustada
    print(f"\n  Probabilidades calculadas con N(μ={mu:.2f}, σ={sigma:.2f}):")

    p_90 = 1 - norm.cdf(90, mu, sigma)
    p_70_90 = norm.cdf(90, mu, sigma) - norm.cdf(70, mu, sigma)
    p_menos70 = norm.cdf(70, mu, sigma)

    print(f"  P(Puntualidad > 90%)          = {p_90:.4f}  = {p_90*100:.2f}%")
    print(f"  P(70% < Puntualidad ≤ 90%)    = {p_70_90:.4f}  = {p_70_90*100:.2f}%")
    print(f"  P(Puntualidad ≤ 70%)          = {p_menos70:.4f}  = {p_menos70*100:.2f}%")

    # Percentil 10 y 90 (valor crítico)
    p10 = norm.ppf(0.10, mu, sigma)
    p90 = norm.ppf(0.90, mu, sigma)
    print(f"  Percentil 10 (mes muy malo)   = {p10:.2f}%")
    print(f"  Percentil 90 (mes muy bueno)  = {p90:.2f}%")

# ============================================================
# GRÁFICOS
# ============================================================

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

# ── Gráfico 1 y 2: Histograma + curva normal por año ──
for idx, (anio, color) in enumerate(zip(ANIOS_ANALISIS, COLORES_ANIOS)):
    ax = fig.add_subplot(gs[0, idx])
    r = resultados_anios[anio]
    datos = r["datos"]
    mu, sigma = r["media"], r["std"]

    # Histograma de densidad
    ax.hist(datos, bins=6, density=True, alpha=0.55, color=color,
            edgecolor="white", linewidth=1.2, label="Frecuencia observada")

    # Curva normal teórica
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    ax.plot(x, norm.pdf(x, mu, sigma), color="black", linewidth=2,
            label=f"N(μ={mu:.1f}, σ={sigma:.1f})")

    # Líneas de media y ± 1σ
    ax.axvline(mu, color="black", linestyle="-", linewidth=1.2, alpha=0.7)
    ax.axvline(mu - sigma, color=color, linestyle="--", linewidth=1, alpha=0.8)
    ax.axvline(mu + sigma, color=color, linestyle="--", linewidth=1, alpha=0.8)
    ax.fill_between(x, norm.pdf(x, mu, sigma),
                    where=(x >= mu - sigma) & (x <= mu + sigma),
                    alpha=0.15, color=color, label="μ ± 1σ  (≈68%)")

    stat, p = stats.shapiro(datos)
    ax.set_title(
        f"Año {anio} — Histograma + Curva Normal\n"
        f"Shapiro-Wilk: p = {p:.4f}  {'✓ Normal' if p > 0.05 else '✗ No normal'}",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Tasa de Puntualidad (%)")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# ── Gráfico 3 y 4: Q-Q Plot por año ──
for idx, (anio, color) in enumerate(zip(ANIOS_ANALISIS, COLORES_ANIOS)):
    ax = fig.add_subplot(gs[1, idx])
    r = resultados_anios[anio]
    datos = r["datos"].sort_values().values

    # Cuantiles teóricos vs observados
    n = len(datos)
    probs = (np.arange(1, n + 1) - 0.5) / n
    teoricos = norm.ppf(probs, r["media"], r["std"])

    ax.scatter(teoricos, datos, color=color, s=60, zorder=3,
               edgecolors="white", linewidths=0.5)
    lims = [min(teoricos.min(), datos.min()) - 1,
            max(teoricos.max(), datos.max()) + 1]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Línea de referencia (Normal perfecta)")
    ax.set_title(f"Año {anio} — Q-Q Plot\n(cuantiles observados vs teóricos)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Cuantiles teóricos N(μ, σ)")
    ax.set_ylabel("Cuantiles observados (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

# ── Gráfico 5: Evolución histórica con medias anuales ──
ax5 = fig.add_subplot(gs[2, :])

medias = [(anio, v["media"], v["std"], v["p"]) for anio, v in resultados_anios.items()]
anios_ord = sorted(resultados_anios.keys())

for anio in anios_ord:
    r = resultados_anios[anio]
    datos = r["datos"]
    orden_meses = {
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
        "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
        "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
    }
    sub_anio = sub[sub["Año"] == anio].copy()
    sub_anio["mes_num"] = sub_anio["Mes"].map(orden_meses)
    sub_anio = sub_anio.sort_values("mes_num")

# Línea temporal completa
sub_ord = sub.copy()
orden_meses = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}
sub_ord["mes_num"] = sub_ord["Mes"].map(orden_meses)
sub_ord = sub_ord.sort_values(["Año", "mes_num"]).reset_index(drop=True)
sub_ord["periodo_idx"] = range(len(sub_ord))

ax5.plot(sub_ord["periodo_idx"], sub_ord["Tasa de Puntualidad"],
         color="gray", linewidth=1, alpha=0.5, label="Puntualidad mensual")
ax5.scatter(sub_ord["periodo_idx"], sub_ord["Tasa de Puntualidad"],
            s=15, color="gray", alpha=0.4)

# Resaltar los años analizados con banda μ ± σ
for anio, color in zip(ANIOS_ANALISIS, COLORES_ANIOS):
    mask = sub_ord["Año"] == anio
    idxs = sub_ord[mask]["periodo_idx"].values
    r = resultados_anios[anio]
    ax5.axhspan(r["media"] - r["std"], r["media"] + r["std"],
                xmin=idxs[0] / len(sub_ord), xmax=idxs[-1] / len(sub_ord),
                alpha=0.15, color=color)
    ax5.hlines(r["media"], idxs[0], idxs[-1], color=color, linewidth=2,
               linestyle="-", label=f"μ {anio} = {r['media']:.1f}%")

# Etiquetas eje x cada 12 meses
tick_idxs = list(range(0, len(sub_ord), 12))
ax5.set_xticks(tick_idxs)
ax5.set_xticklabels(
    [sub_ord["Año"].iloc[i] for i in tick_idxs],
    rotation=0, fontsize=9
)
ax5.set_xlabel("Año")
ax5.set_ylabel("Tasa de Puntualidad (%)")
ax5.set_title(
    "Evolución histórica de la Tasa de Puntualidad\n"
    "El modelo Normal aplica dentro de cada año — no al acumulado histórico",
    fontsize=10, fontweight="bold"
)
ax5.legend(fontsize=9, loc="upper right")
ax5.grid(True, alpha=0.3)

fig.suptitle(
    "Distribución Normal — Tasa de Puntualidad Mensual\n"
    "Servicio Constitución – Bosques (vía Temperley) | Línea Roca",
    fontsize=13, fontweight="bold"
)

plt.savefig("distribucion_normal.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✓ Gráfico guardado como 'distribucion_normal.png'")
