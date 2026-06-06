"""
=============================================================
  KNN aplicado a dataset_E.csv — CS3061 ML | UTEC
  Reusa el KNN desde cero de knn_completo.py (sin reescribirlo).
=============================================================

Dataset: horas_estudio, nota_parcial -> rendimiento (3 clases: 0,1,2)
Responde las 5 preguntas de análisis con evidencia numérica + gráfico.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

# Reuso directo del código del profe (importar NO ejecuta el main)
from knn_completo import (
    KNN,
    train_test_split_manual,
    normalizar,
    graficar_frontera,
    graficar_accuracy_vs_k,
)

# La semilla 42 ya se fija al importar knn_completo -> split reproducible.

# ── Cargar dataset ────────────────────────────────────────────
data = np.genfromtxt("dataset_E.csv", delimiter=",", skip_header=1)
X = data[:, :2]              # horas_estudio, nota_parcial
y = data[:, 2].astype(int)   # rendimiento (3 clases)

print("=" * 60)
print("  KNN sobre dataset_E.csv")
print("=" * 60)
print(f"  n total = {len(X)} | clases = {sorted(set(y))}")
for c in sorted(set(y)):
    print(f"    clase {c}: {np.sum(y == c)} muestras")

# ── Split + normalización (mismas funciones del profe) ────────
X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.3)
X_train, X_test = normalizar(X_train, X_test)

n_train = len(X_train)
k_sqrt = max(1, int(np.sqrt(n_train)))
k_range = range(1, 41)

print(f"\n  n_train = {n_train} | n_test = {len(X_test)} | K=√n = {k_sqrt}")
print(f"\n  {'K':>4} | {'Train Acc':>10} | {'Test Acc':>10}")
print(f"  {'-'*32}")

acc_train, acc_test = [], []
mejor_k, mejor_acc = 1, 0.0
for k in k_range:
    m = KNN(k=k)
    m.fit(X_train, y_train)
    atr = m.accuracy(X_train, y_train)
    ate = m.accuracy(X_test, y_test)
    acc_train.append(atr)
    acc_test.append(ate)
    marca = " ← √n" if k == k_sqrt else ""
    print(f"  K={k:2d} | Train={atr:.2%} | Test={ate:.2%}{marca}")
    if ate > mejor_acc:
        mejor_acc, mejor_k = ate, k

acc_train = np.array(acc_train)
acc_test = np.array(acc_test)

print("\n" + "=" * 60)
print("  RESUMEN PARA EL ANÁLISIS")
print("=" * 60)
print(f"  K=1            -> Train={acc_train[0]:.2%} | Test={acc_test[0]:.2%}")
print(f"  K=√n={k_sqrt:<2d}        -> Train={acc_train[k_sqrt-1]:.2%} | Test={acc_test[k_sqrt-1]:.2%}")
print(f"  K óptimo={mejor_k:<2d}     -> Test={mejor_acc:.2%}")
print(f"  Gap train-test en K=1  : {acc_train[0]-acc_test[0]:+.2%}")
print(f"  Gap train-test en K=√n : {acc_train[k_sqrt-1]-acc_test[k_sqrt-1]:+.2%}")
print(f"  Train acc: K=1={acc_train[0]:.2%} -> K=40={acc_train[-1]:.2%} (caída {acc_train[0]-acc_train[-1]:+.2%})")
print(f"  Test acc:  min={acc_test.min():.2%} | max={acc_test.max():.2%} (rango {acc_test.max()-acc_test.min():.2%})")
print(f"  ¿K=√n == K óptimo? {'Sí' if mejor_k == k_sqrt else 'No (óptimo=%d)' % mejor_k}")

# ── Gráfico: fronteras (K=1, √n, óptimo) + curva acc vs K ─────
colores_mapa = ListedColormap(["#AADDFF", "#FFAAAA", "#AAFFAA"])
fig = plt.figure(figsize=(16, 10), constrained_layout=True)
fig.suptitle("dataset_E — KNN: frontera de decisión y accuracy vs K",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

ks_a_graficar = [1, k_sqrt, mejor_k]
etiquetas = ["K=1 (mínimo)", f"K={k_sqrt} = √n", f"K={mejor_k} (óptimo)"]
for idx, (k_val, etq) in enumerate(zip(ks_a_graficar, etiquetas)):
    m = KNN(k=k_val)
    m.fit(X_train, y_train)
    acc = m.accuracy(X_test, y_test)
    ax = fig.add_subplot(gs[0, idx])
    graficar_frontera(ax, m, X_test, y_test, etq, k_val, acc,
                      colores_mapa, ["blue", "red", "green"])

ax_curve = fig.add_subplot(gs[1, :])
graficar_accuracy_vs_k(ax_curve, X_train, y_train, X_test, y_test,
                       k_range, k_sqrt, "Accuracy vs K — dataset_E")

plt.savefig("knn_dataset_E.png", dpi=120, bbox_inches="tight")
print("\n  ✅ Gráfico guardado: knn_dataset_E.png")
