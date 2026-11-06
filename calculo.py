"""Verifica la solucion analitica (8)/(10) contra integracion numerica (RK45)."""
import numpy as np
from scipy.integrate import solve_ivp

r, K, N0 = 0.5, 100.0, 10.0
t = np.linspace(0, 40, 401)

def analitica(t, c):
    a, b = r - c, r / K
    if abs(a) < 1e-12:
        return N0 / (1 + b * N0 * t)
    return a * N0 * np.exp(a * t) / (a + b * N0 * (np.exp(a * t) - 1))

for c in [0.2, 0.4, 0.5, 0.8]:
    f = lambda tt, N: r * N * (1 - N / K) - c * N
    sol = solve_ivp(f, (0, 40), [N0], t_eval=t, rtol=1e-10, atol=1e-12)
    err = np.max(np.abs(sol.y[0] - analitica(t, c)))
    print(f"c = {c}: error maximo |analitica - RK45| = {err:.2e}")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
# Valores nominales e incertidumbres ASUMIDAS (1 sigma)
r0, K0 = 0.5, 100.0
sr, sc, sK = 0.05, 0.05, 5.0

def Nstar(r, c, K): return np.maximum(K*(1-c/r), 0)

# ---------- Tabla 1: propagacion lineal vs Monte Carlo ----------
rows = []
M = 20000
for c0 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.8]:
    # propagacion lineal (solo valida si c0<r0, lejos del umbral)
    dK = 1 - c0/r0; dc = -K0/r0; dr = K0*c0/r0**2
    sN_lin = np.sqrt((dK*sK)**2 + (dc*sc)**2 + (dr*sr)**2)
    tau = 1/(r0-c0) if r0 > c0 else np.nan
    s_tau = np.sqrt(sr**2+sc**2)/(r0-c0)**2 if r0 > c0 else np.nan
    # Monte Carlo
    r = rng.normal(r0, sr, M); c = rng.normal(c0, sc, M); K = rng.normal(K0, sK, M)
    r = np.clip(r, 1e-3, None); c = np.clip(c, 0, None)
    Ns = Nstar(r, c, K)
    p_col = np.mean(c >= r)
    rows.append(dict(c=c0, c_over_r=c0/r0, N_star_nom=max(K0*(1-c0/r0),0),
                     sigma_N_lin=sN_lin, N_star_MC_mean=Ns.mean(), N_star_MC_std=Ns.std(),
                     tau_nom=tau, sigma_tau_lin=s_tau, P_colapso=p_col))
T = pd.DataFrame(rows).round(3)
T.to_csv("tabla_equilibrio_incertidumbre.csv", index=False)
print(T.to_string(index=False))

# ---------- Figura 1: N* vs c con barras de error ----------
cs = np.linspace(0, 0.8, 33)
mc_m, mc_s = [], []
for c0 in cs:
    r = np.clip(rng.normal(r0, sr, M),1e-3,None); c = np.clip(rng.normal(c0, sc, M),0,None); K = rng.normal(K0, sK, M)
    Ns = Nstar(r, c, K); mc_m.append(Ns.mean()); mc_s.append(Ns.std())
fig, ax = plt.subplots(figsize=(7,4.5))
ax.plot(cs, np.maximum(K0*(1-cs/r0),0), 'k-', label=r'Nominal $N^*=K(1-c/r)$')
sel = slice(0, None, 3)
ax.errorbar(cs[sel], np.array(mc_m)[sel], yerr=np.array(mc_s)[sel], fmt='o', capsize=3, color='C0', label=r'Monte Carlo (media $\pm1\sigma$)')
ax.axvline(r0, ls='--', color='r', label=r'Umbral $c=r$')
ax.set_xlabel('c (tasa de pérdida)'); ax.set_ylabel(r'$N^*$ (instancias)')
ax.set_title('Diagrama de bifurcación con incertidumbre paramétrica'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("fig1_bifurcacion_errores.png", dpi=200); plt.close(fig)

# ---------- Figura 2: tau vs c con barras de error ----------
cc = np.array([0.0,0.1,0.2,0.3,0.35,0.4,0.43,0.45])
tau = 1/(r0-cc); stau = np.sqrt(sr**2+sc**2)/(r0-cc)**2
fig, ax = plt.subplots(figsize=(7,4.5))
ax.errorbar(cc, tau, yerr=stau, fmt='o-', capsize=3, color='C1')
ax.set_yscale('log'); ax.axvline(r0, ls='--', color='r')
ax.set_xlabel('c'); ax.set_ylabel(r'$\tau = 1/(r-c)$'); ax.set_title('Critical slowing down')
ax.grid(alpha=.3, which='both'); fig.tight_layout(); fig.savefig("fig2_tau_errores.png", dpi=200); plt.close(fig)

# ---------- Figura 3: trayectorias N(t) media +- std (Monte Carlo) ----------
def N_t(t, N0, r, c, K):
    a = r-c; b = r/K
    out = np.empty_like(t, dtype=float)
    small = np.abs(a) < 1e-9
    if small: return N0/(1+b*N0*t)
    return a*N0*np.exp(a*t)/(a + b*N0*(np.exp(a*t)-1))
t = np.linspace(0, 40, 200); N0 = 10.0
fig, ax = plt.subplots(figsize=(7.5,4.8))
for c0, col in [(0.2,'C2'),(0.4,'C0'),(0.5,'C1'),(0.8,'C3')]:
    Ms = 2000; trajs = []
    for _ in range(Ms):
        r = max(rng.normal(r0, sr),1e-3); c = max(rng.normal(c0, sc),0); K = rng.normal(K0, sK)
        trajs.append(N_t(t, N0, r, c, K))
    trajs = np.array(trajs); m, s = trajs.mean(0), trajs.std(0)
    ax.plot(t, m, color=col, label=f'c = {c0}'); ax.fill_between(t, m-s, m+s, color=col, alpha=.2)
ax.set_xlabel('t'); ax.set_ylabel('N(t)'); ax.set_title(r'$N(t)$, media $\pm1\sigma$ ($N(0)=10$, $r=0.5$, $K=100$)')
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("fig3_trayectorias.png", dpi=200); plt.close(fig)
    