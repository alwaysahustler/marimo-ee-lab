import marimo

app = marimo.App(width="full", app_title="PWM Visualisation")


# ─────────────────────────────────────────────
# CELL 1 — Imports
# ─────────────────────────────────────────────
@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    return mo, np, plt, gridspec


# ─────────────────────────────────────────────
# CELL 2 — Title
# ─────────────────────────────────────────────
@app.cell
def __(mo):
    mo.md("""
# ⚡ Sinusoidal PWM Visualisation
### Unipolar vs Bipolar SPWM
""")
    return


# ─────────────────────────────────────────────
# CELL 3 — UI Controls
# ─────────────────────────────────────────────
@app.cell
def __(mo):
    ma_slider = mo.ui.slider(0.1, 1.2, step=0.05, value=0.8, label="Modulation Index mₐ")
    mf_slider = mo.ui.slider(3, 21, step=2, value=9, label="Frequency Ratio m_f")
    vdc_slider = mo.ui.slider(100, 600, step=50, value=300, label="DC Bus Voltage V_dc")
    cycles_slider = mo.ui.slider(1, 4, step=1, value=2, label="Cycles")

    mo.hstack([ma_slider, mf_slider, vdc_slider, cycles_slider])

    return ma_slider, mf_slider, vdc_slider, cycles_slider


# ─────────────────────────────────────────────
# CELL 4 — Core Simulation + Plot
# ─────────────────────────────────────────────
@app.cell
def __(ma_slider, mf_slider, vdc_slider, cycles_slider, np, plt, gridspec):

    # Parameters
    ma = ma_slider.value
    mf = int(mf_slider.value)
    Vdc = vdc_slider.value
    n_cycles = int(cycles_slider.value)

    f_ref = 50
    f_carrier = mf * f_ref

    T_ref = 1 / f_ref
    T_total = n_cycles * T_ref

    N = 10000
    t = np.linspace(0, T_total, N, endpoint=False)

    # Signals
    v_ref = ma * np.sin(2 * np.pi * f_ref * t)

    carrier_phase = (f_carrier * t) % 1.0
    v_tri = 2 * np.abs(2 * carrier_phase - 1) - 1

    # Unipolar PWM
    V_AN = np.where(v_ref > v_tri, Vdc, 0.0)
    V_BN = np.where(v_ref > -v_tri, Vdc, 0.0)
    v_uni = V_AN - V_BN

    # Bipolar PWM
    v_bip = np.where(v_ref > v_tri, Vdc, -Vdc)

    # Fundamental
    V1_uni = ma * Vdc
    V1_bip = ma * Vdc

    t_ms = t * 1e3

    # Plot
    fig = plt.figure(figsize=(12, 8))

    plt.subplot(2,1,1)
    plt.plot(t_ms, v_uni, label="Unipolar")
    plt.plot(t_ms, V1_uni * np.sin(2*np.pi*f_ref*t), '--', label="Fundamental")
    plt.title("Unipolar Output")
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(t_ms, v_bip, label="Bipolar")
    plt.plot(t_ms, V1_bip * np.sin(2*np.pi*f_ref*t), '--', label="Fundamental")
    plt.title("Bipolar Output")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    return v_uni, v_bip, t, f_ref, V1_uni, V1_bip


# ─────────────────────────────────────────────
# CELL 5 — THD Calculation
# ─────────────────────────────────────────────
@app.cell
def __(v_uni, v_bip, t, f_ref, np):

    def rms(x):
        return np.sqrt(np.mean(x**2))

    def thd(sig, f0):
        N = len(sig)
        freq = np.fft.rfftfreq(N, d=t[1]-t[0])
        mag = (2/N) * np.abs(np.fft.rfft(sig))

        idx = np.argmin(np.abs(freq - f0))
        V1 = mag[idx]

        Vrms_total = rms(sig)
        Vh = np.sqrt(max(Vrms_total**2 - (V1/np.sqrt(2))**2, 0))

        return (Vh / (V1/np.sqrt(2))) * 100

    thd_uni = thd(v_uni, f_ref)
    thd_bip = thd(v_bip, f_ref)

    return thd_uni, thd_bip


# ─────────────────────────────────────────────
# CELL 6 — Results Display
# ─────────────────────────────────────────────
@app.cell
def __(mo, thd_uni, thd_bip, V1_uni, V1_bip):

    mo.md(f"""
## 📊 Results

- Fundamental Voltage ≈ {V1_uni:.2f} V  
- THD (Unipolar): {thd_uni:.2f}%  
- THD (Bipolar): {thd_bip:.2f}%  

👉 Unipolar has lower THD because harmonics shift to higher frequency (2mf)
""")
    return


if __name__ == "__main__":
    app.run()