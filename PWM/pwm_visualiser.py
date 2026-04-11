import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full", app_title="PWM Visualisation")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    return mo, np, plt, gridspec


@app.cell
def __(mo):
    ma_slider = mo.ui.slider(
        start=0.1, stop=1.2, step=0.05, value=0.8,
        label="Modulation Index  *mₐ*"
    )
    mf_slider = mo.ui.slider(
        start=3, stop=21, step=2, value=9,
        label="Frequency Ratio  *m_f*  (carrier / reference)"
    )
    vdc_slider = mo.ui.slider(
        start=100, stop=600, step=50, value=300,
        label="DC Bus Voltage  *V_dc* (V)"
    )
    cycles_slider = mo.ui.slider(
        start=1, stop=4, step=1, value=2,
        label="Reference cycles to display"
    )

    mo.hstack([ma_slider, mf_slider, vdc_slider, cycles_slider], gap=2)
    return ma_slider, mf_slider, vdc_slider, cycles_slider


@app.cell
def __(mo, ma_slider, mf_slider, vdc_slider, cycles_slider, np, plt, gridspec):
    ma   = ma_slider.value
    mf   = int(mf_slider.value)
    Vdc  = vdc_slider.value
    n_cycles = int(cycles_slider.value)

    _f_ref    = 50
    f_carrier = mf * _f_ref
    T_ref     = 1 / _f_ref
    T_total   = n_cycles * T_ref

    N = 10_000
    t = np.linspace(0, T_total, N, endpoint=False)

    v_ref = ma * np.sin(2 * np.pi * _f_ref * t)

    carrier_phase = (f_carrier * t) % 1.0
    v_tri = 2 * np.abs(2 * carrier_phase - 1) - 1

    V_AN = np.where(v_ref >  v_tri,  Vdc, 0.0)
    V_BN = np.where(v_ref > -v_tri,  Vdc, 0.0)
    v_uni = V_AN - V_BN

    v_bip = np.where(v_ref > v_tri, Vdc, -Vdc)

    _V1_uni = ma * Vdc         
    _V1_bip = ma * Vdc

    t_ms = t * 1e3             

    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#0f1117",
        "axes.edgecolor":   "#2a2d3e",
        "axes.labelcolor":  "#c8ccd8",
        "xtick.color":      "#6b7280",
        "ytick.color":      "#6b7280",
        "grid.color":       "#1e2130",
        "grid.linewidth":   0.6,
        "text.color":       "#c8ccd8",
        "font.family":      "monospace",
    })

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"SPWM  |  mₐ = {ma:.2f}   m_f = {mf}   V_dc = {Vdc} V   f_ref = {_f_ref} Hz",
        fontsize=13, color="#e2e8f0", y=0.98, fontweight="bold"
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    CLR_TRI  = "#60a5fa"
    CLR_REF  = "#f97316"
    CLR_UNI  = "#34d399"
    CLR_BIP  = "#a78bfa"
    CLR_NEG  = "#fb7185"
    ALPHA_C  = 0.55

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t_ms, v_tri,  color=CLR_TRI,  lw=0.9, alpha=ALPHA_C, label="Carrier  v_tri")
    ax0.plot(t_ms, -v_tri, color=CLR_NEG,  lw=0.9, alpha=ALPHA_C, ls="--", label="−v_tri")
    ax0.plot(t_ms, v_ref,  color=CLR_REF,  lw=1.5, label=f"Reference  (mₐ={ma:.2f})")
    ax0.set_title("UNIPOLAR — Carrier & Reference", color="#94a3b8", fontsize=9)
    ax0.set_ylabel("Normalised", fontsize=8)
    ax0.legend(fontsize=7, loc="upper right", framealpha=0.15)
    ax0.set_ylim(-1.4, 1.4)
    ax0.axhline(0, color="#374151", lw=0.5)
    ax0.grid(True)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t_ms, v_tri, color=CLR_TRI, lw=0.9, alpha=ALPHA_C, label="Carrier  v_tri")
    ax1.plot(t_ms, v_ref, color=CLR_REF, lw=1.5, label=f"Reference  (mₐ={ma:.2f})")
    ax1.set_title("BIPOLAR — Carrier & Reference", color="#94a3b8", fontsize=9)
    ax1.legend(fontsize=7, loc="upper right", framealpha=0.15)
    ax1.set_ylim(-1.4, 1.4)
    ax1.axhline(0, color="#374151", lw=0.5)
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.step(t_ms, V_AN, color="#38bdf8", lw=0.8, where="post", alpha=0.9, label="V_AN")
    ax2.step(t_ms, V_BN, color=CLR_REF,  lw=0.8, where="post", alpha=0.7, label="V_BN")
    ax2.set_title("UNIPOLAR — Leg voltages V_AN, V_BN", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("Voltage (V)", fontsize=8)
    ax2.set_ylim(-50, Vdc + 80)
    ax2.axhline(0, color="#374151", lw=0.5)
    ax2.legend(fontsize=7, loc="upper right", framealpha=0.15)
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 1])
    cond = v_ref > v_tri
    ax3.fill_between(t_ms, 0, 1, where=cond,  color="#a78bfa", alpha=0.3,
                     transform=ax3.get_xaxis_transform(), label="S1/S4 ON  (+Vdc)")
    ax3.fill_between(t_ms, 0, 1, where=~cond, color="#fb7185", alpha=0.25,
                     transform=ax3.get_xaxis_transform(), label="S2/S3 ON  (−Vdc)")
    ax3.plot(t_ms, v_tri, color=CLR_TRI, lw=0.7, alpha=0.5)
    ax3.plot(t_ms, v_ref, color=CLR_REF, lw=1.2)
    ax3.set_title("BIPOLAR — Switching regions", color="#94a3b8", fontsize=9)
    ax3.set_ylim(-1.4, 1.4)
    ax3.axhline(0, color="#374151", lw=0.5)
    ax3.legend(fontsize=7, loc="upper right", framealpha=0.15)
    ax3.grid(True)

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.step(t_ms, v_uni, color=CLR_UNI, lw=0.9, where="post")
    fund_uni = _V1_uni * np.sin(2 * np.pi * _f_ref * t)
    ax4.plot(t_ms, fund_uni, color=CLR_REF, lw=1.5, ls="--", alpha=0.85,
             label=f"Fundamental  ({_V1_uni:.0f} V pk)")
    ax4.set_title("UNIPOLAR — Output V_AB  (3-level)", color="#94a3b8", fontsize=9)
    ax4.set_ylabel("Voltage (V)", fontsize=8)
    ax4.set_ylim(-Vdc * 1.3, Vdc * 1.3)
    ax4.axhline(0, color="#374151", lw=0.5)
    ax4.legend(fontsize=7, loc="upper right", framealpha=0.15)
    ax4.grid(True)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.step(t_ms, v_bip, color=CLR_BIP, lw=0.9, where="post")
    fund_bip = _V1_bip * np.sin(2 * np.pi * _f_ref * t)
    ax5.plot(t_ms, fund_bip, color=CLR_REF, lw=1.5, ls="--", alpha=0.85,
             label=f"Fundamental  ({_V1_bip:.0f} V pk)")
    ax5.set_title("BIPOLAR — Output V_AB  (2-level)", color="#94a3b8", fontsize=9)
    ax5.set_ylim(-Vdc * 1.3, Vdc * 1.3)
    ax5.axhline(0, color="#374151", lw=0.5)
    ax5.legend(fontsize=7, loc="upper right", framealpha=0.15)
    ax5.grid(True)

    def compute_fft(sig, dt):
        n    = len(sig)
        freq = np.fft.rfftfreq(n, d=dt)
        mag  = (2 / n) * np.abs(np.fft.rfft(sig))
        return freq, mag

    dt = t[1] - t[0]
    freq_u, mag_u = compute_fft(v_uni, dt)
    freq_b, mag_b = compute_fft(v_bip, dt)

    f_max = (mf + 4) * _f_ref + 50
    mask_u = freq_u <= f_max
    mask_b = freq_b <= f_max

    ax6 = fig.add_subplot(gs[3, 0])
    ax6.bar(freq_u[mask_u], mag_u[mask_u], width=8, color=CLR_UNI, alpha=0.85)
    ax6.axvline(_f_ref, color=CLR_REF, lw=1, ls="--", label=f"f_ref = {_f_ref} Hz")
    ax6.set_title("UNIPOLAR — Harmonic Spectrum", color="#94a3b8", fontsize=9)
    ax6.set_xlabel("Frequency (Hz)", fontsize=8)
    ax6.set_ylabel("|V| (V)", fontsize=8)
    ax6.legend(fontsize=7, framealpha=0.15)
    ax6.set_xlim(0, f_max)
    ax6.grid(True, axis="y")

    ax7 = fig.add_subplot(gs[3, 1])
    ax7.bar(freq_b[mask_b], mag_b[mask_b], width=8, color=CLR_BIP, alpha=0.85)
    ax7.axvline(_f_ref, color=CLR_REF, lw=1, ls="--", label=f"f_ref = {_f_ref} Hz")
    ax7.set_title("BIPOLAR — Harmonic Spectrum", color="#94a3b8", fontsize=9)
    ax7.set_xlabel("Frequency (Hz)", fontsize=8)
    ax7.legend(fontsize=7, framealpha=0.15)
    ax7.set_xlim(0, f_max)
    ax7.grid(True, axis="y")

    for ax in [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]:
        ax.set_xlabel("Time (ms)", fontsize=8) if ax not in [ax6,ax7] else None
        ax.tick_params(labelsize=7)

    plt.gcf()
    return fig, ma, mf, Vdc, N, T_total, t, v_ref, v_tri, V_AN, V_BN, v_uni, v_bip


@app.cell
def __(mo, ma, mf, Vdc, v_uni, v_bip, np):
    def rms(x): return np.sqrt(np.mean(x**2))
    def thd(sig, f0, t, N):
        freq = np.fft.rfftfreq(N, d=t[1]-t[0])
        mag  = (2/N) * np.abs(np.fft.rfft(sig))
        idx1 = np.argmin(np.abs(freq - f0))
        V1   = mag[idx1]
        Vrms_total = rms(sig)
        Vh_rms = np.sqrt(max(Vrms_total**2 - (V1/np.sqrt(2))**2, 0))
        return (Vh_rms / (V1/np.sqrt(2))) * 100 if V1 > 0 else 0

    _f_ref = 50
    _N = len(v_uni)
    _t = np.linspace(0, 2/_f_ref, _N, endpoint=False)

    thd_uni = thd(v_uni, _f_ref, _t, _N)
    thd_bip = thd(v_bip, _f_ref, _t, _N)
    _V1_uni  = ma * Vdc
    _V1_bip  = ma * Vdc

    mo.md(f"""
    ---
    ## 📊 Performance Metrics

    | Metric | Unipolar | Bipolar |
    |--------|----------|---------|
    | Output levels | **3** (0, ±{Vdc} V) | **2** (±{Vdc} V) |
    | Fundamental peak | **{_V1_uni:.1f} V** | **{_V1_bip:.1f} V** |
    | Effective sw. freq. | **{2*mf*_f_ref} Hz** (2 × carrier) | **{mf*_f_ref} Hz** |
    | Approx. THD | **{thd_uni:.1f} %** | **{thd_bip:.1f} %** |

    > **Unipolar advantage**: first harmonic cluster appears at *2 m_f*, not *m_f* — easier to filter and lower THD for the same carrier frequency.
    """)
    return



if __name__ == "__main__":
    app.run()
