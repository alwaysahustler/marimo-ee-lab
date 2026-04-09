import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import signal
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    return matplotlib, mo, np, plt, signal, ticker


@app.cell
def _(mo):
    mo.md(r"""
    # PID Controller Analysis

    Plant model:

    $$
    G(s) = \frac{1}{s^2 + 2s + 3}
    $$

    Use the controller mode and gains below to evaluate closed-loop step-response behavior.
    """)
    return


@app.cell
def _(mo):
    controller_type = mo.ui.radio(
        options=["P", "PI", "PID"],
        value="PID",
        label="Controller mode",
        inline=True,
    )
    controller_type
    return (controller_type,)


@app.cell
def _(controller_type, mo):
    _is_p = controller_type.value == "P"
    _is_pi_or_p = controller_type.value in ["P", "PI"]

    kp = mo.ui.slider(
        start=0.1, stop=30.0, value=5.0, step=0.1,
        label="Kp (proportional gain)",
        show_value=True,
    )
    ki = mo.ui.slider(
        start=0.0, stop=15.0, value=(0.0 if _is_p else 1.5), step=0.1,
        label="Ki (integral gain)",
        disabled=_is_p,
        show_value=True,
    )
    kd = mo.ui.slider(
        start=0.0, stop=5.0, value=(0.0 if _is_pi_or_p else 0.8), step=0.05,
        label="Kd (derivative gain)",
        disabled=_is_pi_or_p,
        show_value=True,
    )
    mo.vstack([kp, ki, kd], gap="0.6rem")
    return kd, ki, kp


@app.cell
def _(controller_type, kd, ki, kp, mo, np, plt, signal, ticker):
    # ── Gains (respect the mode toggle) ───────────────────────────────────────
    _Kp = kp.value
    _Ki = ki.value if controller_type.value in ["PI", "PID"] else 0.0
    _Kd = kd.value if controller_type.value == "PID"          else 0.0

    # ── Plant  G(s) = 1 / (s² + 2s + 3) ─────────────────────────────────────
    _Gnum = np.array([1.0])
    _Gden = np.array([1.0, 2.0, 3.0])

    # ── PID controller  C(s) = (Kd·s² + Kp·s + Ki) / s ──────────────────────
    if _Ki == 0 and _Kd == 0:            # P only
        _Cnum, _Cden = np.array([_Kp]),             np.array([1.0])
    elif _Kd == 0:                       # PI
        _Cnum, _Cden = np.array([_Kp, _Ki]),        np.array([1.0, 0.0])
    else:                                # PID
        _Cnum, _Cden = np.array([_Kd, _Kp, _Ki]),  np.array([1.0, 0.0])

    # ── Open-loop then closed-loop ────────────────────────────────────────────
    _OLnum = np.polymul(_Cnum, _Gnum)
    _OLden = np.polymul(_Cden, _Gden)
    _CLnum = _OLnum.copy()
    _CLden = np.polyadd(_OLnum, _OLden)

    _cl = signal.TransferFunction(_CLnum, _CLden)

    # ── Step simulation ───────────────────────────────────────────────────────
    _t     = np.linspace(0, 25, 4000)
    _t_out, _y_raw = signal.step(_cl, T=_t)
    _y     = np.clip(_y_raw, -8.0, 8.0)          # guard against blow-up in plots
    _e     = 1.0 - _y

    # ── Metrics ───────────────────────────────────────────────────────────────
    _ss_val    = float(_y[-1])
    _ss_err    = abs(float(_e[-1]))
    _peak      = float(np.max(_y))
    _overshoot = max(0.0, (_peak - _ss_val) / _ss_val * 100) if _ss_val > 1e-6 else 0.0

    # Rise time  10 % → 90 %
    _i10 = int(np.argmax(_y >= 0.10 * 1.0))
    _i90 = int(np.argmax(_y >= 0.90 * 1.0))
    _rise = float(_t_out[_i90] - _t_out[_i10]) if _i90 > _i10 > 0 else float("inf")

    # Settling time  ±2 % band
    _band  = 0.02
    _settle_idx = len(_t_out) - 1
    for _j in range(len(_t_out) - 1, -1, -1):
        if abs(_y[_j] - _ss_val) > _band:
            _settle_idx = _j + 1
            break
    _settle = float(_t_out[_settle_idx]) if _settle_idx < len(_t_out) else float("inf")

    # Stability: all closed-loop poles in LHP?
    _poles  = np.roots(_CLden)
    _stable = bool(np.all(np.real(_poles) < 0))

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.style.use("default")
    _fig, (_ax1, _ax2) = plt.subplots(
        2, 1, figsize=(10, 7.5),
        constrained_layout=True,
    )

    _ax1.grid(True, linewidth=0.6, alpha=0.5)
    _ax2.grid(True, linewidth=0.6, alpha=0.5)

    # ── Step response ─────────────────────────────────────────────────────────
    _ax1.plot(_t_out, _y, color="C0", lw=2.0, label="Output  y(t)", zorder=3)
    _ax1.axhline(1.0, color="C3", ls="--", lw=1.2, alpha=0.9, label="Setpoint  r = 1")
    _ax1.axhline(_ss_val, color="C2", ls=":", lw=1.1, alpha=0.9,
                 label=f"Steady-state ≈ {_ss_val:.3f}")
    # ±2 % settling band
    _ax1.axhspan(1 - _band, 1 + _band, alpha=0.08, color="C2")

    if _overshoot > 0.1:
        _pk_i = int(np.argmax(_y))
        _ax1.plot(_t_out[_pk_i], _peak, marker="v", color="C1", ms=8, zorder=5,
                  label=f"Peak = {_peak:.3f}  ({_overshoot:.1f}% OS)")

    # Rise & settle markers
    if _rise < float("inf") and _i90 > 0:
        _ax1.axvline(_t_out[_i90], color="0.4", ls=":", lw=1, alpha=0.8)
    if _settle < float("inf"):
        _ax1.axvline(_settle, color="0.2", ls=":", lw=1, alpha=0.8,
                     label=f"Settle ≈ {_settle:.1f} s")

    _ax1.set_xlabel("Time (s)", fontsize=10)
    _ax1.set_ylabel("Output", fontsize=10)
    _ax1.set_title(
        f"Closed-Loop Step Response [{controller_type.value}]  "
        f"Kp = {_Kp}   Ki = {_Ki}   Kd = {_Kd}",
        fontsize=10, pad=8,
    )
    _ax1.legend(fontsize=8, framealpha=0.95)

    # ── Error vs time ─────────────────────────────────────────────────────────
    _ax2.plot(_t_out, _e, color="C3", lw=1.8, zorder=3)
    _ax2.axhline(0, color="0.4", ls="--", lw=1)
    _ax2.fill_between(_t_out, _e, 0, where=(_e > 0),
                      alpha=0.18, color="C3", label="Positive error")
    _ax2.fill_between(_t_out, _e, 0, where=(_e < 0),
                      alpha=0.18, color="C0", label="Negative error")
    _ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    _ax2.set_xlabel("Time (s)", fontsize=10)
    _ax2.set_ylabel("Error e(t) = 1 - y(t)", fontsize=10)
    _ax2.set_title("Error vs Time", fontsize=10, pad=8)
    _ax2.legend(fontsize=8, framealpha=0.95)

    _stab_txt = "Stable" if _stable else "Unstable"
    _rise_txt = f"{_rise:.2f} s" if _rise < float("inf") else "—"
    _settl_txt = f"{_settle:.2f} s" if _settle < float("inf") else "—"

    _metrics_table = mo.md(f"""
| Metric | Value |
|---|---|
| Stability | {_stab_txt} |
| Steady-state error | {_ss_err:.4f} |
| Overshoot | {_overshoot:.1f}% |
| Rise time (10%-90%) | {_rise_txt} |
| Settling time (±2%) | {_settl_txt} |
""")

    mo.vstack([_metrics_table, mo.as_html(_fig)])
    return