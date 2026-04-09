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
    # 🎛️ PID Controller Simulator

    **Plant model:** $\displaystyle G(s) = \frac{1}{s^2 + 2s + 3}$ — second-order underdamped system

    Tune the gains below and watch the closed-loop response update in real time.
    """)
    return


@app.cell
def _(mo):
    controller_type = mo.ui.radio(
        options=["P", "PI", "PID"],
        value="PID",
        label="**Controller Mode**",
        inline=True,
    )
    controller_type
    return (controller_type,)


@app.cell
def _(mo):
    kp = mo.ui.slider(
        start=0.1, stop=30.0, value=5.0, step=0.1,
        label="**Kp** — Proportional gain",
        show_value=True,
    )
    ki = mo.ui.slider(
        start=0.0, stop=15.0, value=1.5, step=0.1,
        label="**Ki** — Integral gain",
        show_value=True,
    )
    kd = mo.ui.slider(
        start=0.0, stop=5.0, value=0.8, step=0.05,
        label="**Kd** — Derivative gain",
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

    # ── Colour palette (Catppuccin Mocha) ─────────────────────────────────────
    _BG   = "#1e1e2e"
    _SURF = "#181825"
    _GRID = "#313244"
    _TEXT = "#cdd6f4"
    _SUB  = "#a6adc8"
    _BLUE = "#89b4fa"
    _RED  = "#f38ba8"
    _GRN  = "#a6e3a1"
    _YEL  = "#f9e2af"
    _PEACH= "#fab387"
    _MAUVE= "#cba6f7"
    _TEAL = "#94e2d5"

    # ── Figure ────────────────────────────────────────────────────────────────
    _fig, (_ax1, _ax2) = plt.subplots(
        2, 1, figsize=(10, 7.5),
        facecolor=_SURF,
        gridspec_kw={"hspace": 0.45},
    )

    def _style_ax(ax):
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_SUB, labelsize=9)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)

    _style_ax(_ax1)
    _style_ax(_ax2)

    # ── Step response ─────────────────────────────────────────────────────────
    _ax1.plot(_t_out, _y, color=_BLUE, lw=2.2, label="Output  y(t)", zorder=3)
    _ax1.axhline(1.0, color=_RED,  ls="--", lw=1.3, alpha=0.8, label="Setpoint  r = 1")
    _ax1.axhline(_ss_val, color=_GRN, ls=":",  lw=1.1, alpha=0.9,
                 label=f"Steady-state ≈ {_ss_val:.3f}")
    # ±2 % settling band
    _ax1.axhspan(1 - _band, 1 + _band, alpha=0.07, color=_GRN)

    if _overshoot > 0.1:
        _pk_i = int(np.argmax(_y))
        _ax1.plot(_t_out[_pk_i], _peak, marker="v", color=_PEACH, ms=9, zorder=5,
                  label=f"Peak = {_peak:.3f}  ({_overshoot:.1f}% OS)")

    # Rise & settle markers
    if _rise < float("inf") and _i90 > 0:
        _ax1.axvline(_t_out[_i90], color=_MAUVE, ls=":", lw=1, alpha=0.7)
    if _settle < float("inf"):
        _ax1.axvline(_settle, color=_TEAL, ls=":", lw=1, alpha=0.7,
                     label=f"Settle ≈ {_settle:.1f} s")

    _ax1.set_xlabel("Time  (s)", fontsize=10)
    _ax1.set_ylabel("Output", fontsize=10)
    _ax1.set_title(
        f"Closed-Loop Step Response  ·  [{controller_type.value}]  "
        f"Kp = {_Kp}   Ki = {_Ki}   Kd = {_Kd}",
        fontsize=10, pad=8,
    )
    _leg1 = _ax1.legend(
        facecolor="#313244", edgecolor=_GRID, labelcolor=_TEXT,
        fontsize=8, framealpha=0.9,
    )

    # ── Error vs time ─────────────────────────────────────────────────────────
    _ax2.plot(_t_out, _e, color=_RED, lw=2, zorder=3)
    _ax2.axhline(0, color=_GRID, ls="--", lw=1)
    _ax2.fill_between(_t_out, _e, 0, where=(_e > 0),
                      alpha=0.18, color=_RED,   label="Positive error")
    _ax2.fill_between(_t_out, _e, 0, where=(_e < 0),
                      alpha=0.18, color=_BLUE,  label="Negative error")
    _ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    _ax2.set_xlabel("Time  (s)", fontsize=10)
    _ax2.set_ylabel("Error  e(t) = 1 − y(t)", fontsize=10)
    _ax2.set_title("Error vs Time", fontsize=10, pad=8)
    _ax2.legend(
        facecolor="#313244", edgecolor=_GRID, labelcolor=_TEXT,
        fontsize=8, framealpha=0.9,
    )

    plt.tight_layout(pad=2.5)

    # ── Metrics cards (HTML) ──────────────────────────────────────────────────
    def _card(label, val, color):
        return f"""
        <div style="background:#181825;border:1px solid #313244;border-top:3px solid {color};
                    border-radius:8px;padding:14px 22px;min-width:130px;flex:1">
          <div style="font-size:10px;color:#a6adc8;text-transform:uppercase;
                      letter-spacing:1.2px;margin-bottom:4px">{label}</div>
          <div style="font-size:20px;font-weight:700;color:{color}">{val}</div>
        </div>"""

    _stab_col = _GRN if _stable else _RED
    _stab_txt = "✅  Stable" if _stable else "❌  Unstable"
    _os_col   = _GRN if _overshoot < 10 else (_YEL if _overshoot < 25 else _RED)
    _sse_col  = _GRN if _ss_err < 0.05  else (_YEL if _ss_err  < 0.2  else _RED)
    _rise_txt = f"{_rise:.2f} s" if _rise < float("inf") else "—"
    _settl_txt= f"{_settle:.2f} s" if _settle < float("inf") else "—"

    _cards = mo.md(f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin:4px 0 10px 0;font-family:monospace">
      {_card("Stability",          _stab_txt,            _stab_col)}
      {_card("Steady-State Error", f"{_ss_err:.4f}",     _sse_col)}
      {_card("Overshoot",          f"{_overshoot:.1f}%", _os_col)}
      {_card("Rise Time (10→90%)", _rise_txt,            _MAUVE)}
      {_card("Settling Time (±2%)",_settl_txt,           _TEAL)}
    </div>
    """)

    mo.vstack([_cards, mo.as_html(_fig)])
    return