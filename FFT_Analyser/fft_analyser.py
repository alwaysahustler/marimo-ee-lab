"""
Interactive FFT Analyzer — marimo template
==========================================
A reactive notebook for exploring time-domain signals and their
frequency-domain representations. Designed as a reusable teaching
and experimentation template.

Run with:
    marimo edit fft_analyzer.py
    marimo run  fft_analyzer.py   # read-only, shareable
"""

import marimo as mo

app = mo.App(width="full")


# ---------------------------------------------------------------------------
# 1. SHARED IMPORTS
#    Return every third-party name used by downstream cells so marimo can
#    wire the reactive DAG correctly.
# ---------------------------------------------------------------------------
@app.cell
def _imports():
    import numpy as np
    from scipy import signal
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend — required in marimo
    import matplotlib.pyplot as plt

    return np, signal, plt


# ---------------------------------------------------------------------------
# 2. INTRO
# ---------------------------------------------------------------------------
@app.cell
def _intro():
    mo.md(
        r"""
# 📡 Interactive FFT Analyzer

A ready-to-use marimo notebook for exploring how signals look in the
**time domain** and the **frequency domain**.

**What makes this a good marimo template:**
- Every control is reactive — change a slider, all plots update instantly.
- The cell DAG makes data-flow explicit and auditable.
- No hidden global state; replace the signal-generation cell with your
  own CSV / WAV / sensor source and everything else stays intact.

---
"""
    )


# ---------------------------------------------------------------------------
# 3. CONTROLS
#    All UI elements live in one cell so they are easy to find and extend.
# ---------------------------------------------------------------------------
@app.cell
def _controls():
    signal_type = mo.ui.dropdown(
        options=[
            "Single tone",
            "Two-tone",
            "Chirp",
            "Square wave",
            "Noisy sine",
            "AM signal",
        ],
        value="Two-tone",
        label="Signal preset",
    )
    fs = mo.ui.slider(500, 20_000, value=4_000, step=100, label="Sampling rate (Hz)")
    duration = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Duration (s)")
    f1 = mo.ui.slider(1, 2_000, value=120, step=1, label="Tone 1 (Hz)")
    f2 = mo.ui.slider(1, 2_000, value=360, step=1, label="Tone 2 (Hz)")
    amplitude = mo.ui.slider(0.0, 2.0, value=1.0, step=0.05, label="Amplitude")
    noise = mo.ui.slider(0.0, 1.0, value=0.15, step=0.01, label="Noise level")
    phase = mo.ui.slider(
        0.0, 6.28, value=0.0, step=0.01, label="Phase shift (rad)"
    )
    chirp_end = mo.ui.slider(
        10, 5_000, value=1_500, step=10, label="Chirp end frequency (Hz)"
    )
    am_mod = mo.ui.slider(1, 200, value=8, step=1, label="AM modulation freq (Hz)")
    window_name = mo.ui.dropdown(
        options=["rectangular", "hann", "hamming", "blackman"],
        value="hann",
        label="FFT window",
    )
    n_fft = mo.ui.slider(256, 16_384, value=4_096, step=256, label="FFT points")

    mo.md(
        f"""
### ⚙️ Controls

{mo.hstack([
    mo.vstack([signal_type, fs, duration, f1, f2]),
    mo.vstack([amplitude, noise, phase, chirp_end, am_mod]),
    mo.vstack([window_name, n_fft]),
], align="start")}
"""
    )

    return (
        signal_type, fs, duration, f1, f2,
        amplitude, noise, phase, chirp_end, am_mod,
        window_name, n_fft,
    )


# ---------------------------------------------------------------------------
# 4. SIGNAL GENERATION
#    Uses .value on every UI element — a marimo element object is NOT its
#    value; you must unwrap it explicitly.
# ---------------------------------------------------------------------------
@app.cell
def _generate(
    np, signal,
    signal_type, fs, duration, f1, f2,
    amplitude, noise, phase, chirp_end, am_mod,
):
    # Unwrap UI element values
    fs_v       = float(fs.value)
    dur_v      = float(duration.value)
    f1_v       = float(f1.value)
    f2_v       = float(f2.value)
    amp_v      = float(amplitude.value)
    noise_v    = float(noise.value)
    phase_v    = float(phase.value)
    chirp_v    = float(chirp_end.value)
    am_mod_v   = float(am_mod.value)
    sig_type   = signal_type.value   # string

    t = np.arange(0, dur_v, 1.0 / fs_v)

    if sig_type == "Single tone":
        x = amp_v * np.sin(2 * np.pi * f1_v * t + phase_v)

    elif sig_type == "Two-tone":
        x = amp_v * (
            0.7 * np.sin(2 * np.pi * f1_v * t + phase_v)
            + 0.4 * np.sin(2 * np.pi * f2_v * t)
        )

    elif sig_type == "Chirp":
        x = amp_v * signal.chirp(
            t, f0=f1_v, f1=chirp_v, t1=dur_v, method="linear"
        )

    elif sig_type == "Square wave":
        x = amp_v * signal.square(2 * np.pi * f1_v * t + phase_v)

    elif sig_type == "Noisy sine":
        rng = np.random.default_rng(7)
        x = (
            amp_v * np.sin(2 * np.pi * f1_v * t + phase_v)
            + noise_v * rng.standard_normal(len(t))
        )

    elif sig_type == "AM signal":
        carrier  = np.sin(2 * np.pi * f2_v * t)
        envelope = 1.0 + 0.65 * np.sin(2 * np.pi * am_mod_v * t)
        x = amp_v * envelope * carrier

    else:
        x = amp_v * np.sin(2 * np.pi * f1_v * t + phase_v)

    # Add a small background noise to all non-noisy presets so the
    # spectrogram is more interesting to look at.
    if sig_type != "Noisy sine":
        rng = np.random.default_rng(7)
        x = x + noise_v * 0.15 * rng.standard_normal(len(t))

    x = np.asarray(x, dtype=float)
    return t, x


# ---------------------------------------------------------------------------
# 5. FFT COMPUTATION
# ---------------------------------------------------------------------------
@app.cell
def _fft(np, signal, x, fs, n_fft, window_name):
    n       = len(x)
    fft_len = int(min(max(256, int(n_fft.value)), max(256, n)))
    win_nm  = window_name.value   # string, not element

    # Build window
    if win_nm == "rectangular":
        w = np.ones(n)
    elif win_nm == "hann":
        w = signal.windows.hann(n, sym=False)
    elif win_nm == "hamming":
        w = signal.windows.hamming(n, sym=False)
    else:  # blackman
        w = signal.windows.blackman(n, sym=False)

    xw        = x * w
    X         = np.fft.rfft(xw, n=fft_len)
    freqs     = np.fft.rfftfreq(fft_len, d=1.0 / float(fs.value))
    magnitude = np.abs(X) / max(1, n)
    power     = magnitude ** 2
    phase_spec = np.unwrap(np.angle(X))

    # Top-5 dominant peaks (excluding DC)
    dc_mask      = freqs > 0
    mag_masked   = np.where(dc_mask, magnitude, 0.0)
    peak_indices = np.argsort(mag_masked)[-5:][::-1]
    peak_table   = [
        (float(freqs[i]), float(magnitude[i]), float(power[i]))
        for i in peak_indices
        if freqs[i] > 0
    ]

    return fft_len, freqs, magnitude, power, phase_spec, peak_table


# ---------------------------------------------------------------------------
# 6. PLOTS
# ---------------------------------------------------------------------------
@app.cell
def _plots(plt, t, x, freqs, magnitude, power, phase_spec):
    fmax = freqs[-1]

    def _fig(title, xlabel, ylabel, xs, ys, xlim=None):
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(xs, ys, linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if xlim:
            ax.set_xlim(*xlim)
        fig.tight_layout()
        return fig

    fig_time  = _fig("Time-domain signal",  "Time (s)",       "Amplitude", t,     x)
    fig_mag   = _fig("FFT magnitude",       "Frequency (Hz)", "Magnitude", freqs, magnitude,  xlim=(0, fmax))
    fig_power = _fig("Power spectrum",      "Frequency (Hz)", "Power",     freqs, power,       xlim=(0, fmax))
    fig_phase = _fig("Phase spectrum",      "Frequency (Hz)", "Phase (rad)", freqs, phase_spec, xlim=(0, fmax))

    return fig_time, fig_mag, fig_power, fig_phase


# ---------------------------------------------------------------------------
# 7. STATS + TABLE
# ---------------------------------------------------------------------------
@app.cell
def _stats(peak_table):
    peak_rows = [
        {
            "Frequency (Hz)": round(f, 2),
            "Magnitude":      round(m, 5),
            "Power":          round(p, 5),
        }
        for f, m, p in peak_table
    ]
    peak_table_ui = mo.ui.table(data=peak_rows, label="Dominant frequency bins")

    if peak_table:
        dom_f, dom_m, _ = peak_table[0]
        dominant_stat = mo.stat(
            label="Dominant frequency",
            value=f"{dom_f:.2f} Hz",
            caption=f"Magnitude: {dom_m:.5f}",
        )
    else:
        dominant_stat = mo.stat(
            label="Dominant frequency",
            value="N/A",
            caption="No non-DC peaks found",
        )

    return dominant_stat, peak_table_ui


# ---------------------------------------------------------------------------
# 8. LAYOUT
# ---------------------------------------------------------------------------
@app.cell
def _layout(dominant_stat, peak_table_ui, fig_time, fig_mag, fig_power, fig_phase):
    mo.vstack([
        mo.md("## 📊 Views"),
        mo.hstack(
            [
                mo.vstack([dominant_stat, peak_table_ui]),
                mo.vstack([mo.image(fig_time), mo.image(fig_mag)]),
            ],
            widths=[1, 2],
            align="start",
        ),
        mo.hstack(
            [mo.image(fig_power), mo.image(fig_phase)],
            align="start",
        ),
    ])


# ---------------------------------------------------------------------------
# 9. SPECTROGRAM  (named fig_spec to avoid collision with plot figs above)
# ---------------------------------------------------------------------------
@app.cell
def _spectrogram(np, plt, signal, x, fs):
    fs_v     = float(fs.value)
    nperseg  = min(256, max(32, len(x) // 8))
    noverlap = nperseg // 2

    f_sg, t_sg, Sxx = signal.spectrogram(
        x, fs=fs_v, nperseg=nperseg, noverlap=noverlap, scaling="density"
    )

    fig_spec, ax_sg = plt.subplots(figsize=(10, 3.8))
    mesh = ax_sg.pcolormesh(
        t_sg, f_sg, 10 * np.log10(Sxx + 1e-12), shading="auto"
    )
    ax_sg.set_title("Spectrogram")
    ax_sg.set_xlabel("Time (s)")
    ax_sg.set_ylabel("Frequency (Hz)")
    fig_spec.colorbar(mesh, ax=ax_sg, label="Power (dB)")
    fig_spec.tight_layout()

    return fig_spec


# ---------------------------------------------------------------------------
# 10. SPECTROGRAM DISPLAY
# ---------------------------------------------------------------------------
@app.cell
def _spectrogram_display(fig_spec):
    mo.vstack([
        mo.md("## 🌈 Spectrogram"),
        mo.image(fig_spec),
    ])


# ---------------------------------------------------------------------------
# 11. USAGE GUIDE
# ---------------------------------------------------------------------------
@app.cell
def _guide():
    mo.md(
        r"""
---

## How to use this template

| Step | What to do |
|------|-----------|
| 1 | Pick a **signal preset** — each highlights different FFT concepts. |
| 2 | Tweak **frequency, amplitude, noise, phase** and watch every plot update reactively. |
| 3 | Change the **FFT window** to see spectral leakage effects in real time. |
| 4 | Increase **FFT points** for finer frequency resolution. |
| 5 | Replace the signal-generation cell with your own data source (CSV, WAV, sensor stream) — nothing else needs to change. |

### marimo patterns demonstrated here

- **One imports cell** — return shared libraries (`np`, `signal`, `plt`) so the reactive DAG is explicit.
- **`.value` unwrapping** — always use `slider.value` / `dropdown.value` inside computation cells; comparing the element object to a plain string always evaluates `False`.
- **Named cell functions** — `def _imports():`, `def _fft():` instead of anonymous `def __():` makes the DAG readable in marimo's graph view.
- **`matplotlib.use("Agg")`** — set the non-interactive backend once at import time; marimo displays figures via `mo.image(fig)`.
"""
    )


if __name__ == "__main__":
    app.run()