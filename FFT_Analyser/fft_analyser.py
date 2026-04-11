import marimo as mo
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

app = mo.App(width="full")


@app.cell
def __():
    mo.md(
        r"""
# Interactive FFT Analyzer

A ready-to-use marimo notebook for exploring how signals look in time and frequency domains.

This template is built to show what marimo does well:
- reactive controls
- clean notebook state
- linked plots and derived computations
- quick experimentation with signal parameters

Use it as a base for teaching, demos, or your open-source template library.
"""
    )
    return


@app.cell
def __():
    import marimo as mo

    # Core controls
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

    fs = mo.ui.slider(500, 20000, value=4000, step=100, label="Sampling rate (Hz)")
    duration = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Duration (s)")
    f1 = mo.ui.slider(1, 2000, value=120, step=1, label="Tone 1 (Hz)")
    f2 = mo.ui.slider(1, 2000, value=360, step=1, label="Tone 2 (Hz)")
    amplitude = mo.ui.slider(0.0, 2.0, value=1.0, step=0.05, label="Amplitude")
    noise = mo.ui.slider(0.0, 1.0, value=0.15, step=0.01, label="Noise level")
    phase = mo.ui.slider(0.0, 2 * np.pi, value=0.0, step=0.01, label="Phase shift (rad)")
    chirp_end = mo.ui.slider(10, 5000, value=1500, step=10, label="Chirp end frequency (Hz)")
    am_mod = mo.ui.slider(1, 200, value=8, step=1, label="AM mod frequency (Hz)")
    window_name = mo.ui.dropdown(
        options=["rectangular", "hann", "hamming", "blackman"],
        value="hann",
        label="FFT window",
    )
    n_fft = mo.ui.slider(256, 16384, value=4096, step=256, label="FFT points")

    mo.md(
        f"""
**Controls**\n\n{signal_type}\n\n{fs}\n\n{duration}\n\n{f1}\n\n{f2}\n\n{amplitude}\n\n{noise}\n\n{phase}\n\n{chirp_end}\n\n{am_mod}\n\n{window_name}\n\n{n_fft}
"""
    )
    return (
        am_mod,
        amplitude,
        chirp_end,
        duration,
        f1,
        f2,
        fs,
        n_fft,
        noise,
        phase,
        signal_type,
        window_name,
    )


@app.cell
def __(am_mod, amplitude, chirp_end, duration, f1, f2, fs, noise, phase, signal_type):
    t = np.arange(0, float(duration), 1.0 / float(fs))

    if signal_type == "Single tone":
        x = amplitude * np.sin(2 * np.pi * float(f1) * t + float(phase))
    elif signal_type == "Two-tone":
        x = amplitude * (
            0.7 * np.sin(2 * np.pi * float(f1) * t + float(phase))
            + 0.4 * np.sin(2 * np.pi * float(f2) * t)
        )
    elif signal_type == "Chirp":
        x = amplitude * signal.chirp(t, f0=float(f1), f1=float(chirp_end), t1=float(duration), method="linear")
    elif signal_type == "Square wave":
        x = amplitude * signal.square(2 * np.pi * float(f1) * t + float(phase))
    elif signal_type == "Noisy sine":
        rng = np.random.default_rng(7)
        x = amplitude * np.sin(2 * np.pi * float(f1) * t + float(phase)) + float(noise) * rng.standard_normal(len(t))
    elif signal_type == "AM signal":
        carrier = np.sin(2 * np.pi * float(f2) * t)
        envelope = 1.0 + 0.65 * np.sin(2 * np.pi * float(am_mod) * t)
        x = amplitude * envelope * carrier
    else:
        x = amplitude * np.sin(2 * np.pi * float(f1) * t + float(phase))

    if signal_type not in {"Noisy sine"}:
        rng = np.random.default_rng(7)
        x = x + float(noise) * 0.15 * rng.standard_normal(len(t))

    x = np.asarray(x, dtype=float)
    return t, x


@app.cell
def __(window_name, x, fs, n_fft):
    n = len(x)
    fft_len = int(min(max(256, int(n_fft)), max(256, n)))

    if window_name == "rectangular":
        w = np.ones(n)
    elif window_name == "hann":
        w = signal.windows.hann(n, sym=False)
    elif window_name == "hamming":
        w = signal.windows.hamming(n, sym=False)
    else:
        w = signal.windows.blackman(n, sym=False)

    xw = x * w
    X = np.fft.rfft(xw, n=fft_len)
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / float(fs))
    magnitude = np.abs(X) / max(1, n)
    power = magnitude**2
    phase_spec = np.unwrap(np.angle(X))

    # Dominant peaks for display
    peak_indices = np.argsort(magnitude)[-5:][::-1]
    peak_table = [
        (float(freqs[i]), float(magnitude[i]), float(power[i]))
        for i in peak_indices
        if freqs[i] > 0
    ]

    return fft_len, freqs, magnitude, peak_table, phase_spec, power, w, xw


@app.cell
def __(freqs, magnitude, peak_table, phase_spec, power, t, x):
    import marimo as mo

    fig1, ax1 = plt.subplots(figsize=(10, 3.8))
    ax1.plot(t, x)
    ax1.set_title("Time-domain signal")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(10, 3.8))
    ax2.plot(freqs, magnitude)
    ax2.set_title("FFT magnitude")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(freqs[-1], 0.5 * float(freqs[-1]) + 1))

    fig3, ax3 = plt.subplots(figsize=(10, 3.8))
    ax3.plot(freqs, power)
    ax3.set_title("Power spectrum")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Power")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, min(freqs[-1], 0.5 * float(freqs[-1]) + 1))

    fig4, ax4 = plt.subplots(figsize=(10, 3.8))
    ax4.plot(freqs, phase_spec)
    ax4.set_title("Phase spectrum")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Phase (rad)")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, min(freqs[-1], 0.5 * float(freqs[-1]) + 1))

    peak_rows = [
        {
            "Frequency (Hz)": round(f, 2),
            "Magnitude": round(m, 5),
            "Power": round(p, 5),
        }
        for f, m, p in peak_table
    ]
    peak_table_ui = mo.ui.table(
        data=peak_rows,
        label="Dominant frequency bins",
    )

    if peak_table:
        dominant_f, dominant_m, _ = peak_table[0]
        dominant_frequency_stat = mo.stat(
            label="Dominant frequency",
            value=f"{dominant_f:.2f} Hz",
            caption=f"Magnitude: {dominant_m:.5f}",
        )
    else:
        dominant_frequency_stat = mo.stat(
            label="Dominant frequency",
            value="N/A",
            caption="No non-DC peaks found",
        )

    return (
        dominant_frequency_stat,
        fig1,
        fig2,
        fig3,
        fig4,
        peak_table_ui,
    )


@app.cell
def __(dominant_frequency_stat, fig1, fig2, fig3, fig4, peak_table_ui):
    import marimo as mo

    mo.vstack(
        [
            mo.md("## Views"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            dominant_frequency_stat,
                            peak_table_ui,
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.image(fig1),
                            mo.image(fig2),
                        ]
                    ),
                ],
                widths=[1, 2],
                align="start",
            ),
            mo.hstack(
                [
                    mo.image(fig3),
                    mo.image(fig4),
                ],
                widths=[1, 1],
                align="start",
            ),
        ]
    )
    return


@app.cell
def __(fs, t, x):
    import marimo as mo

    fig, ax = plt.subplots(figsize=(10, 3.8))
    nperseg = min(256, max(32, len(x) // 8))
    noverlap = nperseg // 2
    f, tt, Sxx = signal.spectrogram(x, fs=float(fs), nperseg=nperseg, noverlap=noverlap, scaling="density")
    mesh = ax.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), shading="auto")
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=ax, label="Power (dB)")
    return fig


@app.cell
def __(fig):
    import marimo as mo
    mo.md("## Spectrogram")
    mo.image(fig)
    return


@app.cell
def __():
    import marimo as mo

    mo.md(
        r"""
---

## How to use this template

1. Start with a preset signal.
2. Tweak frequency, sampling rate, amplitude, noise, and windowing.
3. Watch the time plot, spectrum, power, phase, and spectrogram update together.
4. Replace the synthetic signal cell with your own data source later, such as CSV, WAV, or sensor streams.

This makes a strong reusable starting point for a marimo-based signal processing lab notebook.
"""
    )
    return


if __name__ == "__main__":
    app.run()
