import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full", app_title="Audio Effects Lab")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import scipy.signal as signal
    import scipy.io.wavfile as wavfile
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import io
    import soundfile as sf
    import warnings
    warnings.filterwarnings("ignore")

    # Optional: noisereduce (pip install noisereduce)
    try:
        import noisereduce as nr
        HAS_NR = True
    except ImportError:
        HAS_NR = False

    mo.md("""
    # 🎛️ Audio Effects Lab
    **FFT + DSP Signal Processing Notebook**

    ---
    Upload a `.wav` or `.flac` file, tune parameters with the sliders,
    and watch the FFT + spectrogram update reactively.

    > **Requirements:** `pip install marimo numpy scipy matplotlib soundfile noisereduce`
    """)
    return HAS_NR, io, mo, np, nr, sf, signal, wavfile, gridspec, plt, warnings


@app.cell
def __(mo):
    # ── File Upload ──────────────────────────────────────────────────────────
    file_upload = mo.ui.file(
        accept=[".wav", ".flac", ".ogg"],
        label="📂 Upload Audio File (.wav / .flac / .ogg)",
        multiple=False,
    )
    file_upload
    return (file_upload,)


@app.cell
def __(file_upload, io, mo, np, sf):
    # ── Load Audio ───────────────────────────────────────────────────────────
    audio_data = None
    sample_rate = None
    load_status = ""

    if file_upload.value:
        try:
            raw_bytes = file_upload.value[0].contents
            buf = io.BytesIO(raw_bytes)
            audio_np, sr = sf.read(buf, dtype="float32")

            # Mono-ify if stereo
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            # Trim to max 30 seconds to keep processing fast
            MAX_SAMPLES = 30 * sr
            if len(audio_np) > MAX_SAMPLES:
                audio_np = audio_np[:MAX_SAMPLES]

            audio_data = audio_np
            sample_rate = sr
            duration = len(audio_data) / sample_rate
            load_status = f"✅ Loaded **{file_upload.value[0].name}** — {sr} Hz · {duration:.2f}s · {len(audio_data):,} samples (mono)"
        except Exception as e:
            load_status = f"❌ Error loading file: {e}"
    else:
        # Generate a demo signal if no file uploaded
        sr = 44100
        t = np.linspace(0, 3, sr * 3, endpoint=False)
        demo = (
            0.5 * np.sin(2 * np.pi * 440 * t)     # A4
            + 0.3 * np.sin(2 * np.pi * 880 * t)   # A5
            + 0.15 * np.sin(2 * np.pi * 3000 * t) # 3 kHz
            + 0.05 * np.random.randn(len(t))       # noise
        ).astype(np.float32)
        audio_data = demo
        sample_rate = sr
        load_status = "🎵 No file uploaded — using **demo signal** (440 Hz + 880 Hz + 3 kHz + noise)"

    mo.md(load_status)
    return MAX_SAMPLES, audio_data, duration, load_status, raw_bytes, sample_rate, sr


@app.cell
def __(mo):
    # ── Filter Controls ──────────────────────────────────────────────────────
    mo.md("## 🎚️ Filter Controls")
    return


@app.cell
def __(mo):
    filter_type = mo.ui.dropdown(
        options=["none", "lowpass", "highpass", "bandpass"],
        value="none",
        label="Filter Type",
    )
    cutoff_low = mo.ui.slider(
        start=20, stop=20000, step=10, value=1000,
        label="Cutoff / Low-cut frequency (Hz)",
        show_value=True,
    )
    cutoff_high = mo.ui.slider(
        start=20, stop=20000, step=10, value=5000,
        label="High-cut frequency (Hz) [bandpass only]",
        show_value=True,
    )
    filter_order = mo.ui.slider(
        start=2, stop=10, step=2, value=4,
        label="Filter Order",
        show_value=True,
    )
    mo.hstack([filter_type, filter_order], justify="start", gap=2)
    return cutoff_high, cutoff_low, filter_order, filter_type


@app.cell
def __(cutoff_high, cutoff_low, mo):
    mo.hstack([cutoff_low, cutoff_high], justify="start", gap=2)
    return


@app.cell
def __(mo):
    # ── Echo / Reverb Controls ───────────────────────────────────────────────
    mo.md("## 🌊 Echo & Reverb Controls")
    return


@app.cell
def __(mo):
    enable_echo = mo.ui.switch(label="Enable Echo", value=False)
    echo_delay_ms = mo.ui.slider(
        start=50, stop=1000, step=10, value=300,
        label="Echo Delay (ms)",
        show_value=True,
    )
    echo_decay = mo.ui.slider(
        start=0.1, stop=0.9, step=0.05, value=0.4,
        label="Echo Decay (0–1)",
        show_value=True,
    )
    enable_reverb = mo.ui.switch(label="Enable Reverb", value=False)
    reverb_wet = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.3,
        label="Reverb Wet Mix (0–1)",
        show_value=True,
    )
    mo.hstack([enable_echo, echo_delay_ms, echo_decay], justify="start", gap=2)
    return echo_decay, echo_delay_ms, enable_echo, enable_reverb, reverb_wet


@app.cell
def __(enable_reverb, mo, reverb_wet):
    mo.hstack([enable_reverb, reverb_wet], justify="start", gap=2)
    return


@app.cell
def __(HAS_NR, mo):
    # ── Noise Removal Controls ───────────────────────────────────────────────
    mo.md("## 🔇 Noise Removal Controls")
    enable_nr = mo.ui.switch(
        label=f"Enable Noise Reduction {'✅' if HAS_NR else '(install noisereduce)'}",
        value=False,
    )
    nr_strength = mo.ui.slider(
        start=0.5, stop=3.0, step=0.1, value=1.5,
        label="NR Strength",
        show_value=True,
    )
    mo.hstack([enable_nr, nr_strength], justify="start", gap=2)
    return enable_nr, nr_strength


@app.cell
def __(
    HAS_NR,
    audio_data,
    cutoff_high,
    cutoff_low,
    echo_decay,
    echo_delay_ms,
    enable_echo,
    enable_nr,
    enable_reverb,
    filter_order,
    filter_type,
    mo,
    np,
    nr,
    nr_strength,
    reverb_wet,
    sample_rate,
    signal,
):
    # ── DSP Processing Pipeline ──────────────────────────────────────────────
    def apply_filter(data, sr, ftype, fc_low, fc_high, order):
        nyq = sr / 2.0
        if ftype == "lowpass":
            sos = signal.butter(order, fc_low / nyq, btype="low", output="sos")
            return signal.sosfilt(sos, data)
        elif ftype == "highpass":
            sos = signal.butter(order, fc_low / nyq, btype="high", output="sos")
            return signal.sosfilt(sos, data)
        elif ftype == "bandpass":
            lo = min(fc_low, fc_high) / nyq
            hi = max(fc_low, fc_high) / nyq
            lo = np.clip(lo, 1e-4, 0.9999)
            hi = np.clip(hi, 1e-4, 0.9999)
            if lo >= hi:
                return data
            sos = signal.butter(order, [lo, hi], btype="band", output="sos")
            return signal.sosfilt(sos, data)
        return data

    def apply_echo(data, sr, delay_ms, decay):
        delay_samples = int(sr * delay_ms / 1000)
        out = data.copy()
        if delay_samples < len(out):
            out[delay_samples:] += decay * data[:-delay_samples]
        return out

    def apply_reverb(data, sr, wet):
        # Simple FDN reverb via multiple comb filters
        delays = [int(sr * d) for d in [0.029, 0.037, 0.041, 0.043]]
        decays = [0.85, 0.82, 0.80, 0.78]
        out = np.zeros_like(data)
        for d, dc in zip(delays, decays):
            tmp = np.zeros(len(data) + d)
            tmp[:len(data)] += data
            for i in range(d, len(tmp)):
                tmp[i] += dc * tmp[i - d]
            out += tmp[:len(data)]
        out /= len(delays)
        return (1 - wet) * data + wet * out

    def make_impulse_response(sr, duration=0.5):
        n = int(sr * duration)
        t = np.linspace(0, duration, n)
        ir = np.random.randn(n) * np.exp(-6 * t)
        return ir / np.max(np.abs(ir))

    processed = audio_data.copy()

    # 1. Filter
    if filter_type.value != "none":
        processed = apply_filter(
            processed, sample_rate,
            filter_type.value,
            cutoff_low.value, cutoff_high.value,
            filter_order.value,
        )

    # 2. Echo
    if enable_echo.value:
        processed = apply_echo(processed, sample_rate, echo_delay_ms.value, echo_decay.value)

    # 3. Reverb
    if enable_reverb.value:
        processed = apply_reverb(processed, sample_rate, reverb_wet.value)

    # 4. Noise Reduction
    if enable_nr.value and HAS_NR:
        processed = nr.reduce_noise(
            y=processed,
            sr=sample_rate,
            prop_decrease=float(np.clip(nr_strength.value / 3.0, 0, 1)),
        )

    # Normalize
    peak = np.max(np.abs(processed))
    if peak > 0:
        processed = processed / peak * 0.9

    mo.md("### ✅ DSP pipeline applied")
    return (
        apply_echo,
        apply_filter,
        apply_reverb,
        make_impulse_response,
        peak,
        processed,
    )


@app.cell
def __(audio_data, mo, np, plt, processed, sample_rate):
    # ── FFT Before / After ───────────────────────────────────────────────────
    def compute_fft(data, sr):
        N = len(data)
        window = np.hanning(N)
        yf = np.fft.rfft(data * window)
        xf = np.fft.rfftfreq(N, 1 / sr)
        mag_db = 20 * np.log10(np.abs(yf) / N + 1e-10)
        return xf, mag_db

    xf_orig, mag_orig = compute_fft(audio_data, sample_rate)
    xf_proc, mag_proc = compute_fft(processed, sample_rate)

    fig_fft, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0d0d0d")
    fig_fft.suptitle("FFT Magnitude Spectrum", color="#e0e0e0", fontsize=13, fontweight="bold")

    for ax, xf, mag, label, color in [
        (axes[0], xf_orig, mag_orig, "Original", "#00d4ff"),
        (axes[1], xf_proc, mag_proc, "Processed", "#ff6b35"),
    ]:
        ax.set_facecolor("#111111")
        ax.plot(xf, mag, color=color, linewidth=0.8, alpha=0.9)
        ax.fill_between(xf, mag, mag.min(), alpha=0.15, color=color)
        ax.set_xscale("log")
        ax.set_xlim(20, sample_rate / 2)
        ax.set_ylim(-100, 0)
        ax.set_xlabel("Frequency (Hz)", color="#aaaaaa")
        ax.set_ylabel("Magnitude (dB)", color="#aaaaaa")
        ax.set_title(label, color=color, fontsize=11)
        ax.tick_params(colors="#888888")
        ax.grid(True, color="#222222", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    plt.tight_layout()
    mo.pyplot(fig_fft)
    return axes, compute_fft, fig_fft, mag_orig, mag_proc, xf_orig, xf_proc


@app.cell
def __(audio_data, mo, np, plt, processed, sample_rate):
    # ── Spectrogram Before / After ───────────────────────────────────────────
    NFFT = 1024
    noverlap = 768

    fig_spec, sp_axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0d0d0d")
    fig_spec.suptitle("Spectrogram", color="#e0e0e0", fontsize=13, fontweight="bold")

    for ax, data, label in [
        (sp_axes[0], audio_data, "Original"),
        (sp_axes[1], processed, "Processed"),
    ]:
        ax.set_facecolor("#0a0a0a")
        Pxx, freqs, bins, im = ax.specgram(
            data,
            NFFT=NFFT,
            Fs=sample_rate,
            noverlap=noverlap,
            cmap="inferno",
            scale="dB",
        )
        ax.set_xlabel("Time (s)", color="#aaaaaa")
        ax.set_ylabel("Frequency (Hz)", color="#aaaaaa")
        ax.set_title(label, color="#e0e0e0", fontsize=11)
        ax.tick_params(colors="#888888")
        ax.set_ylim(0, min(sample_rate / 2, 12000))
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        fig_spec.colorbar(im, ax=ax, label="dB").ax.yaxis.set_tick_params(color="#888888")

    plt.tight_layout()
    mo.pyplot(fig_spec)
    return NFFT, Pxx, bins, fig_spec, freqs, im, noverlap, sp_axes


@app.cell
def __(audio_data, mo, np, plt, processed, sample_rate):
    # ── Waveform Before / After ──────────────────────────────────────────────
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    time_axis_proc = np.linspace(0, len(processed) / sample_rate, len(processed))

    fig_wave, wave_axes = plt.subplots(2, 1, figsize=(14, 4), facecolor="#0d0d0d", sharex=False)
    fig_wave.suptitle("Waveform", color="#e0e0e0", fontsize=13, fontweight="bold")

    for ax, t, data, label, color in [
        (wave_axes[0], time_axis, audio_data, "Original", "#00d4ff"),
        (wave_axes[1], time_axis_proc, processed, "Processed", "#ff6b35"),
    ]:
        ax.set_facecolor("#111111")
        ax.plot(t, data, color=color, linewidth=0.4, alpha=0.85)
        ax.set_ylabel("Amplitude", color="#aaaaaa", fontsize=8)
        ax.set_title(label, color=color, fontsize=10, pad=3)
        ax.tick_params(colors="#888888", labelsize=7)
        ax.grid(True, color="#1e1e1e", linewidth=0.4)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-1.05, 1.05)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")

    wave_axes[1].set_xlabel("Time (s)", color="#aaaaaa", fontsize=8)
    plt.tight_layout()
    mo.pyplot(fig_wave)
    return fig_wave, time_axis, time_axis_proc, wave_axes


@app.cell
def __(io, mo, processed, sample_rate, sf):
    # ── Audio Playback ───────────────────────────────────────────────────────
    mo.md("## 🔊 Playback")

    def audio_to_bytes(data, sr):
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    processed_bytes = audio_to_bytes(processed, sample_rate)
    mo.audio(src=processed_bytes)
    return audio_to_bytes, processed_bytes


@app.cell
def __(mo, processed_bytes):
    # ── Download Button ──────────────────────────────────────────────────────
    mo.download(
        data=processed_bytes,
        filename="processed_audio.wav",
        label="⬇️ Download Processed Audio",
        mimetype="audio/wav",
    )
    return


@app.cell
def __(HAS_NR, mo):
    # ── Help / Info Panel ────────────────────────────────────────────────────
    nr_status = "✅ installed" if HAS_NR else "❌ not installed — `pip install noisereduce`"
    mo.callout(
        mo.md(f"""
    ### 📖 Quick Reference

    | Effect | How it works |
    |---|---|
    | **Lowpass** | Butterworth IIR — keeps frequencies below cutoff |
    | **Highpass** | Butterworth IIR — keeps frequencies above cutoff |
    | **Bandpass** | Keeps a band between low-cut and high-cut |
    | **Echo** | Adds a delayed + attenuated copy of the signal |
    | **Reverb** | FDN (feedback delay network) with 4 comb filters |
    | **Noise Reduction** | Spectral gating via `noisereduce` ({nr_status}) |

    **Tips:**
    - Use **log-scale FFT** to spot filter rolloff clearly
    - Echo delay of **150–400 ms** sounds most natural
    - Bandpass: set low-cut < high-cut (they can be swapped safely)
    - Max clip: **30 seconds** of audio to keep processing fast
    """),
        kind="info",
    )
    return (nr_status,)


if __name__ == "__main__":
    app.run()