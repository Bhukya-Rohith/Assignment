"""
Asset generator & downloader script:
1. Downloads curated real estate photographs from Unsplash.
2. Synthesizes ambient royalty-free soundtrack MP3 files.
"""
import io
import math
import os
import subprocess
import wave
from pathlib import Path
import urllib.request
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).parent.parent.resolve()
ASSETS_AUDIO_DIR = BASE_DIR / "assets" / "audio"
SAMPLES_DIR = BASE_DIR / "samples"
VILLA_DIR = SAMPLES_DIR / "images" / "villa"
PENTHOUSE_DIR = SAMPLES_DIR / "images" / "penthouse"
GENERAL_IMAGES_DIR = SAMPLES_DIR / "images"

# High-resolution architectural photography IDs
VILLA_URLS = {
    "01_exterior.jpg": "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=1920&q=85",
    "02_living_room.jpg": "https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?w=1920&q=85",
    "03_kitchen.jpg": "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=1920&q=85",
    "04_master_bedroom.jpg": "https://images.unsplash.com/photo-1598928506311-c55ded91a20c?w=1920&q=85",
    "05_pool_terrace.jpg": "https://images.unsplash.com/photo-1580587771525-78b9dba3b914?w=1920&q=85",
}

PENTHOUSE_URLS = {
    "01_exterior.jpg": "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?w=1920&q=85",
    "02_living_room.jpg": "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?w=1920&q=85",
    "03_terrace.jpg": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=1920&q=85",
    "04_fireplace.jpg": "https://images.unsplash.com/photo-1600565193348-f74bd3c7ccdf?w=1920&q=85",
    "05_wine_cellar.jpg": "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=1920&q=85",
}


def download_or_generate_image(url: str, dest_path: Path, label: str):
    """Downloads authentic photography; if offline, generates high-res architectural gradient card."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file() and dest_path.stat().st_size > 10000:
        return

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.save(dest_path, "JPEG", quality=90)
            print(f"Downloaded: {dest_path.name} ({len(data)} bytes)")
            return
    except Exception as exc:
        print(f"Network download failed for {dest_path.name} ({exc}). Generating fallback visual...")

    # Fallback: create architectural rendered card
    w, h = 1920, 1080
    grad = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        r = int(25 + 40 * (y / h))
        g = int(35 + 50 * (y / h))
        b = int(55 + 75 * (y / h))
        grad[y, :] = [r, g, b]
    img = Image.fromarray(grad)
    img.save(dest_path, "JPEG", quality=90)


def synthesize_ambient_audio(output_mp3: Path, duration_sec: float = 40.0, acoustic_type: str = "luxury"):
    """
    Synthesizes smooth, harmonic, broadcast-grade ambient music bed (44.1kHz stereo).
    - 'luxury': Rich warm minor/major ninth chords with gentle harmonics and slow breathing envelope.
    - 'acoustic': Gentle warm acoustic arpeggiated bells with soft decay.
    """
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    sr = 44100
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Chord progression: Cmaj9 -> Am9 -> Fmaj9 -> Gsus4 (each chord ~10 seconds)
    if acoustic_type == "luxury":
        chord_freqs = [
            [130.81, 164.81, 196.00, 246.94, 293.66],  # C3, E3, G3, B3, D4 (Cmaj9)
            [110.00, 130.81, 164.81, 196.00, 246.94],  # A2, C3, E3, G3, B3 (Am9)
            [87.31, 130.81, 174.61, 220.00, 261.63],   # F2, C3, F3, A3, C4 (Fmaj7)
            [98.00, 146.83, 196.00, 261.63, 293.66],   # G2, D3, G3, C4, D4 (Gsus4)
        ]
    else:
        chord_freqs = [
            [146.83, 185.00, 220.00, 277.18, 329.63],  # D3, F#3, A3, C#4, E4 (Dmaj9)
            [123.47, 146.83, 185.00, 220.00, 277.18],  # B2, D3, F#3, A3, C#4 (Bm9)
            [98.00, 146.83, 196.00, 246.94, 293.66],   # G2, D3, G3, B3, D4 (Gmaj7)
            [110.00, 164.81, 220.00, 293.66, 329.63],  # A2, E3, A3, D4, E4 (Asus4)
        ]

    chord_len = duration_sec / len(chord_freqs)
    left_channel = np.zeros(n_samples)
    right_channel = np.zeros(n_samples)

    for idx, freqs in enumerate(chord_freqs):
        start_idx = int(idx * chord_len * sr)
        end_idx = int((idx + 1) * chord_len * sr)
        segment_len = end_idx - start_idx
        t_seg = t[start_idx:end_idx]

        # Smooth envelope (fade-in, sustain, fade-out for each chord)
        attack_len = int(sr * 1.5)
        release_len = int(sr * 1.5)
        env = np.ones(segment_len)
        if segment_len > attack_len + release_len:
            env[:attack_len] = np.linspace(0, 1, attack_len)
            env[-release_len:] = np.linspace(1, 0, release_len)

        chord_signal_l = np.zeros(segment_len)
        chord_signal_r = np.zeros(segment_len)

        for f_idx, f in enumerate(freqs):
            # Fundamental + soft second harmonic + detuned stereo shimmer
            detune = 1.002
            s_l = np.sin(2 * np.pi * f * t_seg) * 0.4 + np.sin(2 * np.pi * 2 * f * t_seg) * 0.1
            s_r = np.sin(2 * np.pi * f * detune * t_seg) * 0.4 + np.sin(2 * np.pi * 2 * f * detune * t_seg) * 0.1
            weight = 1.0 / (f_idx + 1) ** 0.5
            chord_signal_l += s_l * weight
            chord_signal_r += s_r * weight

        left_channel[start_idx:end_idx] += chord_signal_l * env
        right_channel[start_idx:end_idx] += chord_signal_r * env

    # Master fade in and fade out
    master_fade = int(sr * 2.0)
    left_channel[:master_fade] *= np.linspace(0, 1, master_fade)
    right_channel[:master_fade] *= np.linspace(0, 1, master_fade)
    left_channel[-master_fade:] *= np.linspace(1, 0, master_fade)
    right_channel[-master_fade:] *= np.linspace(1, 0, master_fade)

    # Normalize to -3dB
    max_amp = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)), 1e-6)
    left_channel = (left_channel / max_amp) * 0.70
    right_channel = (right_channel / max_amp) * 0.70

    stereo_16 = np.column_stack((
        (left_channel * 32767).astype(np.int16),
        (right_channel * 32767).astype(np.int16),
    ))

    # Write temporary WAV file
    temp_wav = output_mp3.with_suffix(".wav")
    with wave.open(str(temp_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo_16.tobytes())

    # Convert to MP3 using imageio_ffmpeg
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(temp_wav),
        "-codec:a", "libmp3lame",
        "-b:a", "192k",
        str(output_mp3),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if temp_wav.exists():
        temp_wav.unlink()
    print(f"Generated soundtrack: {output_mp3} ({output_mp3.stat().st_size} bytes)")


def main():
    print("Setting up sample photographic assets...")
    for filename, url in VILLA_URLS.items():
        dest = VILLA_DIR / filename
        download_or_generate_image(url, dest, "Villa")
        # Also copy into general samples/images for single-directory CLI testing
        gen_dest = GENERAL_IMAGES_DIR / filename
        if not gen_dest.exists() and dest.exists():
            import shutil
            shutil.copy(dest, gen_dest)

    for filename, url in PENTHOUSE_URLS.items():
        dest = PENTHOUSE_DIR / filename
        download_or_generate_image(url, dest, "Penthouse")

    print("Synthesizing ambient background soundtracks...")
    synthesize_ambient_audio(ASSETS_AUDIO_DIR / "ambient_luxury.mp3", duration_sec=45.0, acoustic_type="luxury")
    synthesize_ambient_audio(ASSETS_AUDIO_DIR / "modern_acoustic.mp3", duration_sec=45.0, acoustic_type="acoustic")
    print("Asset setup completed successfully!")


if __name__ == "__main__":
    main()
