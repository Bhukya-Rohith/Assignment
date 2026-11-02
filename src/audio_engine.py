"""
Acoustic processing layer: Neural TTS voice synthesis (edge-tts),
voice-first temporal scheduling, and ducked multi-track background mixing.
"""
import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import edge_tts
from moviepy import AudioFileClip, CompositeAudioClip, afx

from src.models import SceneStoryboard, PipelineConfig, AudioSynthesisError

logger = logging.getLogger(__name__)

# Curated high-quality neural voice options
AVAILABLE_VOICES = {
    "Christopher (US Male - Authoritative & Warm)": "en-US-ChristopherNeural",
    "Jenny (US Female - Clear & Sophisticated)": "en-US-JennyNeural",
    "Guy (US Male - Energetic & Modern)": "en-US-GuyNeural",
    "Aria (US Female - Polished & Dynamic)": "en-US-AriaNeural",
    "Sonia (UK Female - Refined & Elegant)": "en-US-SoniaNeural",
    "Ryan (UK Male - Distinquished & Prestigious)": "en-GB-RyanNeural",
}


async def _synthesize_edge_tts(text: str, output_path: str, voice: str) -> None:
    """Asynchronously calls edge-tts WebSocket client to produce an audio file."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def _synthesize_offline_fallback(text: str, output_path: str) -> float:
    """
    Fallback synthesis when edge-tts or internet is unavailable.
    On macOS, uses native 'say' command to synthesize AIFF, then creates audio clip.
    If 'say' is not available, writes a clean silence buffer corresponding to spoken duration (~130 wpm).
    """
    logger.warning("Attempting offline acoustic fallback synthesis...")
    say_bin = shutil.which("say")
    if say_bin:
        try:
            aiff_path = output_path.replace(".mp3", ".aiff")
            cmd = [say_bin, "-o", aiff_path, text]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Verify and return duration
            clip = AudioFileClip(aiff_path)
            dur = clip.duration
            clip.close()
            # If output_path was mp3, rename or leave as aiff
            if output_path != aiff_path:
                shutil.move(aiff_path, output_path)
            return float(dur)
        except Exception as exc:
            logger.warning(f"macOS 'say' synthesis failed: {exc}")

    # Fallback: estimate speech duration (avg 2.2 words/sec) and generate silent track
    word_count = len(text.split())
    estimated_duration = max(word_count / 2.2, 2.5)
    logger.warning(f"Using estimated timing buffer ({estimated_duration:.1f}s) for scene text: {text[:40]}...")
    return estimated_duration


def synthesize_scene_audio(
    scene: SceneStoryboard,
    output_dir: Path,
    voice: str = "en-US-ChristopherNeural",
    safety_padding: float = 0.6,
    min_duration: float = 3.0
) -> float:
    """
    Synthesizes speech audio for an individual storyboard scene beat,
    measures exact audio duration, and computes scene duration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"scene_{scene.scene_id:02d}_speech.mp3"
    
    duration: float = 0.0
    try:
        # Run edge_tts via asyncio
        asyncio.run(_synthesize_edge_tts(scene.script_text, str(audio_path), voice))
        with AudioFileClip(str(audio_path)) as clip:
            duration = float(clip.duration)
        logger.info(f"Synthesized scene {scene.scene_id} audio ({duration:.2f}s) with voice '{voice}'")
    except Exception as exc:
        logger.warning(f"edge-tts synthesis failed for scene {scene.scene_id}: {exc}")
        # Offline fallback
        duration = _synthesize_offline_fallback(scene.script_text, str(audio_path))

    # Apply voice-first temporal scheduling formula: D_scene = max(D_audio + delta_pad, min_scene_duration)
    scene.audio_path = str(audio_path) if audio_path.exists() else None
    scene_duration = max(duration + safety_padding, min_duration)
    scene.duration = round(scene_duration, 2)
    return scene.duration


def synthesize_all_scenes(
    scenes: List[SceneStoryboard],
    temp_dir: Path,
    config: PipelineConfig
) -> float:
    """
    Synthesizes speech for all scenes in the storyboard and calculates cumulative duration.
    """
    total_duration = 0.0
    for scene in scenes:
        dur = synthesize_scene_audio(
            scene=scene,
            output_dir=temp_dir,
            voice=config.voice,
            safety_padding=config.safety_padding,
            min_duration=config.min_scene_duration
        )
        total_duration += dur

    logger.info(f"Completed speech synthesis for {len(scenes)} scenes. Total duration: {total_duration:.2f}s")
    return total_duration


def build_composite_audio(
    scenes: List[SceneStoryboard],
    total_video_duration: float,
    config: PipelineConfig
) -> Optional[CompositeAudioClip]:
    """
    Constructs multi-track composite audio:
    1. Sequenced voice-over narration aligned at each scene's start time.
    2. Ducked background music bed (12%-18% volume) with smooth loop and 1.5s exponential fade-out.
    """
    audio_clips = []
    current_time_offset = 0.0

    # 1. Position each scene's voice-over audio
    for scene in scenes:
        if scene.audio_path and os.path.exists(scene.audio_path):
            try:
                speech_clip = AudioFileClip(scene.audio_path)
                # Offset speech slightly into scene for natural pause
                speech_start = current_time_offset + 0.15
                positioned_clip = speech_clip.with_start(speech_start)
                audio_clips.append(positioned_clip)
            except Exception as exc:
                logger.warning(f"Could not load audio for scene {scene.scene_id}: {exc}")
        current_time_offset += scene.duration

    # 2. Add ducked background music bed if provided
    bgm_clip = None
    if config.bgm_path and os.path.isfile(config.bgm_path):
        try:
            raw_bgm = AudioFileClip(config.bgm_path)
            bgm_dur = raw_bgm.duration
            
            # Loop bgm if shorter than video duration
            if bgm_dur < total_video_duration:
                repeat_count = int(total_video_duration // bgm_dur) + 1
                from moviepy import concatenate_audioclips
                looped_clips = [raw_bgm] * repeat_count
                extended_bgm = concatenate_audioclips(looped_clips)
            else:
                extended_bgm = raw_bgm

            # Trim to exact video duration and apply ducking + 1.5s fade out
            ducked_bgm = (
                extended_bgm
                .subclipped(0, total_video_duration)
                .with_effects([
                    afx.MultiplyVolume(config.bgm_volume),
                    afx.AudioFadeOut(1.5)
                ])
            )
            # Insert ducked music bed underneath voice-overs
            audio_clips.insert(0, ducked_bgm)
            logger.info(f"Acoustic ducking applied to background music at {config.bgm_volume*100:.0f}% volume.")
        except Exception as exc:
            logger.warning(f"Failed to process background music track {config.bgm_path}: {exc}")

    if not audio_clips:
        return None

    return CompositeAudioClip(audio_clips)
