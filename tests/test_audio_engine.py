"""
Unit tests for audio synthesis and voice-first temporal scheduling.
"""
from pathlib import Path
from src.models import SceneStoryboard, PipelineConfig
from src.audio_engine import AVAILABLE_VOICES, synthesize_scene_audio, _synthesize_offline_fallback


def test_available_voices():
    assert len(AVAILABLE_VOICES) >= 4
    assert "en-US-ChristopherNeural" in AVAILABLE_VOICES.values()
    assert "en-US-JennyNeural" in AVAILABLE_VOICES.values()


def test_offline_fallback_timing():
    text = "Welcome to this magnificent estate situated in prestigious Beverly Hills."
    duration = _synthesize_offline_fallback(text, "dummy_output.mp3")
    assert duration > 2.0
    assert duration < 10.0


def test_voice_first_scheduling_calculation(tmp_path):
    scene = SceneStoryboard(
        scene_id=1,
        image_path="dummy.jpg",
        script_text="Short prompt.",
        overlay_title="Title",
        overlay_subtitle="Sub",
    )
    # Using small padding
    dur = synthesize_scene_audio(
        scene=scene,
        output_dir=tmp_path,
        safety_padding=0.7,
        min_duration=3.0,
    )
    assert scene.duration >= 3.0
    assert scene.duration == dur
