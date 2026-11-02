"""
Unit tests for visual compositing, Lanczos cover-cropping, and Pillow overlays.
"""
import numpy as np
from PIL import Image
from src.models import PipelineConfig
from src.video_engine import crop_and_scale_image, render_lower_third_card


def test_crop_and_scale_image(tmp_path):
    # Test normalization of a 4:3 image to 16:9 canvas
    img_path = tmp_path / "sample_4_3.jpg"
    img = Image.new("RGB", (800, 600), color=(120, 150, 180))
    img.save(img_path)

    cropped_16_9 = crop_and_scale_image(img_path, target_size=(1920, 1080))
    assert cropped_16_9.size == (1920, 1080)

    # Test normalization of a 16:9 image to 9:16 portrait canvas
    cropped_9_16 = crop_and_scale_image(img_path, target_size=(1080, 1920))
    assert cropped_9_16.size == (1080, 1920)


def test_render_lower_third_card():
    # Landscape hook scene
    overlay_hook = render_lower_third_card(
        title="The Obsidian Villa",
        subtitle="Bel Air • Luxury Architectural Villa",
        badge="$14,500,000",
        canvas_size=(1920, 1080),
        scene_type="hook",
    )
    assert isinstance(overlay_hook, np.ndarray)
    assert overlay_hook.shape == (1080, 1920, 4)

    # Landscape exploration scene
    overlay_explore = render_lower_third_card(
        title="Infinity Edge Pool",
        subtitle="The Obsidian Villa • Bel Air",
        badge="Featured Space",
        canvas_size=(1920, 1080),
        scene_type="exploration",
    )
    assert isinstance(overlay_explore, np.ndarray)
    assert overlay_explore.shape == (1080, 1920, 4)

    # Portrait outro scene
    overlay_outro = render_lower_third_card(
        title="Schedule Your Private Tour",
        subtitle="Tribeca • Duplex Penthouse",
        badge="$8,200,000",
        canvas_size=(1080, 1920),
        scene_type="outro",
    )
    assert isinstance(overlay_outro, np.ndarray)
    assert overlay_outro.shape == (1920, 1080, 4)


def test_pipeline_config_dimensions():
    cfg_landscape = PipelineConfig(aspect_ratio="16:9")
    assert cfg_landscape.dimensions == (1920, 1080)

    cfg_portrait = PipelineConfig(aspect_ratio="9:16")
    assert cfg_portrait.dimensions == (1080, 1920)
