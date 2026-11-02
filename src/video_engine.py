"""
Visual compositing engine: aspect ratio normalization (Lanczos cover-crop),
Pillow typographic lower-third rasterization, Ken Burns affine scaling,
and MoviePy v2 timeline assembly.
"""
import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    concatenate_videoclips,
    vfx,
)

from src.models import SceneStoryboard, PipelineConfig, VideoCompositingError

logger = logging.getLogger(__name__)

# Search paths for cross-platform TrueType fonts
CANDIDATE_FONT_PATHS = [
    Path("assets/fonts/DejaVuSans-Bold.ttf"),
    Path("assets/fonts/Roboto-Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]


def resolve_font(bold: bool = True, size: int = 36) -> ImageFont.FreeTypeFont:
    """Attempts to locate a crisp TrueType font across local bundled assets and system paths."""
    for p in CANDIDATE_FONT_PATHS:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    try:
        # Fallback to default truetype font if available in PIL
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def crop_and_scale_image(image_path: Union[str, Path], target_size: Tuple[int, int]) -> Image.Image:
    """
    Applies centered cover-crop normalization using Lanczos resampling.
    Ensures input image covers target canvas without letterbox black bars or distortion.
    """
    w_target, h_target = target_size
    with Image.open(image_path) as raw_img:
        img = raw_img.convert("RGB")
        w_src, h_src = img.size

        # Cover-crop scaling factor
        scale = max(w_target / w_src, h_target / h_src)
        new_w = int(math.ceil(w_src * scale))
        new_h = int(math.ceil(h_src * scale))

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Symmetric center crop
        left = (new_w - w_target) // 2
        top = (new_h - h_target) // 2
        right = left + w_target
        bottom = top + h_target

        return resized.crop((left, top, right, bottom))


def render_lower_third_card(
    title: str,
    subtitle: str,
    badge: Optional[str] = None,
    canvas_size: Tuple[int, int] = (1920, 1080),
) -> np.ndarray:
    """
    Renders a broadcast-grade typography card onto a transparent RGBA canvas using Pillow.
    Features dark slate glassmorphism with accent pill badge and crisp typography.
    """
    w, h = canvas_size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    is_portrait = h > w

    if is_portrait:
        # 9:16 layout (Mobile / Reels)
        card_w = int(w * 0.90)
        card_h = int(h * 0.22)
        card_x = (w - card_w) // 2
        card_y = int(h * 0.70)
        font_title = resolve_font(bold=True, size=int(w * 0.048))
        font_sub = resolve_font(bold=False, size=int(w * 0.034))
        font_badge = resolve_font(bold=True, size=int(w * 0.028))
    else:
        # 16:9 layout (Landscape / Broadcast)
        card_w = int(w * 0.85)
        card_h = int(h * 0.22)
        card_x = (w - card_w) // 2
        card_y = int(h * 0.72)
        font_title = resolve_font(bold=True, size=int(h * 0.046))
        font_sub = resolve_font(bold=False, size=int(h * 0.030))
        font_badge = resolve_font(bold=True, size=int(h * 0.024))

    # Glassmorphism container background (dark navy with rounded border)
    container_bg = (15, 23, 42, 220)  # Slate-900 at 86% opacity
    border_color = (255, 255, 255, 45)

    # Draw card shadow & card background
    draw.rounded_rectangle(
        [(card_x, card_y), (card_x + card_w, card_y + card_h)],
        radius=20,
        fill=container_bg,
        outline=border_color,
        width=2,
    )

    pad_x = 35
    curr_y = card_y + 20

    # Draw Badge if present
    if badge:
        badge_text = badge.upper()
        # Measure badge dimensions roughly
        badge_w = int(len(badge_text) * 12 + 24)
        badge_h = 32
        badge_rect = [(card_x + pad_x, curr_y), (card_x + pad_x + badge_w, curr_y + badge_h)]
        # Amber/Gold luxury accent
        draw.rounded_rectangle(badge_rect, radius=8, fill=(217, 119, 6, 235))
        draw.text((card_x + pad_x + 12, curr_y + 6), badge_text, font=font_badge, fill=(255, 255, 255, 255))
        curr_y += badge_h + 12
    else:
        curr_y += 8

    # Draw Headline / Title (truncate if too long)
    max_title_len = 50 if not is_portrait else 35
    display_title = title if len(title) <= max_title_len else title[:max_title_len - 3] + "..."
    draw.text((card_x + pad_x, curr_y), display_title, font=font_title, fill=(255, 255, 255, 255))
    curr_y += int(font_title.size * 1.3)

    # Draw Subtitle
    max_sub_len = 70 if not is_portrait else 48
    display_sub = subtitle if len(subtitle) <= max_sub_len else subtitle[:max_sub_len - 3] + "..."
    draw.text((card_x + pad_x, curr_y), display_sub, font=font_sub, fill=(203, 213, 225, 255))

    return np.array(overlay)


def build_scene_clip(
    storyboard_scene: SceneStoryboard,
    target_size: Tuple[int, int] = (1920, 1080)
) -> CompositeVideoClip:
    """
    Constructs a dynamic video clip for a single scene:
    1. Lanczos normalized base image.
    2. Dynamic Ken Burns motion effect (alternating zoom-in and zoom-out).
    3. Pillow-rasterized lower-third glass card overlay.
    """
    w, h = target_size
    duration = max(storyboard_scene.duration, 1.0)

    # Prepare cover-cropped image
    cropped_img = crop_and_scale_image(storyboard_scene.image_path, target_size)
    img_array = np.array(cropped_img)

    base_clip = ImageClip(img_array).with_duration(duration)

    # Ken Burns Affine Transformation:
    # Even scenes: subtle zoom-in 1.0 -> 1.10
    # Odd scenes: subtle zoom-out 1.10 -> 1.00
    if storyboard_scene.scene_id % 2 == 0:
        animated_clip = base_clip.with_effects([
            vfx.Resize(lambda t: 1.0 + 0.10 * (t / duration))
        ])
    else:
        animated_clip = base_clip.with_effects([
            vfx.Resize(lambda t: 1.10 - 0.10 * (t / duration))
        ])

    # Rasterize typography overlay
    overlay_array = render_lower_third_card(
        title=storyboard_scene.overlay_title,
        subtitle=storyboard_scene.overlay_subtitle,
        badge=storyboard_scene.overlay_badge,
        canvas_size=target_size,
    )
    overlay_clip = ImageClip(overlay_array).with_duration(duration)

    # Composite base motion with fixed overlay card
    composite = CompositeVideoClip(
        [animated_clip, overlay_clip],
        size=target_size
    ).with_duration(duration)

    return composite


def render_complete_video(
    scenes: List[SceneStoryboard],
    composite_audio: Optional[CompositeAudioClip],
    output_path: Union[str, Path],
    config: PipelineConfig,
    progress_callback: Optional[callable] = None,
) -> Path:
    """
    Assembles scene clips into a complete video, attaches multi-track composite audio,
    and encodes to H.264/AAC MP4.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    target_size = config.dimensions

    logger.info(f"Building {len(scenes)} visual scene clips at resolution {target_size}...")
    scene_clips = []
    for idx, sc in enumerate(scenes):
        if progress_callback:
            progress_callback(f"Compositing Scene {sc.scene_id} ({sc.scene_type})...", 0.4 + (0.3 * (idx / len(scenes))))
        clip = build_scene_clip(sc, target_size=target_size)
        scene_clips.append(clip)

    if progress_callback:
        progress_callback("Sequencing timeline and stitching scenes...", 0.75)

    # Concatenate scene clips sequentially
    final_video = concatenate_videoclips(scene_clips, method="compose")

    # Attach composite multi-track audio (voice-over + ducked BGM)
    if composite_audio:
        final_video = final_video.with_audio(composite_audio)

    if progress_callback:
        progress_callback("Encoding MP4 container with H.264 & AAC...", 0.85)

    logger.info(f"Encoding final MP4 to {out_file} at {config.fps} fps...")
    final_video.write_videofile(
        str(out_file),
        fps=config.fps,
        codec="libx264",
        audio_codec="aac",
        preset="fast",
        threads=4,
        logger=None,  # Suppress verbose ffmpeg stdout
    )

    # Dereference and clean up clips
    final_video.close()
    for c in scene_clips:
        c.close()

    logger.info(f"Video rendered successfully: {out_file} (Duration: {final_video.duration:.2f}s)")
    if progress_callback:
        progress_callback("Video generation complete!", 1.0)

    return out_file
