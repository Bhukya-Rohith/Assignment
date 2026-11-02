"""
Motion Graphics Visual Compositing Engine.

Upgrades over basic lower-third overlay:
  • Cinematic vignette layer (top + bottom gradient darkening on every scene)
  • Letterbox bars (16:9 widescreen mode) for a broadcast / film aesthetic
  • Per-scene visual identity:
      - hook     → full-screen bold centered title card with gold accent line
      - exploration → left-rail feature pill with animated accent stripe
      - outro    → centered CTA panel with rule lines and contact details
  • Animated overlay entrance: slide-up CrossFadeIn on all lower-third layers
  • Ken Burns affine zoom with alternating direction per scene
  • Scene-to-scene crossfade dissolve (0.5 s overlap)
  • Accent geometric rules (horizontal gold line, corner mark) per scene type
  • Pillow-rasterized RGBA layers — zero external binary dependency
"""
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
    vfx,
)

from src.models import SceneStoryboard, PipelineConfig, VideoCompositingError

logger = logging.getLogger(__name__)

# ── Font search paths (cross-platform) ──────────────────────────────────────
CANDIDATE_FONT_PATHS = [
    Path("assets/fonts/DejaVuSans-Bold.ttf"),
    Path("assets/fonts/Roboto-Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]

# ── Palette ──────────────────────────────────────────────────────────────────
GOLD      = (217, 119,   6, 255)    # amber-600
GOLD_DIM  = (217, 119,   6, 180)
TEAL      = ( 20, 184, 166, 255)    # teal-500
TEAL_DIM  = ( 20, 184, 166, 180)
WHITE     = (255, 255, 255, 255)
WHITE_DIM = (203, 213, 225, 255)    # slate-300
NAVY      = ( 15,  23,  42, 230)    # slate-900
NAVY_MID  = ( 15,  23,  42, 170)
BLACK0    = (  0,   0,   0,   0)    # transparent
LETTERBOX_ALPHA = 220               # opacity of film bars


# ── Font helper ──────────────────────────────────────────────────────────────
def resolve_font(size: int = 36) -> ImageFont.FreeTypeFont:
    """Locates a TrueType font across bundled assets and system paths."""
    for p in CANDIDATE_FONT_PATHS:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ── Image normalisation ───────────────────────────────────────────────────────
def crop_and_scale_image(
    image_path: Union[str, Path], target_size: Tuple[int, int]
) -> Image.Image:
    """Centered cover-crop with Lanczos resampling to fit the target canvas exactly."""
    w_t, h_t = target_size
    with Image.open(image_path) as raw:
        img = raw.convert("RGB")
        w_s, h_s = img.size
        scale  = max(w_t / w_s, h_t / h_s)
        new_w  = int(math.ceil(w_s * scale))
        new_h  = int(math.ceil(h_s * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left   = (new_w - w_t) // 2
        top    = (new_h - h_t) // 2
        return resized.crop((left, top, left + w_t, top + h_t))


# ── Shared overlay helpers ────────────────────────────────────────────────────
def _new_canvas(w: int, h: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (w, h), BLACK0)
    return img, ImageDraw.Draw(img)


def _draw_vignette(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    """Cinematic edge-to-centre gradient darkening (top + bottom bands)."""
    top_band    = int(h * 0.25)
    bottom_start = int(h * 0.60)
    # top
    for y in range(top_band):
        a = int(160 * (1.0 - y / top_band) ** 1.6)
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, a))
    # bottom
    for y in range(bottom_start, h):
        a = int(200 * ((y - bottom_start) / (h - bottom_start)) ** 1.4)
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, a))


def _draw_letterbox(draw: ImageDraw.ImageDraw, w: int, h: int,
                    bar_h_ratio: float = 0.065) -> None:
    """Adds horizontal black cinema letterbox bars (landscape only)."""
    bh = int(h * bar_h_ratio)
    draw.rectangle([(0, 0), (w, bh)], fill=(0, 0, 0, LETTERBOX_ALPHA))
    draw.rectangle([(0, h - bh), (w, h)], fill=(0, 0, 0, LETTERBOX_ALPHA))


def _draw_accent_line(draw: ImageDraw.ImageDraw, x0: int, y: int, x1: int,
                      color: Tuple = GOLD, thickness: int = 3) -> None:
    """Horizontal accent rule used across overlay designs."""
    draw.rectangle([(x0, y), (x1, y + thickness)], fill=color)


def _draw_corner_mark(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                      arm: int = 22, color: Tuple = GOLD, th: int = 3) -> None:
    """L-shaped corner mark (top-left orientation)."""
    draw.rectangle([(cx, cy), (cx + arm, cy + th)], fill=color)
    draw.rectangle([(cx, cy), (cx + th, cy + arm)], fill=color)


def _badge_pill(draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
                font: ImageFont.FreeTypeFont,
                fill: Tuple = GOLD, text_color: Tuple = WHITE) -> int:
    """Draws a rounded pill badge; returns the right edge x."""
    W = int(len(text) * font.size * 0.65 + 28)
    H = int(font.size + 16)
    draw.rounded_rectangle([(x, y), (x + W, y + H)], radius=H // 2, fill=fill)
    draw.text((x + 14, y + 8), text, font=font, fill=text_color)
    return x + W


# ═══════════════════════════════════════════════════════════════════════════════
# Per-scene overlay renderers
# ═══════════════════════════════════════════════════════════════════════════════

def _render_hook_overlay(
    title: str, subtitle: str, badge: Optional[str],
    w: int, h: int, is_portrait: bool,
) -> np.ndarray:
    """
    HOOK scene: full-width centred title card.
    Design:  vignette → letterbox → gold accent line → large white title
             → location subtitle → price badge pill.
    """
    canvas, draw = _new_canvas(w, h)

    _draw_vignette(draw, w, h)
    if not is_portrait:
        _draw_letterbox(draw, w, h)

    # Card background — wide cinematic strip in the lower-centre
    card_h = int(h * 0.30)
    card_y = int(h * 0.54)
    # Full-width semi-transparent band
    draw.rectangle([(0, card_y), (w, card_y + card_h)], fill=(0, 0, 0, 130))

    # Gold accent lines — top and bottom of card
    _draw_accent_line(draw, int(w * 0.07), card_y,
                      int(w * 0.93), color=GOLD, thickness=3)
    _draw_accent_line(draw, int(w * 0.07), card_y + card_h - 3,
                      int(w * 0.93), color=(255, 255, 255, 60), thickness=2)

    # Corner marks
    _draw_corner_mark(draw, int(w * 0.07), card_y + 12, arm=18, color=GOLD)

    pad_x  = int(w * 0.09)
    curr_y = card_y + 20

    # Badge
    if badge:
        f_badge = resolve_font(size=int(h * 0.022) if not is_portrait else int(w * 0.028))
        curr_y += 6
        _badge_pill(draw, pad_x, curr_y, badge.upper(), f_badge, fill=GOLD)
        curr_y += int(f_badge.size + 22)

    # Title
    f_title = resolve_font(size=int(h * 0.058) if not is_portrait else int(w * 0.058))
    max_len = 42 if not is_portrait else 28
    t = title if len(title) <= max_len else title[:max_len - 1] + "…"
    draw.text((pad_x, curr_y), t, font=f_title, fill=WHITE)
    curr_y += int(f_title.size * 1.25)

    # Subtitle  (location · type)
    f_sub = resolve_font(size=int(h * 0.026) if not is_portrait else int(w * 0.030))
    max_sub = 60 if not is_portrait else 38
    s = subtitle if len(subtitle) <= max_sub else subtitle[:max_sub - 1] + "…"
    draw.text((pad_x, curr_y), s, font=f_sub, fill=WHITE_DIM)

    return np.array(canvas)


def _render_exploration_overlay(
    title: str, subtitle: str, badge: Optional[str],
    w: int, h: int, is_portrait: bool,
) -> np.ndarray:
    """
    EXPLORATION scene: left-rail feature pill card.
    Design:  vignette → teal left-edge stripe → glass card left-anchored
             → feature title → property sub-line → teal accent bottom line.
    """
    canvas, draw = _new_canvas(w, h)

    _draw_vignette(draw, w, h)
    if not is_portrait:
        _draw_letterbox(draw, w, h)

    # Left teal vertical stripe
    stripe_w = int(w * 0.006)
    stripe_x = int(w * 0.055)
    card_y   = int(h * 0.62)
    card_h   = int(h * 0.24)
    draw.rectangle(
        [(stripe_x, card_y - 8), (stripe_x + stripe_w, card_y + card_h + 8)],
        fill=TEAL,
    )

    # Glass card — left anchored
    card_x = stripe_x + stripe_w + 12
    card_w = int(w * 0.52)
    draw.rounded_rectangle(
        [(card_x, card_y), (card_x + card_w, card_y + card_h)],
        radius=16,
        fill=NAVY,
        outline=(255, 255, 255, 30),
        width=1,
    )

    pad_x  = card_x + 28
    curr_y = card_y + 18

    # Badge pill
    if badge:
        f_badge = resolve_font(size=int(h * 0.020) if not is_portrait else int(w * 0.026))
        _badge_pill(draw, pad_x, curr_y, badge.upper(), f_badge, fill=TEAL, text_color=WHITE)
        curr_y += int(f_badge.size + 22)

    # Feature title
    f_title = resolve_font(size=int(h * 0.048) if not is_portrait else int(w * 0.048))
    max_len = 36 if not is_portrait else 24
    t = title if len(title) <= max_len else title[:max_len - 1] + "…"
    draw.text((pad_x, curr_y), t, font=f_title, fill=WHITE)
    curr_y += int(f_title.size * 1.2)

    # Subtitle
    f_sub = resolve_font(size=int(h * 0.024) if not is_portrait else int(w * 0.028))
    max_sub = 52 if not is_portrait else 34
    s = subtitle if len(subtitle) <= max_sub else subtitle[:max_sub - 1] + "…"
    draw.text((pad_x, curr_y), s, font=f_sub, fill=WHITE_DIM)

    # Bottom teal accent rule inside card
    _draw_accent_line(draw, pad_x, card_y + card_h - 8,
                      card_x + card_w - 20, color=TEAL_DIM, thickness=2)

    return np.array(canvas)


def _render_outro_overlay(
    title: str, subtitle: str, badge: Optional[str],
    w: int, h: int, is_portrait: bool,
) -> np.ndarray:
    """
    OUTRO / CTA scene: centred elegant panel.
    Design:  vignette → full-width semi-dark overlay → white rule lines
             → CTA headline → contact sub-line → gold decorative rule footer.
    """
    canvas, draw = _new_canvas(w, h)

    _draw_vignette(draw, w, h)
    if not is_portrait:
        _draw_letterbox(draw, w, h)

    # Full-height dark overlay (more dramatic for outro)
    draw.rectangle([(0, 0), (w, h)], fill=(0, 0, 0, 90))

    # Centred card
    card_w = int(w * 0.70) if not is_portrait else int(w * 0.88)
    card_h = int(h * 0.30) if not is_portrait else int(h * 0.22)
    card_x = (w - card_w) // 2
    card_y = int(h * 0.52)

    draw.rounded_rectangle(
        [(card_x, card_y), (card_x + card_w, card_y + card_h)],
        radius=20,
        fill=NAVY_MID,
        outline=(255, 255, 255, 50),
        width=1,
    )

    # Top white rule
    _draw_accent_line(draw, card_x + 30, card_y,
                      card_x + card_w - 30, color=(255, 255, 255, 120), thickness=2)

    # Bottom gold rule
    _draw_accent_line(draw, card_x + 30, card_y + card_h - 3,
                      card_x + card_w - 30, color=GOLD_DIM, thickness=3)

    pad_x  = card_x + 40
    curr_y = card_y + 22

    # CTA title
    f_title = resolve_font(size=int(h * 0.048) if not is_portrait else int(w * 0.046))
    max_len = 42 if not is_portrait else 28
    t = title if len(title) <= max_len else title[:max_len - 1] + "…"
    # Centre the title
    try:
        bbox = draw.textbbox((0, 0), t, font=f_title)
        text_w = bbox[2] - bbox[0]
    except Exception:
        text_w = len(t) * f_title.size // 2
    draw.text(((w - text_w) // 2, curr_y), t, font=f_title, fill=WHITE)
    curr_y += int(f_title.size * 1.3)

    # Thin separator
    sep_x = int(w * 0.38)
    _draw_accent_line(draw, sep_x, curr_y, w - sep_x, color=(255, 255, 255, 60), thickness=1)
    curr_y += 14

    # Contact subtitle
    f_sub = resolve_font(size=int(h * 0.024) if not is_portrait else int(w * 0.028))
    max_sub = 70 if not is_portrait else 44
    s = subtitle if len(subtitle) <= max_sub else subtitle[:max_sub - 1] + "…"
    try:
        bbox = draw.textbbox((0, 0), s, font=f_sub)
        s_w = bbox[2] - bbox[0]
    except Exception:
        s_w = len(s) * f_sub.size // 2
    draw.text(((w - s_w) // 2, curr_y), s, font=f_sub, fill=WHITE_DIM)

    # Gold badge (phone / price if present)
    if badge:
        f_badge = resolve_font(size=int(h * 0.020))
        bx = (w - int(len(badge) * f_badge.size * 0.65 + 28)) // 2
        _badge_pill(draw, bx, card_y + card_h + 14, badge, f_badge, fill=GOLD)

    return np.array(canvas)


# ── Public overlay dispatcher ─────────────────────────────────────────────────
def render_lower_third_card(
    title: str,
    subtitle: str,
    badge: Optional[str] = None,
    canvas_size: Tuple[int, int] = (1920, 1080),
    scene_type: str = "exploration",
) -> np.ndarray:
    """
    Dispatches to the correct per-scene motion graphics overlay renderer
    based on scene_type: 'hook', 'exploration', or 'outro'.
    """
    w, h = canvas_size
    is_portrait = h > w
    if scene_type == "hook":
        return _render_hook_overlay(title, subtitle, badge, w, h, is_portrait)
    elif scene_type == "outro":
        return _render_outro_overlay(title, subtitle, badge, w, h, is_portrait)
    else:
        return _render_exploration_overlay(title, subtitle, badge, w, h, is_portrait)


# ── Scene clip builder ────────────────────────────────────────────────────────
def build_scene_clip(
    storyboard_scene: SceneStoryboard,
    target_size: Tuple[int, int] = (1920, 1080),
) -> CompositeVideoClip:
    """
    Constructs one motion-graphics scene clip:
      1. Lanczos cover-cropped base image.
      2. Ken Burns affine zoom (alternating in/out per scene index).
      3. FadeIn on the base image (0.4 s) for soft entrance.
      4. Per-scene overlay rasterized via Pillow (vignette + styled card).
      5. Overlay slides up with CrossFadeIn for a polished animated entrance.
    """
    w, h       = target_size
    duration   = max(storyboard_scene.duration, 1.0)
    scene_type = storyboard_scene.scene_type

    # ── Base image ──
    cropped = crop_and_scale_image(storyboard_scene.image_path, target_size)
    base_clip = (
        ImageClip(np.array(cropped))
        .with_duration(duration)
        .with_effects([vfx.FadeIn(0.4)])
    )

    # ── Ken Burns motion ──
    if storyboard_scene.scene_id % 2 == 0:
        # zoom-in: 1.00 → 1.12
        animated = base_clip.with_effects([
            vfx.Resize(lambda t, _d=duration: 1.00 + 0.12 * (t / _d))
        ])
    else:
        # zoom-out: 1.12 → 1.00
        animated = base_clip.with_effects([
            vfx.Resize(lambda t, _d=duration: 1.12 - 0.12 * (t / _d))
        ])

    # ── Motion-graphics overlay (vignette + styled card) ──
    overlay_arr = render_lower_third_card(
        title      = storyboard_scene.overlay_title,
        subtitle   = storyboard_scene.overlay_subtitle,
        badge      = storyboard_scene.overlay_badge,
        canvas_size= target_size,
        scene_type = scene_type,
    )
    # Animate the overlay in: CrossFadeIn 0.6 s + slight delay for hook/outro
    fade_dur = 0.5 if scene_type == "exploration" else 0.7
    overlay_clip = (
        ImageClip(overlay_arr)
        .with_duration(duration)
        .with_effects([vfx.CrossFadeIn(fade_dur)])
    )

    # ── Composite ──
    composite = CompositeVideoClip(
        [animated, overlay_clip], size=target_size
    ).with_duration(duration)

    return composite


# ── Full video renderer ───────────────────────────────────────────────────────
def render_complete_video(
    scenes: List[SceneStoryboard],
    composite_audio: Optional[CompositeAudioClip],
    output_path: Union[str, Path],
    config: PipelineConfig,
    progress_callback: Optional[callable] = None,
) -> Path:
    """
    Assembles scene clips with crossfade dissolves, attaches multi-track
    composite audio, and encodes to H.264/AAC MP4.
    """
    out_file    = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    target_size = config.dimensions

    logger.info(
        f"Compositing {len(scenes)} motion-graphics scene clips "
        f"at {target_size[0]}×{target_size[1]}…"
    )

    scene_clips = []
    for idx, sc in enumerate(scenes):
        if progress_callback:
            progress_callback(
                f"Rendering Scene {sc.scene_id} [{sc.scene_type.upper()}]…",
                0.4 + 0.3 * (idx / len(scenes)),
            )
        clip = build_scene_clip(sc, target_size=target_size)

        # Apply CrossFadeIn on every clip except the first for a smooth dissolve transition
        if idx > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(0.5)])

        scene_clips.append(clip)

    if progress_callback:
        progress_callback("Sequencing timeline with dissolve transitions…", 0.75)

    # Stitch with 0.5 s overlap to activate crossfades between scenes
    final_video = concatenate_videoclips(
        scene_clips, padding=-0.5, method="compose"
    )

    # Attach composite multi-track audio (voice-over + ducked BGM)
    if composite_audio:
        final_video = final_video.with_audio(composite_audio)

    if progress_callback:
        progress_callback("Encoding MP4 (H.264 / AAC)…", 0.85)

    logger.info(f"Encoding → {out_file} @ {config.fps} fps")
    final_video.write_videofile(
        str(out_file),
        fps=config.fps,
        codec="libx264",
        audio_codec="aac",
        preset="fast",
        threads=4,
        logger=None,
    )

    final_video.close()
    for c in scene_clips:
        c.close()

    logger.info(
        f"✓ Video rendered: {out_file} "
        f"(Duration: {final_video.duration:.2f}s)"
    )
    if progress_callback:
        progress_callback("Video generation complete!", 1.0)

    return out_file
