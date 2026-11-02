"""
Top-level orchestration pipeline: connects ingestion, script generation,
audio synthesis, visual compositing, and final video delivery.
"""
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from src.models import (
    PropertyListing,
    PipelineConfig,
    RealtyPipelineError,
)
from src.ingestion import normalize_listing_data, load_listing_from_json
from src.script_engine import build_storyboard
from src.audio_engine import synthesize_all_scenes, build_composite_audio
from src.video_engine import render_complete_video

logger = logging.getLogger(__name__)


def run_pipeline(
    listing_input: Union[PropertyListing, Dict[str, Any], str, Path],
    config: Optional[PipelineConfig] = None,
    images_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """
    Executes the end-to-end automated video generation pipeline:
    1. Ingestion & Pre-flight validation
    2. Zero-hallucination storyboard scripting
    3. Voice-first speech synthesis & timeline scheduling
    4. Multi-track audio composition & acoustic ducking
    5. Visual compositing with Ken Burns affine transformations & lower-third cards
    6. H.264/AAC MP4 encoding
    """
    if config is None:
        config = PipelineConfig()

    def update_progress(msg: str, frac: float):
        logger.info(f"[{int(frac*100)}%] {msg}")
        if progress_callback:
            progress_callback(msg, frac)

    update_progress("Initializing video pipeline & validating assets...", 0.05)

    # 1. Ingestion stage
    if isinstance(listing_input, PropertyListing):
        listing = listing_input
    elif isinstance(listing_input, (str, Path)):
        listing = load_listing_from_json(listing_input, images_dir=images_dir)
    elif isinstance(listing_input, dict):
        listing = normalize_listing_data(listing_input)
    else:
        raise RealtyPipelineError(f"Unsupported listing input type: {type(listing_input)}")

    logger.info(f"Loaded listing: '{listing.name}' with {len(listing.image_paths)} image assets.")

    # 2. Scripting & Storyboard stage
    update_progress("Synthesizing zero-hallucination storyboard narrative...", 0.15)
    scenes = build_storyboard(listing, api_key=config.api_key)
    logger.info(f"Generated storyboard containing {len(scenes)} narrative beats.")

    # Temporary directory for intermediate acoustic assets
    temp_dir = Path(tempfile.mkdtemp(prefix="realty_vid_"))

    try:
        # 3. Audio Synthesis stage (Voice-First Temporal Scheduling)
        update_progress("Synthesizing neural voice-over narration via edge-tts...", 0.25)
        total_duration = synthesize_all_scenes(scenes, temp_dir=temp_dir, config=config)

        # 4. Multi-track Audio Ducking & Composition
        update_progress("Composing multi-track audio bed with acoustic ducking...", 0.35)
        composite_audio = build_composite_audio(
            scenes=scenes,
            total_video_duration=total_duration,
            config=config,
        )

        # 5. Visual Compositing & Rendering
        update_progress("Rendering dynamic visual scenes with Ken Burns scaling...", 0.45)
        final_mp4 = render_complete_video(
            scenes=scenes,
            composite_audio=composite_audio,
            output_path=config.output_path,
            config=config,
            progress_callback=update_progress,
        )

        update_progress(f"Promo video generated successfully: {final_mp4.name}", 1.0)
        return final_mp4

    finally:
        # Clean up intermediate acoustic files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
