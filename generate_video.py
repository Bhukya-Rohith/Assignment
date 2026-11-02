"""
Command-Line Interface (CLI) for automated, headless real estate video generation.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.models import PipelineConfig, RealtyPipelineError
from src.pipeline import run_pipeline
from src.audio_engine import AVAILABLE_VOICES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("realty_cli")


def main():
    parser = argparse.ArgumentParser(
        description="Automated Real Estate Video Generation Pipeline (Headless CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to JSON listing configuration file (e.g., samples/sample_listing_01.json)",
    )
    parser.add_argument(
        "--images",
        "-i",
        default=None,
        help="Optional directory containing property images. If not set, reads image paths from JSON config.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="exports/generated_promo.mp4",
        help="Destination path for the compiled MP4 video.",
    )
    parser.add_argument(
        "--voice",
        "-v",
        default="en-US-ChristopherNeural",
        help=f"Neural TTS voice identifier. Options: {list(AVAILABLE_VOICES.values())}",
    )
    parser.add_argument(
        "--aspect-ratio",
        "-a",
        choices=["16:9", "9:16"],
        default="16:9",
        help="Output aspect ratio: '16:9' for web/desktop portals, '9:16' for mobile/Reels.",
    )
    parser.add_argument(
        "--bgm",
        "-b",
        default="assets/audio/ambient_luxury.mp3",
        help="Path to background music track (optional).",
    )
    parser.add_argument(
        "--bgm-volume",
        type=float,
        default=0.15,
        help="Volume ducking multiplier for background music (0.10 to 0.25 recommended).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target video frames per second.",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        default=None,
        help="Optional API key for LLM structured script synthesis (OpenAI/compatible).",
    )

    args = parser.parse_args()

    bgm_path = args.bgm if (args.bgm and Path(args.bgm).is_file()) else None
    if args.bgm and not bgm_path:
        logger.warning(f"Background music file not found at '{args.bgm}'. Proceeding without BGM.")

    config = PipelineConfig(
        aspect_ratio=args.aspect_ratio,
        voice=args.voice,
        bgm_path=bgm_path,
        bgm_volume=args.bgm_volume,
        fps=args.fps,
        output_path=args.output,
        api_key=args.api_key,
    )

    logger.info("=" * 60)
    logger.info("Starting Realty Assistant Video Generation")
    logger.info(f"Listing Config : {args.config}")
    logger.info(f"Aspect Ratio   : {args.aspect_ratio}")
    logger.info(f"Voice Profile  : {args.voice}")
    logger.info(f"Output Target  : {args.output}")
    logger.info("=" * 60)

    try:
        output_file = run_pipeline(
            listing_input=args.config,
            config=config,
            images_dir=args.images,
        )
        logger.info("=" * 60)
        logger.info(f"SUCCESS: Promo video saved to: {output_file}")
        logger.info("=" * 60)
        sys.exit(0)
    except RealtyPipelineError as exc:
        logger.error(f"Pipeline Execution Failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Unexpected Fatal Error: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
