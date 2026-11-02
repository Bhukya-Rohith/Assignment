"""
Ingestion, asset verification, and metadata normalization layer.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from src.models import PropertyListing, IngestionError

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def verify_image_file(image_path: Union[str, Path]) -> bool:
    """
    Verifies that an image file exists, is non-empty, and has a valid, readable header.
    """
    path = Path(image_path)
    if not path.is_file():
        logger.warning(f"Image path does not exist or is not a file: {path}")
        return False

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        logger.warning(f"Unsupported image extension for {path}. Expected {SUPPORTED_IMAGE_EXTENSIONS}")
        return False

    try:
        with Image.open(path) as img:
            img.verify()
        # Verify only reads header; test opening to ensure pixel buffer is loadable
        with Image.open(path) as img:
            img.load()
        return True
    except Exception as exc:
        logger.warning(f"Corrupted or unreadable image file {path}: {exc}")
        return False


def normalize_listing_data(data: Dict[str, Any], image_paths: Optional[List[str]] = None) -> PropertyListing:
    """
    Normalizes raw listing dictionary into a validated PropertyListing dataclass,
    substituting semantic fallbacks for missing or blank attributes.
    """
    raw_name = str(data.get("name", "")).strip()
    name = raw_name if raw_name else "Luxury Residence"

    raw_location = str(data.get("location", "")).strip()
    location = raw_location if raw_location else "Prime Location"

    raw_price = str(data.get("price", "")).strip()
    price = raw_price if raw_price else "Price on Request"

    raw_type = str(data.get("property_type", "")).strip()
    property_type = raw_type if raw_type else "Exclusive Property"

    raw_features = data.get("features", [])
    if isinstance(raw_features, str):
        # Support comma-separated or newline-separated strings
        features = [f.strip() for f in raw_features.replace("\n", ",").split(",") if f.strip()]
    elif isinstance(raw_features, list):
        features = [str(f).strip() for f in raw_features if str(f).strip()]
    else:
        features = []

    if not features:
        features = ["Architectural Excellence", "Premium Interior Finishes", "Prime Accessibility"]

    raw_cta = str(data.get("contact_cta", "")).strip()
    contact_cta = raw_cta if raw_cta else "Contact our advisory team today for exclusive viewings."

    # Process image paths
    candidate_images = image_paths if image_paths is not None else data.get("image_paths", [])
    if isinstance(candidate_images, str):
        candidate_images = [candidate_images]

    valid_images: List[str] = []
    for p in candidate_images:
        p_str = str(p).strip()
        if p_str and verify_image_file(p_str):
            valid_images.append(str(Path(p_str).resolve()))

    if not valid_images:
        raise IngestionError(
            "No valid, readable image assets provided. Please supply at least one valid image file (.jpg, .jpeg, .png, .webp)."
        )

    # Dynamic asset cycling if fewer than 4 images supplied
    if len(valid_images) < 4:
        logger.warning(
            f"Only {len(valid_images)} valid image(s) provided. Cycling assets to satisfy 4-6 scene narrative arc."
        )
        original_count = len(valid_images)
        cycle_idx = 0
        while len(valid_images) < 4:
            valid_images.append(valid_images[cycle_idx % original_count])
            cycle_idx += 1

    return PropertyListing(
        name=name,
        location=location,
        price=price,
        property_type=property_type,
        features=features,
        contact_cta=contact_cta,
        image_paths=valid_images,
        agent_name=data.get("agent_name"),
        phone=data.get("phone"),
        email=data.get("email"),
    )


def load_listing_from_json(json_path: Union[str, Path], images_dir: Optional[Union[str, Path]] = None) -> PropertyListing:
    """
    Loads listing data from a JSON file and resolves relative image paths.
    """
    path = Path(json_path)
    if not path.is_file():
        raise IngestionError(f"Listing configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise IngestionError(f"Failed to parse JSON listing file {path}: {exc}") from exc

    # If images_dir is specified, discover valid images from that directory
    image_paths: List[str] = []
    if images_dir:
        img_dir_path = Path(images_dir)
        if img_dir_path.is_dir():
            found = sorted([
                str(p.resolve())
                for p in img_dir_path.iterdir()
                if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ])
            image_paths.extend(found)
        else:
            logger.warning(f"Specified images directory does not exist: {images_dir}")

    # Fallback to images listed in JSON if images_dir was empty or not given
    if not image_paths and "image_paths" in data:
        base_dir = path.parent
        for rel_p in data["image_paths"]:
            p = Path(rel_p)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            image_paths.append(str(p))

    return normalize_listing_data(data, image_paths=image_paths)
