"""
Unit tests for metadata ingestion, validation, and defensive degradation.
"""
import json
import pytest
from pathlib import Path
from PIL import Image

from src.models import PropertyListing, IngestionError
from src.ingestion import normalize_listing_data, verify_image_file, load_listing_from_json


def test_verify_image_file(tmp_path):
    # Create valid JPEG
    valid_img_path = tmp_path / "valid.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(valid_img_path, "JPEG")

    assert verify_image_file(valid_img_path) is True

    # Create corrupted file
    corrupt_path = tmp_path / "corrupt.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_AN_IMAGE_HEADER")

    assert verify_image_file(corrupt_path) is False

    # Non-existent file
    assert verify_image_file(tmp_path / "ghost.jpg") is False


def test_normalize_listing_fallbacks(tmp_path):
    # Dummy valid image
    img_path = tmp_path / "photo.png"
    Image.new("RGB", (200, 200), color=(0, 255, 0)).save(img_path, "PNG")

    raw_data = {
        "name": "",  # Empty
        "location": "",  # Empty
        "price": "",  # Empty
        "property_type": "",  # Empty
        "features": [],  # Empty
        "contact_cta": "",  # Empty
        "image_paths": [str(img_path)],
    }

    normalized = normalize_listing_data(raw_data)

    # Check that defensive semantic fallbacks were injected
    assert normalized.name == "Luxury Residence"
    assert normalized.location == "Prime Location"
    assert normalized.price == "Price on Request"
    assert normalized.property_type == "Exclusive Property"
    assert len(normalized.features) >= 1
    assert "Contact" in normalized.contact_cta
    # Check asset cycling (should cycle single image to at least 4 scenes)
    assert len(normalized.image_paths) == 4


def test_no_valid_images_raises_error():
    raw_data = {
        "name": "Test Home",
        "image_paths": ["non_existent_1.jpg", "non_existent_2.jpg"],
    }
    with pytest.raises(IngestionError):
        normalize_listing_data(raw_data)


def test_load_sample_json():
    listing = load_listing_from_json("samples/sample_listing_01.json")
    assert listing.name == "The Obsidian Villa"
    assert listing.price == "$14,500,000"
    assert len(listing.features) >= 4
    assert len(listing.image_paths) >= 4
