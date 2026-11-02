"""
Unit tests for zero-hallucination scripting and entity reconciliation.
"""
from src.models import PropertyListing, SceneStoryboard
from src.script_engine import (
    extract_numeric_tokens,
    reconcile_entities,
    generate_deterministic_storyboard,
    build_storyboard,
)


def test_extract_numeric_tokens():
    text = "Offered at $14,500,000 with 5 beds, 6.5 baths across 7500 sqft."
    tokens = extract_numeric_tokens(text)
    assert "14500000" in tokens
    assert "5" in tokens
    assert "6.5" in tokens
    assert "7500" in tokens


def test_entity_reconciliation_detects_hallucinations():
    listing = PropertyListing(
        name="The Obsidian Villa",
        location="Bel Air",
        price="$14,500,000",
        property_type="Architectural Villa",
        features=["Infinity Pool", "Chef Kitchen"],
        contact_cta="Call now",
        image_paths=["dummy1.jpg", "dummy2.jpg", "dummy3.jpg", "dummy4.jpg"],
    )

    # Valid grounded statement
    grounded_script = "Welcome to The Obsidian Villa in Bel Air, offered at $14,500,000."
    assert reconcile_entities(grounded_script, listing) is True

    # Hallucinated number (e.g. ungrounded square footage or altered price)
    hallucinated_script = "This home spans 9500 sqft with 10 bedrooms for $20,000,000."
    assert reconcile_entities(hallucinated_script, listing) is False


def test_storyboard_narrative_arc():
    listing = PropertyListing(
        name="The Glass House",
        location="Aspen",
        price="$9,500,000",
        property_type="Alpine Chalet",
        features=["Heated Terrace", "Wine Cellar", "Private Sauna"],
        contact_cta="Contact Prime Advisory",
        image_paths=["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
    )

    scenes = generate_deterministic_storyboard(listing)
    assert len(scenes) >= 3

    # Check 3-phase narrative structure
    assert scenes[0].scene_type == "hook"
    assert scenes[-1].scene_type == "outro"
    assert any(sc.scene_type == "exploration" for sc in scenes)

    # Verify that hook contains property name & location
    assert listing.name in scenes[0].overlay_title or listing.name in scenes[0].script_text
    assert listing.location in scenes[0].overlay_subtitle or listing.location in scenes[0].script_text


def test_build_storyboard_api_key_graceful_fallback():
    listing = PropertyListing(
        name="The Obsidian Villa",
        location="Bel Air",
        price="$14,500,000",
        property_type="Architectural Villa",
        features=["Infinity Pool", "Chef Kitchen"],
        contact_cta="Call now",
        image_paths=["dummy1.jpg", "dummy2.jpg", "dummy3.jpg", "dummy4.jpg"],
    )

    # Calling with an invalid / unauthenticated API key must never crash;
    # it must gracefully cascade to the validated fallback template.
    scenes = build_storyboard(listing, api_key="sk_dummy_invalid_key")
    assert len(scenes) >= 3
    assert scenes[0].scene_type == "hook"
    assert scenes[-1].scene_type == "outro"
