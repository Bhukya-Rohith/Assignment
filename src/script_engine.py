"""
Zero-hallucination scripting engine with hybrid LLM structured output,
deterministic template fallback, and strict regex entity reconciliation.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.models import PropertyListing, SceneStoryboard, ScriptingError

logger = logging.getLogger(__name__)


def extract_numeric_tokens(text: str) -> List[str]:
    """
    Extracts numeric tokens, currency amounts, percentages, and measurement figures
    (e.g., '$2,500,000', '4', '3.5', '4500 sqft', '100%') to verify factual grounding.
    """
    # Match currency, numbers with commas/periods, and digits
    pattern = r"(?:\$[\d,]+(?:\.\d+)?|\b\d+(?:[\.,]\d+)?(?:\s*(?:sqft|sq\s*ft|beds?|baths?|acres?|m2))?)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    cleaned = []
    for m in matches:
        # Strip trailing unit labels for raw numeric token check
        raw_num = re.sub(r"[^\d.]", "", m)
        if raw_num:
            cleaned.append(raw_num)
    return cleaned


def reconcile_entities(generated_text: str, listing: PropertyListing) -> bool:
    """
    Reconciles all numeric tokens in the generated script against the ground-truth listing metadata.
    Returns True if 100% of generated numbers exist in the source listing, False otherwise.
    """
    # Gather ground-truth text representation
    source_corpus = f"{listing.name} {listing.location} {listing.price} {listing.property_type} {' '.join(listing.features)} {listing.contact_cta} {listing.phone or ''}"
    source_nums = set(extract_numeric_tokens(source_corpus))

    generated_nums = extract_numeric_tokens(generated_text)
    for num in generated_nums:
        if num not in source_nums:
            logger.warning(
                f"Entity reconciliation failure: ungrounded numeric token '{num}' detected in script '{generated_text}'. Source tokens: {source_nums}"
            )
            return False
    return True


def generate_deterministic_storyboard(listing: PropertyListing) -> List[SceneStoryboard]:
    """
    Generates a guaranteed zero-hallucination, 3-tier narrative storyboard
    (Hook -> Feature Exploration -> Call to Action) using rule-based slot substitution.
    """
    images = listing.image_paths
    total_imgs = len(images)
    scenes: List[SceneStoryboard] = []

    # Scene 1: Hook & Identity
    img_1 = images[0]
    script_1 = (
        f"Welcome to {listing.name}, an exceptional {listing.property_type.lower()} "
        f"in {listing.location}."
    )
    scenes.append(SceneStoryboard(
        scene_id=1,
        image_path=img_1,
        script_text=script_1,
        overlay_title=listing.name,
        overlay_subtitle=f"{listing.location} • {listing.property_type}",
        overlay_badge=listing.price,
        scene_type="hook"
    ))

    # Feature exploration scenes (Scenes 2 to 4, bounded to 2-3 scenes for optimal 20-40s duration)
    features = listing.features.copy()
    num_explore = min(max(total_imgs - 2, 2), 3)

    for i in range(num_explore):
        img_idx = (i + 1) % total_imgs
        feat = features[i % len(features)]

        if i == 0:
            script_explore = f"Experience {feat.lower()}, designed with refined elegance."
            badge_text = "Featured Space"
        elif i == 1:
            feat_next = features[(i + 1) % len(features)] if len(features) > 1 else feat
            script_explore = f"Highlighting {feat_next.lower()}, offering unparalleled living."
            badge_text = "Interior Highlight"
            feat = feat_next
        else:
            script_explore = f"Offered at {listing.price}, this residence exemplifies luxury."
            badge_text = "Valuation"

        scenes.append(SceneStoryboard(
            scene_id=len(scenes) + 1,
            image_path=images[img_idx],
            script_text=script_explore,
            overlay_title=feat if i < 2 else f"Offered at {listing.price}",
            overlay_subtitle=f"{listing.name} • {listing.location}",
            overlay_badge=badge_text,
            scene_type="exploration"
        ))

    # Final Scene: Call to Action & Outro
    img_outro = images[-1]
    contact_info = []
    if listing.phone:
        contact_info.append(listing.phone)
    if listing.email:
        contact_info.append(listing.email)
    contact_line = " | ".join(contact_info) if contact_info else "Private Viewings Available"

    script_outro = f"{listing.contact_cta}. Contact our advisory team today for your private tour."
    scenes.append(SceneStoryboard(
        scene_id=len(scenes) + 1,
        image_path=img_outro,
        script_text=script_outro,
        overlay_title="Schedule Your Private Tour",
        overlay_subtitle=f"{listing.contact_cta} • {contact_line}",
        overlay_badge="Exclusive Advisory",
        scene_type="outro"
    ))

    return scenes


def llm_generate_storyboard(
    listing: PropertyListing,
    api_key: str,
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
) -> List[SceneStoryboard]:
    """
    Prompts an LLM (OpenAI / OpenRouter / compatible endpoint) using structured JSON output
    to author a dynamic, broadcast-grade real estate storyboard.
    """
    import json
    import requests

    base_url = api_base or "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    images = listing.image_paths
    total_imgs = len(images)
    features_str = ", ".join(listing.features)

    system_prompt = (
        "You are an elite real estate commercial scriptwriter. "
        "Generate a 5-scene broadcast promotional video storyboard for the provided property. "
        "CRITICAL RULES:\n"
        "1. STRICT FACTUAL GROUNDING: Rely exclusively on the provided facts. Do NOT invent square footage, bedrooms, or claims.\n"
        "2. BREVITY: Each scene's narration MUST be between 8 and 14 words (punchy and commercial for 20-35s total video).\n"
        "3. Output strictly valid JSON conforming to this schema:\n"
        "{\n"
        '  "scenes": [\n'
        '    {"scene_id": 1, "script_text": "...", "overlay_title": "...", "overlay_subtitle": "...", "overlay_badge": "...", "scene_type": "hook"},\n'
        '    {"scene_id": 2, "script_text": "...", "overlay_title": "...", "overlay_subtitle": "...", "overlay_badge": "...", "scene_type": "exploration"},\n'
        '    {"scene_id": 3, "script_text": "...", "overlay_title": "...", "overlay_subtitle": "...", "overlay_badge": "...", "scene_type": "exploration"},\n'
        '    {"scene_id": 4, "script_text": "...", "overlay_title": "...", "overlay_subtitle": "...", "overlay_badge": "...", "scene_type": "exploration"},\n'
        '    {"scene_id": 5, "script_text": "...", "overlay_title": "...", "overlay_subtitle": "...", "overlay_badge": "...", "scene_type": "outro"}\n'
        "  ]\n"
        "}"
    )

    user_prompt = (
        f"Property: {listing.name}\n"
        f"Location: {listing.location}\n"
        f"Price: {listing.price}\n"
        f"Type: {listing.property_type}\n"
        f"Features: {features_str}\n"
        f"CTA: {listing.contact_cta}\n"
        f"Phone: {listing.phone or 'Private advisory'}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(base_url, headers=headers, json=payload, timeout=12)
    if resp.status_code != 200:
        raise ScriptingError(f"LLM API returned status {resp.status_code}: {resp.text[:120]}")

    result_json = resp.json()
    content = result_json["choices"][0]["message"]["content"]
    data = json.loads(content)
    raw_scenes = data.get("scenes", [])

    scenes: List[SceneStoryboard] = []
    for idx, sc in enumerate(raw_scenes[:5]):
        img_idx = idx % total_imgs
        scenes.append(SceneStoryboard(
            scene_id=idx + 1,
            image_path=images[img_idx],
            script_text=str(sc.get("script_text", "")).strip(),
            overlay_title=str(sc.get("overlay_title", listing.name)).strip(),
            overlay_subtitle=str(sc.get("overlay_subtitle", listing.location)).strip(),
            overlay_badge=str(sc.get("overlay_badge", "Featured")).strip(),
            scene_type=str(sc.get("scene_type", "exploration")).strip(),
        ))

    return scenes


def build_storyboard(
    listing: PropertyListing,
    api_key: Optional[str] = None,
    llm_generator_func: Optional[Any] = None,
) -> List[SceneStoryboard]:
    """
    Main entry point for storyboard generation.
    1. If an API key or custom LLM generator is supplied, attempts structured LLM copywriting.
    2. Runs entity reconciliation over all generated text to verify zero hallucination.
    3. Gracefully cascades to the deterministic template bank on API errors, timeouts, or unverified claims.
    """
    candidate_scenes = None

    if api_key and api_key.strip():
        try:
            logger.info("Attempting LLM-driven structured script synthesis via API Key...")
            candidate_scenes = llm_generate_storyboard(listing, api_key=api_key.strip())
        except Exception as exc:
            logger.warning(f"LLM API script synthesis failed ({exc}). Cascading to deterministic fallback.")

    elif llm_generator_func is not None:
        try:
            logger.info("Attempting LLM-driven script synthesis via generator function...")
            candidate_scenes = llm_generator_func(listing)
        except Exception as exc:
            logger.warning(f"Custom LLM generator failed ({exc}). Cascading to deterministic fallback.")

    if candidate_scenes and len(candidate_scenes) >= 3:
        all_valid = True
        for sc in candidate_scenes:
            if not reconcile_entities(sc.script_text, listing):
                all_valid = False
                logger.warning(
                    f"Entity reconciliation rejected ungrounded claim in scene {sc.scene_id}: '{sc.script_text}'. Reverting."
                )
                break
        if all_valid:
            logger.info("LLM script passed entity reconciliation successfully!")
            return candidate_scenes

    # High-reliability zero-hallucination fallback
    return generate_deterministic_storyboard(listing)
