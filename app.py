"""
Streamlit Web Application for Interactive Real Estate Video Generation.
Features two-column layout, instant sample listing presets, live progress,
inline video preview, and direct MP4 export downloads.
"""
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.models import PropertyListing, PipelineConfig, RealtyPipelineError
from src.pipeline import run_pipeline
from src.audio_engine import AVAILABLE_VOICES

# Configure page metadata & layout
st.set_page_config(
    page_title="Realty Studio | AI Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Premium Dark Glassmorphism CSS
CUSTOM_CSS = """
<style>
    /* Dark Theme Core */
    .main {
        background: radial-gradient(circle at top right, #0f172a, #020617);
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Gradient & Badges */
    .title-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .badge-pill {
        display: inline-block;
        background: rgba(217, 119, 6, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    /* Form Cards */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px 0 rgba(217, 119, 6, 0.4) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(217, 119, 6, 0.6) !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Pre-packaged Sample Listing Presets for 1-click evaluation
SAMPLE_PRESETS = {
    "Bel Air Modern Villa ($14.5M)": {
        "name": "The Obsidian Villa",
        "location": "Bel Air, Los Angeles",
        "price": "$14,500,000",
        "property_type": "Ultra-Luxury Architectural Villa",
        "features": (
            "Infinity Edge Horizon Pool\n"
            "Sub-Zero & Gaggenau Chef Kitchen\n"
            "Panoramic Downtown Skyline Views\n"
            "Automated Crestron Smart Home"
        ),
        "contact_cta": "Presented by Beverly Hills Luxury Estates",
        "phone": "+1 (310) 555-0199",
        "email": "inquiries@bhresidences.com",
        "sample_dir": "samples/images/villa",
    },
    "Tribeca Sky Penthouse ($8.2M)": {
        "name": "The Duane Penthouse",
        "location": "Tribeca, New York City",
        "price": "$8,200,000",
        "property_type": "Duplex Penthouse Loft",
        "features": (
            "Private Keyed Elevator Access\n"
            "Wrap-Around Sunset Terrace\n"
            "Calacatta Gold Marble Fireplace\n"
            "Custom Glass Wine Cellar"
        ),
        "contact_cta": "Schedule a private preview with Manhattan Advisory",
        "phone": "+1 (212) 555-0144",
        "email": "tribeca@luxuryadvisory.com",
        "sample_dir": "samples/images/penthouse",
    },
}


def main():
    # Top Header Banner
    st.markdown(
        """
        <div class="title-banner">
            <span class="badge-pill">BROADCAST MULTIMEDIA ENGINE</span>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #ffffff;">
                Realty Studio AI
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                Transform residential listings and photography into broadcast-grade promotional videos in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preset Quick-Loader Row
    st.markdown("### ⚡ Quick-Start Presets")
    preset_cols = st.columns([2, 1, 1, 2])
    selected_preset = preset_cols[0].selectbox(
        "Choose an Evaluation Preset:",
        options=["Custom Listing"] + list(SAMPLE_PRESETS.keys()),
        index=1,
    )

    preset_data = SAMPLE_PRESETS.get(selected_preset, {})

    col_left, col_right = st.columns([5, 5], gap="large")

    with col_left:
        st.subheader("📋 Listing Details & Media")

        prop_name = st.text_input(
            "Property Title",
            value=preset_data.get("name", "The Horizon Estate"),
            placeholder="e.g., The Skyline Penthouse",
        )
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            prop_location = st.text_input(
                "Location / Enclave",
                value=preset_data.get("location", "Aspen, Colorado"),
                placeholder="e.g., Beverly Hills, CA",
            )
            prop_type = st.text_input(
                "Property Type",
                value=preset_data.get("property_type", "Contemporary Alpine Chalet"),
                placeholder="e.g., Luxury Modern Residence",
            )
        with col_meta2:
            prop_price = st.text_input(
                "Listing Price",
                value=preset_data.get("price", "$6,900,000"),
                placeholder="e.g., $4,250,000",
            )
            prop_phone = st.text_input(
                "Contact Phone",
                value=preset_data.get("phone", "+1 (800) 555-0199"),
                placeholder="e.g., (310) 555-0123",
            )

        prop_features_raw = st.text_area(
            "Key Features & Amenities (One per line or comma-separated)",
            value=preset_data.get(
                "features",
                "Floor-to-Ceiling Mountain Windows\nHeated Outdoor Stone Terrace\nPrivate Spa & Cedar Sauna\nDesigner Kitchen Suite",
            ),
            rows=4,
        )

        prop_cta = st.text_input(
            "Call to Action & Agency Headline",
            value=preset_data.get("contact_cta", "Offered exclusively by Prime Luxury Advisory"),
        )

        st.markdown("#### 📸 Photographic Assets")
        uploaded_files = st.file_uploader(
            "Upload 4 to 8 Property Photographs (.jpg, .png, .webp)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )

        if not uploaded_files and "sample_dir" in preset_data:
            st.info(f"Using pre-packaged high-res photos from `{preset_data['sample_dir']}`")

        st.markdown("#### 🎛️ Production Controls")
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            aspect_choice = st.selectbox(
                "Aspect Ratio",
                options=["16:9 Landscape (YouTube/Web)", "9:16 Portrait (Reels/TikTok)"],
                index=0,
            )
            aspect_ratio = "9:16" if "9:16" in aspect_choice else "16:9"

        with col_cfg2:
            voice_label = st.selectbox(
                "Neural TTS Voice",
                options=list(AVAILABLE_VOICES.keys()),
                index=0,
            )
            voice_id = AVAILABLE_VOICES[voice_label]

        with col_cfg3:
            bgm_choice = st.selectbox(
                "Soundtrack Bed",
                options=["Ambient Luxury (Chilled Strings)", "Modern Acoustic (Warm & Lively)", "None (Narration Only)"],
                index=0,
            )

        api_key_input = st.text_input(
            "🔑 Optional LLM API Key (OpenAI / OpenRouter / Compatible)",
            value="",
            type="password",
            help="Provide an API key to enable dynamic neural scriptwriting. If left blank or invalid, the zero-hallucination deterministic template bank is used.",
        )

        generate_clicked = st.button("🚀 Render Broadcast Video", use_container_width=True)

    with col_right:
        st.subheader("🎥 Output Staging & Preview")

        # Map BGM selection
        bgm_path = None
        if "Ambient Luxury" in bgm_choice:
            bgm_path = "assets/audio/ambient_luxury.mp3"
        elif "Modern Acoustic" in bgm_choice:
            bgm_path = "assets/audio/modern_acoustic.mp3"

        if generate_clicked:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def web_progress(msg: str, frac: float):
                progress_bar.progress(frac)
                status_text.markdown(f"**Status:** {msg}")

            try:
                # Resolve image paths: custom uploads or preset directory
                resolved_images: List[str] = []
                temp_upload_dir = None

                if uploaded_files:
                    temp_upload_dir = tempfile.mkdtemp(prefix="realty_uploads_")
                    for up_f in uploaded_files:
                        tmp_path = Path(temp_upload_dir) / up_f.name
                        with open(tmp_path, "wb") as f:
                            f.write(up_f.getbuffer())
                        resolved_images.append(str(tmp_path))
                elif "sample_dir" in preset_data:
                    preset_path = Path(preset_data["sample_dir"])
                    if preset_path.is_dir():
                        resolved_images = sorted([
                            str(p.resolve())
                            for p in preset_path.iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                        ])

                # Fallback to general samples/images if still empty
                if not resolved_images:
                    general_sample_dir = Path("samples/images")
                    if general_sample_dir.is_dir():
                        resolved_images = sorted([
                            str(p.resolve())
                            for p in general_sample_dir.glob("*.jpg")
                        ])

                # Parse features
                features_list = [f.strip() for f in prop_features_raw.replace("\n", ",").split(",") if f.strip()]

                listing = PropertyListing(
                    name=prop_name,
                    location=prop_location,
                    price=prop_price,
                    property_type=prop_type,
                    features=features_list,
                    contact_cta=prop_cta,
                    image_paths=resolved_images,
                    phone=prop_phone,
                )

                output_dest = Path("exports") / f"{prop_name.lower().replace(' ', '_')}_{aspect_ratio.replace(':', '_')}.mp4"

                config = PipelineConfig(
                    aspect_ratio=aspect_ratio,
                    voice=voice_id,
                    bgm_path=bgm_path if (bgm_path and os.path.isfile(bgm_path)) else None,
                    bgm_volume=0.15,
                    output_path=str(output_dest),
                    api_key=api_key_input.strip() if api_key_input else None,
                )

                # Execute pipeline
                final_video_path = run_pipeline(
                    listing_input=listing,
                    config=config,
                    progress_callback=web_progress,
                )

                status_text.success("🎉 Video rendered and synchronized successfully!")
                st.video(str(final_video_path))

                # Download button
                with open(final_video_path, "rb") as vid_file:
                    st.download_button(
                        label="⬇️ Download Broadcast MP4",
                        data=vid_file.read(),
                        file_name=final_video_path.name,
                        mime="video/mp4",
                        use_container_width=True,
                    )

            except RealtyPipelineError as exc:
                status_text.error(f"❌ Processing Error: {exc}")
            except Exception as exc:
                status_text.error(f"❌ Unexpected Error: {exc}")
            finally:
                if temp_upload_dir and os.path.exists(temp_upload_dir):
                    shutil.rmtree(temp_upload_dir, ignore_errors=True)
        else:
            # Placeholder display
            default_preview = Path("exports/sample_output.mp4")
            if default_preview.is_file():
                st.info("Demonstration pre-rendered video:")
                st.video(str(default_preview))
            else:
                st.markdown(
                    """
                    <div style="border: 2px dashed #334155; border-radius: 12px; padding: 48px; text-align: center; color: #64748b;">
                        <h3>Ready for Production</h3>
                        <p>Configure listing parameters on the left and click <b>Render Broadcast Video</b> to generate your synchronized promotional video.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
