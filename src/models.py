"""
Data models and custom exception classes for the Real Estate Video Generation Pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


class RealtyPipelineError(Exception):
    """Base exception for all video generation errors."""
    pass


class IngestionError(RealtyPipelineError):
    """Raised when listing metadata or image assets fail validation."""
    pass


class ScriptingError(RealtyPipelineError):
    """Raised when script generation or entity verification fails."""
    pass


class AudioSynthesisError(RealtyPipelineError):
    """Raised when TTS synthesis or audio mixing encounters an error."""
    pass


class VideoCompositingError(RealtyPipelineError):
    """Raised when frame compositing, animation, or video muxing fails."""
    pass


@dataclass
class PropertyListing:
    """Represents normalized, validated real estate listing attributes."""
    name: str
    location: str
    price: str
    property_type: str
    features: List[str] = field(default_factory=list)
    contact_cta: str = "Contact us today to schedule your private tour"
    image_paths: List[str] = field(default_factory=list)
    agent_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "location": self.location,
            "price": self.price,
            "property_type": self.property_type,
            "features": self.features,
            "contact_cta": self.contact_cta,
            "image_paths": self.image_paths,
            "agent_name": self.agent_name,
            "phone": self.phone,
            "email": self.email,
        }


@dataclass
class SceneStoryboard:
    """Represents an atomic storyboard scene in the promotional narrative."""
    scene_id: int
    image_path: str
    script_text: str
    overlay_title: str
    overlay_subtitle: str
    overlay_badge: Optional[str] = None
    duration: float = 0.0
    audio_path: Optional[str] = None
    scene_type: str = "exploration"  # "hook", "exploration", "outro"


@dataclass
class PipelineConfig:
    """Configuration options controlling video output format, audio, and pacing."""
    aspect_ratio: str = "16:9"  # "16:9" (1920x1080) or "9:16" (1080x1920)
    voice: str = "en-US-ChristopherNeural"
    bgm_path: Optional[str] = None
    bgm_volume: float = 0.15  # 15% ducked volume for speech clarity
    min_scene_duration: float = 3.0
    safety_padding: float = 0.6  # Delta pad in seconds added to speech duration
    target_duration_range: Tuple[float, float] = (20.0, 40.0)
    fps: int = 24
    output_path: str = "exports/output.mp4"
    api_key: Optional[str] = None  # Optional LLM / Voice API Key
    llm_model: str = "gpt-4o-mini"

    @property
    def dimensions(self) -> Tuple[int, int]:
        if self.aspect_ratio == "9:16":
            return (1080, 1920)
        return (1920, 1080)
