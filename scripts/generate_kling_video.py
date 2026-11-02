"""
Kling AI Image-to-Video generation script.
Interfaces directly with the Kling AI API (Singapore & Global endpoints)
using the Kling API key.
"""
import argparse
import json
import logging
import time
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kling_generator")

KLING_BASE_URL = "https://api-singapore.klingai.com"


def submit_image_to_video(
    api_key: str,
    image_url: str,
    prompt: str = "Slow cinematic aerial drone pan of luxury modern estate at golden hour, 4k ultra-realistic real estate video",
    model_name: str = "kling-v2-5-turbo",
    duration: str = "5",
    mode: str = "std",
) -> str:
    """Submits an Image-to-Video generation task to Kling AI."""
    endpoint = f"{KLING_BASE_URL}/v1/videos/image2video"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model_name": model_name,
        "image": image_url,
        "prompt": prompt,
        "duration": duration,
        "mode": mode,
    }

    logger.info(f"Submitting Kling AI Image2Video task (Model: {model_name}, Duration: {duration}s)...")
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=15)
    
    if resp.status_code != 200:
        raise RuntimeError(f"Kling API returned HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Kling API Error (code {data.get('code')}): {data.get('message')}")

    task_id = data["data"]["task_id"]
    logger.info(f"Task successfully submitted! Task ID: {task_id}")
    return task_id


def poll_kling_task(api_key: str, task_id: str, timeout_sec: int = 600, poll_interval: int = 10) -> str:
    """Polls Kling AI task until completion and returns the generated MP4 video URL."""
    endpoint = f"{KLING_BASE_URL}/v1/videos/image2video/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    start = time.time()
    while time.time() - start < timeout_sec:
        r = requests.get(endpoint, headers=headers, timeout=10)
        if r.status_code == 200:
            res = r.json()
            if res.get("code") == 0:
                tdata = res.get("data", {})
                status = tdata.get("task_status")
                logger.info(f"Task {task_id} status: {status}")
                if status == "succeed":
                    videos = tdata.get("task_result", {}).get("videos", [])
                    if videos:
                        return videos[0].get("url")
                elif status == "failed":
                    msg = tdata.get("task_status_msg", "Unknown error")
                    raise RuntimeError(f"Kling AI video generation failed: {msg}")
        time.sleep(poll_interval)

    raise TimeoutError("Kling AI video polling timed out.")


def main():
    parser = argparse.ArgumentParser(description="Kling AI Video Generator")
    parser.add_argument("--key", default="api-key-kling-uXMtBWAVKKGv1cFC7Jfq0AuuKWzb0OqJ4KEx4kamwss")
    parser.add_argument("--image", default="https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=1280&q=80")
    parser.add_argument("--output", default="exports/kling_output.mp4")
    args = parser.parse_args()

    try:
        task_id = submit_image_to_video(api_key=args.key, image_url=args.image)
        video_url = poll_kling_task(api_key=args.key, task_id=task_id)
        logger.info(f"Downloading generated video from {video_url} to {args.output}...")
        v_resp = requests.get(video_url, stream=True, timeout=30)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "wb") as f:
            for chunk in v_resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Video saved successfully to {args.output}")
    except Exception as exc:
        logger.error(f"Kling AI generation error: {exc}")


if __name__ == "__main__":
    main()
