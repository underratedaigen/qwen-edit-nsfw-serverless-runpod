from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Tuple

import gradio as gr
import requests
from PIL import Image


DEFAULT_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "biqd9c2lr7dqjn")
DEFAULT_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
DEFAULT_API_BASE = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai/v2")
DEFAULT_POLL_INTERVAL = float(os.environ.get("RUNPOD_POLL_INTERVAL", "5"))
DEFAULT_TIMEOUT = int(os.environ.get("RUNPOD_JOB_TIMEOUT", "900"))
MAX_REQUEST_BODY_BYTES = int(os.environ.get("RUNPOD_MAX_REQUEST_BODY_BYTES", "9500000"))
DEFAULT_INPUT_MAX_LONG_EDGE = int(os.environ.get("RUNPOD_INPUT_MAX_LONG_EDGE", "1920"))
DEFAULT_INPUT_MAX_SHORT_EDGE = int(os.environ.get("RUNPOD_INPUT_MAX_SHORT_EDGE", "1080"))
DEFAULT_INPUT_MAX_PIXELS = int(os.environ.get("RUNPOD_INPUT_MAX_PIXELS", str(1920 * 1080)))
INPUT_LONG_EDGE_CANDIDATES = [None, 1792, 1536, 1408, 1280, 1152, 1024]
INPUT_JPEG_QUALITIES = [95, 90, 85, 80, 75, 70, 65]


def image_to_data_uri(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def data_uri_to_image(image_url: str) -> Image.Image:
    payload = image_url.split(",", 1)[1] if image_url.startswith("data:image/") else image_url
    return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")


def remote_url_to_image(image_url: str) -> Image.Image:
    response = requests.get(image_url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _flatten_for_upload(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.getchannel("A") if "A" in image.getbands() else image.getchannel(image.getbands()[-1])
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")


def _resize_to_long_edge(image: Image.Image, max_long_edge: int | None) -> Image.Image:
    if max_long_edge is None:
        return image

    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image

    scale = max_long_edge / float(long_edge)
    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        resample=Image.Resampling.LANCZOS,
    )
    return resized


def _fit_image_to_limits(image: Image.Image, max_long_edge: int, max_short_edge: int, max_pixels: int) -> Image.Image:
    width, height = image.size
    scale = 1.0
    long_edge = max(width, height)
    short_edge = min(width, height)
    area = width * height

    if max_long_edge > 0 and long_edge > max_long_edge:
        scale = min(scale, max_long_edge / float(long_edge))
    if max_short_edge > 0 and short_edge > max_short_edge:
        scale = min(scale, max_short_edge / float(short_edge))
    if max_pixels > 0 and area > max_pixels:
        scale = min(scale, (max_pixels / float(area)) ** 0.5)

    if scale >= 0.9999:
        return image

    return image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        resample=Image.Resampling.LANCZOS,
    )


def _encode_upload_candidate(image: Image.Image, image_format: str, quality: int | None = None) -> tuple[str, int]:
    buffer = BytesIO()
    save_kwargs: Dict[str, Any] = {}
    if image_format.upper() == "JPEG":
        save_kwargs.update(
            {
                "format": "JPEG",
                "quality": int(quality if quality is not None else 90),
                "optimize": True,
                "progressive": True,
                "subsampling": 0,
            }
        )
    else:
        save_kwargs.update({"format": "PNG", "optimize": True, "compress_level": 9})

    image.save(buffer, **save_kwargs)
    encoded_bytes = buffer.getvalue()
    encoded = base64.b64encode(encoded_bytes).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{encoded}", len(encoded_bytes)


def _build_request_payload(
    image_data_uri: str,
    prompt: str,
    seed: int,
    auto_steps: bool,
    num_inference_steps: int,
    auto_guidance: bool,
    true_guidance_scale: float,
    quality_mode: str,
    rewrite_prompt: bool,
    lock_face_identity: bool,
    face_mask_strategy: str,
    face_mask_mode: str,
    face_mask_strength: float,
    body_cleanup: bool,
    body_cleanup_strength: float,
    debug_masks: bool,
    postprocess_upscale_mode: str,
    width: str,
    height: str,
    num_images_per_prompt: int,
    output_format: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "input": {
            "prompt": prompt,
            "images": [{"base64": image_data_uri}],
            "seed": seed,
            "randomize_seed": False,
            "quality_mode": quality_mode,
            "rewrite_prompt": rewrite_prompt,
            "lock_face_identity": lock_face_identity,
            "face_mask_strategy": face_mask_strategy,
            "face_mask_mode": face_mask_mode,
            "face_mask_strength": face_mask_strength,
            "body_cleanup": body_cleanup,
            "body_cleanup_strength": body_cleanup_strength,
            "debug_masks": debug_masks,
            "postprocess_upscale_mode": postprocess_upscale_mode,
            "num_images_per_prompt": num_images_per_prompt,
            "output_format": output_format,
        }
    }

    if not auto_guidance:
        payload["input"]["true_guidance_scale"] = true_guidance_scale
    if not auto_steps:
        payload["input"]["num_inference_steps"] = num_inference_steps
    if str(width).strip():
        payload["input"]["width"] = int(float(width))
    if str(height).strip():
        payload["input"]["height"] = int(float(height))

    return payload


def build_payload(
    image: Image.Image,
    prompt: str,
    seed: int,
    auto_steps: bool,
    num_inference_steps: int,
    auto_guidance: bool,
    true_guidance_scale: float,
    quality_mode: str,
    rewrite_prompt: bool,
    lock_face_identity: bool,
    face_mask_strategy: str,
    face_mask_mode: str,
    face_mask_strength: float,
    body_cleanup: bool,
    body_cleanup_strength: float,
    debug_masks: bool,
    postprocess_upscale_mode: str,
    width: str,
    height: str,
    num_images_per_prompt: int,
    output_format: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    base_image = _fit_image_to_limits(
        _flatten_for_upload(image),
        max_long_edge=DEFAULT_INPUT_MAX_LONG_EDGE,
        max_short_edge=DEFAULT_INPUT_MAX_SHORT_EDGE,
        max_pixels=DEFAULT_INPUT_MAX_PIXELS,
    )
    best_failure: Dict[str, Any] | None = None

    for max_long_edge in INPUT_LONG_EDGE_CANDIDATES:
        resized = _resize_to_long_edge(base_image, max_long_edge)

        for image_format, quality in [("PNG", None)] + [("JPEG", q) for q in INPUT_JPEG_QUALITIES]:
            try:
                image_data_uri, encoded_size = _encode_upload_candidate(resized, image_format=image_format, quality=quality)
            except OSError:
                continue

            payload = _build_request_payload(
                image_data_uri=image_data_uri,
                prompt=prompt,
                seed=seed,
                auto_steps=auto_steps,
                num_inference_steps=num_inference_steps,
                auto_guidance=auto_guidance,
                true_guidance_scale=true_guidance_scale,
                quality_mode=quality_mode,
                rewrite_prompt=rewrite_prompt,
                lock_face_identity=lock_face_identity,
                face_mask_strategy=face_mask_strategy,
                face_mask_mode=face_mask_mode,
                face_mask_strength=face_mask_strength,
                body_cleanup=body_cleanup,
                body_cleanup_strength=body_cleanup_strength,
                debug_masks=debug_masks,
                postprocess_upscale_mode=postprocess_upscale_mode,
                width=width,
                height=height,
                num_images_per_prompt=num_images_per_prompt,
                output_format=output_format,
            )
            body_size = len(json.dumps(payload).encode("utf-8"))
            metadata = {
                "upload_format": image_format.lower(),
                "upload_quality": quality,
                "upload_width": resized.width,
                "upload_height": resized.height,
                "upload_bytes": encoded_size,
                "request_body_bytes": body_size,
            }
            if body_size <= MAX_REQUEST_BODY_BYTES:
                return payload, metadata
            best_failure = metadata

    failure = best_failure or {}
    raise ValueError(
        "Input image is still too large for Runpod after automatic compression. "
        f"Last attempt: {failure.get('upload_width', 'n/a')}x{failure.get('upload_height', 'n/a')} "
        f"{failure.get('upload_format', 'n/a')} request body {failure.get('request_body_bytes', 'n/a')} bytes."
    )

def submit_job(
    endpoint_id: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{DEFAULT_API_BASE}/{endpoint_id}/run",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def poll_job(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        response = requests.get(
            f"{DEFAULT_API_BASE}/{endpoint_id}/status/{job_id}",
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return data

        time.sleep(poll_interval)

    raise TimeoutError(f"Job {job_id} did not finish within {timeout_seconds} seconds.")


def extract_image(result: Dict[str, Any]) -> Image.Image:
    output = result.get("output", {})
    images = output.get("images", [])
    if not images:
        raise ValueError("No images found in the Runpod response.")

    image_url = images[0].get("image_url")
    if not image_url:
        raise ValueError("The first output image did not contain an image_url.")

    if image_url.startswith("data:image/"):
        return data_uri_to_image(image_url)

    return remote_url_to_image(image_url)


def extract_debug_gallery(result: Dict[str, Any]) -> List[Tuple[Image.Image, str]]:
    output = result.get("output", {})
    debug_masks = output.get("debug_masks", [])
    if not debug_masks:
        return []

    items = debug_masks[0].get("items", [])
    gallery: List[Tuple[Image.Image, str]] = []
    for item in items:
        image_url = item.get("image_url")
        if not image_url:
            continue
        image = data_uri_to_image(image_url) if image_url.startswith("data:image/") else remote_url_to_image(image_url)
        gallery.append((image, str(item.get("name", "debug"))))
    return gallery


def run_inference(
    endpoint_id: str,
    api_key: str,
    image: Image.Image,
    prompt: str,
    seed: int,
    auto_steps: bool,
    num_inference_steps: int,
    auto_guidance: bool,
    true_guidance_scale: float,
    quality_mode: str,
    rewrite_prompt: bool,
    lock_face_identity: bool,
    face_mask_strategy: str,
    face_mask_mode: str,
    face_mask_strength: float,
    body_cleanup: bool,
    body_cleanup_strength: float,
    debug_masks: bool,
    postprocess_upscale_mode: str,
    width: str,
    height: str,
    num_images_per_prompt: int,
    output_format: str,
) -> Tuple[Image.Image | None, str, str, List[Tuple[Image.Image, str]]]:
    if not endpoint_id.strip():
        raise gr.Error("Endpoint ID is required.")
    if not api_key.strip():
        raise gr.Error("Runpod API key is required.")
    if image is None:
        raise gr.Error("Upload an image first.")
    if not prompt.strip():
        raise gr.Error("Prompt is required.")

    payload, input_upload_meta = build_payload(
        image=image,
        prompt=prompt,
        seed=int(seed),
        auto_steps=bool(auto_steps),
        num_inference_steps=int(num_inference_steps),
        auto_guidance=bool(auto_guidance),
        true_guidance_scale=float(true_guidance_scale),
        quality_mode=quality_mode,
        rewrite_prompt=bool(rewrite_prompt),
        lock_face_identity=bool(lock_face_identity),
        face_mask_strategy=face_mask_strategy,
        face_mask_mode=face_mask_mode,
        face_mask_strength=float(face_mask_strength),
        body_cleanup=bool(body_cleanup),
        body_cleanup_strength=float(body_cleanup_strength),
        debug_masks=bool(debug_masks),
        postprocess_upscale_mode=postprocess_upscale_mode,
        width=width,
        height=height,
        num_images_per_prompt=int(num_images_per_prompt),
        output_format=output_format.lower(),
    )

    try:
        submitted = submit_job(endpoint_id.strip(), api_key.strip(), payload)
        job_id = submitted["id"]
        result = poll_job(endpoint_id.strip(), api_key.strip(), job_id)
        output_image = extract_image(result) if result.get("status") == "COMPLETED" else None
        debug_gallery = extract_debug_gallery(result)
        output = result.get("output") or {}
        output_images = ((result.get("output") or {}).get("images") or [])
        first_image = output_images[0] if output_images else {}
        generation = output.get("generation") or {}
        face_masking = output.get("face_masking") or []
        first_mask = face_masking[0] if face_masking else {}
        first_drift = first_mask.get("identity_drift") or {}
        body_cleanup_items = output.get("body_cleanup") or []
        first_cleanup = body_cleanup_items[0] if body_cleanup_items else {}
        strategy_text = first_mask.get("strategy_used") or first_mask.get("strategy_requested") or output.get("face_mask_strategy", "n/a")
        mode_text = first_mask.get("mode_used") or first_mask.get("mode") or output.get("face_mask_mode", "n/a")
        delivered_resolution_text = (
            f"{first_image.get('width', 'n/a')}x{first_image.get('height', 'n/a')}"
            if first_image
            else "n/a"
        )
        generated_resolution_text = (
            f"{first_image.get('original_width', 'n/a')}x{first_image.get('original_height', 'n/a')}"
            if first_image
            else "n/a"
        )
        source_target_resolution_text = (
            f"{generation.get('source_output_width', 'n/a')}x{generation.get('source_output_height', 'n/a')}"
            if generation.get("preserve_source_exact_size")
            else "n/a"
        )
        attempt_count = len(generation.get("attempts") or [])
        upload_quality_text = (
            f" q{input_upload_meta.get('upload_quality')}" if input_upload_meta.get("upload_quality") else ""
        )
        status_text = (
            f"Job {job_id}\n"
            f"Status: {result.get('status')}\n"
            f"Upload prep: {input_upload_meta.get('upload_width', 'n/a')}x{input_upload_meta.get('upload_height', 'n/a')} "
            f"{input_upload_meta.get('upload_format', 'n/a')}"
            f"{upload_quality_text} "
            f"({input_upload_meta.get('request_body_bytes', 'n/a')} bytes)\n"
            f"Mask: {first_mask.get('engine', 'n/a')} / {strategy_text} / {mode_text}\n"
            f"Regime: {first_mask.get('edit_regime', 'n/a')} ({first_mask.get('routing_reason', 'n/a')})\n"
            f"Quality: {generation.get('quality_mode', 'n/a')} ({generation.get('prompt_intent', 'n/a')})\n"
            f"Source target: {source_target_resolution_text}\n"
            f"Generated: {generated_resolution_text}\n"
            f"Delivered: {delivered_resolution_text}\n"
            f"Identity drift: {first_drift.get('score', 'n/a')}\n"
            f"Body cleanup: {'used' if first_cleanup.get('body_cleanup_applied') else first_cleanup.get('body_cleanup_reason', 'n/a')}\n"
            f"Surface effects: {'used' if first_mask.get('surface_effect_recovery_applied') else first_mask.get('surface_effect_recovery_reason', first_mask.get('liquid_recovery_reason', 'not requested'))}\n"
            f"Face coverage: {generation.get('face_coverage', 'n/a')}\n"
            f"Attempts: {attempt_count}\n"
            f"Delay: {result.get('delayTime', 'n/a')} ms\n"
            f"Execution: {result.get('executionTime', 'n/a')} ms"
        )
        return output_image, status_text, json.dumps(result, indent=2), debug_gallery
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else str(exc)
        raise gr.Error(f"Runpod request failed: {body}") from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="Runpod Qwen Image Test Client") as demo:
    gr.Markdown("# Runpod Qwen Image Test Client")
    gr.Markdown(
        "Upload one image, enter a prompt, and test your Runpod endpoint. "
        "The worker now uses a strict identity-lock path, source-dominant face reinforcement, optional face-surface recovery for wet or makeup-style prompts, "
        "non-face body cleanup for skin/clothing artifacts, adaptive quality planning, and debug masks."
    )

    with gr.Row():
        endpoint_id = gr.Textbox(label="Endpoint ID", value=DEFAULT_ENDPOINT_ID)
        api_key = gr.Textbox(label="Runpod API Key", value=DEFAULT_API_KEY, type="password")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="pil")
        output_image = gr.Image(label="Output Image", type="pil")

    prompt = gr.Textbox(
        label="Prompt",
        lines=3,
        placeholder='Example: Replace the sign text with "Runpod ready" while preserving the style.',
    )

    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            seed = gr.Number(label="Seed", value=42, precision=0)
            auto_steps = gr.Checkbox(label="Auto Steps", value=True)
            num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=12, step=1, value=6)
            auto_guidance = gr.Checkbox(label="Auto Guidance", value=True)
            true_guidance_scale = gr.Slider(label="True Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.3)
            quality_mode = gr.Dropdown(label="Quality Mode", choices=["speed", "balanced", "quality"], value="balanced")

        with gr.Row():
            rewrite_prompt = gr.Checkbox(label="Rewrite Prompt", value=False)
            lock_face_identity = gr.Checkbox(label="Lock Face Identity", value=True)
            face_mask_strategy = gr.Dropdown(
                label="Face Strategy",
                choices=["auto", "strict_identity", "smart", "preserve_skin", "legacy", "off"],
                value="auto",
            )
            face_mask_mode = gr.Dropdown(
                label="Face Mode",
                choices=["strict", "balanced", "surface_fx", "off"],
                value="strict",
            )
            face_mask_strength = gr.Slider(label="Mask Strength", minimum=0.0, maximum=1.0, step=0.01, value=0.86)
            body_cleanup = gr.Checkbox(label="Body Cleanup", value=False)
            body_cleanup_strength = gr.Slider(label="Cleanup Strength", minimum=0.0, maximum=1.0, step=0.01, value=0.45)
            debug_masks = gr.Checkbox(label="Debug Masks", value=False)
            num_images_per_prompt = gr.Slider(label="Images Per Prompt", minimum=1, maximum=4, step=1, value=1)
            output_format = gr.Dropdown(label="Output Format", choices=["png", "jpeg"], value="png")

        with gr.Row():
            postprocess_upscale_mode = gr.Dropdown(
                label="Upscale Mode",
                choices=["detail", "classic", "auto", "off"],
                value="detail",
            )
            width = gr.Textbox(label="Width (optional)", placeholder="Leave blank to match source size")
            height = gr.Textbox(label="Height (optional)", placeholder="Leave blank to match source size")

    run_button = gr.Button("Run Test", variant="primary")
    status_box = gr.Textbox(label="Status", lines=8)
    raw_response = gr.Code(label="Raw Runpod Response", language="json")
    debug_gallery = gr.Gallery(label="Debug Masks", columns=4, height=320, preview=True)

    run_button.click(
        fn=run_inference,
        inputs=[
            endpoint_id,
            api_key,
            input_image,
            prompt,
            seed,
            auto_steps,
            num_inference_steps,
            auto_guidance,
            true_guidance_scale,
            quality_mode,
            rewrite_prompt,
            lock_face_identity,
            face_mask_strategy,
            face_mask_mode,
            face_mask_strength,
            body_cleanup,
            body_cleanup_strength,
            debug_masks,
            postprocess_upscale_mode,
            width,
            height,
            num_images_per_prompt,
            output_format,
        ],
        outputs=[output_image, status_box, raw_response, debug_gallery],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("RUNPOD_TEST_CLIENT_PORT", "7861")))
