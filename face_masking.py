from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


try:
    import mediapipe as mp
except Exception:  # pragma: no cover - dependency is only required in the Runpod worker image
    mp = None

try:
    import torch
    import facer
except Exception:  # pragma: no cover - parser is lazily loaded in the Runpod worker image
    torch = None
    facer = None


NOSE_INDICES = [1, 2, 4, 5, 6, 19, 45, 48, 49, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 279, 289, 344, 440]
LEFT_EYE_CENTER_INDICES = [33, 133, 159, 145]
RIGHT_EYE_CENTER_INDICES = [362, 263, 386, 374]
MOUTH_CORNER_LEFT_INDEX = 61
MOUTH_CORNER_RIGHT_INDEX = 291
NOSE_TIP_INDEX = 1

PARSER_SKIN_LABELS = {"face"}
PARSER_BROW_LABELS = {"lb", "rb"}
PARSER_EYE_LABELS = {"le", "re"}
PARSER_NOSE_LABELS = {"nose"}
PARSER_LIP_LABELS = {"ulip", "llip", "imouth"}
PARSER_HAIR_LABELS = {"hair"}
PARSER_FACE_REGION_LABELS = (
    PARSER_SKIN_LABELS
    | PARSER_BROW_LABELS
    | PARSER_EYE_LABELS
    | PARSER_NOSE_LABELS
    | PARSER_LIP_LABELS
)

DEBUG_REGION_COLORS = {
    "face": (76, 201, 240),
    "surface": (67, 170, 139),
    "core": (244, 63, 94),
    "contour": (251, 191, 36),
    "hairline": (168, 85, 247),
}
PARSER_LABEL_COLORS = {
    "background": (0, 0, 0),
    "face": (255, 250, 79),
    "lb": (255, 125, 138),
    "rb": (213, 32, 29),
    "le": (0, 144, 187),
    "re": (0, 196, 253),
    "nose": (255, 129, 54),
    "ulip": (88, 233, 135),
    "imouth": (255, 76, 249),
    "llip": (0, 117, 27),
    "hair": (255, 0, 0),
}

POSITION_CHANGE_HINTS = (
    "different pose",
    "new pose",
    "change pose",
    "reposition",
    "change position",
    "different position",
    "turn around",
    "turn sideways",
    "side profile",
    "profile view",
    "back view",
    "facing away",
    "look away",
    "walk",
    "walking",
    "run",
    "running",
    "jump",
    "jumping",
    "dance",
    "dancing",
    "sit down",
    "sitting",
    "stand up",
    "standing",
    "kneeling",
    "crouching",
    "lying down",
    "laying down",
    "lean back",
    "lean forward",
    "raise arm",
    "raise arms",
    "lift arm",
    "lift arms",
    "hands up",
    "arms up",
    "move closer",
    "move farther",
    "closer to camera",
    "farther from camera",
    "full body",
    "wide shot",
    "medium shot",
    "close-up",
    "close up",
)

POSITION_CHANGE_VERBS = (
    "raise",
    "lift",
    "move",
    "turn",
    "rotate",
    "bend",
    "lean",
    "sit",
    "stand",
    "kneel",
    "crouch",
    "lie",
    "lay",
    "walk",
    "run",
    "jump",
    "dance",
    "step",
)

POSITION_CHANGE_TARGETS = (
    "arm",
    "arms",
    "hand",
    "hands",
    "leg",
    "legs",
    "body",
    "torso",
    "hip",
    "hips",
    "shoulder",
    "shoulders",
    "head",
    "pose",
    "position",
)

LIQUID_HINTS = (
    "water drop",
    "water drops",
    "water droplet",
    "water droplets",
    "droplet",
    "droplets",
    "water bead",
    "water beads",
    "wet",
    "wet face",
    "wet skin",
    "sweat",
    "sweaty",
    "sweating",
    "tear",
    "tears",
    "tear streak",
    "teardrop",
    "dew",
    "dewy",
    "moist",
    "moisture",
    "condensation",
    "fresh from the pool",
    "just in the pool",
    "just out of the pool",
    "pool water",
    "raindrops",
    "rain drops",
)

SHEEN_HINTS = (
    "wet",
    "wet face",
    "wet skin",
    "sweat",
    "sweaty",
    "sweating",
    "dew",
    "dewy",
    "moist",
    "moisture",
    "glistening",
    "shimmer",
)

SURFACE_EFFECT_HINTS = tuple(
    dict.fromkeys(
        (
            *LIQUID_HINTS,
            "glitter",
            "glittery",
            "sparkle",
            "sparkly",
            "shimmer",
            "shimmery",
            "makeup",
            "eyelash",
            "eyelashes",
            "lashes",
            "eyeliner",
            "mascara",
            "lipstick",
            "blush",
            "freckles",
        )
    )
)

SURFACE_EFFECT_CORE_HINTS = (
    "eyelash",
    "eyelashes",
    "lashes",
    "eyeliner",
    "mascara",
    "lipstick",
)

SURFACE_EFFECT_COLOR_HINTS = (
    "makeup",
    "lipstick",
    "blush",
    "glitter",
    "glittery",
    "sparkle",
    "sparkly",
    "shimmer",
    "shimmery",
    "freckles",
)


def _connection_ids(connections: Iterable[tuple[int, int]]) -> list[int]:
    ids: set[int] = set()
    for left, right in connections:
        ids.add(int(left))
        ids.add(int(right))
    return sorted(ids)


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) <= 2:
        return list(unique)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _mean_point(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    array = np.asarray(points, dtype=np.float32)
    return float(array[:, 0].mean()), float(array[:, 1].mean())


def _fit_affine(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    rows = []
    values = []
    for (sx, sy), (tx, ty) in zip(source_points, target_points):
        rows.append([sx, sy, 1.0, 0.0, 0.0, 0.0])
        rows.append([0.0, 0.0, 0.0, sx, sy, 1.0])
        values.extend([tx, ty])

    matrix, _, _, _ = np.linalg.lstsq(np.asarray(rows, dtype=np.float32), np.asarray(values, dtype=np.float32), rcond=None)
    affine = np.array(
        [
            [matrix[0], matrix[1], matrix[2]],
            [matrix[3], matrix[4], matrix[5]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return affine


def _apply_transform(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be a Nx2 array")

    homogenous = np.concatenate(
        [points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    transformed = (affine @ homogenous.T).T
    return transformed[:, :2]


def _fit_similarity(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    if source_points.shape != target_points.shape:
        raise ValueError("source_points and target_points must have matching shapes")
    if source_points.shape[0] < 2:
        raise ValueError("At least two points are required for similarity alignment")

    source = source_points.astype(np.float32)
    target = target_points.astype(np.float32)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = (source_centered.T @ target_centered) / float(source.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_variance <= 1e-6:
        raise ValueError("source points do not have enough variance for similarity alignment")

    scale = float(np.sum(singular_values) / source_variance)
    translation = target_mean - (scale * (rotation @ source_mean))

    affine = np.array(
        [
            [scale * rotation[0, 0], scale * rotation[0, 1], translation[0]],
            [scale * rotation[1, 0], scale * rotation[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return affine


def _normalized_point_error(source_points: np.ndarray, target_points: np.ndarray, affine: np.ndarray) -> float:
    transformed = _apply_transform(source_points, affine)
    residuals = np.linalg.norm(transformed - target_points.astype(np.float32), axis=1)
    target_extent = np.linalg.norm(target_points.max(axis=0) - target_points.min(axis=0))
    return float(residuals.mean() / max(float(target_extent), 1.0))


def _warp_image(image: Image.Image, affine: np.ndarray, output_size: tuple[int, int]) -> Image.Image:
    inverse = np.linalg.inv(affine)
    coeffs = (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(inverse[0, 2]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(inverse[1, 2]),
    )
    return image.convert("RGB").transform(output_size, Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC)


def _expand_mask(mask: Image.Image, expand_px: int, blur_radius: float) -> Image.Image:
    expanded = mask
    if expand_px > 0:
        filter_size = max(3, (expand_px * 2) + 1)
        expanded = expanded.filter(ImageFilter.MaxFilter(size=filter_size))
    if blur_radius > 0:
        expanded = expanded.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return expanded


def _contract_mask(mask: Image.Image, contract_px: int, blur_radius: float) -> Image.Image:
    contracted = mask
    if contract_px > 0:
        filter_size = max(3, (contract_px * 2) + 1)
        contracted = contracted.filter(ImageFilter.MinFilter(size=filter_size))
    if blur_radius > 0:
        contracted = contracted.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return contracted


def _mask_to_array(mask: Image.Image) -> np.ndarray:
    return (np.asarray(mask, dtype=np.float32) / 255.0)[..., None]


def _luminance_array(image_or_array: Image.Image | np.ndarray) -> np.ndarray:
    array = image_or_array
    if isinstance(image_or_array, Image.Image):
        array = np.asarray(image_or_array.convert("RGB"), dtype=np.float32)
    elif array.dtype != np.float32:
        array = array.astype(np.float32)

    return (0.299 * array[..., 0]) + (0.587 * array[..., 1]) + (0.114 * array[..., 2])


def _blur_rgb_array(image_or_array: Image.Image | np.ndarray, radius: float) -> np.ndarray:
    if isinstance(image_or_array, Image.Image):
        image = image_or_array.convert("RGB")
    else:
        image = Image.fromarray(np.clip(image_or_array, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)


def _mask_from_points(
    size: tuple[int, int],
    points: Sequence[tuple[float, float]] | np.ndarray,
    blur_radius: float,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    polygon = _convex_hull([(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float32)])
    if len(polygon) >= 3:
        draw.polygon(polygon, fill=255)
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return mask


def _mask_from_ellipse(
    size: tuple[int, int],
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    blur_radius: float,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = center
    if radius_x <= 0.0 or radius_y <= 0.0:
        return mask
    draw.ellipse(
        (
            float(cx - radius_x),
            float(cy - radius_y),
            float(cx + radius_x),
            float(cy + radius_y),
        ),
        fill=255,
    )
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return mask


def _scale_points(
    points: np.ndarray,
    center: tuple[float, float],
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    scaled = points.astype(np.float32).copy()
    scaled[:, 0] = ((scaled[:, 0] - center[0]) * scale_x) + center[0]
    scaled[:, 1] = ((scaled[:, 1] - center[1]) * scale_y) + center[1]
    return scaled


def _scale_image_about(
    image: Image.Image,
    center: tuple[float, float],
    scale: float,
    output_size: tuple[int, int],
) -> Image.Image:
    affine = np.array(
        [
            [scale, 0.0, center[0] - (scale * center[0])],
            [0.0, scale, center[1] - (scale * center[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return _warp_image(image, affine, output_size)


def _union_masks(*masks: Image.Image) -> Image.Image:
    valid_masks = [np.asarray(mask, dtype=np.uint8) for mask in masks if mask is not None]
    if not valid_masks:
        raise ValueError("At least one mask is required for union.")
    union = valid_masks[0]
    for mask_array in valid_masks[1:]:
        union = np.maximum(union, mask_array)
    return Image.fromarray(union, mode="L")


def _subtract_masks(base: Image.Image, *subtract_masks: Image.Image, blur_radius: float = 0.0) -> Image.Image:
    result = np.asarray(base, dtype=np.float32)
    for mask in subtract_masks:
        if mask is not None:
            result -= np.asarray(mask, dtype=np.float32)
    result = np.clip(result, 0.0, 255.0)
    output = Image.fromarray(result.astype(np.uint8), mode="L")
    if blur_radius > 0:
        output = output.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return output


def _intersect_masks(*masks: Image.Image) -> Image.Image:
    valid_masks = [np.asarray(mask, dtype=np.uint8) for mask in masks if mask is not None]
    if not valid_masks:
        raise ValueError("At least one mask is required for intersection.")
    intersection = valid_masks[0]
    for mask_array in valid_masks[1:]:
        intersection = np.minimum(intersection, mask_array)
    return Image.fromarray(intersection, mode="L")


def _mask_from_labels(
    labels: np.ndarray,
    label_names: Sequence[str],
    names: set[str],
    expand_px: int = 0,
    blur_radius: float = 0.0,
) -> Image.Image:
    label_map = {str(name): index for index, name in enumerate(label_names)}
    active_ids = [label_map[name] for name in names if name in label_map]
    if not active_ids:
        return Image.new("L", (labels.shape[1], labels.shape[0]), 0)

    mask = np.isin(labels, active_ids).astype(np.uint8) * 255
    return _expand_mask(Image.fromarray(mask, mode="L"), expand_px=expand_px, blur_radius=blur_radius)


def _colorize_labels(labels: np.ndarray, label_names: Sequence[str]) -> Image.Image:
    height, width = labels.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    for index, label_name in enumerate(label_names):
        colored[labels == index] = PARSER_LABEL_COLORS.get(label_name, (255, 255, 255))
    return Image.fromarray(colored, mode="RGB")


def _overlay_regions(base_image: Image.Image, regions: Dict[str, Image.Image]) -> Image.Image:
    overlay = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    for name, mask in regions.items():
        color = np.asarray(DEBUG_REGION_COLORS.get(name, (255, 255, 255)), dtype=np.float32)
        alpha = _mask_to_array(mask) * 0.35
        overlay = overlay * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def _normalize_prompt(prompt: str | None) -> str:
    return " ".join(str(prompt or "").lower().split())


def _position_change_hints(prompt: str | None) -> list[str]:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        return []

    matches: list[str] = []
    for term in POSITION_CHANGE_HINTS:
        if term in normalized:
            matches.append(term)

    if any(term in normalized for term in POSITION_CHANGE_VERBS) and any(
        term in normalized for term in POSITION_CHANGE_TARGETS
    ):
        matches.append("body-motion")

    ordered: list[str] = []
    seen: set[str] = set()
    for match in matches:
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered


def _is_liquid_request(prompt: str | None) -> bool:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        return False
    return any(term in normalized for term in LIQUID_HINTS)


def _uses_broad_surface_sheen(prompt: str | None) -> bool:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        return False
    return any(term in normalized for term in SHEEN_HINTS)


def _surface_effect_terms(prompt: str | None) -> list[str]:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        return []

    ordered: list[str] = []
    seen: set[str] = set()
    for term in SURFACE_EFFECT_HINTS:
        if term in normalized and term not in seen:
            seen.add(term)
            ordered.append(term)
    return ordered


def _requests_surface_effect(prompt: str | None) -> bool:
    return bool(_surface_effect_terms(prompt))


def _uses_core_surface_effects(prompt: str | None) -> bool:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        return False
    return any(term in normalized for term in SURFACE_EFFECT_CORE_HINTS)


def _uses_color_surface_effects(prompt: str | None) -> bool:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        return False
    return any(term in normalized for term in SURFACE_EFFECT_COLOR_HINTS)


def _detect_exposed_skin_mask(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)

    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    h = hsv[..., 0].astype(np.float32) * (360.0 / 255.0)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    cb = 128.0 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
    cr = 128.0 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)

    rgb_rule = (
        (r > 45)
        & (g > 25)
        & (b > 10)
        & ((max_channel - min_channel) > 15)
        & (np.abs(r - g) > 10)
        & (r > g)
        & (r > b)
    )
    ycrcb_rule = (cr >= 132) & (cr <= 178) & (cb >= 82) & (cb <= 135)
    hsv_rule = (((h <= 45) | (h >= 335)) & (s >= 0.08) & (s <= 0.72) & (v >= 0.18))
    near_white = (r > 238) & (g > 232) & (b > 226) & (np.abs(r - g) < 10) & (np.abs(r - b) < 18)

    mask = ((rgb_rule & hsv_rule) | (ycrcb_rule & hsv_rule)) & (luminance > 30) & ~near_white
    pil_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

    base = max(image.size)
    pil_mask = _contract_mask(
        pil_mask,
        contract_px=max(1, int(base / 520)),
        blur_radius=max(1.0, base / 560.0),
    )
    pil_mask = _expand_mask(
        pil_mask,
        expand_px=max(1, int(base / 420)),
        blur_radius=max(1.5, base / 360.0),
    )
    return pil_mask


@dataclass
class FaceMaskResult:
    image: Image.Image
    applied: bool
    mode: str
    reason: str
    engine: str = "legacy"
    metadata: Dict[str, Any] = field(default_factory=dict)
    debug_images: Dict[str, Image.Image] = field(default_factory=dict)


@dataclass
class ParserFaceData:
    labels: np.ndarray
    label_names: list[str]
    rect: tuple[float, float, float, float]
    score: float | None = None


class _FacerParserRuntime:
    def __init__(self, parser_model: str = "farl/lapa/448", device: str | None = None) -> None:
        if torch is None or facer is None:
            raise ImportError("pyfacer and torch are required for parser-driven face masking.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = facer.face_detector("retinaface/mobilenet", device=self.device)
        self.parser = facer.face_parser(parser_model, device=self.device)
        self.parser_model = parser_model

    def parse(self, image: Image.Image) -> ParserFaceData | None:
        image_tensor = facer.hwc2bchw(np.asarray(image.convert("RGB"), dtype=np.uint8)).to(device=self.device)

        with torch.inference_mode():
            faces = self.detector(image_tensor)
            if "rects" not in faces or faces["rects"].shape[0] == 0:
                return None
            faces = self.parser(image_tensor, faces)

        rects = faces["rects"].detach().cpu().numpy()
        areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
        face_index = int(np.argmax(areas))
        seg_logits = faces["seg"]["logits"][face_index]
        labels = seg_logits.argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
        label_names = list(faces["seg"].get("label_names") or self.parser.label_names)
        score = float(faces["scores"][face_index].item()) if "scores" in faces else None

        return ParserFaceData(
            labels=labels,
            label_names=label_names,
            rect=tuple(float(value) for value in rects[face_index]),
            score=score,
        )


class FaceIdentityMasker:
    def __init__(self, parser_model: str = "farl/lapa/448") -> None:
        if mp is None:
            raise ImportError("mediapipe is required for face masking.")

        face_mesh = mp.solutions.face_mesh
        self._mesh = face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        self._face_oval_ids = _connection_ids(face_mesh.FACEMESH_FACE_OVAL)
        self._left_eye_ids = _connection_ids(face_mesh.FACEMESH_LEFT_EYE)
        self._right_eye_ids = _connection_ids(face_mesh.FACEMESH_RIGHT_EYE)
        self._left_brow_ids = _connection_ids(face_mesh.FACEMESH_LEFT_EYEBROW)
        self._right_brow_ids = _connection_ids(face_mesh.FACEMESH_RIGHT_EYEBROW)
        self._lip_ids = _connection_ids(face_mesh.FACEMESH_LIPS)

        self._parser_model = parser_model
        self._parser_runtime: _FacerParserRuntime | None = None
        self._parser_failed = False

    def _align_source_to_generated(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
    ) -> tuple[Image.Image, list[tuple[float, float]] | None, list[tuple[float, float]] | None, str]:
        source_rgb = source_image.convert("RGB")
        generated_rgb = generated_image.convert("RGB")
        source_landmarks = self._extract_landmarks(source_rgb)
        generated_landmarks = self._extract_landmarks(generated_rgb)

        if source_landmarks and generated_landmarks:
            source_anchors = self._anchors(source_landmarks)
            generated_anchors = self._anchors(generated_landmarks)

            similarity_error: float | None = None
            affine_error: float | None = None
            similarity_affine: np.ndarray | None = None
            affine_transform: np.ndarray | None = None

            try:
                similarity_affine = _fit_similarity(source_anchors, generated_anchors)
                similarity_error = _normalized_point_error(source_anchors, generated_anchors, similarity_affine)
            except Exception:
                similarity_affine = None

            try:
                affine_transform = _fit_affine(source_anchors, generated_anchors)
                affine_error = _normalized_point_error(source_anchors, generated_anchors, affine_transform)
            except Exception:
                affine_transform = None

            chosen_transform: np.ndarray | None = None
            chosen_method = "direct"
            if similarity_affine is not None:
                similarity_scale = float(
                    (np.linalg.norm(similarity_affine[:2, 0]) + np.linalg.norm(similarity_affine[:2, 1])) / 2.0
                )
                if (
                    similarity_error is not None
                    and similarity_error <= 0.115
                    and 0.78 <= similarity_scale <= 1.28
                    and (
                        affine_error is None
                        or similarity_error <= (affine_error + 0.01)
                        or affine_error > 0.055
                    )
                ):
                    chosen_transform = similarity_affine
                    chosen_method = f"face_similarity:{similarity_error:.4f}"

            if chosen_transform is None and affine_transform is not None and affine_error is not None:
                affine_scale_x = float(np.linalg.norm(affine_transform[:2, 0]))
                affine_scale_y = float(np.linalg.norm(affine_transform[:2, 1]))
                min_affine_scale = max(min(affine_scale_x, affine_scale_y), 1e-6)
                affine_anisotropy = max(affine_scale_x, affine_scale_y) / min_affine_scale
                if (
                    affine_error <= 0.055
                    and 0.8 <= affine_scale_x <= 1.24
                    and 0.8 <= affine_scale_y <= 1.24
                    and affine_anisotropy <= 1.08
                ):
                    chosen_transform = affine_transform
                    chosen_method = f"face_affine:{affine_error:.4f}"

            if chosen_transform is not None:
                return (
                    _warp_image(source_rgb, chosen_transform, generated_rgb.size),
                    source_landmarks,
                    generated_landmarks,
                    chosen_method,
                )

        if source_rgb.size != generated_rgb.size:
            return (
                source_rgb.resize(generated_rgb.size, resample=Image.Resampling.BICUBIC),
                source_landmarks,
                generated_landmarks,
                "resized",
            )

        return source_rgb, source_landmarks, generated_landmarks, "direct"

    def _extract_landmarks(self, image: Image.Image) -> list[tuple[float, float]] | None:
        rgb = np.asarray(image.convert("RGB"))
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        width, height = image.size
        points: list[tuple[float, float]] = []
        for landmark in result.multi_face_landmarks[0].landmark:
            points.append((landmark.x * width, landmark.y * height))
        return points

    def estimate_face_coverage(self, image: Image.Image) -> float | None:
        landmarks = self._extract_landmarks(image)
        if not landmarks:
            return None

        face_points = self._face_points(landmarks)
        if face_points.size == 0:
            return None

        x1, y1 = face_points.min(axis=0)
        x2, y2 = face_points.max(axis=0)
        bbox_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        image_area = max(1.0, float(image.width * image.height))
        return bbox_area / image_area

    def _anchors(self, landmarks: list[tuple[float, float]]) -> np.ndarray:
        left_eye = _mean_point([landmarks[index] for index in LEFT_EYE_CENTER_INDICES])
        right_eye = _mean_point([landmarks[index] for index in RIGHT_EYE_CENTER_INDICES])
        nose_tip = landmarks[NOSE_TIP_INDEX]
        mouth_left = landmarks[MOUTH_CORNER_LEFT_INDEX]
        mouth_right = landmarks[MOUTH_CORNER_RIGHT_INDEX]
        return np.asarray([left_eye, right_eye, nose_tip, mouth_left, mouth_right], dtype=np.float32)

    def _face_points(self, landmarks: list[tuple[float, float]]) -> np.ndarray:
        return np.asarray([landmarks[index] for index in self._face_oval_ids if index < len(landmarks)], dtype=np.float32)

    def _face_bbox(self, landmarks: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
        face_points = self._face_points(landmarks)
        if face_points.size == 0:
            return None
        x1, y1 = face_points.min(axis=0)
        x2, y2 = face_points.max(axis=0)
        return float(x1), float(y1), float(x2), float(y2)

    def _build_face_shape_lock(
        self,
        aligned_source: Image.Image,
        source_landmarks: list[tuple[float, float]],
        generated_landmarks: list[tuple[float, float]],
        source_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> tuple[Image.Image, Image.Image, Image.Image, Dict[str, float]]:
        source_bbox = self._face_bbox(source_landmarks)
        generated_bbox = self._face_bbox(generated_landmarks)
        if source_bbox is None or generated_bbox is None:
            empty = Image.new("L", output_size, 0)
            return aligned_source, empty, empty, {}

        sx1, sy1, sx2, sy2 = source_bbox
        gx1, gy1, gx2, gy2 = generated_bbox
        source_scale_x = output_size[0] / max(float(source_size[0]), 1.0)
        source_scale_y = output_size[1] / max(float(source_size[1]), 1.0)
        source_width = max(1.0, (sx2 - sx1) * source_scale_x)
        source_height = max(1.0, (sy2 - sy1) * source_scale_y)
        generated_width = max(1.0, gx2 - gx1)
        generated_height = max(1.0, gy2 - gy1)
        source_diag = math.hypot(source_width, source_height)
        generated_diag = max(1.0, math.hypot(generated_width, generated_height))

        # Use a nearly uniform scale so the eyes/lips/nose stay proportionate to the head size.
        image_scale = float(np.clip(source_diag / generated_diag, 0.92, 1.16))
        mask_scale_x = float(
            np.clip(
                source_width / generated_width,
                image_scale * 0.95,
                image_scale * 1.07,
            )
        )
        mask_scale_y = float(
            np.clip(
                source_height / generated_height,
                image_scale * 0.95,
                image_scale * 1.07,
            )
        )
        generated_center = ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)

        shape_locked_source = _scale_image_about(
            aligned_source,
            center=generated_center,
            scale=image_scale,
            output_size=output_size,
        )

        base = max(output_size)
        generated_face_points = self._face_points(generated_landmarks)
        scaled_face_points = _scale_points(
            generated_face_points,
            center=generated_center,
            scale_x=mask_scale_x,
            scale_y=mask_scale_y,
        )
        inner_face_points = _scale_points(
            generated_face_points,
            center=generated_center,
            scale_x=max(0.9, mask_scale_x * 0.975),
            scale_y=max(0.9, mask_scale_y * 0.975),
        )

        shape_mask = _mask_from_points(output_size, scaled_face_points, blur_radius=max(3.0, base / 220.0))
        shape_mask = _expand_mask(
            shape_mask,
            expand_px=max(1, int(base / 520)),
            blur_radius=max(1.5, base / 340.0),
        )
        inner_shape_mask = _mask_from_points(output_size, inner_face_points, blur_radius=max(2.0, base / 260.0))
        inner_shape_mask = _contract_mask(
            inner_shape_mask,
            contract_px=max(1, int(base / 760)),
            blur_radius=max(1.2, base / 420.0),
        )

        metadata = {
            "image_scale": round(image_scale, 4),
            "mask_scale_x": round(mask_scale_x, 4),
            "mask_scale_y": round(mask_scale_y, 4),
            "source_face_width": round(source_width, 2),
            "source_face_height": round(source_height, 2),
            "generated_face_width": round(generated_width, 2),
            "generated_face_height": round(generated_height, 2),
        }
        return shape_locked_source, shape_mask, inner_shape_mask, metadata

    def _mask_from_indices(
        self,
        size: tuple[int, int],
        landmarks: list[tuple[float, float]],
        index_groups: Sequence[Sequence[int]],
        blur_radius: float,
    ) -> Image.Image:
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)

        for indices in index_groups:
            polygon = _convex_hull([landmarks[index] for index in indices if index < len(landmarks)])
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=255)

        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return mask

    def _build_legacy_masks(
        self,
        size: tuple[int, int],
        landmarks: list[tuple[float, float]],
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        base = max(size)
        face_blur = max(8.0, base / 80.0)
        core_blur = max(5.0, base / 120.0)
        face_expand = max(10, int(base / 70))
        core_expand = max(6, int(base / 120))

        face_mask = self._mask_from_indices(size, landmarks, [self._face_oval_ids], blur_radius=face_blur)
        core_groups = [
            [*self._left_eye_ids, *self._left_brow_ids],
            [*self._right_eye_ids, *self._right_brow_ids],
            self._lip_ids,
            NOSE_INDICES,
        ]
        core_mask = self._mask_from_indices(size, landmarks, core_groups, blur_radius=core_blur)
        face_mask = _expand_mask(face_mask, expand_px=face_expand, blur_radius=face_blur)
        core_mask = _expand_mask(core_mask, expand_px=core_expand, blur_radius=core_blur)

        contour_outer = _expand_mask(face_mask, expand_px=max(4, face_expand // 2), blur_radius=max(2.0, face_blur * 0.35))
        contour_inner = _contract_mask(face_mask, contract_px=max(4, face_expand // 2), blur_radius=max(2.0, face_blur * 0.15))
        contour_mask = _subtract_masks(contour_outer, contour_inner, blur_radius=max(2.0, face_blur * 0.25))
        return face_mask, core_mask, contour_mask

    def _build_legacy_debug_images(
        self,
        generated_image: Image.Image,
        aligned_source: Image.Image,
        face_mask: Image.Image,
        core_mask: Image.Image,
        contour_mask: Image.Image,
    ) -> Dict[str, Image.Image]:
        return {
            "source_aligned": aligned_source,
            "mask_face": face_mask.convert("L"),
            "mask_core": core_mask.convert("L"),
            "mask_contour": contour_mask.convert("L"),
            "overlay_regions": _overlay_regions(
                generated_image,
                {
                    "face": face_mask,
                    "core": core_mask,
                    "contour": contour_mask,
                },
            ),
        }

    def _legacy_protect(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        mode: str = "balanced",
        debug: bool = False,
    ) -> FaceMaskResult:
        normalized_mode = (mode or "balanced").strip().lower()
        if normalized_mode == "surface_fx":
            normalized_mode = "balanced"

        generated_rgb = generated_image.convert("RGB")
        aligned_source, source_landmarks, generated_landmarks, alignment_method = self._align_source_to_generated(
            source_image,
            generated_rgb,
        )
        if not source_landmarks:
            return FaceMaskResult(image=generated_image, applied=False, mode=normalized_mode, reason="no-source-face", engine="legacy")

        if not generated_landmarks:
            return FaceMaskResult(image=generated_rgb, applied=False, mode=normalized_mode, reason="no-output-face", engine="legacy")

        face_mask, core_mask, contour_mask = self._build_legacy_masks(generated_rgb.size, generated_landmarks)

        source_array = np.asarray(aligned_source, dtype=np.float32)
        generated_array = np.asarray(generated_rgb, dtype=np.float32)
        face_alpha = _mask_to_array(face_mask)
        core_alpha = _mask_to_array(core_mask)

        if normalized_mode == "strict":
            strict_low = source_array * 0.98 + generated_array * 0.02
            strict = generated_array * (1.0 - face_alpha) + strict_low * face_alpha
            strict = strict * (1.0 - core_alpha) + source_array * core_alpha
            return FaceMaskResult(
                image=Image.fromarray(np.clip(strict, 0, 255).astype(np.uint8), mode="RGB"),
                applied=True,
                mode="strict",
                reason="legacy-strict-protection",
                engine="legacy",
                metadata={"strategy_used": "legacy", "alignment_method": alignment_method},
                debug_images=self._build_legacy_debug_images(generated_rgb, aligned_source, face_mask, core_mask, contour_mask) if debug else {},
            )

        detail_radius = max(6.0, max(generated_rgb.size) / 100.0)
        source_low = np.asarray(aligned_source.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_low = np.asarray(generated_rgb.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_detail = generated_array - generated_low
        source_detail = source_array - source_low

        face_mix_low = source_low * 0.86 + generated_low * 0.14
        core_mix_low = source_low * 0.98 + generated_low * 0.02

        low_mix = generated_low * (1.0 - face_alpha) + face_mix_low * face_alpha
        low_mix = low_mix * (1.0 - core_alpha) + core_mix_low * core_alpha
        detail_scale = np.ones_like(face_alpha, dtype=np.float32)
        detail_scale = detail_scale * (1.0 - face_alpha) + (0.42 * face_alpha)
        detail_scale = detail_scale * (1.0 - core_alpha) + (0.08 * core_alpha)
        source_detail_boost = (core_alpha * 0.72) + (face_alpha * 0.18)
        inside_face = np.clip(low_mix + (generated_detail * detail_scale) + (source_detail * source_detail_boost), 0, 255)
        final = generated_array * (1.0 - face_alpha) + inside_face * face_alpha

        return FaceMaskResult(
            image=Image.fromarray(np.clip(final, 0, 255).astype(np.uint8), mode="RGB"),
            applied=True,
            mode="balanced",
            reason="legacy-balanced-protection",
            engine="legacy",
            metadata={"strategy_used": "legacy", "alignment_method": alignment_method},
            debug_images=self._build_legacy_debug_images(generated_rgb, aligned_source, face_mask, core_mask, contour_mask) if debug else {},
        )

    def _get_parser_runtime(self) -> _FacerParserRuntime:
        if self._parser_runtime is not None:
            return self._parser_runtime
        if self._parser_failed:
            raise RuntimeError("parser-unavailable")

        try:
            self._parser_runtime = _FacerParserRuntime(parser_model=self._parser_model)
            return self._parser_runtime
        except Exception as exc:  # pragma: no cover - exercised in the worker image
            self._parser_failed = True
            raise RuntimeError(f"parser-load-failed: {exc}") from exc

    def _parse_face_regions(self, image: Image.Image) -> ParserFaceData:
        runtime = self._get_parser_runtime()
        parsed = runtime.parse(image)
        if parsed is None:
            raise RuntimeError("no-parser-face")
        return parsed

    def _build_smart_masks(
        self,
        size: tuple[int, int],
        landmarks: list[tuple[float, float]],
        parser_face: ParserFaceData,
    ) -> Dict[str, Image.Image]:
        base = max(size)
        face_mask_land, core_mask_land, contour_mask_land = self._build_legacy_masks(size, landmarks)

        parser_face_mask = _mask_from_labels(
            parser_face.labels,
            parser_face.label_names,
            PARSER_FACE_REGION_LABELS,
            expand_px=max(2, int(base / 220)),
            blur_radius=max(2.0, base / 220.0),
        )
        parser_skin_mask = _mask_from_labels(
            parser_face.labels,
            parser_face.label_names,
            PARSER_SKIN_LABELS,
            expand_px=max(2, int(base / 250)),
            blur_radius=max(2.0, base / 260.0),
        )
        parser_core_mask = _union_masks(
            _mask_from_labels(parser_face.labels, parser_face.label_names, PARSER_BROW_LABELS, blur_radius=max(1.5, base / 260.0)),
            _mask_from_labels(parser_face.labels, parser_face.label_names, PARSER_EYE_LABELS, blur_radius=max(1.5, base / 260.0)),
            _mask_from_labels(parser_face.labels, parser_face.label_names, PARSER_NOSE_LABELS, blur_radius=max(1.5, base / 260.0)),
            _mask_from_labels(parser_face.labels, parser_face.label_names, PARSER_LIP_LABELS, blur_radius=max(1.5, base / 260.0)),
        )
        parser_hair_mask = _mask_from_labels(
            parser_face.labels,
            parser_face.label_names,
            PARSER_HAIR_LABELS,
            expand_px=max(2, int(base / 250)),
            blur_radius=max(2.0, base / 260.0),
        )

        face_mask = _union_masks(face_mask_land, parser_face_mask, parser_core_mask)
        core_mask = _union_masks(core_mask_land, parser_core_mask)
        contour_mask = _union_masks(
            contour_mask_land,
            _subtract_masks(
                _expand_mask(face_mask_land, expand_px=max(4, int(base / 180)), blur_radius=max(2.0, base / 200.0)),
                _contract_mask(face_mask_land, contract_px=max(4, int(base / 220)), blur_radius=max(1.5, base / 280.0)),
                blur_radius=max(2.0, base / 260.0),
            ),
        )
        hairline_mask = _intersect_masks(
            _expand_mask(parser_hair_mask, expand_px=max(2, int(base / 260)), blur_radius=max(1.5, base / 280.0)),
            _expand_mask(face_mask_land, expand_px=max(4, int(base / 180)), blur_radius=max(2.0, base / 220.0)),
        )

        surface_mask = _subtract_masks(
            _union_masks(parser_skin_mask, parser_face_mask),
            core_mask,
            contour_mask,
            hairline_mask,
            blur_radius=max(2.0, base / 260.0),
        )
        surface_mask = _intersect_masks(surface_mask, face_mask)
        remainder_mask = _subtract_masks(
            face_mask,
            core_mask,
            contour_mask,
            hairline_mask,
            surface_mask,
            blur_radius=max(1.0, base / 320.0),
        )

        return {
            "face": face_mask,
            "core": core_mask,
            "contour": contour_mask,
            "hairline": hairline_mask,
            "surface": surface_mask,
            "remainder": remainder_mask,
            "parser_labels": _colorize_labels(parser_face.labels, parser_face.label_names),
        }

    def _profile(self, mode: str, strength: float) -> Dict[str, Dict[str, float]]:
        normalized_mode = (mode or "balanced").strip().lower()
        clamped_strength = float(np.clip(strength, 0.0, 1.0))

        profile = {
            "surface": {
                "source_low": 0.88 + (0.08 * clamped_strength),
                "generated_detail": 0.72 - (0.20 * clamped_strength),
                "source_detail": 0.12 + (0.12 * clamped_strength),
            },
            "remainder": {
                "source_low": 0.84 + (0.08 * clamped_strength),
                "generated_detail": 0.58 - (0.18 * clamped_strength),
                "source_detail": 0.10 + (0.10 * clamped_strength),
            },
            "hairline": {
                "source_low": 0.88 + (0.08 * clamped_strength),
                "generated_detail": 0.30 - (0.18 * clamped_strength),
                "source_detail": 0.14 + (0.14 * clamped_strength),
            },
            "contour": {
                "source_low": 0.92 + (0.07 * clamped_strength),
                "generated_detail": 0.18 - (0.12 * clamped_strength),
                "source_detail": 0.28 + (0.18 * clamped_strength),
            },
            "core": {
                "source_low": 0.95 + (0.045 * clamped_strength),
                "generated_detail": 0.12 - (0.09 * clamped_strength),
                "source_detail": 0.55 + (0.25 * clamped_strength),
            },
        }

        if normalized_mode == "strict":
            profile["surface"] = {
                "source_low": 0.94 + (0.04 * clamped_strength),
                "generated_detail": 0.35 - (0.18 * clamped_strength),
                "source_detail": 0.25 + (0.20 * clamped_strength),
            }
            profile["remainder"] = {
                "source_low": 0.90 + (0.06 * clamped_strength),
                "generated_detail": 0.26 - (0.14 * clamped_strength),
                "source_detail": 0.18 + (0.18 * clamped_strength),
            }
        elif normalized_mode == "surface_fx":
            profile["surface"] = {
                "source_low": 0.84 + (0.08 * clamped_strength),
                "generated_detail": 0.92 - (0.12 * clamped_strength),
                "source_detail": 0.10 + (0.10 * clamped_strength),
            }
            profile["remainder"] = {
                "source_low": 0.82 + (0.08 * clamped_strength),
                "generated_detail": 0.76 - (0.16 * clamped_strength),
                "source_detail": 0.10 + (0.10 * clamped_strength),
            }

        return profile

    def _build_smart_debug_images(
        self,
        generated_image: Image.Image,
        aligned_source: Image.Image,
        region_masks: Dict[str, Image.Image],
    ) -> Dict[str, Image.Image]:
        overlay_regions = {
            "face": region_masks["face"],
            "surface": region_masks["surface"],
            "core": region_masks["core"],
            "contour": region_masks["contour"],
            "hairline": region_masks["hairline"],
        }
        return {
            "source_aligned": aligned_source,
            "mask_face": region_masks["face"].convert("L"),
            "mask_surface": region_masks["surface"].convert("L"),
            "mask_core": region_masks["core"].convert("L"),
            "mask_contour": region_masks["contour"].convert("L"),
            "mask_hairline": region_masks["hairline"].convert("L"),
            "parser_labels": region_masks["parser_labels"],
            "overlay_regions": _overlay_regions(generated_image, overlay_regions),
        }

    def _build_preserve_skin_debug_images(
        self,
        generated_image: Image.Image,
        aligned_source: Image.Image,
        source_skin_mask: Image.Image,
        preserve_mask: Image.Image,
        inner_preserve_mask: Image.Image,
        region_masks: Dict[str, Image.Image] | None,
    ) -> Dict[str, Image.Image]:
        overlay_regions = {
            "surface": preserve_mask,
            "core": inner_preserve_mask,
        }
        if region_masks is not None:
            overlay_regions["face"] = region_masks["face"]
            overlay_regions["hairline"] = region_masks["hairline"]

        debug_images = {
            "source_aligned": aligned_source,
            "mask_source_skin": source_skin_mask.convert("L"),
            "mask_preserve_outer": preserve_mask.convert("L"),
            "mask_preserve_inner": inner_preserve_mask.convert("L"),
            "overlay_regions": _overlay_regions(generated_image, overlay_regions),
        }
        if region_masks is not None:
            debug_images["parser_labels"] = region_masks["parser_labels"]
        return debug_images

    def _build_highlight_lock_mask(
        self,
        aligned_source: Image.Image,
        generated_image: Image.Image,
        face_mask: Image.Image,
    ) -> Image.Image:
        face_array = np.asarray(face_mask, dtype=np.uint8) > 0
        if not bool(face_array.any()):
            return Image.new("L", aligned_source.size, 0)

        base = max(aligned_source.size)
        source_array = np.asarray(aligned_source.convert("RGB"), dtype=np.float32)
        generated_array = np.asarray(generated_image.convert("RGB"), dtype=np.float32)
        source_luminance = _luminance_array(source_array)
        generated_luminance = _luminance_array(generated_array)

        local_luminance = np.asarray(
            Image.fromarray(np.clip(source_luminance, 0, 255).astype(np.uint8), mode="L").filter(
                ImageFilter.GaussianBlur(radius=max(5.0, base / 120.0))
            ),
            dtype=np.float32,
        )
        highlight_energy = source_luminance - local_luminance
        face_luminance = source_luminance[face_array]
        face_energy = highlight_energy[face_array]

        if face_luminance.size < 32:
            return Image.new("L", aligned_source.size, 0)

        bright_floor = float(max(np.percentile(face_luminance, 70), face_luminance.mean()))
        energy_floor = float(max(5.0, np.percentile(face_energy, 72)))
        luminance_gap = source_luminance - generated_luminance

        highlight_mask = (
            face_array
            & (source_luminance >= bright_floor)
            & (highlight_energy >= energy_floor)
            & (luminance_gap >= 3.0)
        )
        if not bool(highlight_mask.any()):
            return Image.new("L", aligned_source.size, 0)

        mask = Image.fromarray((highlight_mask.astype(np.uint8) * 255), mode="L")
        mask = _expand_mask(mask, expand_px=max(1, int(base / 520)), blur_radius=max(1.5, base / 420.0))
        return mask

    def _build_expression_lock_masks(
        self,
        size: tuple[int, int],
        landmarks: list[tuple[float, float]],
        face_mask: Image.Image,
        core_mask: Image.Image,
        contour_mask: Image.Image,
    ) -> Dict[str, Image.Image]:
        empty = Image.new("L", size, 0)
        bbox = self._face_bbox(landmarks)
        if bbox is None:
            return {
                "eyes": empty,
                "smile": empty,
                "jaw_cheeks": empty,
                "expression": empty,
            }

        base = max(size)
        x1, y1, x2, y2 = bbox
        face_width = max(1.0, x2 - x1)
        face_height = max(1.0, y2 - y1)
        face_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        left_eye = _mean_point([landmarks[index] for index in LEFT_EYE_CENTER_INDICES])
        right_eye = _mean_point([landmarks[index] for index in RIGHT_EYE_CENTER_INDICES])
        eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        mouth_left = landmarks[MOUTH_CORNER_LEFT_INDEX]
        mouth_right = landmarks[MOUTH_CORNER_RIGHT_INDEX]
        mouth_center = ((mouth_left[0] + mouth_right[0]) / 2.0, (mouth_left[1] + mouth_right[1]) / 2.0)

        eye_mask = self._mask_from_indices(
            size,
            landmarks,
            [
                [*self._left_eye_ids, *self._left_brow_ids],
                [*self._right_eye_ids, *self._right_brow_ids],
            ],
            blur_radius=max(2.0, base / 260.0),
        )
        eye_mask = _union_masks(
            eye_mask,
            _mask_from_ellipse(
                size,
                center=(eye_center[0], eye_center[1] - (0.02 * face_height)),
                radius_x=face_width * 0.38,
                radius_y=face_height * 0.19,
                blur_radius=max(2.0, base / 240.0),
            ),
        )
        eye_mask = _expand_mask(
            eye_mask,
            expand_px=max(2, int(base / 260)),
            blur_radius=max(1.8, base / 260.0),
        )
        eye_mask = _intersect_masks(eye_mask, face_mask)

        smile_mask = self._mask_from_indices(
            size,
            landmarks,
            [
                self._lip_ids,
                [MOUTH_CORNER_LEFT_INDEX, 13, MOUTH_CORNER_RIGHT_INDEX, 14],
            ],
            blur_radius=max(2.0, base / 260.0),
        )
        smile_mask = _union_masks(
            smile_mask,
            _mask_from_ellipse(
                size,
                center=(mouth_center[0], mouth_center[1] - (0.015 * face_height)),
                radius_x=face_width * 0.34,
                radius_y=face_height * 0.21,
                blur_radius=max(2.0, base / 220.0),
            ),
        )
        smile_mask = _expand_mask(
            smile_mask,
            expand_px=max(2, int(base / 240)),
            blur_radius=max(2.0, base / 240.0),
        )
        smile_mask = _intersect_masks(smile_mask, face_mask)

        jaw_cheek_mask = _mask_from_ellipse(
            size,
            center=(face_center[0], face_center[1] + (0.18 * face_height)),
            radius_x=face_width * 0.46,
            radius_y=face_height * 0.31,
            blur_radius=max(2.0, base / 220.0),
        )
        jaw_cheek_mask = _union_masks(jaw_cheek_mask, contour_mask)
        jaw_cheek_mask = _intersect_masks(jaw_cheek_mask, face_mask)

        expression_mask = _union_masks(eye_mask, smile_mask, jaw_cheek_mask, core_mask)
        expression_mask = _intersect_masks(expression_mask, face_mask)

        return {
            "eyes": eye_mask,
            "smile": smile_mask,
            "jaw_cheeks": jaw_cheek_mask,
            "expression": expression_mask,
        }

    def _reinforce_strict_identity(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        protected_result: FaceMaskResult,
        strength: float,
        debug: bool,
    ) -> FaceMaskResult:
        if not protected_result.applied:
            protected_result.metadata["strict_reinforcement_applied"] = False
            protected_result.metadata["strict_reinforcement_reason"] = "strict-mask-not-applied"
            return protected_result

        protected_image = protected_result.image.convert("RGB")
        raw_generated = generated_image.convert("RGB")
        base = max(protected_image.size)
        clamped_strength = float(np.clip(strength, 0.0, 1.0))

        aligned_source, source_landmarks, protected_landmarks, alignment_method = self._align_source_to_generated(
            source_image,
            protected_image,
        )
        if not source_landmarks:
            protected_result.metadata["strict_reinforcement_applied"] = False
            protected_result.metadata["strict_reinforcement_reason"] = "no-source-face"
            protected_result.metadata["strict_reinforcement_alignment_method"] = alignment_method
            return protected_result
        if not protected_landmarks:
            protected_result.metadata["strict_reinforcement_applied"] = False
            protected_result.metadata["strict_reinforcement_reason"] = "no-output-face"
            protected_result.metadata["strict_reinforcement_alignment_method"] = alignment_method
            return protected_result

        parser_face: ParserFaceData | None = None
        parser_error: str | None = None
        try:
            parser_face = self._parse_face_regions(protected_image)
            region_masks = self._build_smart_masks(protected_image.size, protected_landmarks, parser_face)
            face_mask = _subtract_masks(
                region_masks["face"],
                region_masks["hairline"],
                blur_radius=max(1.0, base / 360.0),
            )
            core_mask = region_masks["core"]
            contour_mask = _subtract_masks(region_masks["contour"], core_mask)
            hairline_mask = _subtract_masks(region_masks["hairline"], core_mask, region_masks["contour"])
            surface_mask = _subtract_masks(
                region_masks["surface"],
                region_masks["core"],
                region_masks["contour"],
                region_masks["hairline"],
                blur_radius=max(1.2, base / 340.0),
            )
            remainder_mask = _subtract_masks(
                region_masks["remainder"],
                region_masks["core"],
                region_masks["contour"],
                region_masks["hairline"],
                region_masks["surface"],
                blur_radius=max(1.0, base / 360.0),
            )
        except Exception as exc:
            parser_error = str(exc)
            face_mask, core_mask, contour_mask = self._build_legacy_masks(protected_image.size, protected_landmarks)
            hairline_mask = Image.new("L", protected_image.size, 0)
            surface_mask = _subtract_masks(
                face_mask,
                core_mask,
                contour_mask,
                blur_radius=max(1.2, base / 340.0),
            )
            remainder_mask = surface_mask

        shape_locked_source, face_shape_mask, face_shape_inner_mask, shape_meta = self._build_face_shape_lock(
            aligned_source=aligned_source,
            source_landmarks=source_landmarks,
            generated_landmarks=protected_landmarks,
            source_size=source_image.size,
            output_size=protected_image.size,
        )
        expression_masks = self._build_expression_lock_masks(
            size=protected_image.size,
            landmarks=protected_landmarks,
            face_mask=face_mask,
            core_mask=core_mask,
            contour_mask=contour_mask,
        )

        full_face_mask = _union_masks(face_mask, face_shape_mask, expression_masks["jaw_cheeks"])
        structure_mask = _union_masks(core_mask, expression_masks["expression"], face_shape_inner_mask)
        surface_structure_mask = _union_masks(surface_mask, remainder_mask, expression_masks["jaw_cheeks"])
        full_face_inner_mask = _union_masks(structure_mask, face_shape_inner_mask)

        source_array = np.asarray(aligned_source.convert("RGB"), dtype=np.float32)
        shape_locked_source_array = np.asarray(shape_locked_source.convert("RGB"), dtype=np.float32)
        protected_array = np.asarray(protected_image, dtype=np.float32)
        raw_array = np.asarray(raw_generated.resize(protected_image.size, resample=Image.Resampling.BICUBIC), dtype=np.float32)

        full_face_source = np.clip((shape_locked_source_array * 0.9) + (source_array * 0.1), 0, 255)
        feature_source = np.clip((source_array * 0.82) + (shape_locked_source_array * 0.18), 0, 255)
        tone_reference = np.clip((shape_locked_source_array * 0.72) + (source_array * 0.28), 0, 255)

        final = protected_array.copy()
        low_freq_alpha = _mask_to_array(full_face_mask) * (0.64 + (0.2 * clamped_strength))
        if float(low_freq_alpha.max()) > 0.0:
            final_low = _blur_rgb_array(final, radius=max(10.0, base / 72.0))
            source_low = _blur_rgb_array(tone_reference, radius=max(10.0, base / 72.0))
            final = np.clip(final + ((source_low - final_low) * low_freq_alpha), 0, 255)

        hairline_alpha = _mask_to_array(hairline_mask) * (0.66 + (0.16 * clamped_strength))
        if float(hairline_alpha.max()) > 0.0:
            hairline_source = np.clip((source_array * 0.97) + (protected_array * 0.03), 0, 255)
            final = (final * (1.0 - hairline_alpha)) + (hairline_source * hairline_alpha)

        full_face_alpha = _mask_to_array(full_face_mask) * (0.58 + (0.18 * clamped_strength))
        if float(full_face_alpha.max()) > 0.0:
            full_face_blend = np.clip((full_face_source * 0.992) + (protected_array * 0.008), 0, 255)
            final = (final * (1.0 - full_face_alpha)) + (full_face_blend * full_face_alpha)

        surface_alpha = _mask_to_array(surface_structure_mask) * (0.74 + (0.14 * clamped_strength))
        if float(surface_alpha.max()) > 0.0:
            surface_blend = np.clip((full_face_source * 0.986) + (protected_array * 0.014), 0, 255)
            final = (final * (1.0 - surface_alpha)) + (surface_blend * surface_alpha)

        expression_alpha = _mask_to_array(expression_masks["expression"]) * (0.86 + (0.12 * clamped_strength))
        if float(expression_alpha.max()) > 0.0:
            expression_blend = np.clip((feature_source * 0.996) + (raw_array * 0.004), 0, 255)
            final = (final * (1.0 - expression_alpha)) + (expression_blend * expression_alpha)

        structure_alpha = _mask_to_array(structure_mask) * min(1.0, 0.94 + (0.08 * clamped_strength))
        if float(structure_alpha.max()) > 0.0:
            structure_blend = np.clip((feature_source * 0.998) + (raw_array * 0.002), 0, 255)
            final = (final * (1.0 - structure_alpha)) + (structure_blend * structure_alpha)

        shape_lock_strength = min(1.0, 0.84 + (0.14 * clamped_strength))
        shape_alpha = _mask_to_array(face_shape_mask) * shape_lock_strength
        if float(shape_alpha.max()) > 0.0:
            shape_blend = np.clip((shape_locked_source_array * 0.998) + (protected_array * 0.002), 0, 255)
            final = (final * (1.0 - shape_alpha)) + (shape_blend * shape_alpha)

        shape_inner_alpha = _mask_to_array(full_face_inner_mask) * min(1.0, shape_lock_strength + 0.08)
        if float(shape_inner_alpha.max()) > 0.0:
            final = (final * (1.0 - shape_inner_alpha)) + (shape_locked_source_array * shape_inner_alpha)

        texture_mask = _union_masks(surface_structure_mask, structure_mask, face_shape_inner_mask)
        final, tone_lock_coverage = self._apply_face_tone_lock(
            working_array=final,
            source_array=tone_reference,
            face_mask=full_face_mask,
            radius=max(8.0, base / 86.0),
            strength=0.78 + (0.16 * clamped_strength),
        )
        final, texture_lock_coverage = self._apply_face_texture_lock(
            working_array=final,
            source_array=feature_source,
            face_mask=texture_mask,
            radius=max(1.8, base / 250.0),
            strength=0.64 + (0.18 * clamped_strength),
        )

        highlight_lock_mask = self._build_highlight_lock_mask(aligned_source, protected_image, full_face_mask)
        highlight_alpha = _mask_to_array(highlight_lock_mask) * (0.92 + (0.08 * clamped_strength))
        if float(highlight_alpha.max()) > 0.0:
            highlight_blend = np.clip((source_array * 0.988) + (feature_source * 0.012), 0, 255)
            final = (final * (1.0 - highlight_alpha)) + (highlight_blend * highlight_alpha)

        metadata = dict(protected_result.metadata)
        metadata["strict_reinforcement_applied"] = True
        metadata["strict_reinforcement_reason"] = "source-dominant-face-composite"
        metadata["strict_reinforcement_alignment_method"] = alignment_method
        metadata["strict_reinforcement_full_face_coverage"] = round(float((np.asarray(full_face_mask, dtype=np.uint8) > 0).mean()), 4)
        metadata["strict_reinforcement_expression_coverage"] = round(float((np.asarray(expression_masks["expression"], dtype=np.uint8) > 0).mean()), 4)
        metadata["strict_reinforcement_structure_coverage"] = round(float((np.asarray(structure_mask, dtype=np.uint8) > 0).mean()), 4)
        metadata["strict_reinforcement_hairline_coverage"] = round(float((np.asarray(hairline_mask, dtype=np.uint8) > 0).mean()), 4)
        metadata["strict_reinforcement_tone_lock_coverage"] = round(tone_lock_coverage, 4)
        metadata["strict_reinforcement_texture_lock_coverage"] = round(texture_lock_coverage, 4)
        metadata["strict_reinforcement_highlight_coverage"] = round(float((np.asarray(highlight_lock_mask, dtype=np.uint8) > 0).mean()), 4)
        metadata["strict_reinforcement_strength"] = round(clamped_strength, 3)
        for key, value in shape_meta.items():
            metadata[f"strict_reinforcement_shape_{key}"] = value
        if parser_face is not None and parser_face.score is not None:
            metadata["strict_reinforcement_parser_score"] = parser_face.score
        if parser_error:
            metadata["strict_reinforcement_parser_fallback"] = parser_error

        debug_images = dict(protected_result.debug_images)
        if debug:
            debug_images["mask_strict_full_face"] = full_face_mask.convert("L")
            debug_images["mask_strict_expression"] = expression_masks["expression"].convert("L")
            debug_images["mask_strict_structure"] = structure_mask.convert("L")
            debug_images["mask_strict_hairline"] = hairline_mask.convert("L")
            debug_images["overlay_strict_identity"] = _overlay_regions(
                protected_image,
                {
                    "face": full_face_mask,
                    "surface": surface_structure_mask,
                    "core": structure_mask,
                    "contour": expression_masks["jaw_cheeks"],
                    "hairline": hairline_mask,
                },
            )

        return FaceMaskResult(
            image=Image.fromarray(np.clip(final, 0, 255).astype(np.uint8), mode="RGB"),
            applied=protected_result.applied,
            mode="strict",
            reason=f"{protected_result.reason}; strict-source-dominant-composite",
            engine=protected_result.engine,
            metadata=metadata,
            debug_images=debug_images,
        )

    def _apply_face_tone_lock(
        self,
        working_array: np.ndarray,
        source_array: np.ndarray,
        face_mask: Image.Image,
        radius: float,
        strength: float,
    ) -> tuple[np.ndarray, float]:
        alpha = _mask_to_array(face_mask) * float(np.clip(strength, 0.0, 1.0))
        if float(alpha.max()) <= 0.0:
            return np.clip(working_array, 0, 255), 0.0

        working_low = _blur_rgb_array(working_array, radius=radius)
        source_low = _blur_rgb_array(source_array, radius=radius)
        corrected = np.clip(working_array + ((source_low - working_low) * alpha), 0, 255)
        coverage = float((np.asarray(face_mask, dtype=np.uint8) > 0).mean())
        return corrected, coverage

    def _apply_face_texture_lock(
        self,
        working_array: np.ndarray,
        source_array: np.ndarray,
        face_mask: Image.Image,
        radius: float,
        strength: float,
    ) -> tuple[np.ndarray, float]:
        alpha = _mask_to_array(face_mask) * float(np.clip(strength, 0.0, 1.0))
        if float(alpha.max()) <= 0.0:
            return np.clip(working_array, 0, 255), 0.0

        working_low = _blur_rgb_array(working_array, radius=radius)
        source_low = _blur_rgb_array(source_array, radius=radius)
        working_texture = working_array - working_low
        source_texture = source_array - source_low
        corrected = np.clip(working_array + ((source_texture - working_texture) * alpha), 0, 255)
        coverage = float((np.asarray(face_mask, dtype=np.uint8) > 0).mean())
        return corrected, coverage

    def _feature_geometry_signature(self, landmarks: list[tuple[float, float]]) -> Dict[str, float] | None:
        bbox = self._face_bbox(landmarks)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        face_width = max(1.0, x2 - x1)
        face_height = max(1.0, y2 - y1)
        left_eye = _mean_point([landmarks[index] for index in LEFT_EYE_CENTER_INDICES])
        right_eye = _mean_point([landmarks[index] for index in RIGHT_EYE_CENTER_INDICES])
        nose_tip = landmarks[NOSE_TIP_INDEX]
        mouth_left = landmarks[MOUTH_CORNER_LEFT_INDEX]
        mouth_right = landmarks[MOUTH_CORNER_RIGHT_INDEX]
        mouth_center = ((mouth_left[0] + mouth_right[0]) / 2.0, (mouth_left[1] + mouth_right[1]) / 2.0)
        eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]

        return {
            "face_aspect_ratio": face_width / face_height,
            "eye_span": math.dist(left_eye, right_eye) / face_width,
            "mouth_span": math.dist(mouth_left, mouth_right) / face_width,
            "eye_to_mouth": abs(mouth_center[1] - eye_center[1]) / face_height,
            "nose_to_mouth": math.dist(nose_tip, mouth_center) / face_height,
            "left_eye_open": abs(landmarks[159][1] - landmarks[145][1]) / face_height,
            "right_eye_open": abs(landmarks[386][1] - landmarks[374][1]) / face_height,
            "mouth_open": abs(upper_lip[1] - lower_lip[1]) / face_height,
            "mouth_tilt": abs(mouth_left[1] - mouth_right[1]) / face_height,
        }

    def assess_identity_drift(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
    ) -> Dict[str, Any]:
        generated_rgb = generated_image.convert("RGB")
        aligned_source, source_landmarks, generated_landmarks, alignment_method = self._align_source_to_generated(
            source_image,
            generated_rgb,
        )

        if not source_landmarks:
            return {
                "available": False,
                "reason": "no-source-face",
                "alignment_method": alignment_method,
            }
        if not generated_landmarks:
            return {
                "available": False,
                "reason": "no-output-face",
                "alignment_method": alignment_method,
            }

        source_anchors = self._anchors(source_landmarks)
        generated_anchors = self._anchors(generated_landmarks)

        try:
            geometry_transform = _fit_similarity(source_anchors, generated_anchors)
            geometry_error = _normalized_point_error(source_anchors, generated_anchors, geometry_transform)
        except Exception:
            geometry_error = 1.0

        region_masks: Dict[str, Image.Image] | None = None
        parser_face: ParserFaceData | None = None
        try:
            parser_face = self._parse_face_regions(generated_rgb)
            region_masks = self._build_smart_masks(generated_rgb.size, generated_landmarks, parser_face)
            face_mask = _subtract_masks(
                region_masks["face"],
                region_masks["hairline"],
                blur_radius=max(1.0, max(generated_rgb.size) / 360.0),
            )
            core_mask = region_masks["core"]
        except Exception:
            face_mask, core_mask, _ = self._build_legacy_masks(generated_rgb.size, generated_landmarks)

        face_bool = np.asarray(face_mask, dtype=np.uint8) > 0
        core_bool = np.asarray(core_mask, dtype=np.uint8) > 0
        if not bool(face_bool.any()):
            return {
                "available": False,
                "reason": "no-face-mask",
                "alignment_method": alignment_method,
            }

        source_array = np.asarray(aligned_source.convert("RGB"), dtype=np.float32)
        generated_array = np.asarray(generated_rgb, dtype=np.float32)
        pixel_delta = np.abs(source_array - generated_array).mean(axis=2)
        face_pixel_delta = float(pixel_delta[face_bool].mean() / 255.0)
        core_pixel_delta = float(pixel_delta[core_bool].mean() / 255.0) if bool(core_bool.any()) else face_pixel_delta
        source_bbox = self._face_bbox(source_landmarks)
        generated_bbox = self._face_bbox(generated_landmarks)
        face_scale_error = 0.0
        if source_bbox is not None and generated_bbox is not None:
            source_width = (source_bbox[2] - source_bbox[0]) / max(float(source_image.width), 1.0)
            source_height = (source_bbox[3] - source_bbox[1]) / max(float(source_image.height), 1.0)
            generated_width = (generated_bbox[2] - generated_bbox[0]) / max(float(generated_image.width), 1.0)
            generated_height = (generated_bbox[3] - generated_bbox[1]) / max(float(generated_image.height), 1.0)
            source_diag = math.hypot(source_width, source_height)
            generated_diag = math.hypot(generated_width, generated_height)
            if source_diag > 1e-6 and generated_diag > 1e-6:
                face_scale_error = float(abs(math.log(generated_diag / source_diag)))

        feature_error = 0.0
        aspect_ratio_error = 0.0
        source_signature = self._feature_geometry_signature(source_landmarks)
        generated_signature = self._feature_geometry_signature(generated_landmarks)
        if source_signature is not None and generated_signature is not None:
            aspect_ratio_error = float(
                abs(
                    math.log(
                        max(generated_signature["face_aspect_ratio"], 1e-6)
                        / max(source_signature["face_aspect_ratio"], 1e-6)
                    )
                )
            )
            feature_components = [
                abs(generated_signature["eye_span"] - source_signature["eye_span"]) / 0.03,
                abs(generated_signature["mouth_span"] - source_signature["mouth_span"]) / 0.035,
                abs(generated_signature["eye_to_mouth"] - source_signature["eye_to_mouth"]) / 0.03,
                abs(generated_signature["nose_to_mouth"] - source_signature["nose_to_mouth"]) / 0.028,
                abs(generated_signature["mouth_tilt"] - source_signature["mouth_tilt"]) / 0.018,
                abs(generated_signature["mouth_open"] - source_signature["mouth_open"]) / 0.022,
                abs(generated_signature["left_eye_open"] - source_signature["left_eye_open"]) / 0.018,
                abs(generated_signature["right_eye_open"] - source_signature["right_eye_open"]) / 0.018,
            ]
            feature_error = float(np.mean(np.clip(feature_components, 0.0, 1.0)))

        highlight_mask = self._build_highlight_lock_mask(aligned_source, generated_rgb, face_mask)
        highlight_bool = np.asarray(highlight_mask, dtype=np.uint8) > 0
        if bool(highlight_bool.any()):
            source_luminance = _luminance_array(source_array)
            generated_luminance = _luminance_array(generated_array)
            highlight_luminance_delta = float(
                np.maximum(source_luminance - generated_luminance, 0.0)[highlight_bool].mean() / 255.0
            )
        else:
            highlight_luminance_delta = 0.0

        geometry_norm = float(np.clip(geometry_error / 0.055, 0.0, 1.0))
        face_norm = float(np.clip(face_pixel_delta / 0.09, 0.0, 1.0))
        core_norm = float(np.clip(core_pixel_delta / 0.07, 0.0, 1.0))
        highlight_norm = float(np.clip(highlight_luminance_delta / 0.08, 0.0, 1.0))
        scale_norm = float(np.clip(face_scale_error / 0.07, 0.0, 1.0))
        feature_norm = float(np.clip(feature_error, 0.0, 1.0))
        aspect_norm = float(np.clip(aspect_ratio_error / 0.055, 0.0, 1.0))
        score = float(
            (0.24 * geometry_norm)
            + (0.17 * face_norm)
            + (0.19 * core_norm)
            + (0.1 * highlight_norm)
            + (0.12 * scale_norm)
            + (0.12 * feature_norm)
            + (0.06 * aspect_norm)
        )

        assessment: Dict[str, Any] = {
            "available": True,
            "score": round(score, 4),
            "alignment_method": alignment_method,
            "geometry_error": round(geometry_error, 4),
            "face_pixel_delta": round(face_pixel_delta, 4),
            "core_pixel_delta": round(core_pixel_delta, 4),
            "highlight_luminance_delta": round(highlight_luminance_delta, 4),
            "face_scale_error": round(face_scale_error, 4),
            "feature_geometry_error": round(feature_error, 4),
            "face_aspect_ratio_error": round(aspect_ratio_error, 4),
            "face_mask_coverage": round(float(face_bool.mean()), 4),
        }
        if parser_face is not None:
            assessment["parser_score"] = parser_face.score
        return assessment

    def _smart_protect(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        mode: str,
        strength: float,
        debug: bool,
    ) -> FaceMaskResult:
        normalized_mode = (mode or "balanced").strip().lower()
        generated_rgb = generated_image.convert("RGB")
        base = max(generated_rgb.size)

        aligned_source, source_landmarks, generated_landmarks, alignment_method = self._align_source_to_generated(
            source_image,
            generated_rgb,
        )
        if not source_landmarks:
            return FaceMaskResult(image=generated_rgb, applied=False, mode=normalized_mode, reason="no-source-face", engine="smart")

        if not generated_landmarks:
            return FaceMaskResult(image=generated_rgb, applied=False, mode=normalized_mode, reason="no-output-face", engine="smart")

        parser_face = self._parse_face_regions(generated_rgb)
        region_masks = self._build_smart_masks(generated_rgb.size, generated_landmarks, parser_face)
        shape_locked_source, face_shape_mask, face_shape_inner_mask, shape_meta = self._build_face_shape_lock(
            aligned_source=aligned_source,
            source_landmarks=source_landmarks,
            generated_landmarks=generated_landmarks,
            source_size=source_image.size,
            output_size=generated_rgb.size,
        )

        source_array = np.asarray(aligned_source, dtype=np.float32)
        shape_locked_source_array = np.asarray(shape_locked_source, dtype=np.float32)
        generated_array = np.asarray(generated_rgb, dtype=np.float32)
        detail_radius = max(5.0, max(generated_rgb.size) / 120.0)
        source_low = np.asarray(aligned_source.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_low = np.asarray(generated_rgb.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_detail = generated_array - generated_low
        source_detail = source_array - source_low

        core_mask = region_masks["core"]
        contour_mask = _subtract_masks(region_masks["contour"], core_mask)
        hairline_mask = _subtract_masks(region_masks["hairline"], core_mask, region_masks["contour"])
        surface_mask = _subtract_masks(region_masks["surface"], core_mask, region_masks["contour"], region_masks["hairline"])
        remainder_mask = _subtract_masks(
            region_masks["remainder"],
            core_mask,
            region_masks["contour"],
            region_masks["hairline"],
            region_masks["surface"],
        )

        profile = self._profile(normalized_mode, strength=strength)
        low_mix = generated_low.copy()
        detail_mix = generated_detail.copy()

        def apply_region(mask: Image.Image, settings: Dict[str, float]) -> None:
            nonlocal low_mix, detail_mix
            alpha = _mask_to_array(mask)
            if float(alpha.max()) <= 0.0:
                return
            region_low = (source_low * settings["source_low"]) + (generated_low * (1.0 - settings["source_low"]))
            region_detail = (generated_detail * settings["generated_detail"]) + (source_detail * settings["source_detail"])
            low_mix = (low_mix * (1.0 - alpha)) + (region_low * alpha)
            detail_mix = (detail_mix * (1.0 - alpha)) + (region_detail * alpha)

        apply_region(surface_mask, profile["surface"])
        apply_region(remainder_mask, profile["remainder"])
        apply_region(hairline_mask, profile["hairline"])
        apply_region(contour_mask, profile["contour"])
        apply_region(core_mask, profile["core"])

        final = np.clip(low_mix + detail_mix, 0, 255)
        face_mask = _subtract_masks(
            region_masks["face"],
            region_masks["hairline"],
            blur_radius=max(1.0, max(generated_rgb.size) / 360.0),
        )
        texture_mask = _union_masks(surface_mask, core_mask, remainder_mask, face_shape_inner_mask)

        if normalized_mode == "strict":
            shape_lock_strength = 0.58 + (0.18 * float(np.clip(strength, 0.0, 1.0)))
        elif normalized_mode == "surface_fx":
            shape_lock_strength = 0.18 + (0.10 * float(np.clip(strength, 0.0, 1.0)))
        else:
            shape_lock_strength = 0.34 + (0.16 * float(np.clip(strength, 0.0, 1.0)))
        shape_alpha = _mask_to_array(face_shape_mask) * shape_lock_strength
        shape_inner_alpha = _mask_to_array(face_shape_inner_mask) * min(1.0, shape_lock_strength + 0.12)
        if float(shape_alpha.max()) > 0.0:
            shape_blend = (shape_locked_source_array * 0.996) + (generated_array * 0.004)
            final = (final * (1.0 - shape_alpha)) + (shape_blend * shape_alpha)
        if float(shape_inner_alpha.max()) > 0.0:
            final = (final * (1.0 - shape_inner_alpha)) + (shape_locked_source_array * shape_inner_alpha)
        tone_face_mask = _union_masks(face_mask, face_shape_mask)

        if normalized_mode == "strict":
            tone_lock_strength = 0.54 + (0.16 * float(np.clip(strength, 0.0, 1.0)))
            texture_lock_strength = 0.48 + (0.18 * float(np.clip(strength, 0.0, 1.0)))
        elif normalized_mode == "surface_fx":
            tone_lock_strength = 0.18 + (0.12 * float(np.clip(strength, 0.0, 1.0)))
            texture_lock_strength = 0.14 + (0.12 * float(np.clip(strength, 0.0, 1.0)))
        else:
            tone_lock_strength = 0.34 + (0.16 * float(np.clip(strength, 0.0, 1.0)))
            texture_lock_strength = 0.28 + (0.18 * float(np.clip(strength, 0.0, 1.0)))

        final, tone_lock_coverage = self._apply_face_tone_lock(
            working_array=final,
            source_array=source_array,
            face_mask=tone_face_mask,
            radius=max(7.0, base / 90.0),
            strength=tone_lock_strength,
        )
        final, texture_lock_coverage = self._apply_face_texture_lock(
            working_array=final,
            source_array=source_array,
            face_mask=texture_mask,
            radius=max(2.0, base / 230.0),
            strength=texture_lock_strength,
        )

        highlight_lock_mask = self._build_highlight_lock_mask(aligned_source, generated_rgb, face_mask)
        highlight_alpha = _mask_to_array(highlight_lock_mask)
        if float(highlight_alpha.max()) > 0.0:
            if normalized_mode == "surface_fx":
                highlight_strength = 0.32 + (0.16 * float(np.clip(strength, 0.0, 1.0)))
            elif normalized_mode == "strict":
                highlight_strength = 0.58 + (0.18 * float(np.clip(strength, 0.0, 1.0)))
            else:
                highlight_strength = 0.44 + (0.18 * float(np.clip(strength, 0.0, 1.0)))
            highlight_blend = (generated_array * (1.0 - highlight_strength)) + (source_array * highlight_strength)
            final = (final * (1.0 - highlight_alpha)) + (highlight_blend * highlight_alpha)

        result_image = Image.fromarray(final.astype(np.uint8), mode="RGB")
        metadata: Dict[str, Any] = {
            "strategy_used": "smart",
            "parser_model": self._parser_model,
            "strength": round(float(np.clip(strength, 0.0, 1.0)), 3),
            "parser_score": parser_face.score,
            "parser_rect": [round(value, 2) for value in parser_face.rect],
            "alignment_method": alignment_method,
            "highlight_lock_coverage": round(float((np.asarray(highlight_lock_mask, dtype=np.uint8) > 0).mean()), 4),
            "tone_lock_coverage": round(tone_lock_coverage, 4),
            "texture_lock_coverage": round(texture_lock_coverage, 4),
            "shape_lock_coverage": round(float((np.asarray(face_shape_mask, dtype=np.uint8) > 0).mean()), 4),
            **{f"shape_lock_{key}": value for key, value in shape_meta.items()},
        }

        return FaceMaskResult(
            image=result_image,
            applied=True,
            mode=normalized_mode,
            reason=f"smart-{normalized_mode}-protection",
            engine="smart",
            metadata=metadata,
            debug_images=self._build_smart_debug_images(generated_rgb, aligned_source, region_masks) if debug else {},
        )

    def _preserve_skin_protect(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        mode: str,
        strength: float,
        prompt: str | None,
        debug: bool,
    ) -> FaceMaskResult:
        normalized_mode = (mode or "balanced").strip().lower()
        motion_hints = _position_change_hints(prompt)

        if motion_hints:
            try:
                relaxed = self._smart_protect(
                    source_image=source_image,
                    generated_image=generated_image,
                    mode=normalized_mode,
                    strength=strength,
                    debug=debug,
                )
            except Exception:
                relaxed = self._legacy_protect(
                    source_image=source_image,
                    generated_image=generated_image,
                    mode=normalized_mode,
                    debug=debug,
                )

            relaxed.engine = "preserve_skin"
            relaxed.reason = f"{relaxed.reason}; preserve-skin-relaxed-for-motion"
            relaxed.metadata["strategy_used"] = "preserve_skin"
            relaxed.metadata["motion_relaxed"] = True
            relaxed.metadata["motion_hints"] = motion_hints
            return relaxed

        generated_rgb = generated_image.convert("RGB")
        aligned_source, source_landmarks, generated_landmarks, alignment_method = self._align_source_to_generated(
            source_image,
            generated_rgb,
        )
        source_skin_mask = _detect_exposed_skin_mask(aligned_source)
        source_skin_mask_array = np.asarray(source_skin_mask, dtype=np.uint8)

        if int(source_skin_mask_array.max()) <= 0:
            try:
                fallback = self._smart_protect(
                    source_image=source_image,
                    generated_image=generated_image,
                    mode=normalized_mode,
                    strength=strength,
                    debug=debug,
                )
            except Exception:
                fallback = self._legacy_protect(
                    source_image=source_image,
                    generated_image=generated_image,
                    mode=normalized_mode,
                    debug=debug,
                )

            fallback.engine = "preserve_skin"
            fallback.reason = f"{fallback.reason}; preserve-skin-fallback-no-source-skin"
            fallback.metadata["strategy_used"] = "preserve_skin"
            fallback.metadata["motion_relaxed"] = False
            fallback.metadata["alignment_method"] = alignment_method
            return fallback

        base = max(generated_rgb.size)
        preserve_mask = _expand_mask(
            source_skin_mask,
            expand_px=max(1, int(base / 420)),
            blur_radius=max(2.0, base / 320.0),
        )
        inner_preserve_mask = _contract_mask(
            source_skin_mask,
            contract_px=max(1, int(base / 420)),
            blur_radius=max(1.2, base / 420.0),
        )

        region_masks: Dict[str, Image.Image] | None = None
        parser_error: str | None = None
        if generated_landmarks:
            try:
                parser_face = self._parse_face_regions(generated_rgb)
                region_masks = self._build_smart_masks(generated_rgb.size, generated_landmarks, parser_face)
                face_skin_mask = _subtract_masks(
                    _union_masks(
                        region_masks["surface"],
                        region_masks["remainder"],
                        region_masks["core"],
                    ),
                    region_masks["hairline"],
                    blur_radius=max(1.2, base / 340.0),
                )
                preserve_mask = _union_masks(preserve_mask, face_skin_mask)
                inner_preserve_mask = _union_masks(inner_preserve_mask, region_masks["core"])
                preserve_mask = _subtract_masks(
                    preserve_mask,
                    region_masks["hairline"],
                    blur_radius=max(1.0, base / 360.0),
                )
            except Exception as exc:
                parser_error = str(exc)

        source_array = np.asarray(aligned_source, dtype=np.float32)
        shape_locked_source = aligned_source
        face_shape_mask = Image.new("L", generated_rgb.size, 0)
        face_shape_inner_mask = Image.new("L", generated_rgb.size, 0)
        shape_meta: Dict[str, float] = {}
        if source_landmarks and generated_landmarks:
            shape_locked_source, face_shape_mask, face_shape_inner_mask, shape_meta = self._build_face_shape_lock(
                aligned_source=aligned_source,
                source_landmarks=source_landmarks,
                generated_landmarks=generated_landmarks,
                source_size=source_image.size,
                output_size=generated_rgb.size,
            )
        shape_locked_source_array = np.asarray(shape_locked_source, dtype=np.float32)
        generated_array = np.asarray(generated_rgb, dtype=np.float32)
        detail_radius = max(4.0, base / 140.0)
        source_low = np.asarray(aligned_source.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        generated_low = np.asarray(generated_rgb.filter(ImageFilter.GaussianBlur(radius=detail_radius)), dtype=np.float32)
        source_detail = source_array - source_low
        generated_detail = generated_array - generated_low

        full_face_mask = preserve_mask
        face_surface_mask = preserve_mask
        face_core_mask = inner_preserve_mask
        if region_masks is not None:
            full_face_mask = _subtract_masks(
                region_masks["face"],
                region_masks["hairline"],
                blur_radius=max(1.0, base / 360.0),
            )
            face_core_mask = _union_masks(face_core_mask, region_masks["core"])
            face_surface_mask = _subtract_masks(
                full_face_mask,
                face_core_mask,
                region_masks["contour"],
                region_masks["hairline"],
                blur_radius=max(1.2, base / 340.0),
            )

        full_face_mask_for_shape = _union_masks(full_face_mask, face_shape_mask)
        full_face_inner_shape_mask = _union_masks(face_core_mask, face_shape_inner_mask)
        low_mix = generated_low.copy()
        detail_mix = generated_detail.copy()

        def apply_region(mask: Image.Image, source_low_weight: float, generated_detail_weight: float, source_detail_weight: float) -> None:
            nonlocal low_mix, detail_mix
            alpha = _mask_to_array(mask)
            if float(alpha.max()) <= 0.0:
                return

            region_low = (source_low * source_low_weight) + (generated_low * (1.0 - source_low_weight))
            region_detail = (generated_detail * generated_detail_weight) + (source_detail * source_detail_weight)
            low_mix = (low_mix * (1.0 - alpha)) + (region_low * alpha)
            detail_mix = (detail_mix * (1.0 - alpha)) + (region_detail * alpha)

        clamped_strength = float(np.clip(strength, 0.0, 1.0))
        if normalized_mode == "strict":
            full_face_source_low_weight = 0.975 + (0.02 * clamped_strength)
            full_face_generated_detail_weight = 0.06 - (0.03 * clamped_strength)
            full_face_source_detail_weight = 0.30 + (0.20 * clamped_strength)
            face_surface_source_low_weight = 0.988 + (0.01 * clamped_strength)
            face_surface_generated_detail_weight = 0.03 - (0.015 * clamped_strength)
            face_surface_source_detail_weight = 0.52 + (0.18 * clamped_strength)
            face_core_source_low_weight = 0.994 + (0.005 * clamped_strength)
            face_core_generated_detail_weight = 0.015 - (0.008 * clamped_strength)
            face_core_source_detail_weight = 0.82 + (0.12 * clamped_strength)
            full_face_lock_strength = 0.60 + (0.18 * clamped_strength)
            tone_lock_strength = 0.78 + (0.14 * clamped_strength)
            texture_lock_strength = 0.66 + (0.16 * clamped_strength)
        elif normalized_mode == "surface_fx":
            full_face_source_low_weight = 0.95 + (0.03 * clamped_strength)
            full_face_generated_detail_weight = 0.12 - (0.05 * clamped_strength)
            full_face_source_detail_weight = 0.22 + (0.18 * clamped_strength)
            face_surface_source_low_weight = 0.978 + (0.012 * clamped_strength)
            face_surface_generated_detail_weight = 0.05 - (0.02 * clamped_strength)
            face_surface_source_detail_weight = 0.40 + (0.18 * clamped_strength)
            face_core_source_low_weight = 0.99 + (0.008 * clamped_strength)
            face_core_generated_detail_weight = 0.025 - (0.01 * clamped_strength)
            face_core_source_detail_weight = 0.76 + (0.14 * clamped_strength)
            full_face_lock_strength = 0.44 + (0.14 * clamped_strength)
            tone_lock_strength = 0.56 + (0.14 * clamped_strength)
            texture_lock_strength = 0.44 + (0.16 * clamped_strength)
        else:
            full_face_source_low_weight = 0.965 + (0.025 * clamped_strength)
            full_face_generated_detail_weight = 0.09 - (0.04 * clamped_strength)
            full_face_source_detail_weight = 0.26 + (0.18 * clamped_strength)
            face_surface_source_low_weight = 0.983 + (0.012 * clamped_strength)
            face_surface_generated_detail_weight = 0.04 - (0.018 * clamped_strength)
            face_surface_source_detail_weight = 0.46 + (0.18 * clamped_strength)
            face_core_source_low_weight = 0.992 + (0.006 * clamped_strength)
            face_core_generated_detail_weight = 0.02 - (0.01 * clamped_strength)
            face_core_source_detail_weight = 0.78 + (0.14 * clamped_strength)
            full_face_lock_strength = 0.52 + (0.16 * clamped_strength)
            tone_lock_strength = 0.66 + (0.14 * clamped_strength)
            texture_lock_strength = 0.54 + (0.16 * clamped_strength)

        apply_region(
            full_face_mask,
            source_low_weight=full_face_source_low_weight,
            generated_detail_weight=full_face_generated_detail_weight,
            source_detail_weight=full_face_source_detail_weight,
        )
        apply_region(
            face_surface_mask,
            source_low_weight=face_surface_source_low_weight,
            generated_detail_weight=face_surface_generated_detail_weight,
            source_detail_weight=face_surface_source_detail_weight,
        )
        apply_region(
            face_core_mask,
            source_low_weight=face_core_source_low_weight,
            generated_detail_weight=face_core_generated_detail_weight,
            source_detail_weight=face_core_source_detail_weight,
        )

        final = np.clip(low_mix + detail_mix, 0, 255)
        outer_alpha = _mask_to_array(preserve_mask) * (0.94 + (0.06 * clamped_strength))
        inner_alpha = _mask_to_array(inner_preserve_mask)
        final = (final * (1.0 - outer_alpha)) + (source_array * outer_alpha)
        final = (final * (1.0 - inner_alpha)) + (source_array * inner_alpha)

        full_face_alpha = _mask_to_array(full_face_mask_for_shape) * full_face_lock_strength
        source_dominant_face = (source_array * 0.992) + (generated_array * 0.008)
        final = (final * (1.0 - full_face_alpha)) + (source_dominant_face * full_face_alpha)

        shape_lock_strength = min(1.0, full_face_lock_strength + 0.16)
        shape_alpha = _mask_to_array(face_shape_mask) * shape_lock_strength
        shape_inner_alpha = _mask_to_array(full_face_inner_shape_mask) * min(1.0, shape_lock_strength + 0.1)
        if float(shape_alpha.max()) > 0.0:
            shape_blend = (shape_locked_source_array * 0.997) + (generated_array * 0.003)
            final = (final * (1.0 - shape_alpha)) + (shape_blend * shape_alpha)
        if float(shape_inner_alpha.max()) > 0.0:
            final = (final * (1.0 - shape_inner_alpha)) + (shape_locked_source_array * shape_inner_alpha)

        texture_mask = _union_masks(face_surface_mask, face_core_mask, face_shape_inner_mask)
        final, tone_lock_coverage = self._apply_face_tone_lock(
            working_array=final,
            source_array=source_array,
            face_mask=full_face_mask_for_shape,
            radius=max(7.0, base / 88.0),
            strength=tone_lock_strength,
        )
        final, texture_lock_coverage = self._apply_face_texture_lock(
            working_array=final,
            source_array=source_array,
            face_mask=texture_mask,
            radius=max(1.8, base / 240.0),
            strength=texture_lock_strength,
        )

        highlight_lock_mask = self._build_highlight_lock_mask(aligned_source, generated_rgb, full_face_mask)
        highlight_alpha = _mask_to_array(highlight_lock_mask) * (0.88 + (0.10 * clamped_strength))
        if float(highlight_alpha.max()) > 0.0:
            highlight_blend = (source_array * 0.98) + (generated_array * 0.02)
            final = (final * (1.0 - highlight_alpha)) + (highlight_blend * highlight_alpha)

        metadata: Dict[str, Any] = {
            "strategy_used": "preserve_skin",
            "motion_relaxed": False,
            "alignment_method": alignment_method,
            "strength": round(clamped_strength, 3),
            "preserved_skin_coverage": round(float((source_skin_mask_array > 0).mean()), 4),
            "highlight_lock_coverage": round(float((np.asarray(highlight_lock_mask, dtype=np.uint8) > 0).mean()), 4),
            "tone_lock_coverage": round(tone_lock_coverage, 4),
            "texture_lock_coverage": round(texture_lock_coverage, 4),
            "shape_lock_coverage": round(float((np.asarray(face_shape_mask, dtype=np.uint8) > 0).mean()), 4),
            **{f"shape_lock_{key}": value for key, value in shape_meta.items()},
        }
        if parser_error:
            metadata["parser_fallback"] = parser_error

        return FaceMaskResult(
            image=Image.fromarray(np.clip(final, 0, 255).astype(np.uint8), mode="RGB"),
            applied=True,
            mode=normalized_mode,
            reason="preserve-skin-source-composite",
            engine="preserve_skin",
            metadata=metadata,
            debug_images=(
                self._build_preserve_skin_debug_images(
                    generated_rgb,
                    aligned_source,
                    source_skin_mask,
                    preserve_mask,
                    inner_preserve_mask,
                    region_masks,
                )
                if debug
                else {}
            ),
        )

    def _strict_identity_protect(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        strength: float,
        debug: bool,
    ) -> FaceMaskResult:
        try:
            base_result = self._smart_protect(
                source_image=source_image,
                generated_image=generated_image,
                mode="strict",
                strength=strength,
                debug=debug,
            )
            base_result.metadata.setdefault("fallback", "none")
            base_result.metadata.setdefault("base_engine", base_result.engine)
        except Exception as exc:
            base_result = self._legacy_protect(
                source_image=source_image,
                generated_image=generated_image,
                mode="strict",
                debug=debug,
            )
            base_result.reason = f"{base_result.reason}; strict-smart-fallback:{exc}"
            base_result.metadata["fallback"] = "legacy"
            base_result.metadata["base_engine"] = base_result.engine

        try:
            result = self._reinforce_strict_identity(
                source_image=source_image,
                generated_image=generated_image,
                protected_result=base_result,
                strength=strength,
                debug=debug,
            )
        except Exception as exc:
            result = base_result
            result.metadata["strict_reinforcement_applied"] = False
            result.metadata["strict_reinforcement_reason"] = f"fallback:{exc}"
            result.reason = f"{result.reason}; strict-reinforcement-fallback:{exc}"

        result.mode = "strict"
        result.engine = "strict_identity"
        result.metadata["strategy_used"] = "strict_identity"
        result.metadata["strict_identity_lock"] = True
        return result

    def _apply_liquid_surface_recovery(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        protected_result: FaceMaskResult,
        prompt: str | None,
        debug: bool,
    ) -> FaceMaskResult:
        effect_terms = _surface_effect_terms(prompt)
        effect_label = "liquid" if _is_liquid_request(prompt) else "surface-effect"

        def mark_not_applied(reason: str, *, alignment: str | None = None) -> FaceMaskResult:
            protected_result.metadata["surface_effect_recovery_applied"] = False
            protected_result.metadata["surface_effect_recovery_reason"] = reason
            protected_result.metadata["surface_effect_terms"] = effect_terms
            if alignment is not None:
                protected_result.metadata["surface_effect_recovery_alignment_method"] = alignment

            protected_result.metadata["liquid_recovery_applied"] = False
            protected_result.metadata["liquid_recovery_reason"] = reason
            if alignment is not None:
                protected_result.metadata["liquid_recovery_alignment_method"] = alignment
            return protected_result

        if not protected_result.applied:
            return mark_not_applied("strict-mask-not-applied")

        raw_generated = generated_image.convert("RGB")
        protected_image = protected_result.image.convert("RGB")
        base = max(raw_generated.size)

        aligned_source, source_landmarks, generated_landmarks, alignment_method = self._align_source_to_generated(
            source_image,
            raw_generated,
        )
        if not source_landmarks:
            return mark_not_applied("no-source-face", alignment=alignment_method)
        if not generated_landmarks:
            return mark_not_applied("no-output-face", alignment=alignment_method)

        parser_face: ParserFaceData | None = None
        region_masks: Dict[str, Image.Image] | None = None
        allow_core_effects = _uses_core_surface_effects(prompt)
        allow_color_effects = _uses_color_surface_effects(prompt)
        try:
            parser_face = self._parse_face_regions(raw_generated)
            region_masks = self._build_smart_masks(raw_generated.size, generated_landmarks, parser_face)
            face_mask = _subtract_masks(
                region_masks["face"],
                region_masks["hairline"],
                blur_radius=max(1.0, base / 360.0),
            )
            editable_region = _subtract_masks(
                _union_masks(
                    region_masks["surface"],
                    region_masks["remainder"],
                    region_masks["contour"],
                    _expand_mask(
                        region_masks["core"],
                        expand_px=max(1, int(base / 540)),
                        blur_radius=max(1.0, base / 420.0),
                    ),
                ),
                region_masks["hairline"],
                blur_radius=max(1.2, base / 340.0),
            )
        except Exception as exc:
            parser_face = None
            face_mask, core_mask, contour_mask = self._build_legacy_masks(raw_generated.size, generated_landmarks)
            editable_region = _subtract_masks(
                _union_masks(face_mask, contour_mask, _expand_mask(core_mask, expand_px=max(1, int(base / 540)), blur_radius=max(1.0, base / 420.0))),
                Image.new("L", raw_generated.size, 0),
                blur_radius=max(1.2, base / 320.0),
            )
            protected_result.metadata["surface_effect_recovery_parser_fallback"] = str(exc)
            protected_result.metadata["liquid_recovery_parser_fallback"] = str(exc)

        if not allow_core_effects and parser_face is not None and region_masks is not None:
            editable_region = _subtract_masks(
                editable_region,
                region_masks["core"],
                blur_radius=max(1.0, base / 420.0),
            )

        effect_region = _intersect_masks(editable_region, face_mask)
        region_bool = np.asarray(effect_region, dtype=np.uint8) > 0
        if not bool(region_bool.any()):
            return mark_not_applied("no-surface-region", alignment=alignment_method)

        source_array = np.asarray(aligned_source.convert("RGB"), dtype=np.float32)
        raw_array = np.asarray(raw_generated, dtype=np.float32)
        protected_array = np.asarray(protected_image, dtype=np.float32)
        blur_radius = max(2.0, base / 220.0)
        raw_low = _blur_rgb_array(raw_array, blur_radius)
        source_low = _blur_rgb_array(source_array, blur_radius)
        protected_low = _blur_rgb_array(protected_array, blur_radius)

        raw_luminance = _luminance_array(raw_array)
        source_luminance = _luminance_array(source_array)
        protected_luminance = _luminance_array(protected_array)
        raw_local_luminance = _luminance_array(raw_low)

        raw_detail_magnitude = np.mean(np.abs(raw_array - raw_low), axis=2)
        source_detail_magnitude = np.mean(np.abs(source_array - source_low), axis=2)
        protected_detail_magnitude = np.mean(np.abs(protected_array - protected_low), axis=2)
        new_detail = np.maximum(raw_detail_magnitude - np.maximum(source_detail_magnitude, protected_detail_magnitude), 0.0)
        new_sheen = np.maximum(raw_luminance - np.maximum(source_luminance, protected_luminance), 0.0)
        local_highlight = np.maximum(raw_luminance - raw_local_luminance, 0.0)

        region_detail = new_detail[region_bool]
        region_sheen = new_sheen[region_bool]
        region_highlight = local_highlight[region_bool]
        if region_detail.size < 32:
            return mark_not_applied("surface-region-too-small", alignment=alignment_method)

        detail_floor = float(max(4.5, np.percentile(region_detail, 80)))
        sheen_floor = float(max(3.0, np.percentile(region_sheen, 76)))
        highlight_floor = float(max(3.0, np.percentile(region_highlight, 72)))
        candidate = region_bool & (
            ((new_detail >= detail_floor) & (new_sheen >= sheen_floor))
            | (
                (local_highlight >= highlight_floor)
                & (new_sheen >= max(2.0, sheen_floor * 0.65))
                & (new_detail >= detail_floor * 0.75)
            )
        )
        if allow_color_effects:
            color_delta = np.mean(np.abs(raw_low - protected_low), axis=2)
            region_color = color_delta[region_bool]
            color_floor = float(max(3.0, np.percentile(region_color, 74)))
            candidate = candidate | (region_bool & (color_delta >= color_floor) & (new_detail >= detail_floor * 0.55))

        candidate_mask = Image.fromarray((candidate.astype(np.uint8) * 255), mode="L")
        candidate_mask = _expand_mask(
            candidate_mask,
            expand_px=max(1, int(base / 560)),
            blur_radius=max(1.4, base / 380.0),
        )
        candidate_mask = _intersect_masks(candidate_mask, _expand_mask(effect_region, 0, max(1.0, base / 480.0)))
        candidate_bool = np.asarray(candidate_mask, dtype=np.uint8) > 0
        if not bool(candidate_bool.any()):
            return mark_not_applied("no-surface-delta-detected", alignment=alignment_method)

        clamped_strength = float(np.clip(protected_result.metadata.get("strength", 0.86), 0.0, 1.0))
        broad_sheen = _uses_broad_surface_sheen(prompt)
        detail_strength = 0.78 + (0.14 * clamped_strength)
        sheen_strength = (0.32 if broad_sheen else 0.18) + (0.12 * clamped_strength)
        color_strength = (0.18 if allow_color_effects else 0.0) + (0.08 * clamped_strength if allow_color_effects else 0.0)
        alpha = _mask_to_array(candidate_mask)
        raw_detail = raw_array - raw_low
        protected_detail = protected_array - protected_low
        detail_transfer = raw_detail - protected_detail
        sheen_transfer = np.clip(raw_low - protected_low, 0.0, 255.0)
        color_transfer = raw_low - protected_low
        recovered = np.clip(
            protected_array
            + (detail_transfer * alpha * detail_strength)
            + (sheen_transfer * alpha * sheen_strength),
            0,
            255,
        )
        if allow_color_effects and color_strength > 0.0:
            recovered = np.clip(recovered + (color_transfer * alpha * color_strength), 0, 255)

        metadata = dict(protected_result.metadata)
        metadata["surface_effect_recovery_applied"] = True
        metadata["surface_effect_recovery_reason"] = "surface-detail-transfer"
        metadata["surface_effect_recovery_alignment_method"] = alignment_method
        metadata["surface_effect_recovery_mask_coverage"] = round(float(candidate_bool.mean()), 4)
        metadata["surface_effect_recovery_detail_strength"] = round(detail_strength, 3)
        metadata["surface_effect_recovery_sheen_strength"] = round(sheen_strength, 3)
        metadata["surface_effect_recovery_color_strength"] = round(color_strength, 3)
        metadata["surface_effect_terms"] = effect_terms
        metadata["liquid_recovery_applied"] = True
        metadata["liquid_recovery_reason"] = "surface-detail-transfer"
        metadata["liquid_recovery_alignment_method"] = alignment_method
        metadata["liquid_recovery_mask_coverage"] = round(float(candidate_bool.mean()), 4)
        metadata["liquid_recovery_detail_strength"] = round(detail_strength, 3)
        metadata["liquid_recovery_sheen_strength"] = round(sheen_strength, 3)
        metadata["liquid_recovery_color_strength"] = round(color_strength, 3)
        if parser_face is not None and parser_face.score is not None:
            metadata["surface_effect_recovery_parser_score"] = parser_face.score
            metadata["liquid_recovery_parser_score"] = parser_face.score

        debug_images = dict(protected_result.debug_images)
        if debug:
            debug_images["mask_surface_effect_region"] = effect_region.convert("L")
            debug_images["mask_surface_effect_recovery"] = candidate_mask.convert("L")
            debug_images["overlay_surface_effect_recovery"] = _overlay_regions(
                raw_generated,
                {
                    "surface": effect_region,
                    "core": candidate_mask,
                },
            )
            debug_images["mask_liquid_region"] = effect_region.convert("L")
            debug_images["mask_liquid_recovery"] = candidate_mask.convert("L")
            debug_images["overlay_liquid_recovery"] = debug_images["overlay_surface_effect_recovery"]

        return FaceMaskResult(
            image=Image.fromarray(recovered.astype(np.uint8), mode="RGB"),
            applied=protected_result.applied,
            mode=protected_result.mode,
            reason=f"{protected_result.reason}; {effect_label}-recovery",
            engine=protected_result.engine,
            metadata=metadata,
            debug_images=debug_images,
        )

    def protect(
        self,
        source_image: Image.Image,
        generated_image: Image.Image,
        mode: str = "balanced",
        strategy: str = "auto",
        strength: float = 0.86,
        prompt: str | None = None,
        debug: bool = False,
    ) -> FaceMaskResult:
        normalized_mode = (mode or "strict").strip().lower()

        if normalized_mode == "off":
            return FaceMaskResult(image=generated_image, applied=False, mode="off", reason="disabled", engine="none")

        result = self._strict_identity_protect(
            source_image=source_image,
            generated_image=generated_image,
            strength=strength,
            debug=debug,
        )
        result.metadata.setdefault("strategy_requested", "strict_identity")

        if _requests_surface_effect(prompt):
            result = self._apply_liquid_surface_recovery(
                source_image=source_image,
                generated_image=generated_image,
                protected_result=result,
                prompt=prompt,
                debug=debug,
            )
        else:
            result.metadata.setdefault("surface_effect_recovery_applied", False)
            result.metadata.setdefault("surface_effect_recovery_reason", "not-requested")
            result.metadata.setdefault("surface_effect_terms", [])
            result.metadata.setdefault("liquid_recovery_applied", False)
            result.metadata.setdefault("liquid_recovery_reason", "not-requested")

        return result
