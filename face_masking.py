from __future__ import annotations

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
                    and similarity_error <= 0.14
                    and 0.72 <= similarity_scale <= 1.38
                    and (
                        affine_error is None
                        or similarity_error <= (affine_error + 0.015)
                        or affine_error > 0.08
                    )
                ):
                    chosen_transform = similarity_affine
                    chosen_method = f"face_similarity:{similarity_error:.4f}"

            if chosen_transform is None and affine_transform is not None and affine_error is not None and affine_error <= 0.085:
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

        face_points = np.asarray([landmarks[index] for index in self._face_oval_ids if index < len(landmarks)], dtype=np.float32)
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
        score = float((0.34 * geometry_norm) + (0.26 * face_norm) + (0.28 * core_norm) + (0.12 * highlight_norm))

        assessment: Dict[str, Any] = {
            "available": True,
            "score": round(score, 4),
            "alignment_method": alignment_method,
            "geometry_error": round(geometry_error, 4),
            "face_pixel_delta": round(face_pixel_delta, 4),
            "core_pixel_delta": round(core_pixel_delta, 4),
            "highlight_luminance_delta": round(highlight_luminance_delta, 4),
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

        source_array = np.asarray(aligned_source, dtype=np.float32)
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
        texture_mask = _union_masks(surface_mask, core_mask, remainder_mask)

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
            face_mask=face_mask,
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
        aligned_source, _, generated_landmarks, alignment_method = self._align_source_to_generated(
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

        full_face_alpha = _mask_to_array(full_face_mask) * full_face_lock_strength
        source_dominant_face = (source_array * 0.992) + (generated_array * 0.008)
        final = (final * (1.0 - full_face_alpha)) + (source_dominant_face * full_face_alpha)

        texture_mask = _union_masks(face_surface_mask, face_core_mask)
        final, tone_lock_coverage = self._apply_face_tone_lock(
            working_array=final,
            source_array=source_array,
            face_mask=full_face_mask,
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
        normalized_mode = (mode or "balanced").strip().lower()
        normalized_strategy = (strategy or "auto").strip().lower()

        if normalized_mode == "off":
            return FaceMaskResult(image=generated_image, applied=False, mode="off", reason="disabled", engine="none")

        if normalized_strategy == "legacy":
            result = self._legacy_protect(source_image, generated_image, mode=normalized_mode, debug=debug)
            result.metadata.setdefault("strategy_requested", "legacy")
            return result

        if normalized_strategy == "preserve_skin":
            result = self._preserve_skin_protect(
                source_image=source_image,
                generated_image=generated_image,
                mode=normalized_mode,
                strength=strength,
                prompt=prompt,
                debug=debug,
            )
            result.metadata.setdefault("strategy_requested", "preserve_skin")
            return result

        try:
            result = self._smart_protect(
                source_image=source_image,
                generated_image=generated_image,
                mode=normalized_mode,
                strength=strength,
                debug=debug,
            )
            result.metadata.setdefault("strategy_requested", normalized_strategy)
            return result
        except Exception as exc:
            if normalized_strategy == "smart":
                return FaceMaskResult(
                    image=generated_image.convert("RGB"),
                    applied=False,
                    mode=normalized_mode,
                    reason=f"smart-failed:{exc}",
                    engine="smart",
                    metadata={"strategy_requested": "smart"},
                )

            fallback = self._legacy_protect(source_image, generated_image, mode=normalized_mode, debug=debug)
            fallback.reason = f"{fallback.reason}; smart-fallback:{exc}"
            fallback.metadata["strategy_requested"] = normalized_strategy
            fallback.metadata["fallback"] = "legacy"
            return fallback
