import unittest

from PIL import Image

from face_masking import (
    EDIT_REGIME_POSE_OR_VIEW,
    EDIT_REGIME_SAME_VIEW_MINOR,
    EDIT_REGIME_SAME_VIEW_SURFACE,
    FaceCandidate,
    FaceIdentityMasker,
    FaceMaskResult,
    STRATEGY_SMART,
    STRATEGY_STRICT_IDENTITY,
)


def _image() -> Image.Image:
    return Image.new("RGB", (64, 64), (128, 96, 82))


def _masker() -> FaceIdentityMasker:
    masker = FaceIdentityMasker.__new__(FaceIdentityMasker)
    masker._parser_failed = False
    return masker


def _result(engine: str, mode: str, reason: str = "stub") -> FaceMaskResult:
    return FaceMaskResult(
        image=_image(),
        applied=True,
        mode=mode,
        reason=reason,
        engine=engine,
        metadata={"strategy_used": engine},
    )


class FaceMaskingRoutingTests(unittest.TestCase):
    def test_protect_honors_explicit_smart_strategy_and_mode(self) -> None:
        masker = _masker()
        calls = []
        masker._safe_assess_identity_drift = lambda *args, **kwargs: {"available": True, "score": 0.01}
        masker._smart_protect = lambda **kwargs: calls.append(kwargs) or _result("smart", kwargs["mode"])
        masker._apply_liquid_surface_recovery = lambda **kwargs: kwargs["protected_result"]

        result = masker.protect(
            _image(),
            _image(),
            strategy="smart",
            mode="surface_fx",
            prompt="change the background color",
        )

        self.assertEqual(result.engine, "smart")
        self.assertEqual(calls[0]["mode"], "surface_fx")
        self.assertEqual(result.metadata["strategy_requested"], "smart")
        self.assertEqual(result.metadata["strategy_used"], "smart")
        self.assertEqual(result.metadata["mode_used"], "surface_fx")

    def test_smart_parser_failure_falls_back_explicitly_to_legacy(self) -> None:
        masker = _masker()
        masker._safe_assess_identity_drift = lambda *args, **kwargs: {"available": True, "score": 0.01}
        masker._smart_protect = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("parser-unavailable"))
        masker._legacy_protect = lambda **kwargs: _result("legacy", kwargs["mode"], reason="legacy-balanced-protection")
        masker._apply_liquid_surface_recovery = lambda **kwargs: kwargs["protected_result"]

        result = masker.protect(_image(), _image(), strategy="smart", mode="balanced", prompt="minor edit")

        self.assertEqual(result.engine, "legacy")
        self.assertEqual(result.metadata["strategy_used"], "smart")
        self.assertEqual(result.metadata["fallback"], "legacy")
        self.assertIn("smart-fallback", result.metadata["fallback_reason"])

    def test_legacy_no_source_and_no_output_face_are_explicit(self) -> None:
        masker = _masker()
        masker._align_source_to_generated = lambda source, generated: (generated, None, None, "stub-align")

        result = FaceIdentityMasker._legacy_protect(masker, _image(), _image(), mode="strict", debug=False)

        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "no-source-face")
        self.assertEqual(result.engine, "legacy")

        landmarks = [(1.0, 1.0)] * 500
        masker._align_source_to_generated = lambda source, generated: (generated, landmarks, None, "stub-align")

        result = FaceIdentityMasker._legacy_protect(masker, _image(), _image(), mode="strict", debug=False)

        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "no-output-face")
        self.assertEqual(result.engine, "legacy")

    def test_pose_prompt_relaxes_strict_identity_rescue(self) -> None:
        masker = _masker()
        calls = []
        masker._safe_assess_identity_drift = lambda *args, **kwargs: {"available": True, "score": 0.01}
        masker._strict_identity_protect = lambda **kwargs: calls.append(kwargs) or _result("strict_identity", kwargs["mode"])
        masker._apply_liquid_surface_recovery = lambda **kwargs: kwargs["protected_result"]

        result = masker.protect(
            _image(),
            _image(),
            strategy="strict_identity",
            mode="strict",
            strength=0.86,
            prompt="turn head to the left",
        )

        self.assertEqual(calls[0]["edit_regime"], EDIT_REGIME_POSE_OR_VIEW)
        self.assertEqual(calls[0]["mode"], "balanced")
        self.assertLess(calls[0]["strength"], 0.86)
        self.assertEqual(result.metadata["edit_regime"], EDIT_REGIME_POSE_OR_VIEW)

    def test_surface_prompt_uses_surface_regime_and_weaker_rescue(self) -> None:
        masker = _masker()
        calls = []
        masker._safe_assess_identity_drift = lambda *args, **kwargs: {"available": True, "score": 0.01}
        masker._strict_identity_protect = lambda **kwargs: calls.append(kwargs) or _result("strict_identity", kwargs["mode"])
        masker._apply_liquid_surface_recovery = lambda **kwargs: kwargs["protected_result"]

        result = masker.protect(
            _image(),
            _image(),
            strategy="strict_identity",
            mode="strict",
            strength=0.86,
            prompt="add water droplets on the face",
        )

        self.assertEqual(calls[0]["edit_regime"], EDIT_REGIME_SAME_VIEW_SURFACE)
        self.assertEqual(calls[0]["mode"], "surface_fx")
        self.assertLess(calls[0]["strength"], 0.86)
        self.assertEqual(result.metadata["edit_regime"], EDIT_REGIME_SAME_VIEW_SURFACE)

    def test_drift_score_can_route_auto_to_pose_safe_smart(self) -> None:
        masker = _masker()
        calls = []
        masker._safe_assess_identity_drift = lambda *args, **kwargs: {
            "available": True,
            "score": 0.4,
            "orientation_error": 0.7,
        }
        masker._smart_protect = lambda **kwargs: calls.append(kwargs) or _result("smart", kwargs["mode"])
        masker._apply_liquid_surface_recovery = lambda **kwargs: kwargs["protected_result"]

        result = masker.protect(_image(), _image(), strategy="auto", mode="strict", prompt="change shirt color")

        self.assertEqual(result.engine, "smart")
        self.assertEqual(calls[0]["mode"], "balanced")
        self.assertEqual(result.metadata["strategy_used"], STRATEGY_SMART)
        self.assertEqual(result.metadata["edit_regime"], EDIT_REGIME_POSE_OR_VIEW)

    def test_same_view_auto_uses_strict_identity(self) -> None:
        masker = _masker()
        calls = []
        masker._safe_assess_identity_drift = lambda *args, **kwargs: {"available": True, "score": 0.01}
        masker._strict_identity_protect = lambda **kwargs: calls.append(kwargs) or _result("strict_identity", kwargs["mode"])
        masker._apply_liquid_surface_recovery = lambda **kwargs: kwargs["protected_result"]

        result = masker.protect(_image(), _image(), strategy="auto", mode="strict", prompt="change shirt color")

        self.assertEqual(result.engine, "strict_identity")
        self.assertEqual(calls[0]["edit_regime"], EDIT_REGIME_SAME_VIEW_MINOR)
        self.assertEqual(result.metadata["strategy_used"], STRATEGY_STRICT_IDENTITY)

    def test_face_selection_scaffolding_prefers_largest_then_reference_match(self) -> None:
        masker = _masker()
        small = FaceCandidate(0, [], (0, 0, 10, 10), 100, (5, 5))
        large = FaceCandidate(1, [], (0, 0, 20, 20), 400, (10, 10))

        self.assertEqual(masker._select_face_candidate([small, large]), large)
        self.assertEqual(masker._select_face_candidate([large, small], reference=small), small)


if __name__ == "__main__":
    unittest.main()
