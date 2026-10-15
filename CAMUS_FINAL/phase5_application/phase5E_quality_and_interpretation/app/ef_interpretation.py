
import numpy as np


def get_ef_band(ef):
    """Use the frozen pipeline's EF-band definitions."""
    ef = float(ef)

    if not np.isfinite(ef):
        return "Unknown"

    if ef <= 40:
        return "Lower_EF"
    elif ef < 50:
        return "Intermediate_EF"
    return "Higher_EF"


def build_ef_interpretation(result, quality_assessment):
    """
    Explain frozen geometric EF and classifier outputs.

    Does not modify measurements or classifier predictions.
    """

    measurements = result["quantitative_measurements"]
    classification = result["classification"]

    edv = float(measurements["EDV_ml"])
    esv = float(measurements["ESV_ml"])
    ef = float(measurements["EF_pct"])

    classifier_band = classification["estimated_EF_band"]
    geometry_band = get_ef_band(ef)

    class_scores = classification.get("class_scores", {})

    if not all(np.isfinite(x) for x in [edv, esv, ef]):
        ef_formula_status = "INVALID_NUMERICAL_VALUES"
    elif edv <= 0:
        ef_formula_status = "INVALID_EDV"
    else:
        reconstructed_ef = 100.0 * (edv - esv) / edv

        ef_formula_status = (
            "CONSISTENT"
            if np.isclose(
                reconstructed_ef,
                ef,
                atol=1e-6,
                rtol=1e-9
            )
            else "INCONSISTENT"
        )

    agreement = geometry_band == classifier_band

    return {
        "geometry_EF_pct": ef,
        "geometry_EF_band": geometry_band,
        "classifier_EF_band": classifier_band,
        "bands_agree": bool(agreement),
        "EF_formula_status": ef_formula_status,
        "class_scores": dict(class_scores),
        "technical_flags": list(
            quality_assessment["warnings"]
        ),
        "method_description": (
            "EF is estimated from EDV and ESV derived "
            "from predicted LV segmentation masks using "
            "a simplified single-plane A4C area-length method."
        ),
        "classifier_description": (
            "The EF-band classifier is a separate learned "
            "model using predicted geometric measurements. "
            "Its band may differ from the band obtained "
            "by directly thresholding the geometric EF."
        ),
        "limitations": [
            (
                "Single-plane geometric EF may differ from "
                "reference EF measurements."
            ),
            (
                "Segmentation errors may propagate into "
                "EDV, ESV, SV and EF."
            ),
            (
                "Classifier scores are uncalibrated and do "
                "not represent clinical reliability."
            ),
            (
                "Absence of technical flags does not "
                "establish quantitative accuracy."
            ),
            (
                "This system is an assistive research "
                "prototype and is not diagnostic."
            )
        ]
    }
