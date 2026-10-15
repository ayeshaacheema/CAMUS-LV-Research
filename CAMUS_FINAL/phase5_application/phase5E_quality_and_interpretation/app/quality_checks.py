
import numpy as np


def ef_band(ef):
    """Use the same EF-band boundaries as the frozen pipeline."""
    if not np.isfinite(float(ef)):
        return "Unknown"

    ef = float(ef)

    if ef <= 40:
        return "Lower_EF"
    elif ef < 50:
        return "Intermediate_EF"
    return "Higher_EF"


def assess_quality(result):
    """
    Technical quality checks for frozen A4C inference.

    These are review flags, not validated clinical
    reliability or diagnostic assessments.
    """

    warnings = []

    segmentation = result["segmentation"]
    measurements = result["quantitative_measurements"]
    classification = result["classification"]

    ed_mask = np.asarray(
        segmentation["ED"]["predicted_mask"],
        dtype=bool
    )

    es_mask = np.asarray(
        segmentation["ES"]["predicted_mask"],
        dtype=bool
    )

    # ------------------------------------------------------------
    # 1. Empty masks
    # ------------------------------------------------------------
    if not ed_mask.any():
        warnings.append(
            "ED segmentation is empty. "
            "LV measurements require review."
        )

    if not es_mask.any():
        warnings.append(
            "ES segmentation is empty. "
            "LV measurements require review."
        )

    # ------------------------------------------------------------
    # 2. Small-mask technical flag
    # ------------------------------------------------------------
    # 0.5% is an exploratory image-coverage heuristic,
    # not a clinically validated cutoff.
    # ------------------------------------------------------------
    min_fraction = 0.005

    ed_fraction = ed_mask.mean()
    es_fraction = es_mask.mean()

    if 0 < ed_fraction < min_fraction:
        warnings.append(
            "ED LV mask occupies less than 0.5% of "
            "the processed image. Review segmentation."
        )

    if 0 < es_fraction < min_fraction:
        warnings.append(
            "ES LV mask occupies less than 0.5% of "
            "the processed image. Review segmentation."
        )

    # ------------------------------------------------------------
    # 3. ED/ES consistency checks
    # ------------------------------------------------------------
    ed_area = float(measurements["ED_area_mm2"])
    es_area = float(measurements["ES_area_mm2"])

    edv = float(measurements["EDV_ml"])
    esv = float(measurements["ESV_ml"])

    if np.isfinite(ed_area) and np.isfinite(es_area):
        if ed_area <= es_area:
            warnings.append(
                "ED LV area is not larger than ES LV area. "
                "Verify frame selection and segmentation."
            )

    if np.isfinite(edv) and np.isfinite(esv):
        if edv <= esv:
            warnings.append(
                "Estimated EDV is not larger than ESV. "
                "Verify frame selection and measurements."
            )

    # ------------------------------------------------------------
    # 4. EF-band disagreement
    # ------------------------------------------------------------
    ef = float(measurements["EF_pct"])

    geometry_band = ef_band(ef)

    classifier_band = classification[
        "estimated_EF_band"
    ]

    if geometry_band != classifier_band:
        warnings.append(
            "Geometry-derived EF band differs from "
            "the frozen classifier's estimated EF band. "
            "Interpret both outputs cautiously."
        )

    # ------------------------------------------------------------
    # 5. Invalid numerical measurements
    # ------------------------------------------------------------
    for key in [
        "ED_area_mm2",
        "ES_area_mm2",
        "ED_length_mm",
        "ES_length_mm",
        "EDV_ml",
        "ESV_ml",
        "SV_ml",
        "EF_pct"
    ]:
        value = float(measurements[key])

        if not np.isfinite(value):
            warnings.append(
                f"{key} is not finite. "
                "Quantitative output requires review."
            )

    return {
        "status": (
            "REVIEW_RECOMMENDED"
            if warnings else "NO_RULE_FLAGS"
        ),
        "warnings": warnings,
        "ED_mask_fraction": float(ed_fraction),
        "ES_mask_fraction": float(es_fraction),
        "geometry_EF_band": geometry_band,
        "classifier_EF_band": classifier_band,
        "interpretation": (
            "Technical review flags only. "
            "Absence of flags does not establish "
            "segmentation accuracy, EF accuracy, "
            "or clinical reliability."
        )
    }
