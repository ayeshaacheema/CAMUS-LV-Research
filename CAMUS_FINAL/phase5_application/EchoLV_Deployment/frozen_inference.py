
# Frozen inference implementation extracted from Phase 5B.
# Do not modify preprocessing, segmentation or geometry methods.

import numpy as np
import pandas as pd
import cv2
import nibabel as nib
import tensorflow as tf
import joblib

from pathlib import Path
from scipy.ndimage import binary_fill_holes
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent

FROZEN_SEGMENTATION_MODEL_PATH = (
    PROJECT_ROOT
    / "phase1_segmentation"
    / "phase1F_final_model"
    / "models"
    / "final_lv_segmentation_model.keras"
)

PHASE3_CLASSIFIER_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "phase3_classification"
    / "phase3C_classifier_development"
    / "models"
    / "phase3C_selected_logistic_regression.joblib"
)



# ================================================================
# PHASE 5B — CELL 30
# EXACT FROZEN PREPROCESSING + PHYSICAL GEOMETRY
# Reconstructed directly from authoritative Phase 2A implementation
# ================================================================

import cv2
import nibabel as nib

from scipy.ndimage import binary_fill_holes


SEGMENTATION_RESOLUTION = 256
SEGMENTATION_THRESHOLD = 0.50


def load_nifti_2d(file_path):

    nii = nib.load(
        str(file_path)
    )

    data = np.asarray(
        nii.get_fdata()
    )

    if (
        data.ndim == 3
        and data.shape[-1] == 1
    ):
        data = data[:, :, 0]

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D NIfTI, got {data.shape}"
        )

    spacing = tuple(
        float(x)
        for x in nii.header.get_zooms()[:2]
    )

    return data, spacing


def preprocess_for_segmentation(
    image,
    target_size=256
):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    original_h, original_w = image.shape

    resized = cv2.resize(
        image,
        (
            target_size,
            target_size
        ),
        interpolation=cv2.INTER_LINEAR
    )

    minimum = float(
        resized.min()
    )

    maximum = float(
        resized.max()
    )

    if maximum > minimum:

        normalized = (
            resized - minimum
        ) / (
            maximum - minimum
        )

    else:

        normalized = np.zeros_like(
            resized
        )

    scale_x = (
        original_w
        /
        target_size
    )

    scale_y = (
        original_h
        /
        target_size
    )

    return (
        normalized.astype(
            np.float32
        ),
        (
            scale_x,
            scale_y
        )
    )


def effective_spacing(
    native_spacing,
    scale_factors
):

    spacing_x, spacing_y = map(
        float,
        native_spacing
    )

    scale_x, scale_y = map(
        float,
        scale_factors
    )

    return (
        spacing_x * scale_x,
        spacing_y * scale_y
    )


def keep_largest_component(mask):

    mask_uint8 = (
        np.asarray(mask)
        .astype(np.uint8)
    )

    if mask_uint8.sum() == 0:

        return np.zeros_like(
            mask_uint8,
            dtype=bool
        )

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8
        )
    )

    if n_labels <= 1:

        return np.zeros_like(
            mask_uint8,
            dtype=bool
        )

    largest_label = (
        1
        +
        np.argmax(
            stats[
                1:,
                cv2.CC_STAT_AREA
            ]
        )
    )

    return (
        labels
        ==
        largest_label
    )


def postprocess_prediction(
    probability_mask,
    threshold=0.50
):

    binary_mask = (
        probability_mask
        >=
        threshold
    )

    largest = keep_largest_component(
        binary_mask
    )

    filled = binary_fill_holes(
        largest
    )

    return np.asarray(
        filled
    ).astype(bool)


def mask_area_mm2(
    mask,
    spacing
):

    mask = np.asarray(
        mask
    ).astype(bool)

    spacing_x, spacing_y = map(
        float,
        spacing
    )

    return float(
        mask.sum()
        *
        spacing_x
        *
        spacing_y
    )


def mask_points_mm(
    mask,
    spacing
):

    mask = np.asarray(
        mask
    ).astype(bool)

    rows, cols = np.where(
        mask
    )

    spacing_x, spacing_y = map(
        float,
        spacing
    )

    # Exact Phase 2A coordinate convention:
    # x = image columns
    # y = image rows

    x_mm = (
        cols.astype(np.float64)
        *
        spacing_x
    )

    y_mm = (
        rows.astype(np.float64)
        *
        spacing_y
    )

    return np.column_stack([
        x_mm,
        y_mm
    ])


def pca_major_axis_mm(
    mask,
    spacing
):

    points = mask_points_mm(
        mask,
        spacing
    )

    if len(points) < 2:
        return np.nan

    centered = (
        points
        -
        points.mean(
            axis=0
        )
    )

    covariance = np.cov(
        centered,
        rowvar=False
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            covariance
        )
    )

    principal_axis = (
        eigenvectors[
            :,
            np.argmax(
                eigenvalues
            )
        ]
    )

    projections = (
        centered
        @
        principal_axis
    )

    return float(
        projections.max()
        -
        projections.min()
    )


def frozen_area_length_volume_ml(
    area_mm2,
    length_mm
):

    if (
        not np.isfinite(area_mm2)
        or not np.isfinite(length_mm)
        or area_mm2 <= 0
        or length_mm <= 0
    ):
        return np.nan

    area_cm2 = (
        area_mm2
        /
        100.0
    )

    length_cm = (
        length_mm
        /
        10.0
    )

    volume_ml = (
        8.0
        /
        (
            3.0
            *
            np.pi
        )
        *
        (
            area_cm2 ** 2
        )
        /
        length_cm
    )

    return float(
        volume_ml
    )


print("=" * 80)
print("FROZEN PHASE 2 MEASUREMENT FUNCTIONS")
print("=" * 80)

print("\n✓ 256×256 stretch preprocessing")
print("✓ Post-resize per-image min-max normalization")
print("✓ Anisotropic effective spacing")
print("✓ Threshold >= 0.50")
print("✓ 8-connected largest component")
print("✓ Binary hole filling")
print("✓ Physical LV area")
print("✓ Exact PCA major-axis length")
print("✓ Single-plane area-length volume")

# ================================================================
# PHASE 5B — CELL 31
# LOAD EXACT FROZEN PHASE 1 SEGMENTATION MODEL
# ================================================================

import gc
import tensorflow as tf


SMOOTH = 1e-6


def dice_coefficient(
    y_true,
    y_pred
):

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred,
        tf.float32
    )

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1, 2, 3]
    )

    denominator = (
        tf.reduce_sum(
            y_true,
            axis=[1, 2, 3]
        )
        +
        tf.reduce_sum(
            y_pred,
            axis=[1, 2, 3]
        )
    )

    dice = (
        2.0 * intersection
        +
        SMOOTH
    ) / (
        denominator
        +
        SMOOTH
    )

    return tf.reduce_mean(
        dice
    )


def dice_loss(
    y_true,
    y_pred
):

    return (
        1.0
        -
        dice_coefficient(
            y_true,
            y_pred
        )
    )


def bce_dice_loss(
    y_true,
    y_pred
):

    bce = (
        tf.keras.losses
        .binary_crossentropy(
            y_true,
            y_pred
        )
    )

    bce = tf.reduce_mean(
        bce
    )

    return (
        bce
        +
        dice_loss(
            y_true,
            y_pred
        )
    )


CUSTOM_OBJECTS = {

    "bce_dice_loss":
        bce_dice_loss,

    "dice_loss":
        dice_loss,

    "dice_coefficient":
        dice_coefficient
}


gc.collect()


frozen_segmentation_model = (
    tf.keras.models.load_model(
        FROZEN_SEGMENTATION_MODEL_PATH,
        custom_objects=CUSTOM_OBJECTS,
        compile=False
    )
)


assert (
    frozen_segmentation_model.input_shape
    ==
    (None, 256, 256, 1)
)

assert (
    frozen_segmentation_model.output_shape
    ==
    (None, 256, 256, 1)
)

assert (
    frozen_segmentation_model.count_params()
    ==
    7_759_521
)


print("=" * 80)
print("FROZEN SEGMENTATION MODEL LOADED")
print("=" * 80)

print(
    "\nParameters:",
    f"{frozen_segmentation_model.count_params():,}"
)

print(
    "Input shape :",
    frozen_segmentation_model.input_shape
)

print(
    "Output shape:",
    frozen_segmentation_model.output_shape
)

print(
    "\n✓ Frozen Phase 1 model ready for inference."
)

# ================================================================
# PHASE 5B — CELL 33
# SINGLE-FRAME FROZEN SEGMENTATION → MEASUREMENT FUNCTION
# ================================================================

def analyze_lv_frame(
    image,
    native_spacing
):

    # ------------------------------------------------------------
    # 1. Frozen preprocessing
    # ------------------------------------------------------------

    processed_image, scale_factors = (
        preprocess_for_segmentation(
            image,
            target_size=
                SEGMENTATION_RESOLUTION
        )
    )


    resized_spacing = effective_spacing(
        native_spacing,
        scale_factors
    )


    # ------------------------------------------------------------
    # 2. Frozen segmentation inference
    # ------------------------------------------------------------

    model_input = (
        processed_image[
            None,
            ...,
            None
        ]
    )


    probability_mask = (
        frozen_segmentation_model(
            model_input,
            training=False
        )
        .numpy()[0, :, :, 0]
    )


    predicted_mask = (
        postprocess_prediction(
            probability_mask,
            threshold=
                SEGMENTATION_THRESHOLD
        )
    )


    # ------------------------------------------------------------
    # 3. Frozen quantitative geometry
    # ------------------------------------------------------------

    area_mm2 = mask_area_mm2(
        predicted_mask,
        resized_spacing
    )


    length_mm = pca_major_axis_mm(
        predicted_mask,
        resized_spacing
    )


    volume_ml = (
        frozen_area_length_volume_ml(
            area_mm2,
            length_mm
        )
    )


    # ------------------------------------------------------------
    # 4. Structured output
    # ------------------------------------------------------------

    return {

        "processed_image":
            processed_image,

        "probability_mask":
            probability_mask,

        "predicted_mask":
            predicted_mask,

        "native_spacing":
            tuple(
                float(x)
                for x in native_spacing
            ),

        "effective_spacing":
            tuple(
                float(x)
                for x in resized_spacing
            ),

        "scale_factors":
            tuple(
                float(x)
                for x in scale_factors
            ),

        "area_mm2":
            float(area_mm2),

        "length_mm":
            float(length_mm),

        "volume_ml":
            float(volume_ml)
    }


print(
    "✓ Single-frame frozen inference function ready."
)


frozen_phase3_classifier = joblib.load(
    PHASE3_CLASSIFIER_ARTIFACT_PATH
)

assert isinstance(frozen_phase3_classifier, Pipeline)

assert list(frozen_phase3_classifier.classes_) == [
    "Higher_EF",
    "Intermediate_EF",
    "Lower_EF"
]


# ================================================================
# PHASE 5B — CELL 38
# COMPLETE FROZEN PATIENT-LEVEL INFERENCE PIPELINE
# ================================================================

def analyze_a4c_pair(
    ed_image,
    es_image,
    ed_native_spacing,
    es_native_spacing
):
    """
    Complete frozen assistive A4C analysis pipeline.

    Required input
    --------------
    ed_image, es_image:
        Identified A4C ED and ES 2D frames.

    ed_native_spacing, es_native_spacing:
        Original physical pixel spacing from the source images.

    Returns
    -------
    Structured segmentation, quantitative, and EF-band outputs.

    Important
    ---------
    This is an assistive, non-diagnostic pipeline.
    The classifier confidence is an uncalibrated model
    confidence score and must not be interpreted as a
    patient-level reliability or safety guarantee.
    """

    # ============================================================
    # PHASE 1 + PHASE 2
    # ============================================================

    ed_result = analyze_lv_frame(
        ed_image,
        ed_native_spacing
    )

    es_result = analyze_lv_frame(
        es_image,
        es_native_spacing
    )


    # ------------------------------------------------------------
    # Quantitative patient-level measurements
    # ------------------------------------------------------------

    edv_ml = float(
        ed_result["volume_ml"]
    )

    esv_ml = float(
        es_result["volume_ml"]
    )


    if (
        not np.isfinite(edv_ml)
        or not np.isfinite(esv_ml)
        or edv_ml <= 0
    ):
        raise ValueError(
            "Valid EDV/ESV could not be derived "
            "from the predicted LV masks."
        )


    sv_ml = float(
        edv_ml
        -
        esv_ml
    )


    ef_pct = float(
        sv_ml
        /
        edv_ml
        *
        100.0
    )


    # ============================================================
    # PHASE 3 FEATURE VECTOR
    # Exact frozen feature order
    # ============================================================

    feature_values = np.array(
        [[
            ed_result["area_mm2"],
            es_result["area_mm2"],

            ed_result["length_mm"],
            es_result["length_mm"],

            edv_ml,
            esv_ml,

            sv_ml,
            ef_pct
        ]],
        dtype=float
    )


    assert (
        feature_values.shape
        ==
        (1, 8)
    )


    # ============================================================
    # PHASE 3 — FROZEN CLASSIFIER
    # ============================================================

    predicted_class = (
        frozen_phase3_classifier
        .predict(
            feature_values
        )[0]
    )


    class_probabilities = (
        frozen_phase3_classifier
        .predict_proba(
            feature_values
        )[0]
    )


    classifier_classes = (
        frozen_phase3_classifier
        .classes_
    )


    class_scores = {

        str(class_name):
            float(score)

        for class_name, score
        in zip(
            classifier_classes,
            class_probabilities
        )
    }


    model_confidence_score = float(
        np.max(
            class_probabilities
        )
    )


    # ============================================================
    # STRUCTURED APPLICATION OUTPUT
    # ============================================================

    return {

        "segmentation": {

            "ED": {
                "processed_image":
                    ed_result[
                        "processed_image"
                    ],

                "probability_mask":
                    ed_result[
                        "probability_mask"
                    ],

                "predicted_mask":
                    ed_result[
                        "predicted_mask"
                    ],

                "effective_spacing":
                    ed_result[
                        "effective_spacing"
                    ]
            },

            "ES": {
                "processed_image":
                    es_result[
                        "processed_image"
                    ],

                "probability_mask":
                    es_result[
                        "probability_mask"
                    ],

                "predicted_mask":
                    es_result[
                        "predicted_mask"
                    ],

                "effective_spacing":
                    es_result[
                        "effective_spacing"
                    ]
            }
        },


        "quantitative_measurements": {

            "ED_area_mm2":
                float(
                    ed_result[
                        "area_mm2"
                    ]
                ),

            "ES_area_mm2":
                float(
                    es_result[
                        "area_mm2"
                    ]
                ),

            "ED_length_mm":
                float(
                    ed_result[
                        "length_mm"
                    ]
                ),

            "ES_length_mm":
                float(
                    es_result[
                        "length_mm"
                    ]
                ),

            "EDV_ml":
                edv_ml,

            "ESV_ml":
                esv_ml,

            "SV_ml":
                sv_ml,

            "EF_pct":
                ef_pct
        },


        "classification": {

            "estimated_EF_band":
                str(
                    predicted_class
                ),

            "class_scores":
                class_scores,

            "model_confidence_score":
                model_confidence_score,

            "confidence_interpretation":
                (
                    "Uncalibrated model confidence score; "
                    "not a patient-level reliability or "
                    "safety guarantee."
                )
        },


        "system": {

            "view":
                "A4C",

            "input_requirement":
                (
                    "Paired identified ED and ES frames "
                    "with physical pixel-spacing metadata."
                ),

            "purpose":
                (
                    "Assistive LV analysis."
                ),

            "diagnostic_status":
                "Non-diagnostic"
        }
    }


print("=" * 85)
print("COMPLETE FROZEN A4C PIPELINE")
print("=" * 85)

print(
    "\nED/ES input"
    "\n   ↓"
    "\nFrozen LV segmentation"
    "\n   ↓"
    "\nPhysical LV geometry"
    "\n   ↓"
    "\nEDV / ESV / SV / EF"
    "\n   ↓"
    "\nFrozen EF-band classifier"
    "\n   ↓"
    "\nEF band + uncalibrated confidence"
)

print(
    "\n✓ analyze_a4c_pair() ready."
)