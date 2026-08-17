import os
from ultralytics import YOLO

# ============================================================
# 21-CLASS VEHICLE PART MODEL
# ============================================================

MODEL_PATH = r"C:\projects\Vehicle_AI_Insurance\runs\segment\runs\insure_ai\vehicle_damage-2\weights\best.pt"

model = YOLO(MODEL_PATH)

print("=" * 70)
print("INSURE AI - VEHICLE PART DETECTOR")
print("=" * 70)
print("Model:", MODEL_PATH)
print("Classes:", model.names)
print("=" * 70)


# Parts that can indicate an area that may be damaged.
# IMPORTANT:
# This model detects PARTS, not actual damage.
# Therefore we should NOT claim that every detected part is damaged.
VEHICLE_PARTS = {
    "Front-bumper",
    "Back-bumper",
    "Front-door",
    "Back-door",
    "Fender",
    "Quarter-panel",
    "Hood",
    "Trunk",
    "Rocker-panel",
    "Mirror",
    "Headlight",
    "Tail-light",
    "Windshield",
    "Back-windshield",
    "Front-window",
    "Back-window",
    "Roof",
    "License-plate",
    "Grille",
    "Front-wheel",
    "Back-wheel",
}


def predict_damage(image_path):

    if not os.path.exists(image_path):
        return {
            "success": False,
            "error": f"Image not found: {image_path}",
            "damage_detected": False,
            "status": "Image not found",
            "detections": [],
            "damaged_parts": [],
            "damage_confidence": 0.0,
        }

    # --------------------------------------------------------
    # YOLO prediction
    # --------------------------------------------------------

    results = model.predict(
        source=image_path,
        conf=0.15,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names.get(
                cls_id,
                str(cls_id)
            )

            xyxy = box.xyxy[0].tolist()

            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [
                    round(float(x), 2)
                    for x in xyxy
                ]
            })

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # The 21-class model detects vehicle PARTS.
    #
    # It does NOT contain a "damage" class.
    #
    # Therefore:
    #
    #     We cannot honestly say that a detected part is damaged.
    #
    # We return detected parts separately.
    # --------------------------------------------------------

    damaged_parts = []

    # Only use detections as "affected candidates".
    # Actual damage confirmation must come from a damage model
    # or a visual damage classifier.

    for detection in detections:

        part = detection["class_name"]

        if part in VEHICLE_PARTS:

            damaged_parts.append({
                "part": part,
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "damage_confirmed": False
            })

    # --------------------------------------------------------
    # Current model status
    # --------------------------------------------------------

    if detections:

        status = "Vehicle Parts Detected"

    else:

        status = "No Vehicle Parts Detected"

    return {
        "success": True,

        # IMPORTANT:
        # Do NOT falsely mark damage as detected.
        "damage_detected": False,

        "status": status,

        "damage_confidence": 0.0,

        "detections": detections,

        "damaged_parts": damaged_parts,

        "model_type": "21-class vehicle-part detector",

        "damage_model_available": False
    }