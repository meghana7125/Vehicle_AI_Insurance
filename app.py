import os
import uuid
import math
import traceback
from datetime import datetime

import cv2

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
)

from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from ultralytics import YOLO

from utils.storage import (
    save_user,
    get_user_by_email,
    update_user,
    save_assessment,
    get_assessment,
    save_claim,
    get_claim,
    get_claims_by_user,
    get_all_claims,
    update_claim_status,
)

from utils.severity import calculate_severity
from utils.cost_estimator import estimate_repair_cost


# ============================================================
# APP CONFIG
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "INSURE_AI_SECRET_KEY",
    "insure-ai-development-secret-key",
)

CORS(app)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads",
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True,
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
}


# ============================================================
# MODEL PATHS
# ============================================================

DAMAGE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best.pt",
)

PART_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "car_parts_best.pt",
)

VEHICLE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo26n.pt",
)


# ============================================================
# MODEL LOADING
# ============================================================

print("=" * 75)
print("INSURE AI - MODEL INITIALIZATION")
print("=" * 75)


if not os.path.exists(DAMAGE_MODEL_PATH):
    raise FileNotFoundError(
        f"Damage model not found:\n{DAMAGE_MODEL_PATH}"
    )


if not os.path.exists(PART_MODEL_PATH):
    raise FileNotFoundError(
        f"Car-part model not found:\n{PART_MODEL_PATH}"
    )


if not os.path.exists(VEHICLE_MODEL_PATH):
    raise FileNotFoundError(
        f"Vehicle model not found:\n{VEHICLE_MODEL_PATH}"
    )


print(
    "[1] Loading DAMAGE model:",
    DAMAGE_MODEL_PATH,
)

damage_model = YOLO(
    DAMAGE_MODEL_PATH
)

print(
    "    classes:",
    damage_model.names
)


print(
    "[2] Loading CAR PART model:",
    PART_MODEL_PATH,
)

part_model = YOLO(
    PART_MODEL_PATH
)

print(
    "    classes:",
    part_model.names
)


print(
    "[3] Loading VEHICLE model:",
    VEHICLE_MODEL_PATH,
)

vehicle_model = YOLO(
    VEHICLE_MODEL_PATH
)

print(
    "    classes:",
    vehicle_model.names
)


print("=" * 75)
print("ALL MODELS LOADED")
print("=" * 75)


# ============================================================
# GENERAL HELPERS
# ============================================================

def allowed_file(filename):

    return (
        bool(filename)
        and "."
        in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


def current_user():

    email = session.get(
        "user_email"
    )

    if not email:
        return None

    return get_user_by_email(
        email
    )


def login_required():

    return (
        session.get("user_email")
        is not None
    )


def model_class_name(
    model,
    class_id,
):

    names = model.names

    if isinstance(names, dict):

        return str(
            names.get(
                class_id,
                class_id,
            )
        )

    if isinstance(
        names,
        (list, tuple),
    ):

        if (
            0
            <= class_id
            < len(names)
        ):

            return str(
                names[class_id]
            )

    return str(class_id)


# ============================================================
# BOX HELPERS
# ============================================================

def box_area(box):

    if (
        not box
        or len(box) != 4
    ):
        return 0.0

    x1, y1, x2, y2 = map(
        float,
        box,
    )

    return (
        max(
            0.0,
            x2 - x1,
        )
        *
        max(
            0.0,
            y2 - y1,
        )
    )


def box_center(box):

    if (
        not box
        or len(box) != 4
    ):
        return 0.0, 0.0

    x1, y1, x2, y2 = map(
        float,
        box,
    )

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def calculate_intersection(
    a,
    b,
):

    if (
        not a
        or not b
        or len(a) != 4
        or len(b) != 4
    ):
        return 0.0

    ax1, ay1, ax2, ay2 = map(
        float,
        a,
    )

    bx1, by1, bx2, by2 = map(
        float,
        b,
    )

    x1 = max(
        ax1,
        bx1,
    )

    y1 = max(
        ay1,
        by1,
    )

    x2 = min(
        ax2,
        bx2,
    )

    y2 = min(
        ay2,
        by2,
    )

    return (
        max(
            0.0,
            x2 - x1,
        )
        *
        max(
            0.0,
            y2 - y1,
        )
    )


def calculate_iou(
    a,
    b,
):

    intersection = calculate_intersection(
        a,
        b,
    )

    union = (
        box_area(a)
        +
        box_area(b)
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        /
        union
    )


def overlap_over_damage(
    damage_box,
    part_box,
):

    damage_area = box_area(
        damage_box
    )

    if damage_area <= 0:
        return 0.0

    return (
        calculate_intersection(
            damage_box,
            part_box,
        )
        /
        damage_area
    )


def overlap_over_part(
    damage_box,
    part_box,
):

    part_area = box_area(
        part_box
    )

    if part_area <= 0:
        return 0.0

    return (
        calculate_intersection(
            damage_box,
            part_box,
        )
        /
        part_area
    )


def center_inside_box(
    point,
    box,
):

    if (
        not box
        or len(box) != 4
    ):
        return False

    x, y = point

    x1, y1, x2, y2 = map(
        float,
        box,
    )

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


def center_proximity_score(
    damage_box,
    part_box,
):

    dcx, dcy = box_center(
        damage_box
    )

    pcx, pcy = box_center(
        part_box
    )

    pw = max(
        1.0,
        float(part_box[2])
        -
        float(part_box[0]),
    )

    ph = max(
        1.0,
        float(part_box[3])
        -
        float(part_box[1]),
    )

    diagonal = max(
        1.0,
        math.sqrt(
            pw * pw
            +
            ph * ph
        ),
    )

    distance = math.sqrt(
        (dcx - pcx) ** 2
        +
        (dcy - pcy) ** 2
    )

    return max(
        0.0,
        1.0
        -
        distance / diagonal,
    )


def clamp_box(
    box,
    width,
    height,
):

    if (
        not box
        or len(box) != 4
    ):
        return None

    x1, y1, x2, y2 = map(
        float,
        box,
    )

    x1 = max(
        0.0,
        min(
            float(width),
            x1,
        ),
    )

    y1 = max(
        0.0,
        min(
            float(height),
            y1,
        ),
    )

    x2 = max(
        0.0,
        min(
            float(width),
            x2,
        ),
    )

    y2 = max(
        0.0,
        min(
            float(height),
            y2,
        ),
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):
        return None

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def unique_by_iou(
    detections,
    iou_threshold=0.50,
    limit=20,
):

    result = []

    ordered = sorted(
        detections,
        key=lambda x: float(
            x.get(
                "confidence",
                0.0,
            )
        ),
        reverse=True,
    )

    for detection in ordered:

        if not detection.get(
            "bbox"
        ):
            continue

        duplicate = False

        for old in result:

            if (
                calculate_iou(
                    detection["bbox"],
                    old["bbox"],
                )
                >= iou_threshold
            ):
                duplicate = True
                break

        if duplicate:
            continue

        result.append(
            detection
        )

        if len(result) >= limit:
            break

    return result


# ============================================================
# VEHICLE DETECTION
# ============================================================

def detect_vehicle_type(
    image_path,
):

    print(
        "\n[STEP 1] VEHICLE DETECTION"
    )

    try:

        results = vehicle_model.predict(
            source=image_path,
            conf=0.10,
            iou=0.45,
            imgsz=1280,
            max_det=30,
            verbose=False,
        )

    except Exception as exc:

        print(
            "[VEHICLE] ERROR:",
            exc,
        )

        return (
            "unknown",
            0.0,
            [],
        )

    if (
        not results
        or
        results[0].boxes is None
    ):

        print(
            "[VEHICLE] No vehicle boxes."
        )

        return (
            "unknown",
            0.0,
            [],
        )

    supported = {
        "car",
        "truck",
        "bus",
        "motorcycle",
    }

    detections = []

    for box in results[0].boxes:

        try:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            name = model_class_name(
                vehicle_model,
                class_id,
            ).strip().lower()

            bbox = [
                round(
                    float(v),
                    2,
                )
                for v in
                box.xyxy[0].tolist()
            ]

        except Exception:

            continue

        if name not in supported:
            continue

        detections.append({
            "class_name": name,
            "confidence": round(
                confidence * 100.0,
                2,
            ),
            "bbox": bbox,
        })

    if not detections:

        print(
            "[VEHICLE] No supported vehicle."
        )

        return (
            "unknown",
            0.0,
            [],
        )

    cars = [
        d
        for d in detections
        if d["class_name"]
        == "car"
    ]

    best = max(
        cars or detections,
        key=lambda d:
        d["confidence"],
    )

    print(
        "[VEHICLE]",
        best["class_name"],
        best["confidence"],
        "%",
    )

    return (
        best["class_name"],
        best["confidence"],
        detections,
    )


# ============================================================
# DAMAGE DETECTION
# ============================================================

def detect_damage(
    image_path,
    vehicle_detections=None,
):

    """
    IMPORTANT:

    The damage model is the ONLY model allowed
    to declare that damage exists.

    Current damage model:

        class 0 = damage
        class 1 = whole

    The parts model NEVER declares damage.
    """

    print(
        "\n[STEP 2] DAMAGE DETECTION"
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "[DAMAGE] Image could not be read."
        )

        return []

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # FIND VEHICLE REGION
    # --------------------------------------------------------

    valid_vehicle_boxes = []

    for detection in (
        vehicle_detections or []
    ):

        bbox = detection.get(
            "bbox",
            [],
        )

        name = str(
            detection.get(
                "class_name",
                "",
            )
        ).lower()

        confidence = float(
            detection.get(
                "confidence",
                0.0,
            )
        )

        if (
            name in {
                "car",
                "truck",
                "bus",
                "motorcycle",
            }
            and
            len(bbox) == 4
        ):

            valid_vehicle_boxes.append(
                (
                    name,
                    confidence,
                    bbox,
                )
            )

    cars = [
        x
        for x in valid_vehicle_boxes
        if x[0] == "car"
    ]

    primary = max(
        cars or valid_vehicle_boxes,
        key=lambda x: x[1],
        default=None,
    )

    if primary:

        _vehicle_name = primary[0]
        vehicle_confidence = primary[1]
        original_vehicle_box = primary[2]

        vx1, vy1, vx2, vy2 = (
            original_vehicle_box
        )

        vw = max(
            1.0,
            vx2 - vx1,
        )

        vh = max(
            1.0,
            vy2 - vy1,
        )

        pad_x = max(
            10.0,
            vw * 0.08,
        )

        pad_y = max(
            10.0,
            vh * 0.08,
        )

        rx1 = max(
            0,
            int(vx1 - pad_x),
        )

        ry1 = max(
            0,
            int(vy1 - pad_y),
        )

        rx2 = min(
            width,
            int(vx2 + pad_x),
        )

        ry2 = min(
            height,
            int(vy2 + pad_y),
        )

        roi = image[
            ry1:ry2,
            rx1:rx2,
        ]

        vehicle_box = [
            float(v)
            for v in original_vehicle_box
        ]

        print(
            "[DAMAGE] Vehicle ROI:",
            vehicle_box,
        )

        print(
            "[DAMAGE] Vehicle confidence:",
            round(
                vehicle_confidence,
                2,
            ),
            "%",
        )

    else:

        rx1 = 0
        ry1 = 0

        roi = image

        vehicle_box = [
            0.0,
            0.0,
            float(width),
            float(height),
        ]

        print(
            "[DAMAGE] No vehicle box."
        )

        print(
            "[DAMAGE] Using full image."
        )

    if (
        roi is None
        or roi.size == 0
    ):

        return []

    detections = []

    # --------------------------------------------------------
    # VEHICLE OVERLAP CHECK
    # --------------------------------------------------------

    def accepts_vehicle_overlap(
        box,
    ):

        if primary is None:
            return True

        damage_area = box_area(
            box
        )

        if damage_area <= 0:
            return False

        overlap = (
            calculate_intersection(
                box,
                vehicle_box,
            )
            /
            damage_area
        )

        cx, cy = box_center(
            box
        )

        center_inside = center_inside_box(
            (
                cx,
                cy,
            ),
            vehicle_box,
        )

        return (
            (
                overlap >= 0.15
                and
                center_inside
            )
            or
            overlap >= 0.45
        )

    # --------------------------------------------------------
    # COLLECT YOLO DAMAGE BOXES
    # --------------------------------------------------------

    def collect(
        result,
        source,
        offset_x=0,
        offset_y=0,
    ):

        if (
            result is None
            or
            result.boxes is None
        ):
            return

        for box in result.boxes:

            try:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                name = model_class_name(
                    damage_model,
                    class_id,
                ).strip().lower()

                raw = [
                    float(v)
                    for v in
                    box.xyxy[0].tolist()
                ]

            except Exception:

                continue

            # ONLY class "damage"
            if name != "damage":
                continue

            # IMPORTANT:
            # 8% minimum allows the current model
            # to recover weak genuine damage.
            if confidence < 0.08:
                continue

            mapped = [
                raw[0]
                + offset_x
                + rx1,

                raw[1]
                + offset_y
                + ry1,

                raw[2]
                + offset_x
                + rx1,

                raw[3]
                + offset_y
                + ry1,
            ]

            mapped = clamp_box(
                mapped,
                width,
                height,
            )

            if mapped is None:
                continue

            area = box_area(
                mapped
            )

            # Reject extremely tiny noise.
            if area < max(
                80.0,
                width
                * height
                * 0.00005,
            ):
                continue

            if not accepts_vehicle_overlap(
                mapped
            ):
                continue

            detections.append({
                "class_id": class_id,
                "class_name": "damage",

                "confidence": round(
                    confidence * 100.0,
                    2,
                ),

                "confidence_percent": round(
                    confidence * 100.0,
                    2,
                ),

                "bbox": [
                    round(
                        v,
                        2,
                    )
                    for v in mapped
                ],

                "source": source,
            })

    # --------------------------------------------------------
    # PASS 1: LARGE VEHICLE ROI
    # --------------------------------------------------------

    print(
        "[DAMAGE] Running full ROI inference..."
    )

    for conf, imgsz in [
        (0.08, 1280),
        (0.06, 1600),
    ]:

        try:

            results = damage_model.predict(
                source=roi,
                conf=conf,
                iou=0.45,
                imgsz=imgsz,
                max_det=50,
                verbose=False,
            )

            if results:

                collect(
                    results[0],
                    "vehicle_roi",
                )

        except Exception as exc:

            print(
                "[DAMAGE] ROI error:",
                exc,
            )

    # --------------------------------------------------------
    # PASS 2: OVERLAPPING TILES
    # --------------------------------------------------------

    print(
        "[DAMAGE] ROI detections:",
        len(detections),
    )

    if not detections:

        print(
            "[DAMAGE] No ROI damage."
        )

        print(
            "[DAMAGE] Starting localized scan..."
        )

        rh, rw = roi.shape[:2]

        # Smaller tiles help with headlights,
        # bumpers, doors and mirrors.
        tile_ratios = [
            0.55,
            0.65,
            0.75,
        ]

        for ratio in tile_ratios:

            tile_w = min(
                rw,
                max(
                    384,
                    int(
                        rw * ratio
                    ),
                ),
            )

            tile_h = min(
                rh,
                max(
                    384,
                    int(
                        rh * ratio
                    ),
                ),
            )

            step_x = max(
                160,
                int(
                    tile_w * 0.40
                ),
            )

            step_y = max(
                160,
                int(
                    tile_h * 0.40
                ),
            )

            xs = list(
                range(
                    0,
                    max(
                        1,
                        rw - tile_w + 1,
                    ),
                    step_x,
                )
            )

            ys = list(
                range(
                    0,
                    max(
                        1,
                        rh - tile_h + 1,
                    ),
                    step_y,
                )
            )

            last_x = max(
                0,
                rw - tile_w,
            )

            last_y = max(
                0,
                rh - tile_h,
            )

            if (
                not xs
                or
                xs[-1] != last_x
            ):
                xs.append(
                    last_x
                )

            if (
                not ys
                or
                ys[-1] != last_y
            ):
                ys.append(
                    last_y
                )

            for ty in ys:

                for tx in xs:

                    tile = roi[
                        ty:
                        ty + tile_h,
                        tx:
                        tx + tile_w,
                    ]

                    if (
                        tile is None
                        or
                        tile.size == 0
                    ):
                        continue

                    try:

                        results = damage_model.predict(
                            source=tile,
                            conf=0.06,
                            iou=0.45,
                            imgsz=960,
                            max_det=30,
                            verbose=False,
                        )

                        if results:

                            before = len(
                                detections
                            )

                            collect(
                                results[0],
                                "localized_tile",
                                tx,
                                ty,
                            )

                            after = len(
                                detections
                            )

                            if after > before:

                                print(
                                    "[DAMAGE] Localized damage found:",
                                    after - before,
                                )

                    except Exception as exc:

                        print(
                            "[DAMAGE] Tile error:",
                            exc,
                        )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    final = unique_by_iou(
        detections,
        iou_threshold=0.45,
        limit=20,
    )

    print(
        "[DAMAGE] FINAL REGIONS:",
        len(final),
    )

    for detection in final:

        print(
            "   DAMAGE:",
            detection["confidence"],
            "%",
            "|",
            detection["bbox"],
            "|",
            detection["source"],
        )

    return final


# ============================================================
# VEHICLE PART DETECTION
# ============================================================

def detect_car_parts(
    image_path,
):

    """
    The 21-class model identifies visible vehicle parts.

    IMPORTANT:
    This function NEVER decides whether a part is damaged.
    """

    print(
        "\n[STEP 3] VEHICLE PART DETECTION"
    )

    try:

        results = part_model.predict(
            source=image_path,

            # Slightly permissive so small
            # headlights/mirrors/bumper parts
            # are not missed.
            conf=0.05,

            iou=0.45,

            imgsz=1536,

            max_det=150,

            verbose=False,
        )

    except Exception as exc:

        print(
            "[PARTS] ERROR:",
            exc,
        )

        return []

    if (
        not results
        or
        results[0].boxes is None
    ):

        print(
            "[PARTS] No parts detected."
        )

        return []

    detections = []

    boxes = results[0].boxes

    for box, conf, cls in zip(
        boxes.xyxy.cpu().tolist(),
        boxes.conf.cpu().tolist(),
        boxes.cls.cpu().tolist(),
    ):

        try:

            class_id = int(
                cls
            )

            confidence = float(
                conf
            ) * 100.0

            part_name = model_class_name(
                part_model,
                class_id,
            )

            bbox = [
                float(v)
                for v in box
            ]

        except Exception:

            continue

        if confidence < 5.0:
            continue

        if box_area(bbox) <= 0:
            continue

        detections.append({
            "part": str(
                part_name
            ),

            "class_name": str(
                part_name
            ),

            "class_id": class_id,

            "confidence": round(
                confidence,
                2,
            ),

            "confidence_percent": round(
                confidence,
                2,
            ),

            "bbox": [
                round(
                    v,
                    2,
                )
                for v in bbox
            ],
        })

    # --------------------------------------------------------
    # KEEP STRONGEST BOX FOR EACH PART
    # --------------------------------------------------------

    strongest = {}

    for detection in detections:

        key = str(
            detection[
                "class_name"
            ]
        ).strip().lower()

        if (
            key not in strongest
            or
            detection["confidence"]
            >
            strongest[key]["confidence"]
        ):

            strongest[key] = detection

    detections = sorted(
        strongest.values(),
        key=lambda d:
        d["confidence"],
        reverse=True,
    )

    print(
        "[PARTS] VISIBLE PARTS:"
    )

    for detection in detections:

        print(
            "   ",
            detection[
                "class_name"
            ],
            detection[
                "confidence"
            ],
            "%",
        )

    return detections


# ============================================================
# DAMAGE -> PART ASSOCIATION
# ============================================================

def find_damage_related_parts(
    damage_detections,
    part_detections,
):

    """
    Associates damage regions with vehicle parts.

    Damage model:
        says WHERE damage exists.

    Parts model:
        says WHICH parts are visible.

    This function:
        connects the two spatially.

    It NEVER marks all visible parts as damaged.
    """

    print(
        "\n[STEP 4] DAMAGE -> PART ASSOCIATION"
    )

    if not damage_detections:

        print(
            "[ASSOCIATION] No damage regions."
        )

        return []

    if not part_detections:

        print(
            "[ASSOCIATION] No visible parts."
        )

        return []

    affected = []

    # --------------------------------------------------------
    # FOR EVERY DAMAGE BOX
    # --------------------------------------------------------

    for damage_index, damage in enumerate(
        damage_detections,
        start=1,
    ):

        damage_box = damage.get(
            "bbox",
            [],
        )

        damage_confidence = float(
            damage.get(
                "confidence",
                0.0,
            )
        )

        if len(damage_box) != 4:
            continue

        candidates = []

        # ----------------------------------------------------
        # COMPARE DAMAGE WITH EVERY VISIBLE PART
        # ----------------------------------------------------

        for part in part_detections:

            part_box = part.get(
                "bbox",
                [],
            )

            part_confidence = float(
                part.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                len(part_box) != 4
                or
                part_confidence < 5.0
            ):
                continue

            damage_coverage = (
                overlap_over_damage(
                    damage_box,
                    part_box,
                )
            )

            part_coverage = (
                overlap_over_part(
                    damage_box,
                    part_box,
                )
            )

            iou = calculate_iou(
                damage_box,
                part_box,
            )

            center_score = (
                center_proximity_score(
                    damage_box,
                    part_box,
                )
            )

            damage_center = box_center(
                damage_box
            )

            center_inside = (
                center_inside_box(
                    damage_center,
                    part_box,
                )
            )

            # ------------------------------------------------
            # BASIC SPATIAL FILTER
            # ------------------------------------------------

            if (
                damage_coverage < 0.03
                and
                part_coverage < 0.01
                and
                iou < 0.005
                and
                not center_inside
                and
                center_score < 0.25
            ):
                continue

            # ------------------------------------------------
            # ASSOCIATION SCORE
            # ------------------------------------------------

            score = (

                # Damage box inside part
                0.50
                *
                damage_coverage

                +

                # Part covered by damage
                0.15
                *
                part_coverage

                +

                # Standard IoU
                0.15
                *
                iou

                +

                # Distance between centers
                0.15
                *
                center_score

                +

                # Part detector confidence
                0.05
                *
                min(
                    1.0,
                    part_confidence
                    /
                    100.0,
                )
            )

            # Strong bonus if damage center
            # is actually inside part.
            if center_inside:

                score += 0.10

            # ------------------------------------------------
            # PART NAME
            # ------------------------------------------------

            part_name = str(
                part.get(
                    "class_name",
                    part.get(
                        "part",
                        "Unknown",
                    ),
                )
            )

            candidates.append({

                "part": part_name,

                "class_name": part_name,

                "class_id": part.get(
                    "class_id",
                    -1,
                ),

                "confidence": round(
                    part_confidence,
                    2,
                ),

                "confidence_percent": round(
                    part_confidence,
                    2,
                ),

                "bbox": [
                    round(
                        float(v),
                        2,
                    )
                    for v in part_box
                ],

                "damage_confidence": round(
                    damage_confidence,
                    2,
                ),

                "damage_bbox": [
                    round(
                        float(v),
                        2,
                    )
                    for v in damage_box
                ],

                "association_score": round(
                    score * 100.0,
                    2,
                ),

                "damage_coverage": round(
                    damage_coverage * 100.0,
                    2,
                ),

                "part_coverage": round(
                    part_coverage * 100.0,
                    2,
                ),

                "iou": round(
                    iou * 100.0,
                    2,
                ),

                "center_score": round(
                    center_score * 100.0,
                    2,
                ),
            })

        # ----------------------------------------------------
        # NO CANDIDATE
        # ----------------------------------------------------

        if not candidates:

            print(
                f"[ASSOCIATION] Damage #{damage_index}: "
                "no matching part"
            )

            continue

        # ----------------------------------------------------
        # BEST PART
        # ----------------------------------------------------

        candidates.sort(
            key=lambda x:
            x["association_score"],
            reverse=True,
        )

        best = candidates[0]

        # ----------------------------------------------------
        # STRONG MATCH RULE
        # ----------------------------------------------------

        strong_match = (

            # Most of damage box is inside part
            best["damage_coverage"]
            >= 10.0

            or

            # Meaningful IoU
            best["iou"]
            >= 3.0

            or

            # Damage center strongly inside part
            (
                best["damage_coverage"]
                >= 5.0
                and
                best["center_score"]
                >= 40.0
            )

            or

            # Very strong center match
            (
                best["center_score"]
                >= 70.0
                and
                best["association_score"]
                >= 35.0
            )
        )

        if not strong_match:

            print(
                f"[ASSOCIATION] Damage #{damage_index}: "
                f"weak -> {best['part']} "
                f"({best['association_score']}%)"
            )

            continue

        affected.append(
            best
        )

        print(
            f"[ASSOCIATION] Damage #{damage_index} -> "
            f"{best['part']} | "
            f"association={best['association_score']}% | "
            f"damage={best['damage_confidence']}% | "
            f"damage-coverage={best['damage_coverage']}% | "
            f"IoU={best['iou']}% | "
            f"center={best['center_score']}%"
        )

    # --------------------------------------------------------
    # UNIQUE PARTS
    # --------------------------------------------------------

    unique = {}

    for item in affected:

        key = str(
            item["class_name"]
        ).strip().lower()

        if (
            key not in unique
            or
            item["association_score"]
            >
            unique[key][
                "association_score"
            ]
        ):

            unique[key] = item

    result = sorted(
        unique.values(),
        key=lambda x:
        x["association_score"],
        reverse=True,
    )

    print(
        "\n[ASSOCIATION] ACTUAL DAMAGED PARTS:"
    )

    if not result:

        print(
            "   NONE"
        )

    for item in result:

        print(
            "   ",
            item["class_name"],
            "|",
            item["association_score"],
            "%",
        )

    return result


# Backward compatibility
associate_damage_with_parts = (
    find_damage_related_parts
)


# ============================================================
# ANNOTATED IMAGE
# ============================================================

def create_annotated_image(
    image_path,
    damage_detections,
    affected_parts,
    vehicle_type,
):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return None

    # --------------------------------------------------------
    # RED = ACTUAL DAMAGE
    # --------------------------------------------------------

    for detection in damage_detections:

        bbox = detection.get(
            "bbox",
            [],
        )

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [
            int(round(v))
            for v in bbox
        ]

        confidence = float(
            detection.get(
                "confidence",
                0.0,
            )
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            4,
        )

        cv2.putText(
            image,
            f"DAMAGE {confidence:.1f}%",
            (
                x1,
                max(
                    30,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # BLUE = ONLY DAMAGED PARTS
    # --------------------------------------------------------

    for part in affected_parts:

        bbox = part.get(
            "bbox",
            [],
        )

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [
            int(round(v))
            for v in bbox
        ]

        name = str(
            part.get(
                "class_name",
                "Damaged Part",
            )
        )

        score = float(
            part.get(
                "association_score",
                0.0,
            )
        )

        # OpenCV BGR:
        # blue = 255,120,0
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 120, 0),
            3,
        )

        label = (
            f"DAMAGED: "
            f"{name} "
            f"({score:.0f}%)"
        )

        text_y = min(
            image.shape[0] - 10,
            y2 + 25,
        )

        cv2.putText(
            image,
            label,
            (
                x1,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 120, 0),
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # VEHICLE LABEL
    # --------------------------------------------------------

    cv2.putText(
        image,
        f"Vehicle: {str(vehicle_type).title()}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    filename = (
        f"annotated_"
        f"{uuid.uuid4().hex}.jpg"
    )

    output_path = os.path.join(
        app.config[
            "UPLOAD_FOLDER"
        ],
        filename,
    )

    cv2.imwrite(
        output_path,
        image,
    )

    return filename


# ============================================================
# COMPLETE AI PIPELINE
# ============================================================

def run_damage_detection(
    image_path,
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        "INSURE AI IMAGE ANALYSIS"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    (
        vehicle_type,
        vehicle_confidence,
        vehicle_detections,
    ) = detect_vehicle_type(
        image_path
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    damage_detections = detect_damage(
        image_path,
        vehicle_detections,
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    visible_parts = detect_car_parts(
        image_path
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    affected_parts = (
        find_damage_related_parts(
            damage_detections,
            visible_parts,
        )
    )

    # --------------------------------------------------------
    # VEHICLE FALLBACK
    # --------------------------------------------------------

    if vehicle_type == "unknown":

        car_keywords = {
            "bumper",
            "door",
            "wheel",
            "window",
            "windshield",
            "headlight",
            "tail-light",
            "taillight",
            "hood",
            "fender",
            "mirror",
            "grille",
            "roof",
            "trunk",
            "quarter-panel",
            "rocker-panel",
        }

        supporting = []

        for part in visible_parts:

            name = str(
                part.get(
                    "class_name",
                    "",
                )
            ).lower()

            confidence = float(
                part.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                confidence >= 30.0
                and
                any(
                    keyword in name
                    for keyword
                    in car_keywords
                )
            ):

                supporting.append(
                    part
                )

        if len(supporting) >= 2:

            vehicle_type = "car"

            vehicle_confidence = max(
                float(
                    p.get(
                        "confidence",
                        0.0,
                    )
                )
                for p in supporting
            )

            print(
                "[VEHICLE] Fallback:",
                vehicle_type,
                vehicle_confidence,
                "%",
            )

    # --------------------------------------------------------
    # DAMAGE CONFIDENCE
    # --------------------------------------------------------

    if damage_detections:

        damage_confidence = max(
            float(
                d.get(
                    "confidence",
                    0.0,
                )
            )
            for d in damage_detections
        )

    else:

        damage_confidence = 0.0

    damage_detected = bool(
        damage_detections
    )

    # --------------------------------------------------------
    # ANNOTATION
    # --------------------------------------------------------

    try:

        annotated_filename = (
            create_annotated_image(
                image_path,
                damage_detections,
                affected_parts,
                vehicle_type,
            )
        )

    except Exception as exc:

        print(
            "[ANNOTATION] ERROR:",
            exc,
        )

        traceback.print_exc()

        annotated_filename = None

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 75
    )

    print(
        "Vehicle:",
        vehicle_type,
    )

    print(
        "Vehicle confidence:",
        round(
            vehicle_confidence,
            2,
        ),
        "%",
    )

    print(
        "Damage detected:",
        damage_detected,
    )

    print(
        "Damage confidence:",
        round(
            damage_confidence,
            2,
        ),
        "%",
    )

    print(
        "Damage regions:",
        len(
            damage_detections
        ),
    )

    print(
        "Visible parts:",
        len(
            visible_parts
        ),
    )

    print(
        "DAMAGED PARTS:",
        [
            p["class_name"]
            for p
            in affected_parts
        ],
    )

    print(
        "=" * 75
    )

    return (
        damage_detections,
        affected_parts,
        annotated_filename,
        vehicle_type,
        vehicle_confidence,
        damage_confidence,
        visible_parts,
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"],
)
def register():

    if request.method == "POST":

        first_name = request.form.get(
            "first_name",
            "",
        ).strip()

        last_name = request.form.get(
            "last_name",
            "",
        ).strip()

        email = request.form.get(
            "email",
            "",
        ).strip().lower()

        password = request.form.get(
            "password",
            "",
        )

        confirm_password = request.form.get(
            "confirm_password",
            "",
        )

        if not first_name or not last_name:

            return render_template(
                "register.html",
                error=(
                    "Please enter your full name."
                ),
            )

        if not email:

            return render_template(
                "register.html",
                error=(
                    "Please enter your email."
                ),
            )

        if len(password) < 6:

            return render_template(
                "register.html",
                error=(
                    "Password must contain "
                    "at least 6 characters."
                ),
            )

        if password != confirm_password:

            return render_template(
                "register.html",
                error=(
                    "Passwords do not match."
                ),
            )

        if get_user_by_email(
            email
        ):

            return render_template(
                "register.html",
                error=(
                    "An account with this "
                    "email already exists."
                ),
            )

        user = {

            "first_name":
                first_name,

            "last_name":
                last_name,

            "email":
                email,

            "password":
                generate_password_hash(
                    password
                ),

            "phone": "",
            "dob": "",
            "address": "",
            "city": "",
            "state": "",
            "pincode": "",
            "vehicle": "",
            "insurance": "",
            "policy": "",

            "is_admin":
                False,

            "created_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
        }

        save_user(
            user
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"],
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            "",
        ).strip().lower()

        password = request.form.get(
            "password",
            "",
        )

        user = get_user_by_email(
            email
        )

        if (
            not user
            or
            not check_password_hash(
                user.get(
                    "password",
                    "",
                ),
                password,
            )
        ):

            return render_template(
                "login.html",
                error=(
                    "Invalid email or password."
                ),
            )

        session.clear()

        session["user_email"] = (
            user["email"]
        )

        session["is_admin"] = bool(
            user.get(
                "is_admin",
                False,
            )
        )

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if not login_required():

        return redirect(
            url_for("login")
        )

    user = current_user()

    if not user:

        session.clear()

        return redirect(
            url_for("login")
        )

    claims = get_claims_by_user(
        user["email"]
    )

    total_claims = len(
        claims
    )

    pending_review = sum(
        1
        for claim in claims
        if claim.get(
            "status",
            "Under Review",
        )
        == "Under Review"
    )

    approved_claims = sum(
        1
        for claim in claims
        if claim.get(
            "status"
        )
        == "Approved"
    )

    total_estimated_damage = sum(
        float(
            claim.get(
                "total_cost",
                0,
            )
            or 0
        )
        for claim in claims
    )

    return render_template(
        "dashboard.html",
        user=user,
        total_claims=total_claims,
        pending_review=pending_review,
        approved_claims=approved_claims,
        total_estimated_damage=(
            total_estimated_damage
        ),
        recent_claims=claims[:5],
    )


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
def profile():

    if not login_required():

        return redirect(
            url_for("login")
        )

    return render_template(
        "profile.html",
        user=current_user(),
    )


# ============================================================
# EDIT PROFILE
# ============================================================

@app.route(
    "/profile/edit",
    methods=["GET", "POST"],
)
def edit_profile():

    if not login_required():

        return redirect(
            url_for("login")
        )

    user = current_user()

    if request.method == "POST":

        fields = [
            "first_name",
            "last_name",
            "phone",
            "dob",
            "address",
            "city",
            "state",
            "pincode",
            "vehicle",
            "insurance",
        ]

        updates = {
            field:
                request.form.get(
                    field,
                    "",
                ).strip()
            for field in fields
        }

        update_user(
            user["email"],
            updates,
        )

        return redirect(
            url_for("profile")
        )

    return render_template(
        "profile_edit.html",
        user=user,
    )


# ============================================================
# ASSESSMENT
# ============================================================

@app.route("/assessment", methods=["GET", "POST"])
def assessment():

    if not login_required():
        return redirect(url_for("login"))

    user = current_user()

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    if request.method == "GET":
        return render_template(
            "assessment.html",
            user=user
        )

    # --------------------------------------------------------
    # GET UPLOADED FILE
    # --------------------------------------------------------

    file = request.files.get("image")

    if file is None:
        return render_template(
            "assessment.html",
            user=user,
            error="Please select a vehicle image."
        )

    if not file.filename:
        return render_template(
            "assessment.html",
            user=user,
            error="Please select a vehicle image."
        )

    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if not allowed_file(file.filename):
        return render_template(
            "assessment.html",
            user=user,
            error="Allowed formats: JPG, JPEG, PNG and WEBP."
        )

    # --------------------------------------------------------
    # SAFE FILENAME
    # --------------------------------------------------------

    original_name = secure_filename(file.filename)

    if not original_name:
        return render_template(
            "assessment.html",
            user=user,
            error="Invalid image filename."
        )

    # --------------------------------------------------------
    # GUARANTEE UPLOAD DIRECTORY EXISTS
    # --------------------------------------------------------

    upload_folder = app.config["UPLOAD_FOLDER"]

    os.makedirs(
        upload_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CREATE UNIQUE FILE NAME
    # --------------------------------------------------------

    unique_name = (
        f"{uuid.uuid4().hex}_{original_name}"
    )

    image_path = os.path.join(
        upload_folder,
        unique_name
    )

    image_path = os.path.abspath(image_path)

    print("\n" + "=" * 75)
    print("IMAGE UPLOAD")
    print("=" * 75)
    print("Upload folder:", upload_folder)
    print("Original name:", original_name)
    print("Saved name:", unique_name)
    print("Full path:", image_path)

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    try:

        # Make absolutely sure the directory still exists.
        os.makedirs(
            os.path.dirname(image_path),
            exist_ok=True
        )

        file.save(image_path)

    except Exception as exc:

        print("[UPLOAD ERROR]", exc)

        import traceback
        traceback.print_exc()

        return render_template(
            "assessment.html",
            user=user,
            error="Unable to save the uploaded image."
        )

    # --------------------------------------------------------
    # VERIFY IMAGE WAS ACTUALLY SAVED
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        print(
            "[UPLOAD ERROR] File does not exist after save:",
            image_path
        )

        return render_template(
            "assessment.html",
            user=user,
            error="Uploaded image could not be saved."
        )

    print(
        "[UPLOAD] SUCCESS:",
        image_path
    )

    # --------------------------------------------------------
    # RUN AI ANALYSIS
    # --------------------------------------------------------

    try:

        (
            damage_detections,
            affected_parts,
            annotated_filename,
            vehicle_type,
            vehicle_confidence,
            damage_confidence,
            visible_parts,
        ) = run_damage_detection(
            image_path
        )

        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------

        severity = calculate_severity(
            damage_detections
        )

        # ----------------------------------------------------
        # REPAIR COST
        # ----------------------------------------------------

        total_cost, breakdown = estimate_repair_cost(
            damage_detections,
            vehicle_type,
            affected_parts
        )

        # ----------------------------------------------------
        # DAMAGE STATUS
        # ----------------------------------------------------

        damage_detected = bool(
            damage_detections
        )

        # ----------------------------------------------------
        # BUILD ASSESSMENT DATA
        # ----------------------------------------------------

        assessment_data = {

            "user_email": user["email"],

            # Original uploaded image
            "image_url": url_for(
                "static",
                filename=f"uploads/{unique_name}"
            ),

            # Annotated image
            "annotated_image_url": (
                url_for(
                    "static",
                    filename=f"uploads/{annotated_filename}"
                )
                if annotated_filename
                else None
            ),

            # Damage status
            "damage_detected": damage_detected,

            # Confidence
            "confidence": round(
                damage_confidence,
                2
            ),

            "damage_confidence": round(
                damage_confidence,
                2
            ),

            # Vehicle
            "vehicle_type": vehicle_type,

            "vehicle_confidence": round(
                vehicle_confidence,
                2
            ),

            # Severity
            "severity": severity,

            # Cost
            "repair_cost": total_cost,

            "cost_breakdown": breakdown,

            # ------------------------------------------------
            # DAMAGE MODEL RESULTS
            # ------------------------------------------------

            "detections": damage_detections,

            "damage_detections": damage_detections,

            # ------------------------------------------------
            # ONLY PARTS ASSOCIATED WITH DAMAGE
            # ------------------------------------------------

            "part_detections": affected_parts,

            "damaged_parts": affected_parts,

            # ------------------------------------------------
            # ALL VISIBLE PARTS
            # ------------------------------------------------

            "visible_part_detections": visible_parts,

            "visible_parts": visible_parts,

            # ------------------------------------------------
            # TIMESTAMP
            # ------------------------------------------------

            "created_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        # ----------------------------------------------------
        # SAVE ASSESSMENT
        # ----------------------------------------------------

        assessment_id = save_assessment(
            assessment_data
        )

        print(
            "[ASSESSMENT] SAVED:",
            assessment_id
        )

        # ----------------------------------------------------
        # RESULT PAGE
        # ----------------------------------------------------

        return redirect(
            url_for(
                "assessment_result",
                assessment_id=assessment_id
            )
        )

    except Exception as exc:

        print("\n" + "=" * 75)
        print("[ASSESSMENT ERROR]")
        print(exc)
        print("=" * 75)

        import traceback
        traceback.print_exc()

        return render_template(
            "assessment.html",
            user=user,
            error=(
                "AI analysis failed. "
                "Please check the server console."
            )
        )
# ============================================================
# ASSESSMENT RESULT
# ============================================================

@app.route(
    "/assessment/result/<assessment_id>"
)
def assessment_result(
    assessment_id,
):

    if not login_required():

        return redirect(
            url_for("login")
        )

    data = get_assessment(
        assessment_id
    )

    if not data:

        return (
            "Assessment not found",
            404,
        )

    if (
        data.get("user_email")
        !=
        session.get("user_email")
    ):

        return (
            "Unauthorized",
            403,
        )

    return render_template(
        "assessment_result.html",
        assessment=data,
        assessment_id=assessment_id,
        user=current_user(),
    )


# ============================================================
# CLAIM PAGE
# ============================================================

@app.route(
    "/claim/<assessment_id>"
)
def claim(
    assessment_id,
):

    if not login_required():

        return redirect(
            url_for("login")
        )

    data = get_assessment(
        assessment_id
    )

    if not data:

        return (
            "Assessment not found",
            404,
        )

    if (
        data.get("user_email")
        !=
        session.get("user_email")
    ):

        return (
            "Unauthorized",
            403,
        )

    return render_template(
        "claim.html",
        assessment=data,
        assessment_id=assessment_id,
        user=current_user(),
    )


# ============================================================
# CLAIM SUBMISSION
# ============================================================

@app.route(
    "/claim/submit/<assessment_id>",
    methods=["POST"],
)
def submit_claim(
    assessment_id,
):

    if not login_required():

        return redirect(
            url_for("login")
        )

    assessment_data = get_assessment(
        assessment_id
    )

    if not assessment_data:

        return (
            "Assessment not found",
            404,
        )

    if (
        assessment_data.get(
            "user_email"
        )
        !=
        session.get(
            "user_email"
        )
    ):

        return (
            "Unauthorized",
            403,
        )

    user = current_user()

    # --------------------------------------------------------
    # CREATE CLAIM
    # --------------------------------------------------------

    claim_data = {

        "user_email":
            user["email"],

        "assessment_id":
            assessment_id,

        "first_name":
            user.get(
                "first_name",
                "",
            ),

        "last_name":
            user.get(
                "last_name",
                "",
            ),

        "vehicle":
            user.get(
                "vehicle",
                "",
            ),

        "vehicle_type":
            assessment_data.get(
                "vehicle_type",
                "car",
            ),

        "damage_detected":
            assessment_data.get(
                "damage_detected",
                False,
            ),

        "damage_confidence":
            assessment_data.get(
                "damage_confidence",
                0,
            ),

        "confidence":
            assessment_data.get(
                "damage_confidence",
                0,
            ),

        "severity":
            assessment_data.get(
                "severity",
                "Minor",
            ),

        "total_cost":
            assessment_data.get(
                "repair_cost",
                0,
            ),

        "repair_cost":
            assessment_data.get(
                "repair_cost",
                0,
            ),

        "cost_breakdown":
            assessment_data.get(
                "cost_breakdown",
                {},
            ),

        "damaged_parts":
            assessment_data.get(
                "damaged_parts",
                [],
            ),

        "part_detections":
            assessment_data.get(
                "part_detections",
                [],
            ),

        "damage_detections":
            assessment_data.get(
                "damage_detections",
                [],
            ),

        "image_url":
            assessment_data.get(
                "image_url"
            ),

        "annotated_image_url":
            assessment_data.get(
                "annotated_image_url"
            ),

        "status":
            "Under Review",

        "remarks":
            "",

        "created_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
    }

    claim_id = save_claim(
        claim_data
    )

    print(
        "[CLAIM] SAVED:",
        claim_id,
    )

    return redirect(
        url_for(
            "claim_success",
            claim_id=claim_id,
        )
    )


# ============================================================
# CLAIM SUCCESS
# ============================================================

@app.route(
    "/claim/success/<claim_id>"
)
def claim_success(
    claim_id,
):

    if not login_required():

        return redirect(
            url_for("login")
        )

    claim_data = get_claim(
        claim_id
    )

    if not claim_data:

        return (
            "Claim not found",
            404,
        )

    if (
        claim_data.get(
            "user_email"
        )
        !=
        session.get(
            "user_email"
        )
    ):

        return (
            "Unauthorized",
            403,
        )

    user = current_user()

    claim_data["first_name"] = (
        user.get(
            "first_name",
            "",
        )
    )

    claim_data["last_name"] = (
        user.get(
            "last_name",
            "",
        )
    )

    claim_data["vehicle"] = (
        user.get(
            "vehicle",
            "",
        )
    )

    claim_data["confidence"] = (
        claim_data.get(
            "damage_confidence",
            0,
        )
    )

    claim_data["repair_cost"] = (
        claim_data.get(
            "total_cost",
            0,
        )
    )

    claim_data["claim_id"] = (
        claim_data.get(
            "claim_id",
            claim_id,
        )
    )

    return render_template(
        "claim_success.html",
        claim=claim_data,
        claim_id=claim_id,
        user=user,
    )


# ============================================================
# CLAIMS
# ============================================================

@app.route("/claims")
def claims():

    if not login_required():

        return redirect(
            url_for("login")
        )

    user = current_user()

    return render_template(
        "claims.html",
        user=user,
        claims=get_claims_by_user(
            user["email"]
        ),
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    if not login_required():

        return redirect(
            url_for("login")
        )

    user = current_user()

    return render_template(
        "history.html",
        user=user,
        claims=get_claims_by_user(
            user["email"]
        ),
    )


# ============================================================
# REPORT
# ============================================================

@app.route(
    "/report/<claim_id>"
)
def insurance_report(
    claim_id,
):

    if not login_required():

        return redirect(
            url_for("login")
        )

    claim_data = get_claim(
        claim_id
    )

    if not claim_data:

        return (
            "Claim not found",
            404,
        )

    if (
        claim_data.get(
            "user_email"
        )
        !=
        session.get(
            "user_email"
        )
    ):

        return (
            "Unauthorized",
            403,
        )

    return render_template(
        "report.html",
        claim=claim_data,
        user=current_user(),
    )


# ============================================================
# SETTINGS
# ============================================================

@app.route("/settings")
def settings():

    if not login_required():

        return redirect(
            url_for("login")
        )

    return render_template(
        "settings.html",
        user=current_user(),
    )


# ============================================================
# ADMIN
# ============================================================

@app.route("/admin")
def admin():

    if not login_required():

        return redirect(
            url_for("login")
        )

    if not session.get(
        "is_admin"
    ):

        return (
            "Unauthorized",
            403,
        )

    return render_template(
        "admin.html",
        user=current_user(),
        claims=get_all_claims(),
    )


# ============================================================
# ADMIN CLAIM UPDATE
# ============================================================

@app.route(
    "/admin/claim/<claim_id>",
    methods=["POST"],
)
def admin_update_claim(
    claim_id,
):

    if not login_required():

        return redirect(
            url_for("login")
        )

    if not session.get(
        "is_admin"
    ):

        return (
            "Unauthorized",
            403,
        )

    allowed_statuses = {
        "Under Review",
        "Approved",
        "Rejected",
        "Additional Evidence Required",
    }

    status = request.form.get(
        "status",
        "Under Review",
    )

    remarks = request.form.get(
        "remarks",
        "",
    ).strip()

    if status not in allowed_statuses:

        status = "Under Review"

    update_claim_status(
        claim_id,
        status,
        remarks,
    )

    return redirect(
        url_for("admin")
    )


# ============================================================
# HEALTH API
# ============================================================

@app.route(
    "/api/health"
)
def health():

    return jsonify({

        "status":
            "online",

        "application":
            "INSURE AI",

        "damage_model_loaded":
            damage_model is not None,

        "part_model_loaded":
            part_model is not None,

        "vehicle_model_loaded":
            vehicle_model is not None,

        "damage_model_path":
            DAMAGE_MODEL_PATH,

        "part_model_path":
            PART_MODEL_PATH,

        "vehicle_model_path":
            VEHICLE_MODEL_PATH,

        "damage_classes":
            damage_model.names,

        "part_classes":
            part_model.names,

        "vehicle_classes":
            vehicle_model.names,

        "pipeline": {

            "damage_model":
                "damage detection only",

            "part_model":
                "vehicle part detection only",

            "vehicle_model":
                "vehicle type detection only",

            "association":
                "damage-to-part spatial association",
        },
    })


# ============================================================
# MODELS API
# ============================================================

@app.route(
    "/api/models"
)
def models():

    return jsonify({

        "damage": {

            "path":
                DAMAGE_MODEL_PATH,

            "exists":
                os.path.exists(
                    DAMAGE_MODEL_PATH
                ),

            "loaded":
                damage_model is not None,

            "classes":
                damage_model.names,
        },

        "parts": {

            "path":
                PART_MODEL_PATH,

            "exists":
                os.path.exists(
                    PART_MODEL_PATH
                ),

            "loaded":
                part_model is not None,

            "classes":
                part_model.names,
        },

        "vehicle": {

            "path":
                VEHICLE_MODEL_PATH,

            "exists":
                os.path.exists(
                    VEHICLE_MODEL_PATH
                ),

            "loaded":
                vehicle_model is not None,

            "classes":
                vehicle_model.names,
        },
    })


# ============================================================
# DEBUG AI API
# ============================================================

@app.route(
    "/api/test-models",
    methods=["POST"],
)
def test_models():

    if not login_required():

        return jsonify({
            "error":
                "Login required"
        }), 401

    file = request.files.get(
        "image"
    )

    if (
        not file
        or
        not file.filename
    ):

        return jsonify({
            "error":
                "No image provided"
        }), 400

    if not allowed_file(
        file.filename
    ):

        return jsonify({
            "error":
                "Unsupported image format"
        }), 400

    filename = (
        f"debug_"
        f"{uuid.uuid4().hex}_"
        f"{secure_filename(file.filename)}"
    )

    path = os.path.join(
        UPLOAD_FOLDER,
        filename,
    )

    file.save(path)

    try:

        (
            damage_detections,
            affected_parts,
            annotated_filename,
            vehicle_type,
            vehicle_confidence,
            damage_confidence,
            visible_parts,
        ) = run_damage_detection(
            path
        )

        return jsonify({

            "success":
                True,

            "vehicle": {

                "type":
                    vehicle_type,

                "confidence":
                    vehicle_confidence,
            },

            "damage": {

                "detected":
                    bool(
                        damage_detections
                    ),

                "confidence":
                    damage_confidence,

                "regions":
                    damage_detections,
            },

            "damaged_parts":
                affected_parts,

            "visible_parts":
                visible_parts,

            "annotated_image":
                (
                    url_for(
                        "static",
                        filename=(
                            f"uploads/"
                            f"{annotated_filename}"
                        ),
                    )
                    if annotated_filename
                    else None
                ),
        })

    except Exception as exc:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(exc),
        }), 500


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def too_large(_error):

    return render_template(
        "assessment.html",
        user=current_user(),
        error=(
            "Image is too large. "
            "Maximum size is 20 MB."
        ),
    ), 413


# ============================================================
# GENERAL ERROR HANDLER
# ============================================================

@app.errorhandler(500)
def internal_error(error):

    print(
        "\n[FLASK 500 ERROR]",
        error,
    )

    traceback.print_exc()

    if request.path.startswith(
        "/api/"
    ):

        return jsonify({
            "success":
                False,
            "error":
                "Internal server error",
        }), 500

    return (
        "Internal server error. "
        "Check the server console.",
        500,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 75
    )

    print(
        "INSURE AI SERVER"
    )

    print(
        "=" * 75
    )

    print(
        "Local:   http://127.0.0.1:5000"
    )

    print(
        "Network: http://0.0.0.0:5000"
    )

    print(
        "=" * 75
    )

    # IMPORTANT:
    # use_reloader=False prevents Flask from
    # loading the YOLO models twice.

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True,
    )