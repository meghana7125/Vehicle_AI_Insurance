import os
import uuid
import math
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


# ============================================================
# MODEL PATHS
# ============================================================

DAMAGE_MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
PART_MODEL_PATH = os.path.join(BASE_DIR, "model", "car_parts_best.pt")
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "yolo26n.pt")


# ============================================================
# MODEL LOADING
# ============================================================

print("=" * 75)
print("INSURE AI - MODEL INITIALIZATION")
print("=" * 75)

if not os.path.exists(DAMAGE_MODEL_PATH):
    raise FileNotFoundError(f"Damage model not found: {DAMAGE_MODEL_PATH}")
if not os.path.exists(PART_MODEL_PATH):
    raise FileNotFoundError(f"Car-part model not found: {PART_MODEL_PATH}")
if not os.path.exists(VEHICLE_MODEL_PATH):
    raise FileNotFoundError(f"Vehicle model not found: {VEHICLE_MODEL_PATH}")

print("[1] Loading DAMAGE model:", DAMAGE_MODEL_PATH)
damage_model = YOLO(DAMAGE_MODEL_PATH)
print("    classes:", damage_model.names)

print("[2] Loading CAR PART model:", PART_MODEL_PATH)
part_model = YOLO(PART_MODEL_PATH)
print("    classes:", part_model.names)

print("[3] Loading VEHICLE model:", VEHICLE_MODEL_PATH)
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
print("    classes:", vehicle_model.names)

print("=" * 75)


# ============================================================
# GENERAL HELPERS
# ============================================================

def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def current_user():
    email = session.get("user_email")
    return get_user_by_email(email) if email else None


def login_required():
    return session.get("user_email") is not None


def model_class_name(model, class_id):
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def box_area(box):
    if not box or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box):
    if not box or len(box) != 4:
        return 0.0, 0.0
    x1, y1, x2, y2 = map(float, box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def calculate_intersection(a, b):
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def calculate_iou(a, b):
    intersection = calculate_intersection(a, b)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def overlap_over_damage(damage_box, part_box):
    """Fraction of the DAMAGE box covered by the PART box."""
    damage_area = box_area(damage_box)
    return calculate_intersection(damage_box, part_box) / damage_area if damage_area else 0.0


def overlap_over_part(damage_box, part_box):
    """Fraction of the PART box covered by the DAMAGE box."""
    part_area = box_area(part_box)
    return calculate_intersection(damage_box, part_box) / part_area if part_area else 0.0


def center_inside_box(point, box):
    if not box or len(box) != 4:
        return False
    x, y = point
    x1, y1, x2, y2 = map(float, box)
    return x1 <= x <= x2 and y1 <= y <= y2


def center_proximity_score(damage_box, part_box):
    dcx, dcy = box_center(damage_box)
    pcx, pcy = box_center(part_box)

    pw = max(1.0, float(part_box[2]) - float(part_box[0]))
    ph = max(1.0, float(part_box[3]) - float(part_box[1]))
    diagonal = max(1.0, math.sqrt(pw * pw + ph * ph))

    distance = math.sqrt((dcx - pcx) ** 2 + (dcy - pcy) ** 2)
    return max(0.0, 1.0 - distance / diagonal)


def clamp_box(box, width, height):
    if not box or len(box) != 4:
        return None

    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def unique_by_iou(detections, iou_threshold=0.50, limit=20):
    result = []

    for d in sorted(
        detections,
        key=lambda x: float(x.get("confidence", 0.0)),
        reverse=True,
    ):
        if any(calculate_iou(d["bbox"], old["bbox"]) >= iou_threshold for old in result):
            continue
        result.append(d)
        if len(result) >= limit:
            break

    return result


# ============================================================
# VEHICLE DETECTION
# ============================================================

def detect_vehicle_type(image_path):
    print("\n[STEP 1] VEHICLE DETECTION")

    try:
        results = vehicle_model.predict(
            source=image_path,
            conf=0.15,
            iou=0.45,
            imgsz=960,
            max_det=20,
            verbose=False,
        )
    except Exception as exc:
        print("[VEHICLE] ERROR:", exc)
        return "unknown", 0.0, []

    if not results or results[0].boxes is None:
        return "unknown", 0.0, []

    supported = {"car", "truck", "bus", "motorcycle"}
    detections = []

    for box in results[0].boxes:
        try:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            name = model_class_name(vehicle_model, class_id).strip().lower()
            bbox = [round(float(v), 2) for v in box.xyxy[0].tolist()]
        except Exception:
            continue

        if name not in supported:
            continue

        detections.append({
            "class_name": name,
            "confidence": round(confidence * 100.0, 2),
            "bbox": bbox,
        })

    if not detections:
        return "unknown", 0.0, []

    cars = [d for d in detections if d["class_name"] == "car"]
    best = max(cars or detections, key=lambda d: d["confidence"])

    print("[VEHICLE]", best["class_name"], best["confidence"], "%")
    return best["class_name"], best["confidence"], detections


# ============================================================
# DAMAGE DETECTION
# ============================================================

def detect_damage(image_path, vehicle_detections=None):
    """
    The 2-class model is the ONLY model allowed to declare damage.

    Class 0 = damage
    Class 1 = whole

    The vehicle detector is used only to restrict damage detections
    to the vehicle region when a vehicle box is available.
    """

    import cv2

    print("\n[STEP 2] DAMAGE DETECTION")

    image = cv2.imread(image_path)
    if image is None:
        print("[DAMAGE] Could not read image.")
        return []

    height, width = image.shape[:2]

    # Find primary vehicle.
    valid_vehicle_boxes = []
    for d in vehicle_detections or []:
        bbox = d.get("bbox", [])
        name = str(d.get("class_name", "")).lower()
        conf = float(d.get("confidence", 0.0))

        if name in {"car", "truck", "bus", "motorcycle"} and len(bbox) == 4:
            valid_vehicle_boxes.append((name, conf, bbox))

    cars = [x for x in valid_vehicle_boxes if x[0] == "car"]
    primary = max(cars or valid_vehicle_boxes, key=lambda x: x[1], default=None)

    if primary:
        _, vehicle_conf, vehicle_box = primary
        vx1, vy1, vx2, vy2 = vehicle_box
        vw = vx2 - vx1
        vh = vy2 - vy1

        pad_x = max(8.0, vw * 0.05)
        pad_y = max(8.0, vh * 0.05)

        rx1 = max(0, int(vx1 - pad_x))
        ry1 = max(0, int(vy1 - pad_y))
        rx2 = min(width, int(vx2 + pad_x))
        ry2 = min(height, int(vy2 + pad_y))

        roi = image[ry1:ry2, rx1:rx2]
        vehicle_box = [float(v) for v in vehicle_box]

        print("[DAMAGE] Vehicle ROI:", vehicle_box)
        print("[DAMAGE] Vehicle confidence:", round(vehicle_conf, 2), "%")
    else:
        rx1, ry1 = 0, 0
        roi = image
        vehicle_box = [0.0, 0.0, float(width), float(height)]
        print("[DAMAGE] No vehicle box; using full image.")

    if roi is None or roi.size == 0:
        return []

    detections = []

    def accepts_vehicle_overlap(box):
        if primary is None:
            return True

        damage_area = box_area(box)
        if damage_area <= 0:
            return False

        overlap = calculate_intersection(box, vehicle_box) / damage_area
        cx, cy = box_center(box)

        return (
            overlap >= 0.20
            and center_inside_box((cx, cy), vehicle_box)
        ) or overlap >= 0.50

    def collect(result, source, offset_x=0, offset_y=0):
        if result is None or result.boxes is None:
            return

        for box in result.boxes:
            try:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                name = model_class_name(damage_model, class_id).strip().lower()
                raw = [float(v) for v in box.xyxy[0].tolist()]
            except Exception:
                continue

            if name != "damage":
                continue

            # Deliberately permissive because the user's current model
            # can produce low-confidence true damage detections.
            if confidence < 0.10:
                continue

            mapped = [
                raw[0] + offset_x + rx1,
                raw[1] + offset_y + ry1,
                raw[2] + offset_x + rx1,
                raw[3] + offset_y + ry1,
            ]

            mapped = clamp_box(mapped, width, height)
            if mapped is None:
                continue

            if box_area(mapped) < max(100.0, width * height * 0.0001):
                continue

            if not accepts_vehicle_overlap(mapped):
                continue

            detections.append({
                "class_id": class_id,
                "class_name": "damage",
                "confidence": round(confidence * 100.0, 2),
                "confidence_percent": round(confidence * 100.0, 2),
                "bbox": [round(v, 2) for v in mapped],
                "source": source,
            })

    # Full vehicle ROI passes.
    for conf, imgsz in ((0.10, 1280), (0.08, 1600)):
        try:
            results = damage_model.predict(
                source=roi,
                conf=conf,
                iou=0.45,
                imgsz=imgsz,
                max_det=40,
                verbose=False,
            )
            if results:
                collect(results[0], "vehicle_roi")
        except Exception as exc:
            print("[DAMAGE] ROI inference error:", exc)

    # If no result, scan overlapping tiles to make small damage easier to see.
    if not detections:
        rh, rw = roi.shape[:2]
        tile_w = min(rw, max(384, int(rw * 0.65)))
        tile_h = min(rh, max(384, int(rh * 0.65)))
        step_x = max(160, int(tile_w * 0.45))
        step_y = max(160, int(tile_h * 0.45))

        xs = list(range(0, max(1, rw - tile_w + 1), step_x))
        ys = list(range(0, max(1, rh - tile_h + 1), step_y))

        last_x = max(0, rw - tile_w)
        last_y = max(0, rh - tile_h)

        if not xs or xs[-1] != last_x:
            xs.append(last_x)
        if not ys or ys[-1] != last_y:
            ys.append(last_y)

        seen = set()

        for ty in ys:
            for tx in xs:
                key = (tx, ty)
                if key in seen:
                    continue
                seen.add(key)

                tile = roi[ty:ty + tile_h, tx:tx + tile_w]
                if tile.size == 0:
                    continue

                try:
                    results = damage_model.predict(
                        source=tile,
                        conf=0.10,
                        iou=0.45,
                        imgsz=960,
                        max_det=20,
                        verbose=False,
                    )
                    if results:
                        collect(results[0], "vehicle_tile", tx, ty)
                except Exception as exc:
                    print("[DAMAGE] Tile error:", exc)

    final = unique_by_iou(detections, 0.50, 15)

    print("[DAMAGE] FINAL REGIONS:", len(final))
    for d in final:
        print("   ", d["confidence"], "%", d["bbox"], d["source"])

    return final


# ============================================================
# VEHICLE PART DETECTION
# ============================================================

def detect_car_parts(image_path):
    """
    The 21-class model identifies visible vehicle parts.
    It DOES NOT declare damage.
    """

    print("\n[STEP 3] VEHICLE PART DETECTION")

    try:
        results = part_model.predict(
            source=image_path,
            conf=0.08,
            iou=0.45,
            imgsz=1536,
            max_det=150,
            verbose=False,
        )
    except Exception as exc:
        print("[PARTS] ERROR:", exc)
        return []

    if not results or results[0].boxes is None:
        print("[PARTS] No parts detected.")
        return []

    detections = []

    for box, conf, cls in zip(
        results[0].boxes.xyxy.cpu().tolist(),
        results[0].boxes.conf.cpu().tolist(),
        results[0].boxes.cls.cpu().tolist(),
    ):
        try:
            class_id = int(cls)
            confidence = float(conf) * 100.0
            part_name = model_class_name(part_model, class_id)
            bbox = [float(v) for v in box]
        except Exception:
            continue

        if confidence < 8.0:
            continue
        if box_area(bbox) <= 0:
            continue

        detections.append({
            "part": str(part_name),
            "class_name": str(part_name),
            "class_id": class_id,
            "confidence": round(confidence, 2),
            "confidence_percent": round(confidence, 2),
            "bbox": [round(v, 2) for v in bbox],
        })

    # One strongest visible detection per part class.
    strongest = {}
    for d in detections:
        key = d["class_name"].strip().lower()
        if key not in strongest or d["confidence"] > strongest[key]["confidence"]:
            strongest[key] = d

    detections = sorted(
        strongest.values(),
        key=lambda d: d["confidence"],
        reverse=True,
    )

    print("[PARTS] VISIBLE PARTS:")
    for d in detections:
        print("   ", d["class_name"], d["confidence"], "%")

    return detections


# ============================================================
# DAMAGE -> PART ASSOCIATION
# ============================================================

def find_damage_related_parts(damage_detections, part_detections):
    """
    This is the ONLY function that decides which visible part is
    associated with each damage region.

    Damage model:
        declares whether/where damage exists.

    Parts model:
        identifies which vehicle parts are visible.

    This function:
        spatially associates a damage box with a part box.

    It NEVER marks every visible part as damaged.
    """

    print("\n[STEP 4] DAMAGE -> PART ASSOCIATION")

    if not damage_detections:
        print("[ASSOCIATION] No damage regions.")
        return []

    if not part_detections:
        print("[ASSOCIATION] No parts.")
        return []

    affected = []

    for damage_index, damage in enumerate(damage_detections, start=1):
        damage_box = damage.get("bbox", [])
        damage_conf = float(damage.get("confidence", 0.0))

        if len(damage_box) != 4:
            continue

        candidates = []

        for part in part_detections:
            part_box = part.get("bbox", [])
            part_conf = float(part.get("confidence", 0.0))

            if len(part_box) != 4 or part_conf < 10.0:
                continue

            damage_coverage = overlap_over_damage(damage_box, part_box)
            part_coverage = overlap_over_part(damage_box, part_box)
            iou = calculate_iou(damage_box, part_box)
            center_score = center_proximity_score(damage_box, part_box)

            damage_center = box_center(damage_box)
            center_inside = center_inside_box(damage_center, part_box)

            # A damage region must have a real spatial relationship
            # with the candidate part.
            if (
                damage_coverage < 0.08
                and iou < 0.01
                and not center_inside
                and center_score < 0.35
            ):
                continue

            # Main signal = how much of the detected DAMAGE is inside
            # this part. This is much better than ordinary IoU when a
            # tiny damage box lies inside a large door/hood/bumper box.
            score = (
                0.50 * damage_coverage
                + 0.20 * part_coverage
                + 0.15 * iou
                + 0.10 * center_score
                + 0.05 * min(1.0, part_conf / 100.0)
            )

            # Extra preference when damage center lies directly inside part.
            if center_inside:
                score += 0.08

            candidates.append({
                "part": part.get("class_name", part.get("part", "Unknown")),
                "class_name": part.get("class_name", part.get("part", "Unknown")),
                "class_id": part.get("class_id", -1),
                "confidence": round(part_conf, 2),
                "confidence_percent": round(part_conf, 2),
                "bbox": [round(float(v), 2) for v in part_box],
                "damage_confidence": round(damage_conf, 2),
                "damage_bbox": [round(float(v), 2) for v in damage_box],
                "association_score": round(score * 100.0, 2),
                "damage_coverage": round(damage_coverage * 100.0, 2),
                "part_coverage": round(part_coverage * 100.0, 2),
                "iou": round(iou * 100.0, 2),
                "center_score": round(center_score * 100.0, 2),
            })

        if not candidates:
            print(f"[ASSOCIATION] Damage #{damage_index}: no matching part")
            continue

        candidates.sort(key=lambda x: x["association_score"], reverse=True)
        best = candidates[0]

        # Require meaningful evidence before calling a part damaged.
        strong_match = (
            best["damage_coverage"] >= 18.0
            or best["iou"] >= 5.0
            or (
                best["damage_coverage"] >= 8.0
                and best["center_score"] >= 45.0
            )
        )

        if not strong_match:
            print(
                f"[ASSOCIATION] Damage #{damage_index}: "
                f"weak match -> {best['part']} "
                f"({best['association_score']}%)"
            )
            continue

        affected.append(best)

        print(
            f"[ASSOCIATION] Damage #{damage_index} -> {best['part']} | "
            f"association={best['association_score']}% | "
            f"damage={best['damage_confidence']}% | "
            f"damage-coverage={best['damage_coverage']}% | "
            f"IoU={best['iou']}%"
        )

    # Keep one strongest damage association per named part.
    unique = {}
    for item in affected:
        key = str(item["class_name"]).strip().lower()
        if key not in unique or (
            item["association_score"] > unique[key]["association_score"]
        ):
            unique[key] = item

    result = sorted(
        unique.values(),
        key=lambda x: x["association_score"],
        reverse=True,
    )

    print("[ASSOCIATION] ACTUAL DAMAGED PARTS:")
    for item in result:
        print(
            "   ",
            item["class_name"],
            "|",
            item["association_score"],
            "%",
        )

    return result


# Backward-compatible alias for any old code using this name.
associate_damage_with_parts = find_damage_related_parts


# ============================================================
# ANNOTATED IMAGE
# ============================================================

def create_annotated_image(
    image_path,
    damage_detections,
    affected_parts,
    vehicle_type,
):
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        return None

    # RED = actual damage boxes.
    for detection in damage_detections:
        bbox = detection.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        conf = float(detection.get("confidence", 0.0))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            4,
        )

        cv2.putText(
            image,
            f"DAMAGE {conf:.1f}%",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # BLUE = only parts associated with actual damage.
    for part in affected_parts:
        bbox = part.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        name = str(part.get("class_name", "Damaged Part"))
        score = float(part.get("association_score", 0.0))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 120, 0),
            3,
        )

        label = f"DAMAGED: {name} ({score:.0f}%)"
        text_y = min(image.shape[0] - 10, y2 + 25)

        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 120, 0),
            2,
            cv2.LINE_AA,
        )

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

    filename = f"annotated_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    cv2.imwrite(output_path, image)
    return filename


# ============================================================
# COMPLETE AI PIPELINE
# ============================================================

def run_damage_detection(image_path):
    print("\n" + "=" * 75)
    print("INSURE AI IMAGE ANALYSIS")
    print("=" * 75)

    vehicle_type, vehicle_confidence, vehicle_detections = detect_vehicle_type(
        image_path
    )

    damage_detections = detect_damage(
        image_path,
        vehicle_detections,
    )

    visible_parts = detect_car_parts(image_path)

    affected_parts = find_damage_related_parts(
        damage_detections,
        visible_parts,
    )

    # If the generic vehicle model missed the car, the 21-class
    # car-parts model can provide supporting evidence.
    if vehicle_type == "unknown":
        car_keywords = {
            "bumper", "door", "wheel", "window", "windshield",
            "headlight", "tail-light", "hood", "fender", "mirror",
            "grille", "roof", "trunk", "quarter-panel", "rocker-panel",
        }

        supporting = 0
        for part in visible_parts:
            name = str(part.get("class_name", "")).lower()
            conf = float(part.get("confidence", 0.0))
            if conf >= 30.0 and any(k in name for k in car_keywords):
                supporting += 1

        if supporting >= 2:
            vehicle_type = "car"
            vehicle_confidence = max(
                float(p.get("confidence", 0.0))
                for p in visible_parts
                if float(p.get("confidence", 0.0)) >= 30.0
                and any(
                    k in str(p.get("class_name", "")).lower()
                    for k in car_keywords
                )
            )

    damage_confidence = (
        max(
            float(d.get("confidence", 0.0))
            for d in damage_detections
        )
        if damage_detections
        else 0.0
    )

    damage_detected = bool(damage_detections)

    try:
        annotated_filename = create_annotated_image(
            image_path,
            damage_detections,
            affected_parts,
            vehicle_type,
        )
    except Exception as exc:
        print("[ANNOTATION] ERROR:", exc)
        annotated_filename = None

    print("\n" + "=" * 75)
    print("FINAL RESULT")
    print("=" * 75)
    print("Vehicle:", vehicle_type)
    print("Vehicle confidence:", round(vehicle_confidence, 2), "%")
    print("Damage detected:", damage_detected)
    print("Damage confidence:", round(damage_confidence, 2), "%")
    print("Damage regions:", len(damage_detections))
    print("Visible parts:", len(visible_parts))
    print(
        "Damaged parts:",
        [p["class_name"] for p in affected_parts],
    )
    print("=" * 75)

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
# HOME / ABOUT
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


# ============================================================
# AUTH
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not first_name or not last_name:
            return render_template("register.html", error="Please enter your full name.")

        if not email:
            return render_template("register.html", error="Please enter your email.")

        if len(password) < 6:
            return render_template(
                "register.html",
                error="Password must contain at least 6 characters.",
            )

        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match.")

        if get_user_by_email(email):
            return render_template(
                "register.html",
                error="An account with this email already exists.",
            )

        user = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password": generate_password_hash(password),
            "phone": "",
            "dob": "",
            "address": "",
            "city": "",
            "state": "",
            "pincode": "",
            "vehicle": "",
            "insurance": "",
            "policy": "",
            "is_admin": False,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        save_user(user)
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = get_user_by_email(email)

        if not user or not check_password_hash(user.get("password", ""), password):
            return render_template(
                "login.html",
                error="Invalid email or password.",
            )

        session.clear()
        session["user_email"] = user["email"]
        session["is_admin"] = bool(user.get("is_admin", False))

        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    user = current_user()
    claims = get_claims_by_user(user["email"])

    total_claims = len(claims)
    pending_review = sum(
        1 for c in claims
        if c.get("status", "Under Review") == "Under Review"
    )
    approved_claims = sum(
        1 for c in claims
        if c.get("status") == "Approved"
    )

    total_estimated_damage = sum(
        float(c.get("total_cost", 0) or 0)
        for c in claims
    )

    return render_template(
        "dashboard.html",
        user=user,
        total_claims=total_claims,
        pending_review=pending_review,
        approved_claims=approved_claims,
        total_estimated_damage=total_estimated_damage,
        recent_claims=claims[:5],
    )


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
def profile():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("profile.html", user=current_user())


@app.route("/profile/edit", methods=["GET", "POST"])
def edit_profile():
    if not login_required():
        return redirect(url_for("login"))

    user = current_user()

    if request.method == "POST":
        fields = [
            "first_name", "last_name", "phone", "dob", "address",
            "city", "state", "pincode", "vehicle", "insurance",
        ]

        updates = {
            field: request.form.get(field, "").strip()
            for field in fields
        }

        update_user(user["email"], updates)
        return redirect(url_for("profile"))

    return render_template("profile_edit.html", user=user)


# ============================================================
# ASSESSMENT
# ============================================================

@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    if not login_required():
        return redirect(url_for("login"))

    user = current_user()

    if request.method == "GET":
        return render_template("assessment.html", user=user)

    file = request.files.get("image")

    if not file or not file.filename:
        return render_template(
            "assessment.html",
            user=user,
            error="Please select a vehicle image.",
        )

    if not allowed_file(file.filename):
        return render_template(
            "assessment.html",
            user=user,
            error="Allowed formats: JPG, JPEG, PNG and WEBP.",
        )

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(image_path)

    try:
        (
            damage_detections,
            affected_parts,
            annotated_filename,
            vehicle_type,
            vehicle_confidence,
            damage_confidence,
            visible_parts,
        ) = run_damage_detection(image_path)

        severity = calculate_severity(damage_detections)

        total_cost, breakdown = estimate_repair_cost(
            damage_detections,
            vehicle_type,
            affected_parts,
        )

        damage_detected = bool(damage_detections)

        assessment_data = {
            "user_email": user["email"],

            "image_url": url_for(
                "static",
                filename=f"uploads/{unique_name}",
            ),

            "annotated_image_url": (
                url_for(
                    "static",
                    filename=f"uploads/{annotated_filename}",
                )
                if annotated_filename
                else None
            ),

            "damage_detected": damage_detected,

            "confidence": round(damage_confidence, 2),
            "damage_confidence": round(damage_confidence, 2),

            "vehicle_type": vehicle_type,
            "vehicle_confidence": round(vehicle_confidence, 2),

            "severity": severity,
            "repair_cost": total_cost,
            "cost_breakdown": breakdown,

            # Actual damage boxes.
            "detections": damage_detections,
            "damage_detections": damage_detections,

            # ONLY parts associated with damage.
            "part_detections": affected_parts,
            "damaged_parts": affected_parts,

            # All parts visible in the image.
            "visible_part_detections": visible_parts,
            "visible_parts": visible_parts,

            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        assessment_id = save_assessment(assessment_data)

        print("[ASSESSMENT] SAVED:", assessment_id)

        return redirect(
            url_for(
                "assessment_result",
                assessment_id=assessment_id,
            )
        )

    except Exception as exc:
        print("\n[ASSESSMENT] ERROR:", exc)
        import traceback
        traceback.print_exc()

        return render_template(
            "assessment.html",
            user=user,
            error="AI analysis failed. Please check the server console.",
        )


# ============================================================
# ASSESSMENT RESULT
# ============================================================

@app.route("/assessment/result/<assessment_id>")
def assessment_result(assessment_id):
    if not login_required():
        return redirect(url_for("login"))

    data = get_assessment(assessment_id)

    if not data:
        return "Assessment not found", 404

    if data.get("user_email") != session.get("user_email"):
        return "Unauthorized", 403

    return render_template(
        "assessment_result.html",
        assessment=data,
        assessment_id=assessment_id,
        user=current_user(),
    )
# ============================================================
# CLAIM SUCCESS
# ============================================================

@app.route("/claim/success/<claim_id>")
def claim_success(claim_id):

    if not login_required():
        return redirect(url_for("login"))

    claim_data = get_claim(claim_id)

    if not claim_data:
        return "Claim not found", 404

    # Security: user can only see their own claim
    if claim_data.get("user_email") != session.get("user_email"):
        return "Unauthorized", 403

    user = current_user()

    # --------------------------------------------------------
    # Add user information expected by claim_success.html
    # --------------------------------------------------------

    claim_data["first_name"] = user.get(
        "first_name",
        ""
    )

    claim_data["last_name"] = user.get(
        "last_name",
        ""
    )

    claim_data["vehicle"] = user.get(
        "vehicle",
        ""
    )

    # Your assessment stores damage_confidence,
    # while the template expects confidence.
    claim_data["confidence"] = claim_data.get(
        "damage_confidence",
        0
    )

    # Your assessment stores repair cost as total_cost
    # inside the claim.
    claim_data["repair_cost"] = claim_data.get(
        "total_cost",
        0
    )

    # Make sure claim_id exists for the template.
    claim_data["claim_id"] = claim_data.get(
        "claim_id",
        claim_id
    )

    return render_template(
        "claim_success.html",
        claim=claim_data,
        claim_id=claim_id,
        user=user,
    )


# ... your imports
# ... Flask app creation
# ... helper functions
# ... login routes
# ... dashboard routes
# ... assessment routes


# ============================================================
# CLAIM
# ============================================================

@app.route("/claim/<assessment_id>")
def claim(assessment_id):
    if not login_required():
        return redirect(url_for("login"))

    data = get_assessment(assessment_id)

    if not data:
        return "Assessment not found", 404

    if data.get("user_email") != session.get("user_email"):
        return "Unauthorized", 403

    return render_template(
        "claim.html",
        assessment=data,
        assessment_id=assessment_id,
        user=current_user()
    )





# ============================================================
# CLAIMS / HISTORY
# ============================================================

@app.route("/claims")
def claims():
    if not login_required():
        return redirect(url_for("login"))

    user = current_user()

    return render_template(
        "claims.html",
        user=user,
        claims=get_claims_by_user(user["email"]),
    )


@app.route("/history")
def history():
    if not login_required():
        return redirect(url_for("login"))

    user = current_user()

    return render_template(
        "history.html",
        user=user,
        claims=get_claims_by_user(user["email"]),
    )


# ============================================================
# REPORT
# ============================================================

@app.route("/report/<claim_id>")
def insurance_report(claim_id):
    if not login_required():
        return redirect(url_for("login"))

    claim_data = get_claim(claim_id)

    if not claim_data:
        return "Claim not found", 404

    if claim_data.get("user_email") != session.get("user_email"):
        return "Unauthorized", 403

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
        return redirect(url_for("login"))

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
        return redirect(url_for("login"))

    if not session.get("is_admin"):
        return "Unauthorized", 403

    return render_template(
        "admin.html",
        user=current_user(),
        claims=get_all_claims(),
    )


@app.route("/admin/claim/<claim_id>", methods=["POST"])
def admin_update_claim(claim_id):
    if not login_required():
        return redirect(url_for("login"))

    if not session.get("is_admin"):
        return "Unauthorized", 403

    allowed_statuses = {
        "Under Review",
        "Approved",
        "Rejected",
        "Additional Evidence Required",
    }

    status = request.form.get("status", "Under Review")
    remarks = request.form.get("remarks", "").strip()

    if status not in allowed_statuses:
        status = "Under Review"

    update_claim_status(claim_id, status, remarks)

    return redirect(url_for("admin"))


# ============================================================
# HEALTH / DEBUG API
# ============================================================

@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "application": "INSURE AI",

        "damage_model_loaded": damage_model is not None,
        "part_model_loaded": part_model is not None,
        "vehicle_model_loaded": vehicle_model is not None,

        "damage_model_path": DAMAGE_MODEL_PATH,
        "part_model_path": PART_MODEL_PATH,
        "vehicle_model_path": VEHICLE_MODEL_PATH,

        "damage_classes": damage_model.names,
        "part_classes": part_model.names,
        "vehicle_classes": vehicle_model.names,

        "pipeline": {
            "damage_model": "damage detection only",
            "part_model": "vehicle part detection only",
            "vehicle_model": "vehicle type detection only",
            "association": "damage-to-part spatial association",
        },
    })


@app.route("/api/models")
def models():
    return jsonify({
        "damage": {
            "path": DAMAGE_MODEL_PATH,
            "exists": os.path.exists(DAMAGE_MODEL_PATH),
            "loaded": damage_model is not None,
            "classes": damage_model.names,
        },
        "parts": {
            "path": PART_MODEL_PATH,
            "exists": os.path.exists(PART_MODEL_PATH),
            "loaded": part_model is not None,
            "classes": part_model.names,
        },
        "vehicle": {
            "path": VEHICLE_MODEL_PATH,
            "exists": os.path.exists(VEHICLE_MODEL_PATH),
            "loaded": vehicle_model is not None,
            "classes": vehicle_model.names,
        },
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "assessment.html",
        user=current_user(),
        error="Image is too large. Maximum size is 20 MB.",
    ), 413


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("INSURE AI SERVER")
    print("=" * 75)
    print("Local:   http://127.0.0.1:5000")
    print("Network: http://0.0.0.0:5000")
    print("=" * 75)

    # use_reloader=False prevents Flask debug mode from loading
    # all three YOLO models twice.
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True,
    )