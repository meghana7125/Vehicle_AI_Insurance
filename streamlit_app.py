
import os
import json
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="INSURE AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

# Make directories safely
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

USERS_FILE = DATA_DIR / "users.json"
CLAIMS_FILE = DATA_DIR / "claims.json"

DAMAGE_MODEL_PATH = MODEL_DIR / "best.pt"
PART_MODEL_PATH = MODEL_DIR / "car_parts_best.pt"
VEHICLE_MODEL_PATH = BASE_DIR / "yolo26n.pt"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background: #f6f8fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a, #1d4ed8);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
    }

    .metric {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .metric h2 {
        margin: 0;
        color: #1d4ed8;
    }

    .metric p {
        color: #64748b;
        margin-bottom: 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path, default):
    try:
        if not path.exists():
            return default

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    except Exception as e:
        print(f"JSON load error: {e}")
        return default


def save_json(path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = path.with_suffix(".tmp")

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        os.replace(temp_path, path)

    except Exception as e:
        print(f"JSON save error: {e}")


def load_users():
    data = load_json(USERS_FILE, [])
    return data if isinstance(data, list) else []


def save_users(users):
    save_json(USERS_FILE, users)


def load_claims():
    data = load_json(CLAIMS_FILE, [])
    return data if isinstance(data, list) else []


def save_claims(claims):
    save_json(CLAIMS_FILE, claims)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    damage = None
    parts = None
    vehicle = None

    errors = []

    # DAMAGE MODEL
    if DAMAGE_MODEL_PATH.exists():

        try:
            print(f"Loading damage model: {DAMAGE_MODEL_PATH}")
            damage = YOLO(str(DAMAGE_MODEL_PATH))
            print(f"Damage classes: {damage.names}")

        except Exception as e:
            errors.append(f"Damage model error: {e}")
            print(f"Damage model error: {e}")

    else:
        errors.append(
            f"Damage model missing: {DAMAGE_MODEL_PATH}"
        )

    # PART MODEL
    if PART_MODEL_PATH.exists():

        try:
            print(f"Loading part model: {PART_MODEL_PATH}")
            parts = YOLO(str(PART_MODEL_PATH))
            print(f"Part classes: {parts.names}")

        except Exception as e:
            errors.append(f"Part model error: {e}")
            print(f"Part model error: {e}")

    else:
        errors.append(
            f"Part model missing: {PART_MODEL_PATH}"
        )

    # VEHICLE MODEL
    if VEHICLE_MODEL_PATH.exists():

        try:
            print(f"Loading vehicle model: {VEHICLE_MODEL_PATH}")
            vehicle = YOLO(str(VEHICLE_MODEL_PATH))
            print(f"Vehicle classes: {vehicle.names}")

        except Exception as e:
            errors.append(f"Vehicle model error: {e}")
            print(f"Vehicle model error: {e}")

    else:
        errors.append(
            f"Vehicle model missing: {VEHICLE_MODEL_PATH}"
        )

    return damage, parts, vehicle, errors


damage_model, part_model, vehicle_model, MODEL_ERRORS = load_models()


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "selected_claim" not in st.session_state:
    st.session_state.selected_claim = None


# ============================================================
# LOGOUT
# ============================================================

def logout():

    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.page = "Login"
    st.session_state.selected_claim = None

    st.rerun()


# ============================================================
# LOGIN
# ============================================================

def login_page():

    st.markdown(
        """
        <div class="hero">
            <h1>🚗 INSURE AI</h1>
            <p>
                AI-Powered Vehicle Damage Assessment
                & Insurance Claims
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.markdown("## 🔐 Sign In")

        with st.form("login_form"):

            email = st.text_input(
                "Email",
                placeholder="Enter your email",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )

            submitted = st.form_submit_button(
                "Sign In",
                use_container_width=True,
            )

        if submitted:

            email = email.strip().lower()

            users = load_users()

            found = None

            for user in users:

                if (
                    user.get("email", "").lower() == email
                    and user.get("password", "") == password
                ):
                    found = user
                    break

            if found:

                st.session_state.logged_in = True
                st.session_state.user = found
                st.session_state.page = "Dashboard"

                st.success("Login successful!")

                st.rerun()

            else:

                st.error(
                    "Invalid email or password."
                )

        st.markdown("---")

        if st.button(
            "Create INSURE AI Account",
            use_container_width=True,
        ):

            st.session_state.page = "Register"

            st.rerun()


# ============================================================
# REGISTER
# ============================================================

def register_page():

    st.markdown(
        """
        <div class="hero">
            <h1>🚗 Create INSURE AI Account</h1>
            <p>
                Register to start assessing vehicle damage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        first_name = st.text_input(
            "First Name",
            placeholder="Meghana",
        )

        last_name = st.text_input(
            "Last Name",
            placeholder="Badugu",
        )

        email = st.text_input(
            "Email",
            placeholder="meghanabadugu14@gmail.com",
        )

        phone = st.text_input(
            "Phone",
            placeholder="Enter phone number",
        )

    with col2:

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Minimum 6 characters",
        )

        confirm = st.text_input(
            "Confirm Password",
            type="password",
        )

        vehicle = st.text_input(
            "Vehicle",
            value="Car",
        )

        insurance = st.text_input(
            "Insurance Provider",
            placeholder="Insurance company name",
        )

    submitted = st.button(
        "Create Account",
        type="primary",
        use_container_width=True,
    )

    if submitted:

        first_name = first_name.strip()
        last_name = last_name.strip()
        email = email.strip().lower()
        phone = phone.strip()
        vehicle = vehicle.strip()
        insurance = insurance.strip()

        if not first_name:

            st.error("Please enter your first name.")
            return

        if not last_name:

            st.error("Please enter your last name.")
            return

        if not email:

            st.error("Please enter your email.")
            return

        if "@" not in email or "." not in email:

            st.error("Please enter a valid email.")
            return

        if not phone:

            st.error("Please enter your phone number.")
            return

        if not password:

            st.error("Please enter a password.")
            return

        if len(password) < 6:

            st.error(
                "Password must contain at least 6 characters."
            )
            return

        if password != confirm:

            st.error("Passwords do not match.")
            return

        users = load_users()

        if any(
            u.get("email", "").lower() == email
            for u in users
        ):

            st.error(
                "An account with this email already exists."
            )
            return

        user = {
            "id": uuid.uuid4().hex[:10],
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
            "password": password,
            "vehicle": vehicle or "Car",
            "insurance": insurance or "Not specified",
            "created_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        users.append(user)

        save_users(users)

        st.success(
            "Account created successfully!"
        )

        st.session_state.page = "Login"

        st.rerun()

    if st.button("← Back to Login"):

        st.session_state.page = "Login"

        st.rerun()


# ============================================================
# IOU
# ============================================================

def iou(a, b):

    if not a or not b:
        return 0.0

    if len(a) != 4 or len(b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# DAMAGE COVERAGE
# ============================================================

def intersection_over_damage(
    damage_box,
    part_box,
):

    x1 = max(
        damage_box[0],
        part_box[0],
    )

    y1 = max(
        damage_box[1],
        part_box[1],
    )

    x2 = min(
        damage_box[2],
        part_box[2],
    )

    y2 = min(
        damage_box[3],
        part_box[3],
    )

    intersection = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    damage_area = max(
        1,
        (damage_box[2] - damage_box[0])
        * (damage_box[3] - damage_box[1]),
    )

    return intersection / damage_area


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(image_path):

    image = Image.open(image_path).convert("RGB")

    damage_detections = []
    parts = []

    vehicle_type = "Car"
    vehicle_confidence = 0.0

    # ========================================================
    # VEHICLE
    # ========================================================

    if vehicle_model:

        try:

            results = vehicle_model.predict(
                source=image,
                conf=0.15,
                imgsz=960,
                verbose=False,
            )

            if (
                results
                and results[0].boxes is not None
            ):

                for box in results[0].boxes:

                    cls = int(
                        box.cls[0].item()
                    )

                    conf = float(
                        box.conf[0].item()
                    )

                    name = vehicle_model.names.get(
                        cls,
                        str(cls),
                    )

                    if name.lower() in {
                        "car",
                        "truck",
                        "bus",
                        "motorcycle",
                    }:

                        if conf > vehicle_confidence:

                            vehicle_type = (
                                name.title()
                            )

                            vehicle_confidence = conf

        except Exception as e:

            print(
                f"Vehicle detection error: {e}"
            )

    # ========================================================
    # DAMAGE
    # ========================================================

    if damage_model:

        try:

            results = damage_model.predict(
                source=image,
                conf=0.08,
                iou=0.45,
                imgsz=1280,
                max_det=40,
                verbose=False,
            )

            if (
                results
                and results[0].boxes is not None
            ):

                for box in results[0].boxes:

                    cls = int(
                        box.cls[0].item()
                    )

                    conf = float(
                        box.conf[0].item()
                    )

                    name = damage_model.names.get(
                        cls,
                        str(cls),
                    )

                    if name.lower() != "damage":
                        continue

                    coords = [
                        float(x)
                        for x in box.xyxy[0].tolist()
                    ]

                    damage_detections.append(
                        {
                            "bbox": coords,
                            "confidence": conf * 100,
                        }
                    )

        except Exception as e:

            print(
                f"Damage detection error: {e}"
            )

    # ========================================================
    # CAR PARTS
    # ========================================================

    if part_model:

        try:

            results = part_model.predict(
                source=image,
                conf=0.08,
                iou=0.45,
                imgsz=1536,
                max_det=150,
                verbose=False,
            )

            if (
                results
                and results[0].boxes is not None
            ):

                strongest = {}

                for box in results[0].boxes:

                    cls = int(
                        box.cls[0].item()
                    )

                    conf = float(
                        box.conf[0].item()
                    )

                    name = part_model.names.get(
                        cls,
                        str(cls),
                    )

                    coords = [
                        float(x)
                        for x in box.xyxy[0].tolist()
                    ]

                    key = name.lower()

                    if (
                        key not in strongest
                        or conf
                        > strongest[key]["confidence"]
                        / 100
                    ):

                        strongest[key] = {
                            "part": name,
                            "bbox": coords,
                            "confidence": conf * 100,
                        }

                parts = list(
                    strongest.values()
                )

        except Exception as e:

            print(
                f"Part detection error: {e}"
            )

    # ========================================================
    # MATCH DAMAGE TO PART
    # ========================================================

    damaged_parts = []

    for damage in damage_detections:

        best = None
        best_score = 0.0

        for part in parts:

            coverage = intersection_over_damage(
                damage["bbox"],
                part["bbox"],
            )

            overlap = iou(
                damage["bbox"],
                part["bbox"],
            )

            score = (
                coverage * 0.75
                + overlap * 0.25
            )

            if score > best_score:

                best_score = score
                best = part

        if best:

            coverage = intersection_over_damage(
                damage["bbox"],
                best["bbox"],
            )

            # Lower threshold intentionally.
            if (
                best_score >= 0.05
                or coverage >= 0.08
            ):

                damaged_parts.append(
                    {
                        **best,
                        "association_score":
                            best_score * 100,
                        "damage_confidence":
                            damage["confidence"],
                    }
                )

    # ========================================================
    # REMOVE DUPLICATE PARTS
    # ========================================================

    unique_parts = {}

    for part in damaged_parts:

        key = part["part"].lower()

        if (
            key not in unique_parts
            or part["association_score"]
            > unique_parts[key]["association_score"]
        ):

            unique_parts[key] = part

    damaged_parts = list(
        unique_parts.values()
    )

    # ========================================================
    # SEVERITY
    # ========================================================

    if not damage_detections:

        severity = "No Damage"
        estimated_cost = 0

    else:

        avg_confidence = sum(
            d["confidence"]
            for d in damage_detections
        ) / len(damage_detections)

        count = len(damage_detections)

        if count >= 4 or avg_confidence >= 80:

            severity = "Severe"

            estimated_cost = (
                50000 + count * 8000
            )

        elif count >= 2 or avg_confidence >= 55:

            severity = "Moderate"

            estimated_cost = (
                25000 + count * 5000
            )

        else:

            severity = "Minor"

            estimated_cost = (
                10000 + count * 3000
            )

    return {
        "vehicle_type": vehicle_type,
        "vehicle_confidence":
            vehicle_confidence * 100,
        "damage_detections":
            damage_detections,
        "visible_parts":
            parts,
        "damaged_parts":
            damaged_parts,
        "severity": severity,
        "estimated_cost":
            estimated_cost,
    }


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():

    user = st.session_state.user

    claims = [
        c
        for c in load_claims()
        if c.get("email") == user["email"]
    ]

    st.markdown(
        f"""
        <div class="hero">
            <h1>Welcome, {user['first_name']} 👋</h1>
            <p>
                INSURE AI Vehicle Insurance Dashboard
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric">
                <h2>{len(claims)}</h2>
                <p>Total Claims</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        pending = sum(
            c.get("status") == "Under Review"
            for c in claims
        )

        st.markdown(
            f"""
            <div class="metric">
                <h2>{pending}</h2>
                <p>Under Review</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        approved = sum(
            c.get("status") == "Approved"
            for c in claims
        )

        st.markdown(
            f"""
            <div class="metric">
                <h2>{approved}</h2>
                <p>Approved</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        total = sum(
            float(
                c.get(
                    "estimated_cost",
                    0,
                )
            )
            for c in claims
        )

        st.markdown(
            f"""
            <div class="metric">
                <h2>₹{total:,.0f}</h2>
                <p>Estimated Damage</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        "## 🚗 Start Vehicle Assessment"
    )

    if st.button(
        "Start New Assessment",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.page = "Assessment"

        st.rerun()

    st.markdown("## Recent Claims")

    if not claims:

        st.info(
            "No claims yet. Upload a vehicle image to begin."
        )

    else:

        for claim in claims[-5:][::-1]:

            with st.container(border=True):

                a, b, c = st.columns(3)

                a.write(
                    f"**Claim:** "
                    f"{claim.get('claim_id', 'N/A')}"
                )

                b.write(
                    f"**Severity:** "
                    f"{claim.get('severity', 'N/A')}"
                )

                c.write(
                    f"**Status:** "
                    f"{claim.get('status', 'N/A')}"
                )


# ============================================================
# ASSESSMENT
# ============================================================

def assessment_page():

    st.title(
        "🔍 Vehicle Damage Assessment"
    )

    st.write(
        "Upload a clear image of the damaged vehicle."
    )

    uploaded = st.file_uploader(
        "Vehicle Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if not uploaded:
        return

    try:

        image = Image.open(
            uploaded
        ).convert("RGB")

    except Exception:

        st.error(
            "The uploaded file is not a valid image."
        )

        return

    st.image(
        image,
        caption="Uploaded Vehicle Image",
        use_container_width=True,
    )

    if not st.button(
        "🤖 Analyze Vehicle",
        type="primary",
        use_container_width=True,
    ):
        return

    # ========================================================
    # IMPORTANT FIX
    #
    # NEVER USE uploaded.name HERE.
    #
    # Long uploaded filenames cause:
    # FileNotFoundError / Windows path length errors.
    # ========================================================

    safe_filename = (
        f"vehicle_{uuid.uuid4().hex}.jpg"
    )

    image_path = (
        UPLOAD_DIR / safe_filename
    )

    try:

        # Ensure directory exists
        UPLOAD_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save as JPEG using short filename
        image.save(
            image_path,
            format="JPEG",
            quality=95,
        )

    except Exception as e:

        st.error(
            f"Could not save uploaded image: {e}"
        )

        return

    # ========================================================
    # ANALYZE
    # ========================================================

    with st.spinner(
        "🤖 AI is analyzing the vehicle..."
    ):

        try:

            result = analyze_image(
                str(image_path)
            )

        except Exception as e:

            st.error(
                f"AI analysis failed: {e}"
            )

            return

    # ========================================================
    # CREATE CLAIM
    # ========================================================

    claim_id = (
        "CLM-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:6].upper()
    )

    user = st.session_state.user

    claim = {

        "claim_id": claim_id,

        "email":
            user["email"],

        "customer":
            (
                user["first_name"]
                + " "
                + user["last_name"]
            ),

        # Store relative path instead of
        # absolute Windows path.
        "image":
            str(
                image_path.relative_to(
                    BASE_DIR
                )
            ).replace("\\", "/"),

        "vehicle_type":
            result["vehicle_type"],

        "vehicle_confidence":
            result["vehicle_confidence"],

        "damage_detections":
            result["damage_detections"],

        "visible_parts":
            result["visible_parts"],

        "damaged_parts":
            result["damaged_parts"],

        "severity":
            result["severity"],

        "estimated_cost":
            result["estimated_cost"],

        "status":
            "Under Review",

        "created_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
    }

    claims = load_claims()

    claims.append(claim)

    save_claims(claims)

    st.session_state.selected_claim = claim_id

    st.session_state.page = "Report"

    st.success(
        "Assessment completed successfully!"
    )

    st.rerun()


# ============================================================
# CLAIMS
# ============================================================

def claims_page():

    st.title("📋 My Claims")

    user = st.session_state.user

    claims = [
        c
        for c in load_claims()
        if c.get("email") == user["email"]
    ]

    if not claims:

        st.info(
            "You haven't submitted any claims yet."
        )

        return

    for claim in claims[::-1]:

        with st.container(border=True):

            c1, c2, c3, c4 = st.columns(4)

            c1.write(
                f"**{claim.get('claim_id', 'N/A')}**"
            )

            c2.write(
                f"**{claim.get('severity', 'N/A')}**"
            )

            c3.write(
                f"₹{float(claim.get('estimated_cost', 0)):,.0f}"
            )

            c4.write(
                f"**{claim.get('status', 'N/A')}**"
            )

            if st.button(
                "View Report",
                key=f"view_{claim.get('claim_id')}",
            ):

                st.session_state.selected_claim = (
                    claim.get("claim_id")
                )

                st.session_state.page = "Report"

                st.rerun()


# ============================================================
# REPORT
# ============================================================

def report_page():

    claim_id = st.session_state.get(
        "selected_claim"
    )

    claims = load_claims()

    claim = next(
        (
            c
            for c in claims
            if c.get("claim_id") == claim_id
        ),
        None,
    )

    if not claim:

        st.error(
            "Claim not found."
        )

        return

    st.title(
        "📄 Insurance Assessment Report"
    )

    st.markdown(
        f"""
        <div class="hero">
            <h1>INSURE AI</h1>
            <p>Vehicle Damage Assessment Report</p>
            <p>
                <b>Assessment ID:</b>
                {claim.get('claim_id', 'N/A')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Vehicle",
        claim.get(
            "vehicle_type",
            "Car",
        ),
    )

    c2.metric(
        "Severity",
        claim.get(
            "severity",
            "Unknown",
        ),
    )

    c3.metric(
        "Estimated Cost",
        f"₹{float(claim.get('estimated_cost', 0)):,.0f}",
    )

    c4.metric(
        "Status",
        claim.get(
            "status",
            "Under Review",
        ),
    )

    # ========================================================
    # IMAGE
    # ========================================================

    st.markdown(
        "## 🖼️ Inspection Image"
    )

    image_value = claim.get(
        "image",
        "",
    )

    if image_value:

        image_path = Path(image_value)

        # Handle new relative paths
        if not image_path.is_absolute():
            image_path = (
                BASE_DIR / image_path
            )

        # Handle old Windows absolute paths
        # if they still exist.
        if image_path.exists():

            try:

                st.image(
                    str(image_path),
                    use_container_width=True,
                )

            except Exception as e:

                st.warning(
                    f"Image could not be displayed: {e}"
                )

        else:

            st.warning(
                "The original inspection image "
                "is no longer available."
            )

    else:

        st.info(
            "No inspection image was stored."
        )

    # ========================================================
    # DAMAGE
    # ========================================================

    st.markdown(
        "## 🚨 Damage Detection"
    )

    damages = claim.get(
        "damage_detections",
        [],
    )

    if damages:

        st.success(
            f"{len(damages)} damage region(s) detected."
        )

        for i, damage in enumerate(
            damages,
            start=1,
        ):

            st.write(
                f"**Damage {i}:** "
                f"{float(damage.get('confidence', 0)):.1f}% "
                f"confidence"
            )

    else:

        st.info(
            "No damage region detected."
        )

    # ========================================================
    # DAMAGED PARTS
    # ========================================================

    st.markdown(
        "## 🔧 Damaged Parts"
    )

    damaged = claim.get(
        "damaged_parts",
        [],
    )

    if damaged:

        for part in damaged:

            st.write(
                f"🔴 **{part.get('part', 'Unknown')}** — "
                f"{float(part.get('association_score', 0)):.1f}% "
                f"association"
            )

    else:

        st.info(
            "No specific damaged part could be associated."
        )

    # ========================================================
    # VISIBLE PARTS
    # ========================================================

    st.markdown(
        "## 🚗 Visible Vehicle Parts"
    )

    visible = claim.get(
        "visible_parts",
        [],
    )

    if visible:

        cols = st.columns(3)

        for i, part in enumerate(visible):

            with cols[i % 3]:

                st.info(
                    f"{part.get('part', 'Unknown')} "
                    f"({float(part.get('confidence', 0)):.1f}%)"
                )

    else:

        st.info(
            "No vehicle parts detected."
        )

    # ========================================================
    # COST
    # ========================================================

    st.markdown(
        "## 💰 Repair Estimate"
    )

    st.success(
        f"Estimated Repair Cost: "
        f"₹{float(claim.get('estimated_cost', 0)):,.0f}"
    )

    st.caption(
        "Final settlement is subject to insurance verification."
    )

    st.write("")

    if st.button(
        "← Back to Dashboard"
    ):

        st.session_state.page = "Dashboard"

        st.rerun()


# ============================================================
# PROFILE
# ============================================================

def profile_page():

    user = st.session_state.user

    st.title("👤 My Profile")

    st.write(
        f"**Name:** "
        f"{user.get('first_name', '')} "
        f"{user.get('last_name', '')}"
    )

    st.write(
        f"**Email:** "
        f"{user.get('email', '')}"
    )

    st.write(
        f"**Phone:** "
        f"{user.get('phone', '')}"
    )

    st.write(
        f"**Vehicle:** "
        f"{user.get('vehicle', 'Car')}"
    )

    st.write(
        f"**Insurance:** "
        f"{user.get('insurance', 'Not specified')}"
    )


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.markdown(
            "# 🚗 INSURE AI"
        )

        st.caption(
            "AI Vehicle Insurance"
        )

        if st.session_state.user:

            st.caption(
                "Welcome, "
                + st.session_state.user.get(
                    "first_name",
                    "User",
                )
            )

        st.divider()

        pages = [
            "Dashboard",
            "Assessment",
            "Claims",
            "Report",
            "Profile",
        ]

        current = st.session_state.page

        if current not in pages:
            current = "Dashboard"

        selected = st.radio(
            "Navigation",
            pages,
            index=pages.index(current),
        )

        st.session_state.page = selected

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True,
        ):

            logout()


# ============================================================
# MAIN
# ============================================================

if not st.session_state.logged_in:

    if st.session_state.page == "Register":

        register_page()

    else:

        login_page()

else:

    show_sidebar()

    if st.session_state.page == "Dashboard":

        dashboard()

    elif st.session_state.page == "Assessment":

        assessment_page()

    elif st.session_state.page == "Claims":

        claims_page()

    elif st.session_state.page == "Report":

        report_page()

    elif st.session_state.page == "Profile":

        profile_page()
