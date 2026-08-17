import os
import json
from datetime import datetime


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data")

USERS_FILE = os.path.join(DATA_DIR, "users.json")
ASSESSMENTS_FILE = os.path.join(DATA_DIR, "assessments.json")
CLAIMS_FILE = os.path.join(DATA_DIR, "claims.json")

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# FILE HELPERS
# ============================================================

def ensure_file(filepath):
    """Create JSON file if it does not exist."""

    if not os.path.exists(filepath):
        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                [],
                file,
                indent=4,
                ensure_ascii=False
            )


ensure_file(USERS_FILE)
ensure_file(ASSESSMENTS_FILE)
ensure_file(CLAIMS_FILE)


# ============================================================
# GENERIC JSON FUNCTIONS
# ============================================================

def load_json(filepath):
    """Safely load a JSON list."""

    ensure_file(filepath)

    try:
        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

            return []

    except (
        json.JSONDecodeError,
        FileNotFoundError,
        PermissionError
    ):
        return []


def save_json(filepath, data):
    """Safely save JSON data."""

    os.makedirs(
        os.path.dirname(filepath),
        exist_ok=True
    )

    temporary_file = filepath + ".tmp"

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Replace old file with new file
    os.replace(
        temporary_file,
        filepath
    )


# ============================================================
# USERS
# ============================================================

def save_user(user_data):
    """
    Save a new user.

    Prevents duplicate email accounts.
    """

    users = load_json(USERS_FILE)

    email = str(
        user_data.get("email", "")
    ).strip().lower()

    if not email:
        return None

    # Normalize email
    user_data["email"] = email

    # Do not create duplicate account
    for existing_user in users:

        existing_email = str(
            existing_user.get("email", "")
        ).strip().lower()

        if existing_email == email:
            return None

    users.append(user_data)

    save_json(
        USERS_FILE,
        users
    )

    return user_data


def get_user_by_email(email):
    """
    Find user using case-insensitive email.
    """

    if not email:
        return None

    email = str(
        email
    ).strip().lower()

    users = load_json(USERS_FILE)

    for user in users:

        stored_email = str(
            user.get("email", "")
        ).strip().lower()

        if stored_email == email:
            return user

    return None


def update_user(email, updates):
    """
    Update an existing user.
    """

    if not email:
        return None

    email = str(
        email
    ).strip().lower()

    users = load_json(USERS_FILE)

    updated_user = None

    for user in users:

        stored_email = str(
            user.get("email", "")
        ).strip().lower()

        if stored_email == email:

            user.update(updates)

            # Keep email normalized
            if "email" in user:
                user["email"] = str(
                    user["email"]
                ).strip().lower()

            updated_user = user

            break

    if updated_user is not None:
        save_json(
            USERS_FILE,
            users
        )

    return updated_user


def get_all_users():
    return load_json(USERS_FILE)


# ============================================================
# ASSESSMENTS
# ============================================================

def save_assessment(assessment_data):

    assessments = load_json(
        ASSESSMENTS_FILE
    )

    assessment_id = (
        f"AST-{len(assessments) + 1:06d}"
    )

    assessment_data["id"] = assessment_id

    if not assessment_data.get("created_at"):

        assessment_data["created_at"] = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    assessments.append(
        assessment_data
    )

    save_json(
        ASSESSMENTS_FILE,
        assessments
    )

    return assessment_id


def get_assessment(assessment_id):

    assessments = load_json(
        ASSESSMENTS_FILE
    )

    for assessment in assessments:

        if assessment.get("id") == assessment_id:
            return assessment

    return None


def get_all_assessments():

    return load_json(
        ASSESSMENTS_FILE
    )


# ============================================================
# CLAIMS
# ============================================================

def save_claim(claim_data):

    claims = load_json(
        CLAIMS_FILE
    )

    year = datetime.now().year

    claim_id = (
        f"INS-{year}-{len(claims) + 1:04d}"
    )

    claim_data["claim_id"] = claim_id

    if not claim_data.get("created_at"):

        claim_data["created_at"] = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    claims.append(
        claim_data
    )

    save_json(
        CLAIMS_FILE,
        claims
    )

    return claim_id


def get_claim(claim_id):

    claims = load_json(
        CLAIMS_FILE
    )

    for claim in claims:

        if claim.get("claim_id") == claim_id:
            return claim

    return None


def get_claims_by_user(user_email):

    claims = load_json(
        CLAIMS_FILE
    )

    if not user_email:
        return []

    user_email = str(
        user_email
    ).strip().lower()

    user_claims = []

    for claim in claims:

        claim_email = str(
            claim.get("user_email", "")
        ).strip().lower()

        if claim_email == user_email:
            user_claims.append(claim)

    user_claims.reverse()

    return user_claims


def get_all_claims():

    claims = load_json(
        CLAIMS_FILE
    )

    claims.reverse()

    return claims


def update_claim_status(
    claim_id,
    status,
    remarks=""
):

    claims = load_json(
        CLAIMS_FILE
    )

    updated_claim = None

    for claim in claims:

        if claim.get("claim_id") == claim_id:

            claim["status"] = status

            claim["remarks"] = remarks

            claim["updated_at"] = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            updated_claim = claim

            break

    if updated_claim is not None:

        save_json(
            CLAIMS_FILE,
            claims
        )

    return updated_claim


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(damage_detections):
    """
    Calculate presentation-friendly severity.

    This is an AI-assisted estimate.
    """

    if not damage_detections:
        return "No Damage"

    confidences = []

    for detection in damage_detections:

        try:

            confidence = float(
                detection.get(
                    "confidence",
                    0
                )
            )

            # Handle either 0-1 or 0-100 confidence
            if 0 < confidence <= 1:
                confidence *= 100

            confidences.append(
                confidence
            )

        except (
            ValueError,
            TypeError,
            AttributeError
        ):
            continue

    if not confidences:
        return "Minor"

    highest = max(
        confidences
    )

    damage_count = len(
        damage_detections
    )

    # Multiple detected damage regions
    # increase severity.
    if highest >= 85 or damage_count >= 4:
        return "Severe"

    if highest >= 65 or damage_count >= 2:
        return "Moderate"

    return "Minor"