"""INSURE AI preliminary repair-cost estimator.

This is a demo/preliminary estimate, not an insurer-approved settlement.
It uses the parts that have been spatially associated with a damage region.
"""


def estimate_repair_cost(damage_detections, vehicle_type="car", affected_parts=None):
    affected_parts = affected_parts or []

    damages = [
        d for d in (damage_detections or [])
        if str(d.get("class_name", "")).strip().lower() == "damage"
    ]

    if not damages:
        return 0, []

    # Confidence is used only as supporting evidence, never as the sole
    # definition of severity.
    max_conf = max(float(d.get("confidence", 0.0)) for d in damages)
    regions = len(damages)

    # Preliminary Indian-market demo rates. These are configurable and are
    # intentionally labeled as estimates in the UI/report.
    part_rates = {
        "front-bumper": 9000,
        "back-bumper": 8500,
        "front-door": 12000,
        "back-door": 11500,
        "front-fender": 9000,
        "back-fender": 9000,
        "hood": 11000,
        "roof": 15000,
        "trunk": 12000,
        "headlight": 7500,
        "tail-light": 6500,
        "mirror": 4500,
        "grille": 5500,
        "front-wheel": 6000,
        "back-wheel": 6000,
        "windshield": 10000,
        "back-window": 9000,
    }

    breakdown = []
    total = 0.0

    for part in affected_parts:
        name = str(part.get("class_name", "Vehicle part")).strip()
        key = name.lower()
        base = part_rates.get(key, 8000)

        coverage = float(part.get("damage_coverage", 0.0))
        overlap = float(part.get("damage_overlap", 0.0))

        # Spatial evidence determines the damage multiplier.
        if coverage >= 0.60 or overlap >= 0.35:
            severity = "Severe"
            multiplier = 1.35
        elif coverage >= 0.35 or overlap >= 0.20:
            severity = "Moderate"
            multiplier = 1.00
        else:
            severity = "Minor"
            multiplier = 0.70

        estimated = round(base * multiplier, -2)
        total += estimated

        breakdown.append({
            "part": name,
            "damage_type": "Detected damage region",
            "confidence": round(float(part.get("damage_confidence", max_conf)), 2),
            "severity": severity,
            "damage_coverage": round(coverage * 100, 1),
            "estimated_cost": int(estimated),
        })

    # If damage exists but no part could be safely associated, don't invent a
    # damaged part. Provide a conservative vehicle-level preliminary estimate.
    if not breakdown:
        if regions >= 3:
            severity = "Severe"
            total = 30000
        elif regions == 2:
            severity = "High"
            total = 22000
        else:
            severity = "Moderate"
            total = 12000

        breakdown.append({
            "part": "Unlocalized damage",
            "damage_type": "Damage detected; specific part not confidently localized",
            "confidence": round(max_conf, 2),
            "severity": severity,
            "estimated_cost": int(total),
        })

    # Vehicle-type adjustment. Car-part estimates are only produced for cars;
    # other vehicle types use the fallback estimate above.
    vehicle_type = str(vehicle_type or "car").lower()
    if vehicle_type == "truck":
        total *= 1.20
    elif vehicle_type == "bus":
        total *= 1.30
    elif vehicle_type == "motorcycle":
        total *= 0.60

    total = int(round(total / 100.0) * 100)

    return total, breakdown
