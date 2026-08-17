def calculate_severity(damage_detections):
    """
    Calculate a presentation-friendly severity level
    from damage model confidence.

    This is an AI-assisted estimate, not a professional
    insurance damage assessment.
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

            confidences.append(
                confidence
            )

        except (
            ValueError,
            TypeError
        ):
            pass

    if not confidences:
        return "Minor"

    highest = max(
        confidences
    )

    # Multiple damage regions increase severity.
    damage_count = len(
        damage_detections
    )

    if highest >= 85 or damage_count >= 4:
        return "Severe"

    if highest >= 65 or damage_count >= 2:
        return "Moderate"

    return "Minor"