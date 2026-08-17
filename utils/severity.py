def calculate_severity(damage_detections):

    if not damage_detections:
        return "No Damage"

    confidences = []

    for detection in damage_detections:

        try:
            confidence = float(
                detection.get("confidence", 0)
            )

            if 0 < confidence <= 1:
                confidence *= 100

            confidences.append(confidence)

        except (ValueError, TypeError):
            continue

    if not confidences:
        return "Minor"

    highest = max(confidences)
    count = len(damage_detections)

    # Multiple independent damage regions
    if count >= 4 and highest >= 55:
        return "Severe"

    if count >= 3 and highest >= 55:
        return "High"

    if count >= 2 and highest >= 45:
        return "Moderate"

    if highest >= 75:
        return "Moderate"

    return "Minor"