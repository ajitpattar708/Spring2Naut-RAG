from typing import Optional, Tuple


def parse_version_components(version: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    cleaned = (version or "").strip().lower()
    if not cleaned:
        raise ValueError("Version cannot be empty.")

    parts = cleaned.split(".")
    normalized = []
    for part in parts[:3]:
        if part in {"x", "*"}:
            normalized.append(None)
        else:
            try:
                normalized.append(int(part))
            except ValueError as exc:
                raise ValueError(f"Invalid version component: {part}") from exc

    while len(normalized) < 3:
        normalized.append(None)

    return tuple(normalized)  # type: ignore[return-value]


def matches_version_spec(version: str, spec: Optional[str]) -> bool:
    if not spec:
        return True

    version_components = parse_version_components(version)
    spec_components = parse_version_components(spec)

    for version_part, spec_part in zip(version_components, spec_components):
        if spec_part is not None and version_part != spec_part:
            return False

    return True


def compare_versions(left: str, right: str) -> int:
    left_components = parse_version_components(left)
    right_components = parse_version_components(right)

    for left_part, right_part in zip(left_components, right_components):
        normalized_left = -1 if left_part is None else left_part
        normalized_right = -1 if right_part is None else right_part
        if normalized_left < normalized_right:
            return -1
        if normalized_left > normalized_right:
            return 1

    return 0


def _compare_to_bound(version: str, bound: str, *, is_maximum: bool) -> int:
    version_components = parse_version_components(version)
    bound_components = parse_version_components(bound)

    for version_part, bound_part in zip(version_components, bound_components):
        normalized_version = -1 if version_part is None else version_part
        if bound_part is None:
            normalized_bound = 10**9 if is_maximum else 0
        else:
            normalized_bound = bound_part

        if normalized_version < normalized_bound:
            return -1
        if normalized_version > normalized_bound:
            return 1

    return 0


def includes_version(
    version: str,
    spec: Optional[str] = None,
    minimum: Optional[str] = None,
    maximum: Optional[str] = None,
) -> bool:
    if spec and not matches_version_spec(version, spec):
        return False
    if minimum and _compare_to_bound(version, minimum, is_maximum=False) < 0:
        return False
    if maximum and _compare_to_bound(version, maximum, is_maximum=True) > 0:
        return False
    return True


def normalize_major_minor(version: str) -> str:
    cleaned = (version or "").strip()
    if not cleaned or any(token in cleaned for token in ("x", "*")):
        return cleaned

    parts = cleaned.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return cleaned


def is_concrete_version(version: str) -> bool:
    cleaned = (version or "").strip().lower()
    if not cleaned or "x" in cleaned or "*" in cleaned:
        return False

    parts = cleaned.split(".")
    if len(parts) != 3:
        return False

    try:
        return all(part.isdigit() for part in parts)
    except Exception:
        return False


def validate_migration_target_versions(spring_version: str, micronaut_version: str) -> None:
    if not is_concrete_version(spring_version):
        raise ValueError(
            "Spring migration source version must be an explicit three-part version like 3.4.5."
        )
    if not is_concrete_version(micronaut_version):
        raise ValueError(
            "Micronaut migration target version must be an explicit three-part version like 4.10.8 or 4.5.7."
        )

    spring_major = spring_version.split(".", 1)[0]
    micronaut_major = micronaut_version.split(".", 1)[0]
    if spring_major != "3":
        raise ValueError("Spring migration source version must belong to the Spring Boot 3.x line.")
    if micronaut_major != "4":
        raise ValueError("Micronaut migration target version must belong to the Micronaut 4.x line.")
