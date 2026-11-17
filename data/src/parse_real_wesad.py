"""Compatibility entry point for parsing WESAD Subject S2."""

if __package__:
    from .parse_real_wesad_full import parse_subject
else:
    from parse_real_wesad_full import parse_subject


def extract_real_wesad():
    result = parse_subject("S2")
    if result is None:
        raise FileNotFoundError("Raw WESAD recording for S2 was not found.")
    return result


if __name__ == "__main__":
    extract_real_wesad()
