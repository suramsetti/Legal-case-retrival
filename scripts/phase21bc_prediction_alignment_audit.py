from phase21bc_common import *


def run() -> pd.DataFrame:
    truth = eligible_truth()
    expected = set(truth["id"])
    rows = validate_prediction_frame(load_predictions("m4"), "M4", expected) + validate_prediction_frame(load_predictions("m8"), "M8", expected)
    if set(load_predictions("m4")["id"]) != set(load_predictions("m8")["id"]):
        rows.append({"model": "M4/M8", "check": "cross_model_ids", "status": "FAIL", "detail": "ID sets differ"})
    if not rows:
        rows = [{"model": "M4/M8", "check": "all_alignment_checks", "status": "PASS", "detail": f"{len(expected)} eligible IDs; canonical order {CLASSES}"}]
    out = pd.DataFrame(rows)
    write(out, "prediction_alignment_audit.csv")
    if (out["status"] == "FAIL").any():
        raise ValueError("Alignment audit failed; comparative metrics were not calculated.")
    return out


if __name__ == "__main__":
    run()
