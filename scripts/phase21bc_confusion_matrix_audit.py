from phase21bc_common import *


def run() -> pd.DataFrame:
    merged, _, _ = aligned_frames()
    rows = []
    for model in ["M4", "M8"]:
        rec, per_class, cm = classification_metrics(merged, model)
        for actual_i, actual in enumerate(CLASSES):
            for pred_i, predicted in enumerate(CLASSES):
                rows.append({"model_id": model, "actual_class": actual, "predicted_class": predicted, "count": int(cm[actual_i, pred_i]),
                             "row_total": int(cm[actual_i].sum()), "column_total": int(cm[:, pred_i].sum()), "n": int(cm.sum())})
    out = pd.DataFrame(rows)
    write(out, "corrected_confusion_matrices.csv")
    return out


if __name__ == "__main__":
    run()
