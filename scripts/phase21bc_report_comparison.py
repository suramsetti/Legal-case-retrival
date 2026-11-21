from phase21bc_common import *


def run() -> pd.DataFrame:
    merged, _, _ = aligned_frames()
    rows = []
    old = pd.read_csv(ROOT / "results" / "tables" / "phase21b_test_results.csv")
    new = pd.DataFrame([classification_metrics(merged, m)[0] for m in ["M4", "M8"]])
    fields = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "roc_auc_ovr", "roc_auc_ovo", "pr_auc_macro", "log_loss"]
    for metric in fields:
        for model in ["M4", "M8"]:
            ov = old.loc[old.model_id == model, metric].iloc[0]
            nv = new.loc[new.model_id == model, metric].iloc[0]
            rows.append({"metric": metric, "model": model, "original": ov, "corrected": nv,
                         "status": "Confirmed" if np.isclose(float(ov), float(nv), atol=5e-4) else "Corrected"})
    for model in ["M4", "M8"]:
        _, _, cm = classification_metrics(merged, model)
        old_cm = json.loads(old.loc[old.model_id == model, "confusion_matrix"].iloc[0])
        for i, actual in enumerate(CLASSES):
            for j, predicted in enumerate(CLASSES):
                rows.append({"metric": f"cm_{actual}_{predicted}", "model": model, "original": old_cm[i][j], "corrected": int(cm[i, j]),
                             "status": "Confirmed" if old_cm[i][j] == int(cm[i, j]) else "Corrected"})
    out = pd.DataFrame(rows)
    write(out, "original_vs_corrected_metrics.csv")
    return out


if __name__ == "__main__":
    run()
