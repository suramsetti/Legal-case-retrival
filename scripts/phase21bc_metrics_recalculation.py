from phase21bc_common import *


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    merged, _, _ = aligned_frames()
    summaries, reports = [], []
    for model in ["M4", "M8"]:
        rec, per_class, _ = classification_metrics(merged, model)
        summaries.append(rec)
        reports.extend(per_class)
    summary = pd.DataFrame(summaries)
    report = pd.DataFrame(reports)
    write(summary, "corrected_statistical_comparison.csv")
    write(report, "corrected_classification_report.csv")
    write(pd.DataFrame([calibration(merged, "M4"), calibration(merged, "M8")]), "corrected_calibration_metrics.csv")
    return summary, report


if __name__ == "__main__":
    run()
