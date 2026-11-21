from phase21bc_common import *


def run() -> dict:
    merged, _, _ = aligned_frames()
    result = mcnemar(merged)
    write(pd.DataFrame([result]), "corrected_statistical_comparison.csv")
    return result


if __name__ == "__main__":
    run()
