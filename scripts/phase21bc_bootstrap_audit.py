from phase21bc_common import *


def run() -> pd.DataFrame:
    merged, _, _ = aligned_frames()
    out = bootstrap(merged)
    write(out, "corrected_bootstrap_results.csv")
    return out


if __name__ == "__main__":
    run()
