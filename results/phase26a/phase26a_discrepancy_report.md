# Phase 26A Discrepancy Report

- Previously reported query counts do not match the repository: actual counts are 827/118/237.
- The existing retrieval service imports optional ranking dependencies; `lightgbm` and `xgboost` were installed from the declared integration requirements so the existing service could initialize.
- The service does not load a standalone neural retrieval checkpoint at request time. It uses the existing precomputed hybrid feature index and frozen fusion configuration, as documented above.
- No unresolved blocker remains for the tested IL-TUR API path.
