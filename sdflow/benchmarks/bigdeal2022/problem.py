"""BigDEAL Challenge 2022 — registered, data not yet ingested (see manifest.yaml)."""

from sdflow.problems import ProblemNotIngestedError


def get_problem():
    raise ProblemNotIngestedError(
        "bigdeal2022 is registered but its dataset has not been ingested yet — "
        "see sdflow/benchmarks/bigdeal2022/manifest.yaml"
    )
