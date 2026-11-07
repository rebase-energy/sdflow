"""BFCom2018 — registered, data not yet ingested (see manifest.yaml)."""

from sdflow.problems import ProblemNotIngestedError


def get_problem():
    raise ProblemNotIngestedError(
        "bfcom2018 is registered but its dataset has not been ingested yet — "
        "see sdflow/benchmarks/bfcom2018/manifest.yaml"
    )
