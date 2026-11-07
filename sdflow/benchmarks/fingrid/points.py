"""NWP sampling points per bidding zone, weighted by wind capacity.

A zone-level wind forecast needs weather where the turbines are, not at the
zone's centroid. Each entry below samples the NWP grid at the centre of a
build-out cluster, weighted by that cluster's rough share of the zone's
installed wind capacity.

Finland's ~8 GW is almost entirely a coastal band: North Ostrobothnia around
Raahe/Oulu is the largest cluster, running south through Kalajoki, Kokkola,
Vaasa and Kristinestad to Satakunta, plus a northern group on the Bothnian
Bay coast at Simo/Kemi/Tornio and a thinner inland-Lapland group. Almost
nothing sits in the south-east interior.

The weights are approximate — they come from the published regional build-out
shares, not from a turbine registry. They matter far less than the coverage:
the capacity-weighted mean of eight points across the coastal band tracks the
zone aggregate well, and a model is free to use the per-point columns
directly and learn its own weighting. Replace with a real per-farm set when
one is available; keep the ``name`` values stable, they are column prefixes.
"""

from __future__ import annotations

import typing as t

from sdflow.data.io.weather import Point

#: Finland (FI). Eight clusters covering the coastal wind band.
FI_POINTS: t.Tuple[Point, ...] = (
    Point("simo_kemi", 65.70, 24.90, weight=0.13),        # Bothnian Bay coast
    Point("lapland_inland", 67.00, 25.50, weight=0.05),   # Sodankylä/Kittilä area
    Point("raahe_oulu", 64.50, 24.70, weight=0.25),       # largest cluster
    Point("kalajoki", 64.10, 24.00, weight=0.12),
    Point("kokkola", 63.85, 23.10, weight=0.10),
    Point("vaasa", 63.10, 21.60, weight=0.15),
    Point("kristinestad", 62.30, 21.40, weight=0.10),
    Point("satakunta", 61.40, 21.50, weight=0.10),        # Pori/Rauma coast
)

#: zone slug -> points. Only zones listed here get a ``gridnwp`` problem.
POINTS_BY_ZONE: t.Dict[str, t.Tuple[Point, ...]] = {
    "fi": FI_POINTS,
}


def points_for(zone_slug: str) -> t.Tuple[Point, ...]:
    try:
        return POINTS_BY_ZONE[zone_slug]
    except KeyError:
        raise KeyError(
            f"no NWP sampling points defined for zone {zone_slug!r}; "
            f"known: {sorted(POINTS_BY_ZONE)} (add them in "
            f"sdflow/benchmarks/gridnwp/points.py)"
        ) from None
