"""Energy-specific pieces of sdflow: the asset classes built on energydatamodel.

Install with ``pip install "sdflow[energy]"``. The core of sdflow never imports this
package; a problem that describes its entities with these classes passes a
collection to :class:`sdflow.Dataset`, and everything else is domain-neutral.
"""

from .assets.base import Asset, Collection, Sensor, TimeSeries
from .assets.battery import Battery
from .assets.building import House
from .assets.energycollection import EnergyCommunity, Portfolio, Site
from .assets.heatpump import HeatPump
from .assets.solar import FixedMount, PVArray, PVSystem, SingleAxisTrackerMount, SolarPowerArea
from .assets.wind import WindFarm, WindPowerArea, WindTurbine

__all__ = [
    "Asset", "TimeSeries", "Sensor", "Collection",
    "House",
    "FixedMount", "SingleAxisTrackerMount", "PVArray", "PVSystem", "SolarPowerArea",
    "WindTurbine", "WindFarm", "WindPowerArea",
    "Battery", "HeatPump",
    "Site", "EnergyCommunity", "Portfolio",
]
