#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
periodic_orbits — periodic-orbit enumeration on closed directional-coupler graphs.

Usage
-----
from periodic_orbits import PeriodicOrbits

Classes
-------
PeriodicOrbits
    Builds the directed-bond transition graph of a closed graph whose real
    vertices are lossless 4-port directional couplers, then enumerates its
    irreducible periodic orbits and all periodic orbits up to a given length
    (Kottos & Smilansky, Ann. Phys. 274 (1999) 76-124).
"""

from .core import PeriodicOrbits


__all__ = [
    "PeriodicOrbits",
]
