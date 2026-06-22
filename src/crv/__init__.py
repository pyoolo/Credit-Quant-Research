"""crv — HY EUR credit relative-value engine.

A compartmentalised, three-track relative-value system for European high-yield
corporate bonds:

    Track 1  ``crv.tracks.cross_section``  — cross-sectional fair-spread Lasso
    Track 2  ``crv.tracks.self_ref``       — structural / self-referential pricer
    Track 3  ``crv.tracks.distressed``     — distressed PD/recovery frontier

A single Bloomberg fetch (``crv.data``) feeds a router that splits the universe
into three disjoint panels, one per track. Each track owns its model, its
artefacts and its relative-value output. The pricing layer (``crv.pricing``)
consumes the fitted artefacts to price new issues, and the app layer
(``crv.app``) exposes the monitor and pricer UIs.
"""

from crv._version import __version__

__all__ = ["__version__"]
