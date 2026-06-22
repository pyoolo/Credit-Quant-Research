"""Configuration: paths and model parameters.

This package replaces the identical ``paths`` block that used to be copy-pasted
into the first cell of every notebook. There is now a single source of truth:

    ``crv.config.paths``    — every file location, folders auto-created
    ``crv.config.settings`` — all model / pipeline parameters

The project root is resolved from the ``CRV_ROOT`` environment variable and
defaults to ``./project`` under the current working directory.
"""

from crv.config import paths, settings

__all__ = ["paths", "settings"]
