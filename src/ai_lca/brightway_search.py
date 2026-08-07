"""Compatibility shim. Edit ecoinvent_search.py at the repository root instead."""

from ecoinvent_search import *  # noqa: F401,F403
from ecoinvent_search import (  # explicit private helpers retained for existing imports/tests
    _activity_type,
    _normalise_location,
    _query_variants,
)
