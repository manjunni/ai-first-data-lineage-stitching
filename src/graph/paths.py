"""
Path identity helpers shared by parsers and the stitcher.

ADLS locations are referenced in different shapes depending on the artifact:

* notebooks:   ``abfss://raw@stactionlakeprod.dfs.core.windows.net/sap/articles``
* ADF dataset: ``fileSystem: raw`` + ``folderPath: sap/articles`` (account lives
  in a linked service that is often not exported)

``adls_location_key`` reduces both to ``raw/sap/articles`` so the stitcher can
treat them as one node. Templated segments (``{country}``) are normalised to
``{*}`` so that ``raw/pos/{country:upper}`` and ``raw/pos/{country_code}``
also collapse to the same key.
"""
from __future__ import annotations

import re

_PH = re.compile(r"\{[A-Za-z_]\w*(?::(?:upper|lower))?\}")
_ABFSS = re.compile(r"^(?:abfss?|wasbs?)://([^@/]+)@[^/]+/?(.*)$", re.IGNORECASE)
_ADLS_SCHEMES = ("abfss://", "abfs://", "wasbs://", "wasb://", "adls://")


def is_adls_url(path: str) -> bool:
    return path.lower().startswith(_ADLS_SCHEMES)


def adls_location_key(path: str) -> str | None:
    """``container/folder/...`` for an ADLS URL, or None if not an ADLS URL."""
    m = _ABFSS.match(path)
    if m:
        container, rest = m.group(1), m.group(2)
    elif path.lower().startswith("adls://"):
        container, _, rest = path[len("adls://"):].partition("/")
    else:
        return None
    return normalise_location(f"{container}/{rest}")


_DATE_PH = re.compile(
    r"\{(?:year|yyyy|month|mm|day|dd|hour|hh|minute|date|dt|ds|run_date|\w+_date|\w+_ts)"
    r"(?::(?:upper|lower))?\}",
    re.IGNORECASE,
)


def is_partition_segment(seg: str) -> bool:
    """A path segment that is only a date/time partition placeholder (e.g. ``{year}``)."""
    return bool(_DATE_PH.fullmatch(seg))


def strip_partition_segments(path: str) -> str:
    """Drop trailing date-partition placeholder segments: ``.../pos/{c}/{year}/{month}`` → ``.../pos/{c}``."""
    segs = path.rstrip("/").split("/")
    while len(segs) > 1 and is_partition_segment(segs[-1]):
        segs.pop()
    return "/".join(segs)


def normalise_location(loc: str) -> str:
    """Strip slashes and date partitions; wildcard remaining placeholders for matching."""
    segs = [s for s in strip_partition_segments(loc.strip().strip("/")).split("/") if s]
    return "/".join(_PH.sub("{*}", s) for s in segs)


def adls_display_name(path: str) -> str:
    """Readable node name for an ADLS URL: ``container/folder/...``."""
    m = _ABFSS.match(path)
    if not m:
        return path
    return "/".join(p for p in (m.group(1), m.group(2).strip("/")) if p)


def is_templated(path: str) -> bool:
    return bool(_PH.search(path))
