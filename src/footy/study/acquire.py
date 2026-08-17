"""Download the football-data.co.uk archive and record its provenance.

Study spec §3 and §9. Files are cached on disk and never re-fetched, and every
downloaded file's SHA-256 goes into a manifest so a result can be traced back
to the exact bytes it was computed from.

The archive is a free community resource. Requests are serialised with a delay
between them rather than parallelised.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from footy.data.football_data import (
    LEAGUES,
    USER_AGENT,
    season_codes,
    season_url,
    sha256_file,
    write_manifest,
)

REQUEST_DELAY_SECONDS = 0.25
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Acquisition:
    """What the download found, so coverage gaps are explicit rather than silent."""

    downloaded: tuple[Path, ...]
    cached: tuple[Path, ...]
    absent: tuple[tuple[str, str], ...]
    failed: tuple[tuple[str, str, str], ...]

    @property
    def available(self) -> tuple[Path, ...]:
        return tuple(sorted(set(self.downloaded) | set(self.cached)))

    def summary(self) -> str:
        return (
            f"{len(self.downloaded)} downloaded, {len(self.cached)} cached, "
            f"{len(self.absent)} absent (404), {len(self.failed)} failed"
        )


def acquire(
    root: Path,
    *,
    leagues: tuple[str, ...] = LEAGUES,
    seasons: tuple[str, ...] | None = None,
    delay: float = REQUEST_DELAY_SECONDS,
) -> Acquisition:
    """Fetch every (league, season) CSV that exists, skipping what is cached.

    A 404 means the league did not run that season, which is ordinary and is
    recorded in ``absent``. Any other error is recorded in ``failed`` rather
    than aborting, so one flaky request cannot cost the whole archive.
    """
    codes = tuple(season_codes()) if seasons is None else seasons
    root.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    cached: list[Path] = []
    absent: list[tuple[str, str]] = []
    failed: list[tuple[str, str, str]] = []

    for season in codes:
        for league in leagues:
            dest = root / season / f"{league}.csv"
            if dest.exists() and dest.stat().st_size > 0:
                cached.append(dest)
                continue

            request = urllib.request.Request(
                season_url(season, league), headers={"User-Agent": USER_AGENT}
            )
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                    body = response.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    absent.append((season, league))
                else:
                    failed.append((season, league, f"HTTP {exc.code}"))
                time.sleep(delay)
                continue
            except Exception as exc:  # noqa: BLE001 - one bad request must not abort the archive
                failed.append((season, league, type(exc).__name__))
                time.sleep(delay)
                continue

            if not body.strip():
                absent.append((season, league))
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(body)
                downloaded.append(dest)
            time.sleep(delay)

    return Acquisition(
        downloaded=tuple(downloaded),
        cached=tuple(cached),
        absent=tuple(absent),
        failed=tuple(failed),
    )


def record_manifest(paths: tuple[Path, ...], dest: Path) -> None:
    """Write the SHA-256 manifest that every result file will reference."""
    write_manifest(list(paths), dest)


def checksum(path: Path) -> str:
    return sha256_file(path)
