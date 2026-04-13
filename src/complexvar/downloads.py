"""Step 1 download pipeline."""

from __future__ import annotations

import logging
import os
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from complexvar.data.burke import build_structure_manifest
from complexvar.utils.download import md5sum, stream_download
from complexvar.utils.io import ensure_parent, write_json, write_tsv

DOWNLOAD_COLUMNS = [
    "dataset",
    "source_url",
    "local_path",
    "status",
    "size_bytes",
    "md5",
    "downloaded_at_utc",
    "source_record",
    "notes",
]

DEFAULT_URLS = {
    "skempi_csv": "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv",
    "intact_micluster": (
        "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/"
        "intact-micluster.zip"
    ),
    "intact_dataset_page": "https://www.ebi.ac.uk/intact/download/datasets#mutations",
    "intact_mutations_fallback": (
        "https://ftp.ebi.ac.uk/pub/databases/intact/current/various/mutations.tsv"
    ),
    "clinvar_vcf": (
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
    ),
    "clinvar_vcf_tbi": (
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi"
    ),
    "clinvar_summary": (
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/"
        "variant_summary.txt.gz"
    ),
    "burke_zenodo_api": "https://zenodo.org/api/records/7505985",
    "burke_figshare_api": "https://api.figshare.com/v2/articles/16945039",
    "burke_summary_csv": (
        "https://archive.bioinfo.se/huintaf2/table_AF2_HURI_HuMap_UNIQUE.csv"
    ),
    "alphafold_api_base": "https://alphafold.ebi.ac.uk/api/prediction",
    "rcsb_pdb_base": "https://files.rcsb.org/download",
}

BURKE_EXPECTED_FILES = {
    "README.txt",
    "manifest.txt",
    "random.zip",
    "humap.zip",
    "HuRI.zip",
    "HuRI-single.zip",
    "HuMap-single.zip",
}
MIN_FREE_SPACE_BYTES = 2 * 1024**3
MIN_DOWNLOAD_HEADROOM_BYTES = 128 * 1024**2
MIN_ALPHAFOLD_FREE_BYTES = 1024**3
MUTATION_LINK_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


@dataclass(slots=True)
class Step1Config:
    datasets: list[str]
    root: Path
    skip_existing: bool = True
    force: bool = False
    extract_high_confidence_only: bool = True
    pdb_limit: int | None = None
    monomer_limit: int | None = None
    workers: int = 4


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_url(name: str, default_key: str) -> str:
    return os.environ.get(name, DEFAULT_URLS[default_key])


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _local_path_str(path: Path) -> str:
    return str(path.as_posix())


def _make_logger(name: str, path: Path) -> logging.Logger:
    logger = logging.getLogger(f"complexvar.{name}.{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    ensure_parent(path)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DOWNLOAD_COLUMNS)
    return pd.read_csv(path, sep="\t")


def _manifest_md5_lookup(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {}
    subset = frame[frame["md5"].notna()].copy()
    return {
        str(local_path): str(value)
        for local_path, value in zip(subset["local_path"], subset["md5"], strict=False)
    }


def _bootstrap_manifest_lookup(manifests_dir: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not manifests_dir.exists():
        return lookup
    for path in sorted(manifests_dir.glob("*.tsv")):
        try:
            frame = pd.read_csv(path, sep="\t")
        except Exception:  # noqa: BLE001
            continue
        required = {"local_path", "md5"}
        if not required.issubset(frame.columns):
            continue
        for row in frame.itertuples(index=False):
            local_path = getattr(row, "local_path", "")
            digest = getattr(row, "md5", "")
            if isinstance(local_path, str) and isinstance(digest, str) and digest:
                lookup[local_path] = digest
    return lookup


def _rename_corrupt(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.corrupt.{stamp}")
    path.rename(target)
    return target


def _rename_unverified(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.unverified.{stamp}")
    path.rename(target)
    return target


def _record_row(
    dataset: str,
    source_url: str,
    local_path: Path,
    status: str,
    source_record: str = "",
    notes: str = "",
) -> dict[str, Any]:
    size_bytes = local_path.stat().st_size if local_path.exists() else 0
    digest = md5sum(local_path) if local_path.exists() and local_path.is_file() else ""
    return {
        "dataset": dataset,
        "source_url": source_url,
        "local_path": _local_path_str(local_path),
        "status": status,
        "size_bytes": size_bytes,
        "md5": digest,
        "downloaded_at_utc": _now_utc(),
        "source_record": source_record,
        "notes": notes,
    }


def _blocked_row(
    dataset: str,
    local_path: Path,
    notes: str,
    source_url: str = "",
    source_record: str = "",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "source_url": source_url,
        "local_path": _local_path_str(local_path),
        "status": "blocked",
        "size_bytes": 0,
        "md5": "",
        "downloaded_at_utc": _now_utc(),
        "source_record": source_record,
        "notes": notes,
    }


def _content_length(url: str) -> int | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        value = response.headers.get("Content-Length")
        return int(value) if value else None
    except Exception:  # noqa: BLE001
        return None


def _download_file(
    *,
    dataset: str,
    url: str,
    destination: Path,
    manifest_lookup: dict[str, str],
    skip_existing: bool,
    force: bool,
    logger: logging.Logger,
    source_record: str = "",
    notes: str = "",
) -> dict[str, Any]:
    destination = ensure_parent(destination)
    path_key = _local_path_str(destination)
    if destination.exists() and not force:
        actual_md5 = md5sum(destination)
        known_md5 = manifest_lookup.get(path_key)
        if skip_existing and known_md5 == actual_md5:
            status = "skipped"
            logger.info("skip existing file %s", destination)
            return _record_row(
                dataset=dataset,
                source_url=url,
                local_path=destination,
                status=status,
                source_record=source_record,
                notes=notes,
            )
        if known_md5 and known_md5 != actual_md5:
            corrupt_path = _rename_corrupt(destination)
            logger.warning("renamed mismatched file to %s", corrupt_path)
        elif not known_md5:
            unverified_path = _rename_unverified(destination)
            logger.warning(
                "renamed unverified existing file to %s before redownload",
                unverified_path,
            )
    expected_size = _content_length(url)
    free_bytes = _free_space_bytes(destination.parent)
    headroom_bytes = _env_int(
        "COMPLEXVAR_MIN_DOWNLOAD_HEADROOM_BYTES", MIN_DOWNLOAD_HEADROOM_BYTES
    )
    if expected_size is not None and free_bytes < expected_size + headroom_bytes:
        logger.error("blocked by free space for %s", destination)
        raise RuntimeError(
            f"no_space_for_download expected={expected_size} free={free_bytes}"
        )
    logger.info("download %s -> %s", url, destination)
    try:
        stream_download(url, destination)
    except OSError as exc:
        if destination.exists():
            destination.unlink()
        if getattr(exc, "errno", None) == 28:
            logger.error("no space left on device while writing %s", destination)
            raise RuntimeError("no_space_left_on_device") from exc
        raise
    return _record_row(
        dataset=dataset,
        source_url=url,
        local_path=destination,
        status="downloaded",
        source_record=source_record,
        notes=notes,
    )


def _write_combined_manifest(rows: list[dict[str, Any]], output: Path) -> Path:
    frame = pd.DataFrame(rows, columns=DOWNLOAD_COLUMNS)
    frame = frame.sort_values(["dataset", "local_path", "source_url"]).reset_index(
        drop=True
    )
    return write_tsv(frame, output)


def _write_checksums(rows: list[dict[str, Any]], output: Path) -> Path:
    output = ensure_parent(output)
    lines: list[str] = []
    for row in sorted(rows, key=lambda item: item["local_path"]):
        if row["md5"] and Path(row["local_path"]).exists():
            lines.append(f'{row["md5"]}  {row["local_path"]}')
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output


def _free_space_bytes(root: Path) -> int:
    return shutil.disk_usage(root).free


def _parse_skempi_pdb_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep=";", engine="python")
    column = "#Pdb" if "#Pdb" in frame.columns else frame.columns[0]
    pdb_ids: set[str] = set()
    for value in frame[column].astype(str):
        match = re.match(r"([0-9A-Za-z]{4})", value.strip())
        if match:
            pdb_ids.add(match.group(1).upper())
    return sorted(pdb_ids)


def _download_rcsb_pdbs(
    pdb_ids: list[str],
    out_dir: Path,
    manifest_lookup: dict[str, str],
    skip_existing: bool,
    force: bool,
    workers: int,
    limit: int | None,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    pdb_ids = pdb_ids[:limit] if limit is not None else pdb_ids
    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    def task(pdb_id: str) -> dict[str, Any]:
        url = f"{_env_url('COMPLEXVAR_RCSB_PDB_BASE', 'rcsb_pdb_base')}/{pdb_id}.pdb"
        return _download_file(
            dataset="rcsb",
            url=url,
            destination=out_dir / f"{pdb_id}.pdb",
            manifest_lookup=manifest_lookup,
            skip_existing=skip_existing,
            force=force,
            logger=logger,
            source_record="rcsb",
            notes=f"pdb_id={pdb_id}",
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(task, pdb_id): pdb_id for pdb_id in pdb_ids}
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def _discover_intact_mutation_links(page_html: str, base_url: str) -> list[str]:
    matches = MUTATION_LINK_RE.findall(page_html)
    urls: list[str] = []
    for href in matches:
        lowered = href.lower()
        if "mutation" not in lowered:
            continue
        if href.startswith("http://") or href.startswith("https://"):
            urls.append(href)
        elif href.startswith("/"):
            urls.append(f"https://www.ebi.ac.uk{href}")
        else:
            urls.append(f"{base_url.rstrip('/')}/{href.lstrip('./')}")
    unique_urls: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def _extract_zip(
    archive_path: Path,
    destination_dir: Path,
    logger: logging.Logger,
    members_to_keep: set[str] | None = None,
) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    state_path = destination_dir / f".{archive_path.name}.md5"
    archive_md5 = md5sum(archive_path)
    if (
        state_path.exists()
        and state_path.read_text(encoding="utf-8").strip() == archive_md5
    ):
        logger.info("skip extraction for %s", archive_path.name)
        return []
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_name = Path(member.filename).name
            if members_to_keep is not None and member_name not in members_to_keep:
                continue
            target = destination_dir / member_name
            with archive.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted.append(target)
    state_path.write_text(archive_md5 + "\n", encoding="utf-8")
    logger.info("extracted %s files from %s", len(extracted), archive_path.name)
    return extracted


def _fetch_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _fetch_text(url: str) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def _resolve_burke_sources(logger: logging.Logger) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    zenodo_url = _env_url("COMPLEXVAR_BURKE_ZENODO_API", "burke_zenodo_api")
    zenodo_payload = _fetch_json(zenodo_url)
    zenodo_files = {entry.get("key", "") for entry in zenodo_payload.get("files", [])}
    if not BURKE_EXPECTED_FILES.intersection(zenodo_files):
        notes.append("zenodo_record_7505985_did_not_match_expected_burke_assets")
        logger.warning("Zenodo record 7505985 did not match expected Burke assets")
    figshare_url = _env_url("COMPLEXVAR_BURKE_FIGSHARE_API", "burke_figshare_api")
    figshare_payload = _fetch_json(figshare_url)
    return figshare_payload, notes


def _select_burke_archive_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in payload.get("files", []):
        name = entry.get("name", "")
        if name not in BURKE_EXPECTED_FILES:
            continue
        rows.append(
            {
                "name": name,
                "url": entry.get("download_url", ""),
                "size": int(entry.get("size", 0)),
            }
        )
    return rows


def _build_burke_high_conf_manifest(summary_csv: Path, output_dir: Path) -> Path:
    output = output_dir / "burke_high_confidence_complexes.tsv"
    return build_structure_manifest(summary_csv, output, pdockq_threshold=0.5)


def _read_high_conf_pdb_targets(path: Path) -> tuple[pd.DataFrame, set[str], set[str]]:
    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame["is_high_confidence"]].copy()
    targets = set(frame["structure_file"].astype(str))
    proteins = set(frame["protein_a"].astype(str)).union(
        set(frame["protein_b"].astype(str))
    )
    return frame, targets, proteins


def _extract_burke_high_confidence(
    archives_dir: Path,
    high_conf_frame: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    datasets = {
        "huri": "HuRI.zip",
        "humap": "humap.zip",
    }
    for source_dataset, archive_name in datasets.items():
        archive_path = archives_dir / archive_name
        if not archive_path.exists():
            continue
        members = set(
            high_conf_frame.loc[
                high_conf_frame["source_dataset"].astype(str).str.lower()
                == source_dataset,
                "structure_file",
            ].astype(str)
        )
        extracted.extend(
            _extract_zip(archive_path, output_dir, logger, members_to_keep=members)
        )
    return extracted


def _select_alphafold_entry(
    entries: list[dict[str, Any]], accession: str
) -> dict[str, Any]:
    if not entries:
        raise ValueError(f"no AlphaFold entries for {accession}")
    filtered = [
        entry
        for entry in entries
        if str(entry.get("uniprotAccession", "")).upper() == accession.upper()
    ]
    if not filtered:
        filtered = entries
    filtered.sort(
        key=lambda entry: (
            int(entry.get("latestVersion", 0)),
            int(entry.get("sequenceEnd", 0)) - int(entry.get("sequenceStart", 1)),
            int(bool(entry.get("isUniProtReviewed"))),
            int(bool(entry.get("isUniProtReferenceProteome"))),
        ),
        reverse=True,
    )
    return filtered[0]


def _download_alphafold_monomers(
    accessions: list[str],
    manifest_lookup: dict[str, str],
    skip_existing: bool,
    force: bool,
    workers: int,
    limit: int | None,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accessions = sorted(accessions)
    if limit is not None:
        accessions = accessions[:limit]
    api_dir = Path("data/raw/alphafold_monomers/api").resolve()
    pdb_dir = Path("data/raw/alphafold_monomers/pdb").resolve()
    api_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    api_base = _env_url("COMPLEXVAR_ALPHAFOLD_API_BASE", "alphafold_api_base")
    rows: list[dict[str, Any]] = []
    monomer_rows: list[dict[str, Any]] = []

    def task(accession: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        api_url = f"{api_base.rstrip('/')}/{accession}"
        api_path = api_dir / f"{accession}.json"
        pdb_path = pdb_dir / f"{accession}.pdb"
        task_rows: list[dict[str, Any]] = []
        try:
            response = requests.get(api_url, timeout=60)
            response.raise_for_status()
            entries = response.json()
            write_json(entries, api_path)
            task_rows.append(
                _record_row(
                    dataset="alphafold",
                    source_url=api_url,
                    local_path=api_path,
                    status="downloaded",
                    source_record="alphafold_api",
                    notes=f"accession={accession}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AlphaFold API unavailable for %s: %s", accession, exc)
            task_rows.append(
                _blocked_row(
                    "alphafold",
                    api_path,
                    f"accession={accession}; api_request_failed={exc}",
                    source_url=api_url,
                    source_record="alphafold_api",
                )
            )
            return task_rows, None
        try:
            entry = _select_alphafold_entry(entries, accession)
            pdb_url = entry.get("pdbUrl")
            if not pdb_url:
                task_rows.append(
                    _blocked_row(
                        "alphafold",
                        pdb_path,
                        f"accession={accession}; missing_pdb_url",
                        source_record="alphafold_api",
                    )
                )
                return task_rows, None
            pdb_row = _download_file(
                dataset="alphafold",
                url=pdb_url,
                destination=pdb_path,
                manifest_lookup=manifest_lookup,
                skip_existing=skip_existing,
                force=force,
                logger=logger,
                source_record="alphafold_api",
                notes=f"accession={accession}",
            )
            return task_rows, pdb_row
        except Exception as exc:  # noqa: BLE001
            logger.warning("AlphaFold PDB unavailable for %s: %s", accession, exc)
            task_rows.append(
                _blocked_row(
                    "alphafold",
                    pdb_path,
                    f"accession={accession}; pdb_download_failed={exc}",
                    source_record="alphafold_api",
                )
            )
            return task_rows, None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(task, accession): accession for accession in accessions
        }
        for future in as_completed(futures):
            task_rows, pdb_row = future.result()
            rows.extend(task_rows)
            if pdb_row is not None:
                monomer_rows.append(pdb_row)
    return rows, monomer_rows


def run_step1(config: Step1Config) -> dict[str, Any]:
    root = config.root.resolve()
    raw_dir = root / "data/raw"
    manifests_dir = root / "data/manifests"
    results_dir = root / "results/download/logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_dir / "download_manifest.tsv"
    existing_manifest = _load_manifest(manifest_path)
    manifest_lookup = _manifest_md5_lookup(existing_manifest)
    manifest_lookup.update(
        {
            path: digest
            for path, digest in _bootstrap_manifest_lookup(manifests_dir).items()
            if path not in manifest_lookup
        }
    )
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    all_logger = _make_logger("download_all", results_dir / "download_all.log")
    all_logger.info("start step 1 with datasets=%s", ",".join(config.datasets))

    if "skempi" in config.datasets:
        logger = _make_logger("download_skempi", results_dir / "download_skempi.log")
        try:
            skempi_url = _env_url("COMPLEXVAR_SKEMPI_URL", "skempi_csv")
            skempi_path = root / "data/raw/skempi/skempi_v2.csv"
            rows.append(
                _download_file(
                    dataset="skempi",
                    url=skempi_url,
                    destination=skempi_path,
                    manifest_lookup=manifest_lookup,
                    skip_existing=config.skip_existing,
                    force=config.force,
                    logger=logger,
                    source_record="skempi_2.0",
                )
            )
            pdb_ids = _parse_skempi_pdb_ids(skempi_path)
            pdb_rows = _download_rcsb_pdbs(
                pdb_ids=pdb_ids,
                out_dir=root / "data/raw/skempi/pdbs",
                manifest_lookup=manifest_lookup,
                skip_existing=config.skip_existing,
                force=config.force,
                workers=config.workers,
                limit=config.pdb_limit,
                logger=logger,
            )
            rows.extend(pdb_rows)
            skempi_pdb_manifest = pd.DataFrame(
                [
                    {
                        "pdb_id": Path(row["local_path"]).stem,
                        "source_url": row["source_url"],
                        "local_path": row["local_path"],
                        "status": row["status"],
                        "size_bytes": row["size_bytes"],
                        "md5": row["md5"],
                    }
                    for row in pdb_rows
                ]
            )
            write_tsv(skempi_pdb_manifest, manifests_dir / "skempi_pdb_manifest.tsv")
        except Exception as exc:  # noqa: BLE001
            blockers.append("skempi")
            rows.append(_blocked_row("skempi", root / "data/raw/skempi", str(exc)))
            logger.error("SKEMPI step blocked: %s", exc)

    if "rcsb" in config.datasets and "skempi" not in config.datasets:
        logger = _make_logger("download_skempi", results_dir / "download_skempi.log")
        try:
            skempi_path = root / "data/raw/skempi/skempi_v2.csv"
            if not skempi_path.exists():
                skempi_url = _env_url("COMPLEXVAR_SKEMPI_URL", "skempi_csv")
                rows.append(
                    _download_file(
                        dataset="skempi",
                        url=skempi_url,
                        destination=skempi_path,
                        manifest_lookup=manifest_lookup,
                        skip_existing=config.skip_existing,
                        force=config.force,
                        logger=logger,
                        source_record="skempi_2.0",
                    )
                )
            pdb_rows = _download_rcsb_pdbs(
                pdb_ids=_parse_skempi_pdb_ids(skempi_path),
                out_dir=root / "data/raw/skempi/pdbs",
                manifest_lookup=manifest_lookup,
                skip_existing=config.skip_existing,
                force=config.force,
                workers=config.workers,
                limit=config.pdb_limit,
                logger=logger,
            )
            rows.extend(pdb_rows)
            write_tsv(
                pd.DataFrame(
                    [
                        {
                            "pdb_id": Path(row["local_path"]).stem,
                            "source_url": row["source_url"],
                            "local_path": row["local_path"],
                            "status": row["status"],
                            "size_bytes": row["size_bytes"],
                            "md5": row["md5"],
                        }
                        for row in pdb_rows
                    ]
                ),
                manifests_dir / "skempi_pdb_manifest.tsv",
            )
        except Exception as exc:  # noqa: BLE001
            blockers.append("rcsb")
            rows.append(_blocked_row("rcsb", root / "data/raw/skempi/pdbs", str(exc)))
            logger.error("RCSB step blocked: %s", exc)

    if "intact" in config.datasets:
        logger = _make_logger("download_intact", results_dir / "download_intact.log")
        try:
            intact_zip_url = _env_url(
                "COMPLEXVAR_INTACT_MICLUSTER_URL", "intact_micluster"
            )
            intact_zip_path = root / "data/raw/intact/intact-micluster.zip"
            zip_row = _download_file(
                dataset="intact",
                url=intact_zip_url,
                destination=intact_zip_path,
                manifest_lookup=manifest_lookup,
                skip_existing=config.skip_existing,
                force=config.force,
                logger=logger,
                source_record="intact_psimitab",
            )
            rows.append(zip_row)
            extracted_paths = _extract_zip(
                intact_zip_path,
                root / "data/raw/intact/psimitab",
                logger=logger,
            )
            for path in extracted_paths:
                rows.append(
                    _record_row(
                        dataset="intact",
                        source_url=intact_zip_url,
                        local_path=path,
                        status="extracted",
                        source_record="intact_psimitab",
                    )
                )
            page_url = _env_url(
                "COMPLEXVAR_INTACT_DATASETS_PAGE", "intact_dataset_page"
            )
            mutation_dir = root / "data/raw/intact/mutations"
            mutation_dir.mkdir(parents=True, exist_ok=True)
            mutation_rows: list[dict[str, Any]] = []
            mutation_rows.append(
                _blocked_row("intact", mutation_dir, "mutation_export_pending")
            )
            try:
                html = _fetch_text(page_url)
                links = _discover_intact_mutation_links(html, page_url)
                if not links:
                    links = [
                        _env_url(
                            "COMPLEXVAR_INTACT_MUTATIONS_FALLBACK",
                            "intact_mutations_fallback",
                        )
                    ]
                mutation_rows = []
                for index, link in enumerate(links):
                    name = Path(link).name or f"mutations_{index}.tsv"
                    mutation_rows.append(
                        _download_file(
                            dataset="intact",
                            url=link,
                            destination=mutation_dir / name,
                            manifest_lookup=manifest_lookup,
                            skip_existing=config.skip_existing,
                            force=config.force,
                            logger=logger,
                            source_record="intact_mutations",
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                fallback_url = _env_url(
                    "COMPLEXVAR_INTACT_MUTATIONS_FALLBACK",
                    "intact_mutations_fallback",
                )
                logger.warning("dedicated mutation export discovery failed: %s", exc)
                mutation_rows = [
                    _download_file(
                        dataset="intact",
                        url=fallback_url,
                        destination=mutation_dir / Path(fallback_url).name,
                        manifest_lookup=manifest_lookup,
                        skip_existing=config.skip_existing,
                        force=config.force,
                        logger=logger,
                        source_record="intact_mutations_fallback",
                        notes="page_discovery_failed",
                    )
                ]
            rows.extend(mutation_rows)
            write_tsv(
                pd.DataFrame([zip_row, *mutation_rows]),
                manifests_dir / "intact_download_manifest.tsv",
            )
        except Exception as exc:  # noqa: BLE001
            blockers.append("intact")
            rows.append(_blocked_row("intact", root / "data/raw/intact", str(exc)))
            logger.error("IntAct step blocked: %s", exc)

    high_conf_manifest_path: Path | None = None
    if "burke" in config.datasets or "alphafold" in config.datasets:
        logger = _make_logger("download_burke", results_dir / "download_burke.log")
        try:
            summary_url = _env_url("COMPLEXVAR_BURKE_SUMMARY_URL", "burke_summary_csv")
            summary_path = root / "data/raw/burke/table_AF2_HURI_HuMap_UNIQUE.csv"
            rows.append(
                _download_file(
                    dataset="burke",
                    url=summary_url,
                    destination=summary_path,
                    manifest_lookup=manifest_lookup,
                    skip_existing=config.skip_existing,
                    force=config.force,
                    logger=logger,
                    source_record="burke_summary_csv",
                )
            )
            high_conf_manifest_path = _build_burke_high_conf_manifest(
                summary_path, manifests_dir
            )
        except Exception as exc:  # noqa: BLE001
            blockers.append("burke_summary")
            rows.append(_blocked_row("burke", root / "data/raw/burke", str(exc)))
            logger.error("Burke summary step blocked: %s", exc)

    if "burke" in config.datasets:
        logger = _make_logger("download_burke", results_dir / "download_burke.log")
        payload, source_notes = _resolve_burke_sources(logger)
        record_json_path = root / "data/raw/burke/record_7505985.json"
        write_json(payload, record_json_path)
        rows.append(
            _record_row(
                dataset="burke",
                source_url=_env_url(
                    "COMPLEXVAR_BURKE_FIGSHARE_API", "burke_figshare_api"
                ),
                local_path=record_json_path,
                status="downloaded",
                source_record="burke_figshare_record",
                notes=";".join(source_notes),
            )
        )
        archive_rows = _select_burke_archive_rows(payload)
        write_tsv(
            pd.DataFrame(archive_rows),
            manifests_dir / "burke_files_manifest.tsv",
        )
        required_bytes = sum(row["size"] for row in archive_rows)
        free_bytes = _free_space_bytes(root)
        min_free_bytes = _env_int(
            "COMPLEXVAR_MIN_FREE_SPACE_BYTES", MIN_FREE_SPACE_BYTES
        )
        if free_bytes < required_bytes + min_free_bytes:
            note = (
                f"insufficient_free_space_required={required_bytes}"
                f"_available={free_bytes}"
            )
            blockers.append("burke_archives")
            for row in archive_rows:
                rows.append(
                    {
                        "dataset": "burke",
                        "source_url": row["url"],
                        "local_path": _local_path_str(
                            root / "data/raw/burke/archives" / row["name"]
                        ),
                        "status": "blocked",
                        "size_bytes": row["size"],
                        "md5": "",
                        "downloaded_at_utc": _now_utc(),
                        "source_record": "burke_figshare_record",
                        "notes": note,
                    }
                )
            logger.error("Burke archive download blocked by free space")
        else:
            archives_dir = root / "data/raw/burke/archives"
            archives_dir.mkdir(parents=True, exist_ok=True)
            downloaded_archives: list[dict[str, Any]] = []
            for row in archive_rows:
                downloaded_archives.append(
                    _download_file(
                        dataset="burke",
                        url=row["url"],
                        destination=archives_dir / row["name"],
                        manifest_lookup=manifest_lookup,
                        skip_existing=config.skip_existing,
                        force=config.force,
                        logger=logger,
                        source_record="burke_figshare_record",
                    )
                )
            rows.extend(downloaded_archives)
            if (
                config.extract_high_confidence_only
                and high_conf_manifest_path is not None
            ):
                high_conf_frame, _, _ = _read_high_conf_pdb_targets(
                    high_conf_manifest_path
                )
                extracted_paths = _extract_burke_high_confidence(
                    archives_dir=archives_dir,
                    high_conf_frame=high_conf_frame,
                    output_dir=root / "data/raw/burke/pdbs/high_confidence",
                    logger=logger,
                )
                for path in extracted_paths:
                    rows.append(
                        _record_row(
                            dataset="burke",
                            source_url="burke_figshare_record",
                            local_path=path,
                            status="extracted",
                            source_record="burke_high_confidence",
                        )
                    )

    if "clinvar" in config.datasets:
        logger = _make_logger("download_clinvar", results_dir / "download_clinvar.log")
        try:
            clinvar_dir = root / "data/raw/clinvar"
            clinvar_rows = [
                _download_file(
                    dataset="clinvar",
                    url=_env_url("COMPLEXVAR_CLINVAR_VCF_URL", "clinvar_vcf"),
                    destination=clinvar_dir / "clinvar.vcf.gz",
                    manifest_lookup=manifest_lookup,
                    skip_existing=config.skip_existing,
                    force=config.force,
                    logger=logger,
                    source_record="clinvar",
                ),
                _download_file(
                    dataset="clinvar",
                    url=_env_url("COMPLEXVAR_CLINVAR_SUMMARY_URL", "clinvar_summary"),
                    destination=clinvar_dir / "variant_summary.txt.gz",
                    manifest_lookup=manifest_lookup,
                    skip_existing=config.skip_existing,
                    force=config.force,
                    logger=logger,
                    source_record="clinvar",
                ),
            ]
            try:
                clinvar_rows.append(
                    _download_file(
                        dataset="clinvar",
                        url=_env_url(
                            "COMPLEXVAR_CLINVAR_VCF_TBI_URL", "clinvar_vcf_tbi"
                        ),
                        destination=clinvar_dir / "clinvar.vcf.gz.tbi",
                        manifest_lookup=manifest_lookup,
                        skip_existing=config.skip_existing,
                        force=config.force,
                        logger=logger,
                        source_record="clinvar",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ClinVar VCF index unavailable: %s", exc)
            rows.extend(clinvar_rows)
            write_tsv(
                pd.DataFrame(clinvar_rows),
                manifests_dir / "clinvar_download_manifest.tsv",
            )
        except Exception as exc:  # noqa: BLE001
            blockers.append("clinvar")
            rows.append(_blocked_row("clinvar", root / "data/raw/clinvar", str(exc)))
            logger.error("ClinVar step blocked: %s", exc)

    if "alphafold" in config.datasets:
        logger = _make_logger(
            "download_alphafold", results_dir / "download_alphafold.log"
        )
        try:
            min_alphafold_free_bytes = _env_int(
                "COMPLEXVAR_MIN_ALPHAFOLD_FREE_BYTES", MIN_ALPHAFOLD_FREE_BYTES
            )
            if _free_space_bytes(root) < min_alphafold_free_bytes:
                raise RuntimeError(
                    "insufficient_free_space_for_alphafold "
                    f"free={_free_space_bytes(root)}"
                )
            if high_conf_manifest_path is None or not high_conf_manifest_path.exists():
                raise RuntimeError("burke_high_confidence_complexes.tsv is required")
            _, _, accessions = _read_high_conf_pdb_targets(high_conf_manifest_path)
            api_rows, pdb_rows = _download_alphafold_monomers(
                accessions=list(accessions),
                manifest_lookup=manifest_lookup,
                skip_existing=config.skip_existing,
                force=config.force,
                workers=config.workers,
                limit=config.monomer_limit,
                logger=logger,
            )
            rows.extend(api_rows)
            rows.extend(pdb_rows)
            write_tsv(
                pd.DataFrame(
                    [
                        {
                            "accession": Path(row["local_path"]).stem,
                            "source_url": row["source_url"],
                            "local_path": row["local_path"],
                            "status": row["status"],
                            "size_bytes": row["size_bytes"],
                            "md5": row["md5"],
                        }
                        for row in pdb_rows
                    ]
                ),
                manifests_dir / "alphafold_monomer_manifest.tsv",
            )
        except Exception as exc:  # noqa: BLE001
            blockers.append("alphafold")
            rows.append(
                _blocked_row(
                    "alphafold", root / "data/raw/alphafold_monomers", str(exc)
                )
            )
            logger.error("AlphaFold monomer step blocked: %s", exc)

    _write_combined_manifest(rows, manifest_path)
    _write_checksums(rows, manifests_dir / "checksums.md5")
    summary = {
        "datasets": config.datasets,
        "rows_written": len(rows),
        "blockers": blockers,
        "manifest_path": _local_path_str(manifest_path),
        "checksums_path": _local_path_str(manifests_dir / "checksums.md5"),
        "free_space_bytes": _free_space_bytes(root),
    }
    write_json(summary, manifests_dir / "download_summary.json")
    if blockers:
        raise RuntimeError("step_1_blocked:" + ",".join(blockers))
    return summary
