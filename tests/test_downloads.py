from __future__ import annotations

import json
import logging
import os
import socketserver
import subprocess
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

import pandas as pd

from complexvar.downloads import (
    _discover_intact_mutation_links,
    _download_alphafold_monomers,
    _parse_skempi_pdb_ids,
    _select_alphafold_entry,
)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_zip(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _serve_directory(directory: Path) -> tuple[socketserver.TCPServer, str]:
    handler = partial(QuietHandler, directory=str(directory))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _sample_server_tree(root: Path) -> None:
    _write_text(
        root / "skempi_v2.csv",
        "\n".join(
            [
                "#Pdb;Mutation(s)_cleaned;Affinity_mut_parsed;Affinity_wt_parsed;Temperature",
                "1ABC_A_B;LV10P;1e-6;1e-9;298",
                "2DEF_A_B;LV11S;1e-7;1e-9;298",
            ]
        )
        + "\n",
    )
    _write_text(root / "1ABC.pdb", "HEADER 1ABC\nEND\n")
    _write_text(root / "2DEF.pdb", "HEADER 2DEF\nEND\n")
    _write_zip(root / "intact-micluster.zip", {"intact.tsv": "a\tb\n1\t2\n"})
    _write_text(root / "mutations.tsv", "col1\tcol2\nx\ty\n")
    _write_text(
        root / "intact_datasets.html",
        '<html><body><a href="/mutations.tsv">mutations</a></body></html>\n',
    )
    _write_text(root / "clinvar.vcf.gz", "fake\n")
    _write_text(root / "clinvar.vcf.gz.tbi", "index\n")
    _write_text(root / "variant_summary.txt.gz", "fake\n")
    _write_text(root / "manifest.txt", "manifest\n")
    _write_text(root / "README.txt", "readme\n")
    _write_zip(root / "HuRI.zip", {"ENSG000001-ENSG000002.pdb": "HEADER HURI\nEND\n"})
    _write_zip(root / "humap.zip", {"ENSG000003-ENSG000004.pdb": "HEADER HMAP\nEND\n"})
    _write_zip(root / "HuRI-single.zip", {"O11111.pdb": "HEADER MONO\nEND\n"})
    _write_zip(root / "HuMap-single.zip", {"Q22222.pdb": "HEADER MONO\nEND\n"})
    _write_zip(root / "random.zip", {"random.txt": "random\n"})
    _write_text(
        root / "burke_summary.csv",
        "\n".join(
            [
                "unique_ID,id1,id2,pDockQ,structure_file,Dataset",
                "O11111_Q22222,O11111,Q22222,0.9,ENSG000001-ENSG000002.pdb,HURI",
                "O33333_Q44444,O33333,Q44444,0.2,ENSG000003-ENSG000004.pdb,humap",
            ]
        )
        + "\n",
    )
    figshare = {
        "title": "Protein models",
        "doi": "10.17044/scilifelab.16945039.v1",
        "files": [
            {
                "name": "random.zip",
                "size": (root / "random.zip").stat().st_size,
                "download_url": "",
            },
            {
                "name": "humap.zip",
                "size": (root / "humap.zip").stat().st_size,
                "download_url": "",
            },
            {
                "name": "HuRI.zip",
                "size": (root / "HuRI.zip").stat().st_size,
                "download_url": "",
            },
            {
                "name": "HuRI-single.zip",
                "size": (root / "HuRI-single.zip").stat().st_size,
                "download_url": "",
            },
            {
                "name": "HuMap-single.zip",
                "size": (root / "HuMap-single.zip").stat().st_size,
                "download_url": "",
            },
            {
                "name": "manifest.txt",
                "size": (root / "manifest.txt").stat().st_size,
                "download_url": "",
            },
            {
                "name": "README.txt",
                "size": (root / "README.txt").stat().st_size,
                "download_url": "",
            },
        ],
    }
    zenodo = {"metadata": {"title": "not the expected record"}, "files": []}
    _write_text(root / "figshare.json", json.dumps(figshare))
    _write_text(root / "zenodo.json", json.dumps(zenodo))
    alphafold_payload = [
        {
            "uniprotAccession": "O11111",
            "latestVersion": 3,
            "sequenceStart": 1,
            "sequenceEnd": 150,
            "isUniProtReviewed": True,
            "isUniProtReferenceProteome": True,
            "pdbUrl": "",
        }
    ]
    _write_text(root / "alphafold_O11111.json", json.dumps(alphafold_payload))


def _bind_server_urls(root: Path, base_url: str) -> None:
    figshare_path = root / "figshare.json"
    payload = json.loads(figshare_path.read_text(encoding="utf-8"))
    for item in payload["files"]:
        item["download_url"] = f"{base_url}/{item['name']}"
    figshare_path.write_text(json.dumps(payload), encoding="utf-8")
    alphafold_path = root / "alphafold_O11111.json"
    payload = json.loads(alphafold_path.read_text(encoding="utf-8"))
    payload[0]["pdbUrl"] = f"{base_url}/O11111_af.pdb"
    alphafold_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_text(root / "O11111_af.pdb", "HEADER AF\nEND\n")


def _env(base_url: str) -> dict[str, str]:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src)
    env["COMPLEXVAR_SKEMPI_URL"] = f"{base_url}/skempi_v2.csv"
    env["COMPLEXVAR_RCSB_PDB_BASE"] = base_url
    env["COMPLEXVAR_INTACT_MICLUSTER_URL"] = f"{base_url}/intact-micluster.zip"
    env["COMPLEXVAR_INTACT_DATASETS_PAGE"] = f"{base_url}/intact_datasets.html"
    env["COMPLEXVAR_INTACT_MUTATIONS_FALLBACK"] = f"{base_url}/mutations.tsv"
    env["COMPLEXVAR_BURKE_ZENODO_API"] = f"{base_url}/zenodo.json"
    env["COMPLEXVAR_BURKE_FIGSHARE_API"] = f"{base_url}/figshare.json"
    env["COMPLEXVAR_BURKE_SUMMARY_URL"] = f"{base_url}/burke_summary.csv"
    env["COMPLEXVAR_CLINVAR_VCF_URL"] = f"{base_url}/clinvar.vcf.gz"
    env["COMPLEXVAR_CLINVAR_VCF_TBI_URL"] = f"{base_url}/clinvar.vcf.gz.tbi"
    env["COMPLEXVAR_CLINVAR_SUMMARY_URL"] = f"{base_url}/variant_summary.txt.gz"
    env["COMPLEXVAR_ALPHAFOLD_API_BASE"] = f"{base_url}/alphafold"
    return env


def _run_script(
    workdir: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parents[1] / "scripts/download/download_all.sh"
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=workdir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_parse_skempi_pdb_ids(tmp_path: Path):
    path = tmp_path / "skempi.csv"
    _write_text(
        path,
        "\n".join(
            [
                "#Pdb;Mutation(s)_cleaned;Affinity_mut_parsed;Affinity_wt_parsed;Temperature",
                "1ABC_A_B;LV10P;1e-6;1e-9;298",
                "2def_A_B;LV11S;1e-7;1e-9;298",
            ]
        )
        + "\n",
    )
    assert _parse_skempi_pdb_ids(path) == ["1ABC", "2DEF"]


def test_discover_intact_mutation_links():
    html = '<a href="/exports/mutations.tsv">mut</a><a href="/other.tsv">other</a>'
    links = _discover_intact_mutation_links(html, "https://example.org")
    assert links == ["https://www.ebi.ac.uk/exports/mutations.tsv"]


def test_select_alphafold_entry_prefers_latest_longest():
    entries = [
        {
            "uniprotAccession": "O11111",
            "latestVersion": 1,
            "sequenceStart": 1,
            "sequenceEnd": 50,
        },
        {
            "uniprotAccession": "O11111",
            "latestVersion": 2,
            "sequenceStart": 1,
            "sequenceEnd": 200,
        },
    ]
    selected = _select_alphafold_entry(entries, "O11111")
    assert selected["latestVersion"] == 2


def test_download_alphafold_monomers_records_missing_accession(
    tmp_path: Path, monkeypatch
):
    server_root = tmp_path / "server"
    _sample_server_tree(server_root)
    server, base_url = _serve_directory(server_root)
    try:
        _bind_server_urls(server_root, base_url)
        alphafold_dir = server_root / "alphafold"
        alphafold_dir.mkdir(parents=True, exist_ok=True)
        (alphafold_dir / "O11111").write_text(
            (server_root / "alphafold_O11111.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("COMPLEXVAR_ALPHAFOLD_API_BASE", f"{base_url}/alphafold")
        rows, pdb_rows = _download_alphafold_monomers(
            accessions=["O11111", "MISSING"],
            manifest_lookup={},
            skip_existing=True,
            force=False,
            workers=2,
            limit=None,
            logger=logging.getLogger("test.alphafold"),
        )
        assert any(row["status"] == "blocked" for row in rows)
        assert any(row["status"] == "downloaded" for row in rows)
        assert len(pdb_rows) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_download_all_skempi_and_intact(tmp_path: Path):
    server_root = tmp_path / "server"
    _sample_server_tree(server_root)
    server, base_url = _serve_directory(server_root)
    try:
        _bind_server_urls(server_root, base_url)
        env = _env(base_url)
        _run_script(tmp_path, env, "--datasets", "skempi,intact", "--root", ".")
        manifest = pd.read_csv(
            tmp_path / "data/manifests/download_manifest.tsv", sep="\t"
        )
        assert {"skempi", "intact", "rcsb"}.issubset(set(manifest["dataset"]))
        assert (tmp_path / "data/raw/skempi/skempi_v2.csv").exists()
        assert (tmp_path / "data/raw/intact/intact-micluster.zip").exists()
        _run_script(tmp_path, env, "--datasets", "skempi,intact", "--root", ".")
        manifest = pd.read_csv(
            tmp_path / "data/manifests/download_manifest.tsv", sep="\t"
        )
        assert "skipped" in set(manifest["status"])
    finally:
        server.shutdown()
        server.server_close()


def test_download_redownloads_unverified_existing_file(tmp_path: Path):
    server_root = tmp_path / "server"
    _sample_server_tree(server_root)
    server, base_url = _serve_directory(server_root)
    try:
        _bind_server_urls(server_root, base_url)
        env = _env(base_url)
        destination = tmp_path / "data/raw/skempi/skempi_v2.csv"
        _write_text(destination, "partial\n")
        _run_script(tmp_path, env, "--datasets", "skempi", "--root", ".")
        content = destination.read_text(encoding="utf-8")
        assert "#Pdb;Mutation(s)_cleaned" in content
        backups = list(destination.parent.glob("skempi_v2.csv.unverified.*"))
        assert backups
    finally:
        server.shutdown()
        server.server_close()


def test_download_all_rcsb_limit(tmp_path: Path):
    server_root = tmp_path / "server"
    _sample_server_tree(server_root)
    server, base_url = _serve_directory(server_root)
    try:
        _bind_server_urls(server_root, base_url)
        env = _env(base_url)
        env["COMPLEXVAR_MIN_FREE_SPACE_BYTES"] = "0"
        _run_script(
            tmp_path,
            env,
            "--datasets",
            "rcsb",
            "--pdb-limit",
            "1",
            "--root",
            ".",
        )
        pdb_files = sorted((tmp_path / "data/raw/skempi/pdbs").glob("*.pdb"))
        assert len(pdb_files) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_download_all_burke_high_confidence(tmp_path: Path):
    server_root = tmp_path / "server"
    _sample_server_tree(server_root)
    server, base_url = _serve_directory(server_root)
    try:
        _bind_server_urls(server_root, base_url)
        env = _env(base_url)
        env["COMPLEXVAR_MIN_FREE_SPACE_BYTES"] = "0"
        _run_script(
            tmp_path,
            env,
            "--datasets",
            "burke",
            "--extract-high-confidence-only",
            "--root",
            ".",
        )
        manifest = pd.read_csv(
            tmp_path / "data/manifests/burke_high_confidence_complexes.tsv",
            sep="\t",
        )
        assert int(manifest["is_high_confidence"].sum()) == 1
        extracted = (
            tmp_path / "data/raw/burke/pdbs/high_confidence/ENSG000001-ENSG000002.pdb"
        )
        assert extracted.exists()
    finally:
        server.shutdown()
        server.server_close()
