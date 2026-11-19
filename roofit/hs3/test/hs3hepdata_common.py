"""Access to the HS3 documents on HEPData, for the HEPData test and its fixture generator.

HEPData lists the HS3 documents of a record as resources of the type "HS3". A
resource is an HS3 document, or an archive of HS3 documents. The functions here
ask HEPData for these resources, download them into a cache directory, and lay
the documents out as a fixtures directory of the HS3 test suite:

    <root>/manifest.json
    <root>/fixtures/<test_id>/hs3.json       the document from HEPData, patched if
                                             hs3hepdata/patches has a patch for it
    <root>/fixtures/<test_id>/metadata.json  from roofit/hs3/test/hs3hepdata
    <root>/fixtures/<test_id>/expected.json  from roofit/hs3/test/hs3hepdata

HEPData is accessed with its own client, the Python package hepdata-cli: HEPData
refuses requests of other scripted clients.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

HEPDATA_URL = "https://www.hepdata.net"
SEARCH_QUERY = "analysis:hs3"
# HEPData is asked for one page of records and one resource at a time, with a
# pause between the resources.
PAGE_SIZE = 25
PAUSE_SECONDS = 1.0
# The list of the documents that have fixtures, in the fixtures directory of the source tree.
DOCUMENTS_FILE = "hepdata.json"
# Patches of documents, <test_id>.json in this directory of the fixtures directory: {"reason": ..., "patch": [...]},
# with a JSON Patch (RFC 6902) of the document.
PATCHES_DIR = "patches"


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True)
    # Lists of numbers and strings on one line, to keep the generated files short.
    scalar = r'(?:-?\d[\d.eE+-]*|"[^"\n]*"|true|false|null)'
    text = re.sub(
        rf"\[\s+({scalar}(?:,\s+{scalar})*)\s+\]",
        lambda match: "[" + re.sub(r",\s+", ", ", match.group(1)) + "]",
        text,
    )
    # ... and lists of such lists, like the points of a scan.
    text = re.sub(
        r"\[\s+((?:\[[^\[\]\n]*\])(?:,\s+\[[^\[\]\n]*\])*)\s+\]",
        lambda match: "[" + re.sub(r"\],\s+\[", "], [", match.group(1)) + "]",
        text,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def hepdata_client():
    """The modules of hepdata-cli that are used to access HEPData."""
    # hepdata-cli replaces sys.excepthook when it is imported, which is undone here.
    excepthook = sys.excepthook
    try:
        import hepdata_cli.api
        import hepdata_cli.resilient_requests
    finally:
        sys.excepthook = excepthook
    return hepdata_cli.api, hepdata_cli.resilient_requests


def fetch(url: str) -> bytes:
    return hepdata_client()[1].resilient_requests("get", url, allow_redirects=True).content


def safe_name(name: str) -> str:
    """A file name from HEPData as a single path component."""
    name = re.sub(r"[^A-Za-z0-9._+-]", "_", name).lstrip(".")
    return name or "resource"


def query_resources() -> list[dict[str, Any]]:
    """Ask HEPData for the HS3 resources of all records that have any.

    Returns a list of dictionaries with the keys "inspire_id", "resource_id",
    "filename" and "title" (of the record), sorted by record and resource.
    """
    resources = {}
    for record in hepdata_client()[0].Client().find(SEARCH_QUERY, matches_per_page=PAGE_SIZE):
        for analysis in record.get("analyses", []):
            if str(analysis.get("type", "")).lower() != "hs3":
                continue
            match = re.search(r"/record/resource/(\d+)", str(analysis.get("analysis", "")))
            if not match:
                continue
            resource_id = int(match.group(1))
            resources[resource_id] = dict(
                inspire_id=str(record.get("inspire_id")),
                resource_id=resource_id,
                filename=safe_name(str(analysis.get("filename") or resource_id)),
                title=str(record.get("title") or ""),
            )
    return sorted(resources.values(), key=lambda r: (r["inspire_id"], r["resource_id"]))


def extract_archive(archive: Path, target: Path) -> None:
    """Extract the regular files of a tar or zip archive below `target`.

    The archives are not trusted: members are written by this function, only if
    they are regular files, and only to paths below `target`.
    """

    def destination(member_name):
        parts = [p for p in Path(member_name).parts if p not in ("", ".")]
        if not parts or Path(member_name).is_absolute() or ".." in parts:
            return None
        return target.joinpath(*parts)

    def write(path, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as out:
            shutil.copyfileobj(source, out, 1 << 20)

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            for member in tar:
                path = destination(member.name)
                if member.isfile() and path is not None:
                    with tar.extractfile(member) as source:
                        write(path, source)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                path = destination(member.filename)
                if not member.is_dir() and path is not None:
                    with zipped.open(member) as source:
                        write(path, source)
    else:
        raise RuntimeError(f"{archive.name} is neither a JSON document nor a tar or zip archive")


def is_archive(filename: str) -> bool:
    return filename.lower().endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz", ".zip"))


def resource_directory(cache: Path, resource: dict[str, Any]) -> Path:
    return cache / f"{resource['inspire_id']}_{resource['resource_id']}"


def prepare_resource(cache: Path, resource: dict[str, Any], offline: bool) -> bool:
    """Download and unpack a resource, unless the cache has it. Returns whether it is there."""
    directory = resource_directory(cache, resource)
    downloaded = directory / resource["filename"]
    if not downloaded.is_file():
        if offline:
            return False
        directory.mkdir(parents=True, exist_ok=True)
        partial = directory / (resource["filename"] + ".part")
        partial.write_bytes(fetch(f"{HEPDATA_URL}/record/resource/{resource['resource_id']}?view=true"))
        partial.rename(downloaded)
        time.sleep(PAUSE_SECONDS)
    if is_archive(resource["filename"]):
        marker = directory / "extracted.done"
        if not marker.is_file():
            extract_archive(downloaded, directory / "extracted")
            marker.touch()
    return True


def is_hs3_document(path: Path) -> bool:
    """Whether the file is a JSON object with HS3 metadata. Archives can contain other JSON files."""
    try:
        document = load_json(path)
    except (OSError, ValueError):
        return False
    return isinstance(document, dict) and "hs3_version" in (document.get("metadata") or {})


def resource_documents(cache: Path, resource: dict[str, Any]) -> list[tuple[str, Path]]:
    """The HS3 documents of a resource that is in the cache, as (member, path).

    The member is the path inside the archive, and empty for a resource that is a document.
    """
    directory = resource_directory(cache, resource)
    if not is_archive(resource["filename"]):
        return [("", directory / resource["filename"])]
    extracted = directory / "extracted"
    return [
        (path.relative_to(extracted).as_posix(), path)
        for path in sorted(extracted.rglob("*.json"))
        if path.is_file() and is_hs3_document(path)
    ]


def make_test_id(resource: dict[str, Any], member: str) -> str:
    """The identifier of a document in the fixtures, e.g. ins2784411_r3741768_likelihood."""
    name = member or resource["filename"]
    name = re.sub(r"\.json$", "", name)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return f"ins{resource['inspire_id']}_r{resource['resource_id']}_{name}"


def document_path(cache: Path, document: dict[str, Any]) -> Path:
    """Where the document of an entry of hepdata.json is in the cache."""
    directory = resource_directory(cache, document)
    if document.get("member"):
        return directory / "extracted" / document["member"]
    return directory / document["filename"]


def prepare_documents(cache: Path, documents: list[dict[str, Any]], offline: bool) -> list[dict[str, Any]]:
    """Download the resources of the documents. Returns the documents that are in the cache."""
    prepared: dict[int, bool] = {}
    available = []
    for document in documents:
        resource_id = document["resource_id"]
        if resource_id not in prepared:
            prepared[resource_id] = prepare_resource(cache, document, offline)
        if prepared[resource_id] and document_path(cache, document).is_file():
            available.append(document)
    return available


def _pointer(document: Any, pointer: str) -> tuple[Any, str | int]:
    """The container and the key of the location of a JSON Pointer (RFC 6901)."""
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer {pointer!r}")
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]
    for token in tokens[:-1]:
        document = document[int(token) if isinstance(document, list) else token]
    key = tokens[-1]
    if isinstance(document, list):
        return document, (len(document) if key == "-" else int(key))
    return document, key


def apply_json_patch(document: Any, operations: list[dict[str, Any]]) -> None:
    """Apply a JSON Patch (RFC 6902) to the document, in place. The operations "add", "remove", "replace" and
    "test" are supported. A failed "test" is an error, so a patch can state what it expects to change."""
    for i, operation in enumerate(operations):
        op, path = operation["op"], operation["path"]
        try:
            container, key = _pointer(document, path)
            if op == "test":
                if container[key] != operation["value"]:
                    raise ValueError(f"is {container[key]!r}, not {operation['value']!r}")
            elif op == "replace":
                container[key]  # the location must exist
                container[key] = operation["value"]
            elif op == "remove":
                del container[key]
            elif op == "add":
                if isinstance(container, list):
                    container.insert(key, operation["value"])
                else:
                    container[key] = operation["value"]
            else:
                raise ValueError(f"unsupported operation {op!r}")
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ValueError(f"operation {i} ({op} {path}) of the patch failed: {e}") from None


def document_patch(fixtures: Path, test_id: str) -> dict[str, Any] | None:
    """The patch of a document, or None."""
    path = fixtures / PATCHES_DIR / f"{test_id}.json"
    return load_json(path) if path.is_file() else None


def staged_document(cache: Path, fixtures: Path, document: dict[str, Any]) -> Path:
    """The document as the checks and fits read it: the one in the cache, or, if the fixtures have a patch of
    it, the patched document, which is written to <cache>/patched."""
    original = document_path(cache, document)
    patch = document_patch(fixtures, document["test_id"])
    if patch is None:
        return original
    payload = load_json(original)
    apply_json_patch(payload, patch["patch"])
    target = cache / "patched" / f"{document['test_id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temporary file first, since the fits of the generator run in parallel processes.
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(temporary, target)
    return target


def link_or_copy(source: Path, target: Path) -> None:
    if target.is_symlink() or target.exists():
        target.unlink()
    try:
        os.symlink(source, target)
    except OSError:
        shutil.copyfile(source, target)


def assemble_suite_root(root: Path, fixtures: Path, cache: Path, documents: list[dict[str, Any]]) -> None:
    """Lay the documents out as a fixtures directory of the HS3 test suite.

    The manifest has the entries of the documents only, since the runner of the
    suite stops if the document of an entry is missing.

    The hashes in the manifest of the fixtures pin the documents as they are on HEPData. A document with a
    patch is patched if it has these hashes, and the manifest of the suite gets the hashes of the patched
    document. Otherwise, the document is left as it is, and the suite reports the mismatch.
    """
    from hs3suite.manifest import canonical_sha256, raw_sha256

    if root.exists():
        shutil.rmtree(root)
    manifest = load_json(fixtures / "manifest.json")
    wanted = {document["test_id"]: document for document in documents}
    entries = [entry for entry in manifest["fixtures"] if entry["test_id"] in wanted]
    for i, entry in enumerate(entries):
        target = root / entry["path"]
        target.mkdir(parents=True)
        document = wanted[entry["test_id"]]
        source = document_path(cache, document)
        pinned = entry["hashes"]
        if document_patch(fixtures, entry["test_id"]) is not None and raw_sha256(source) == pinned["sha256"]:
            source = staged_document(cache, fixtures, document)
            hashes = {"sha256": raw_sha256(source), "canonical_sha256": canonical_sha256(source)}
            entries[i] = dict(entry, hashes=hashes)
        link_or_copy(source, target / "hs3.json")
        for name in ("metadata.json", "expected.json"):
            link_or_copy(fixtures / entry["path"] / name, target / name)
    manifest["fixtures"] = entries
    write_json(root / "manifest.json", manifest)
