"""Import the HS3 documents that are published on HEPData with RooFit.

HEPData lists the HS3 documents of a record as resources of the type "HS3". This
test asks HEPData for all records that have such resources, downloads them, and
checks for every HS3 document that RooFit can import it, export the result, and
import that again.

The test needs network access and downloads a few hundred MB, so it is only
enabled with the CMake option -Dtest_roofit_hs3hepdata=ON, and is not meant to
run in the CI: HEPData should not be queried on every build. Everything that
was downloaded is kept in a cache directory and not downloaded again.

HEPData is accessed with its own client, the Python package hepdata-cli:
HEPData refuses requests of other scripted clients.

Environment variables:

HS3HEPDATA_CACHE
    The cache directory (required).
HS3HEPDATA_EXPECTED_FAILURES
    The list of documents that are known to fail.
HS3HEPDATA_OFFLINE
    If set to 1, HEPData is not contacted: the test runs on what is in the cache.
HS3HEPDATA_SELECT
    A regular expression. Only documents whose identifier matches it are tested,
    and the list of expected failures is not checked for stale entries.
"""

import json
import os
import re
import sys
import tarfile
import time
import unittest
import zipfile
from pathlib import Path

HEPDATA_URL = "https://www.hepdata.net"
SEARCH_QUERY = "analysis:hs3"
# HEPData is asked for one page of records and one resource at a time, with a
# pause between the resources.
PAGE_SIZE = 25
PAUSE_SECONDS = 1.0


def load_expected_failures(path):
    """Read the list of expected failures.

    Every non-empty line that is not a comment has the form
    ``<document>  # <reason>``.
    """
    expected = {}
    if not path:
        return expected
    for line in Path(path).read_text().splitlines():
        entry, _, reason = line.partition("#")
        entry = entry.strip()
        if entry:
            expected[entry] = reason.strip()
    return expected


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


def fetch(url):
    return hepdata_client()[1].resilient_requests("get", url, allow_redirects=True).content


def safe_name(name):
    """A file name from HEPData as a single path component."""
    name = re.sub(r"[^A-Za-z0-9._+-]", "_", name).lstrip(".")
    return name or "resource"


def query_resources():
    """Ask HEPData for the HS3 resources of all records that have any.

    Returns a list of dictionaries with the keys "inspire_id", "resource_id"
    and "filename", sorted by record and resource.
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
            )
    return sorted(resources.values(), key=lambda r: (r["inspire_id"], r["resource_id"]))


def extract_archive(archive, target):
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
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)

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


def is_archive(filename):
    return filename.lower().endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz", ".zip"))


def prepare_resource(cache, resource, offline):
    """Download and unpack a resource, unless the cache has it.

    Returns the HS3 documents of the resource as a list of (identifier, path).
    """
    directory = cache / f"{resource['inspire_id']}_{resource['resource_id']}"
    downloaded = directory / resource["filename"]
    if not downloaded.is_file():
        if offline:
            return []
        directory.mkdir(parents=True, exist_ok=True)
        partial = directory / (resource["filename"] + ".part")
        partial.write_bytes(fetch(f"{HEPDATA_URL}/record/resource/{resource['resource_id']}?view=true"))
        partial.rename(downloaded)
        time.sleep(PAUSE_SECONDS)

    prefix = f"{resource['inspire_id']}/{resource['filename']}"
    if not is_archive(resource["filename"]):
        return [(prefix, downloaded)]

    extracted = directory / "extracted"
    marker = directory / "extracted.done"
    if not marker.is_file():
        extract_archive(downloaded, extracted)
        marker.touch()
    return [
        (f"{prefix}/{path.relative_to(extracted).as_posix()}", path)
        for path in sorted(extracted.rglob("*.json"))
        if path.is_file()
    ]


def collect_documents(cache, offline):
    """The HS3 documents to test, as a sorted list of (identifier, path)."""
    index = cache / "index.json"
    if offline:
        if not index.is_file():
            raise RuntimeError(f"HS3HEPDATA_OFFLINE is set, but there is no {index}")
        resources = json.loads(index.read_text())
    else:
        resources = query_resources()
        cache.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(resources, indent=1))
    documents = []
    for resource in resources:
        documents += prepare_resource(cache, resource, offline)
    return sorted(documents)


def is_hs3_document(path):
    """Whether the file is a JSON object with HS3 metadata.

    Archives can contain other JSON files, which are not tested.
    """
    try:
        with open(path) as f:
            document = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(document, dict) and "hs3_version" in (document.get("metadata") or {})


def make_test(identifier, path, expected_failure):
    def test(self):
        import ROOT

        if not is_hs3_document(path):
            self.skipTest("not an HS3 document")

        level = ROOT.RooMsgService.instance().globalKillBelow()
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)
        try:
            stage = "import"
            try:
                ws = ROOT.RooWorkspace("ws")
                if not ROOT.RooJSONFactoryWSTool(ws).importJSON(str(path)):
                    raise RuntimeError("importJSON() returned false")
                stage = "export"
                exported = str(ROOT.RooJSONFactoryWSTool(ws).exportJSONtoString())
                stage = "import of the exported document"
                ws2 = ROOT.RooWorkspace("ws2")
                if not ROOT.RooJSONFactoryWSTool(ws2).importJSONfromString(exported):
                    raise RuntimeError("importJSONfromString() returned false")
            except Exception as e:
                # The message of a C++ exception starts with the signature of the function that threw.
                message = " ".join(str(e).rpartition("=>")[2].split())
                self.fail(f"{stage} failed: {message[:1000] or type(e).__name__}")
        finally:
            ROOT.RooMsgService.instance().setGlobalKillBelow(level)

    test.__name__ = "test_" + re.sub(r"[^A-Za-z0-9]", "_", identifier)
    test.__doc__ = identifier
    # Known failures are listed in hs3hepdata_expected_failures.txt. An unexpected
    # success fails the test run, so the list has to be shrunk as soon as a
    # document starts to pass.
    if expected_failure:
        test = unittest.expectedFailure(test)
    return test


def make_test_case():
    cache_dir = os.environ.get("HS3HEPDATA_CACHE")
    offline = os.environ.get("HS3HEPDATA_OFFLINE", "") not in ("", "0")
    select = os.environ.get("HS3HEPDATA_SELECT")
    expected_failures = load_expected_failures(os.environ.get("HS3HEPDATA_EXPECTED_FAILURES"))

    namespace = dict(__module__=__name__)

    def failing(name, message):
        def test(self):
            self.fail(message)

        test.__name__ = name
        namespace[name] = test
        return type("TestHS3HEPData", (unittest.TestCase,), namespace)

    if not cache_dir:
        return failing("test_configuration", "HS3HEPDATA_CACHE is not set")
    try:
        documents = collect_documents(Path(cache_dir), offline)
    except Exception as e:
        return failing("test_hepdata_download", f"could not get the HS3 documents from HEPData: {e}")
    if not documents:
        return failing("test_hepdata_download", "HEPData lists no HS3 documents, or the cache is empty")

    selected = [d for d in documents if not select or re.search(select, d[0])]
    for identifier, path in selected:
        test = make_test(identifier, path, identifier in expected_failures)
        namespace[test.__name__] = test

    if not select:
        # Entries that don't correspond to any document (e.g. after a record was
        # updated on HEPData) would silently hide nothing, so they are reported as
        # failures.
        stale = sorted(set(expected_failures).difference(identifier for identifier, _ in documents))

        def test_expected_failures_list_is_current(self):
            self.assertEqual(stale, [], "entries in the list of expected failures that match no document")

        namespace[test_expected_failures_list_is_current.__name__] = test_expected_failures_list_is_current

    return type("TestHS3HEPData", (unittest.TestCase,), namespace)


TestHS3HEPData = make_test_case()

if __name__ == "__main__":
    unittest.main()
