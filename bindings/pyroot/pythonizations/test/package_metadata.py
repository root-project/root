import importlib.metadata as meta
import unittest


class PackageMetadata(unittest.TestCase):
    def test_query_metadata(self):
        try:
            meta.version("ROOT")
        except Exception:
            raise AssertionError("importlib failed to access .dist-info metadata for ROOT package")
