import pytest


class TestRentryGetPtr:
    def test_read_rntuple_entry(self):
        import ROOT

        reader = ROOT.RNTupleReader.Open("ntpl", "test_ntuple_datavector.root")
        entry = reader.CreateEntry()
        reader.LoadEntry(0, entry)
        # The test should not raise exceptions
        entry["my_field"]


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
