import pytest

from DistRDF.Backends import Spark
from DistRDF import LiveVisualize

class TestUnsupportedCall:

    def test_warning(self, connection):
        df = Spark.RDataFrame(100, sparkcontext=connection)
        h = df.Define("x", "rdfentry_").Histo1D(("name", "title", 100, 0, 10), "x")

        with pytest.warns(UserWarning, match="The live visualization feature is not supported for the Spark backend. Skipping LiveVisualize."):
            LiveVisualize([h])
            h.Draw()

    
if __name__ == "__main__":
    pytest.main(args=[__file__])
