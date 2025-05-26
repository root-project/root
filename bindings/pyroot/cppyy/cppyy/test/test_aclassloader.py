import py, pytest, os
from pytest import raises, mark
from support import setup_make, IS_MAC

currpath = os.getcwd()
test_dct = currpath + "/libexample01Dict"


class TestACLASSLOADER:

    def setup_class(cls):
        import cppyy

    @mark.xfail(condition=IS_MAC, reason="Fails on OSX")
    def test01_class_autoloading(self):
        """Test whether a class can be found through .rootmap."""
        import cppyy
        example01_class = cppyy.gbl.example01
        assert example01_class
        cl2 = cppyy.gbl.example01
        assert cl2
        assert example01_class is cl2


if __name__ == "__main__":
    exit(pytest.main(args=['-sv', '-ra', __file__]))
