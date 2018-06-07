import py, os, sys
from pytest import raises
from .support import setup_make


class TestBOOSTANY:
    def setup_class(cls):
        import cppyy
        cppyy.include('boost/any.hpp')

    def test01_any_class(self):
        """Availability of boost::any"""

        import cppyy
        assert cppyy.gbl.boost.any

        from cppyy.gbl import std
        from cppyy.gbl.boost import any

        assert std.list(any)

    def test02_any_usage(self):
        """boost::any assignment and casting"""

        import cppyy
        assert cppyy.gbl.boost

        from cppyy.gbl import std, boost

        val = boost.any()
        val.__assign__(std.vector[int]())
        assert val.type() == cppyy.typeid(std.vector[int])

        extract = boost.any_cast[std.vector[int]](std.move(val))
        assert type(extract) is std.vector[int]
        extract += range(100)

        val.__assign__(std.move(extract))
        assert len(extract) == 0

        raises(Exception, boost.any_cast[int], std.move(val))

        extract = boost.any_cast[std.vector[int]](std.move(val))
        assert len(extract) == 100
