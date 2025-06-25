import py, pytest, os
from pytest import mark, raises
from support import setup_make


currpath = os.getcwd()
test_dct = currpath + "/libstd_streamsDict"


class TestSTDStreams:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.streams = cppyy.load_reflection_info(cls.test_dct)

    def test01_std_ostream(self):
        """Test availability of std::ostream"""

        import cppyy

        assert cppyy.gbl.std is cppyy.gbl.std
        assert cppyy.gbl.std.ostream is cppyy.gbl.std.ostream

        assert callable(cppyy.gbl.std.ostream)

    def test02_std_cout(self):
        """Test access to std::cout"""

        import cppyy

        assert not (cppyy.gbl.std.cout is None)

    def test03_consistent_naming_if_char_traits(self):
        """Naming consistency if char_traits"""

        import cppyy

        cppyy.cppdef("""\
        namespace stringstream_base {
        void pass_through_base(std::ostream& o) {
            o << "TEST STRING";
        } }""")

        s = cppyy.gbl.std.ostringstream();
      # base class used to fail to match
        cppyy.gbl.stringstream_base.pass_through_base(s)
        assert s.str() == "TEST STRING"

    @mark.xfail()
    def test04_naming_of_ostringstream(self):
        """Naming consistency of ostringstream"""

        import cppyy

        short_type = cppyy.gbl.CppyyLegacy.TClassEdit.ShortType
        s0 = short_type("std::basic_ostringstream<char>", 2)
        s1 = short_type("std::basic_ostringstream<char, std::char_traits<char> >", 2)
        s2 = short_type("std::basic_ostringstream<char,struct std::char_traits<char> >", 2)
        s3 = short_type("std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >", 2)
        s4 = short_type("std::basic_ostringstream<char,struct std::char_traits<char>, std::allocator<char> >", 2)

        assert s1 == s0
        assert s2 == s0
        assert s3 == s0
        assert s4 == s0

        get_class = cppyy.gbl.CppyyLegacy.TClass.GetClass
        cl0 = get_class("std::ostringstream")
        cl1 = get_class("std::basic_ostringstream<char>")
        cl2 = get_class("std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >")

        assert cl0 == cl1
        assert cl1 == cl2
        assert cl2 == cl0


if __name__ == "__main__":
    exit(pytest.main(args=['-sv', '-ra', __file__]))
