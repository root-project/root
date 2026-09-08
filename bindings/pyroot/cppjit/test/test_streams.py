import py
from support import setup_make

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/std_streamsDict"))


def setup_module(mod):
    setup_make("std_streams")


class TestSTDStreams:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        cls.streams = cppjit.load_reflection_info(cls.test_dct)

    def test01_std_ostream(self):
        """Test availability of std::ostream"""

        import cppjit

        assert cppjit.gbl.std is cppjit.gbl.std
        assert cppjit.gbl.std.ostream is cppjit.gbl.std.ostream

        assert callable(cppjit.gbl.std.ostream)

    def test02_std_cout(self):
        """Test access to std::cout"""

        import cppjit

        assert cppjit.gbl.std.cout is not None

    def test03_consistent_naming_if_char_traits(self):
        """Naming consistency if char_traits"""

        import cppjit

        cppjit.cppdef("""\
        namespace stringstream_base {
        void pass_through_base(std::ostream& o) {
            o << "TEST STRING";
        } }""")

        s = cppjit.gbl.std.ostringstream()
        # base class used to fail to match
        cppjit.gbl.stringstream_base.pass_through_base(s)
        assert s.str() == "TEST STRING"

    def test04_naming_of_ostringstream(self):
        """Naming consistency of ostringstream"""

        import cppjit

        # Check if the object created is equal in all three cases
        cl0 = cppjit.gbl.std.ostringstream
        cl1 = cppjit.gbl.std.basic_ostringstream["char"]
        cl2 = cppjit.gbl.std.basic_ostringstream[
            "char", cppjit.gbl.std.char_traits["char"], cppjit.gbl.std.allocator["char"]
        ]

        assert cl0 == cl1
        assert cl1 == cl2
        assert cl2 == cl0
