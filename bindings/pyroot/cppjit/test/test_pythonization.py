import sys

import py
from pytest import mark, raises
from support import (
    IS_CLANG_REPL,
    IS_LINUX_ARM,
    IS_MAC,
    IS_VALGRIND,
    setup_make,
)

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/pythonizablesDict"))


def setup_module(mod):
    setup_make("pythonizables")


class TestClassPYTHONIZATION:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        cls.pyzables = cppjit.load_reflection_info(cls.test_dct)

    @mark.xfail(condition=IS_MAC, reason="Fails on OSX")
    def test00_api(self):
        """Test basic semantics of the pythonization API"""

        import cppjit

        raises(TypeError, cppjit.py.add_pythonization, 1)

        def pythonizor1(klass, name):
            pass

        def pythonizor2(klass, name):
            pass

        pythonizor3 = pythonizor1

        cppjit.py.add_pythonization(pythonizor1)
        assert cppjit.py.remove_pythonization(pythonizor2) == False
        assert cppjit.py.remove_pythonization(pythonizor3) == True

        def pythonizor(klass, name):
            if name == "pyzables::SomeDummy1":
                klass.test = 1

        cppjit.py.add_pythonization(pythonizor)
        assert cppjit.gbl.pyzables.SomeDummy1.test == 1

        def pythonizor(klass, name):
            if name == "SomeDummy2":
                klass.test = 2

        cppjit.py.add_pythonization(pythonizor, "pyzables")

        # global pythonizors are still run even if namespaced ones available
        def pythonizor(klass, name):
            if name == "pyzables::SomeDummy2":
                klass.test = 3

        cppjit.py.add_pythonization(pythonizor)

        assert cppjit.gbl.pyzables.SomeDummy2.test == 3

        cppjit.cppdef("""
            namespace pyzables {
                class TObjString {
                public:
                    std::string s;
                    TObjString(std::string ss) : s(ss) {}
                    size_t Sizeof() { return s.size() + 1; }
                };
            }
        """)

        def root_pythonizor(klass, name):
            if name == "pyzables::TObjString":
                klass.__len__ = klass.Sizeof

        cppjit.py.add_pythonization(root_pythonizor)

        assert len(cppjit.gbl.pyzables.TObjString("aap")) == 4  # include '\0'

    def test01_size_mapping(self):
        """Use composites to map GetSize() onto buffer returns"""

        import cppjit

        def set_size(self, buf):
            buf.reshape((self.GetN(),))
            return buf

        cppjit.py.add_pythonization(
            cppjit.py.compose_method("NakedBuffers$", "Get[XY]$", set_size), "pyzables"
        )

        bsize, xval, yval = 3, 2, 5
        m = cppjit.gbl.pyzables.NakedBuffers(bsize, xval, yval)

        x = m.GetX()
        assert len(x) == bsize
        assert list(x) == list(map(lambda x: x * xval, range(bsize)))

        y = m.GetY()
        assert len(y) == bsize
        assert list(y) == list(map(lambda x: x * yval, range(bsize)))

    def test02_size_mapping_of_templated_method(self):
        """Use composites to map GetSize() onto buffer returns"""

        import cppjit

        def set_size(self, buf):
            buf.reshape((self.GetN(),))
            return buf

        cppjit.py.add_pythonization(
            cppjit.py.compose_method("NakedBuffers2.*Vector.*", "Get[XY]$", set_size),
            "pyzables",
        )

        bsize, xval, yval = 3, 2, 5
        m = cppjit.gbl.pyzables.NakedBuffers2[cppjit.gbl.pyzables.Vector](
            bsize, xval, yval
        )

        x = m.GetX()
        assert len(x) == bsize
        assert list(x) == list(map(lambda x: x * xval, range(bsize)))

        y = m.GetY()
        assert len(y) == bsize
        assert list(y) == list(map(lambda x: x * yval, range(bsize)))

    def test03_type_pinning(self):
        """Verify pinnability of returns"""

        import cppjit

        cppjit.gbl.pyzables.GimeDerived.__creates__ = True

        result = cppjit.gbl.pyzables.GimeDerived()
        assert type(result) == cppjit.gbl.pyzables.MyDerived

        cppjit.py.pin_type(cppjit.gbl.pyzables.MyBase)
        assert type(result) == cppjit.gbl.pyzables.MyDerived

    def test04_transparency(self):
        """Transparent use of smart pointers"""

        import cppjit

        Countable = cppjit.gbl.pyzables.Countable
        mine = cppjit.gbl.pyzables.mine

        assert type(mine) == Countable
        assert mine.m_check == 0xCDCDCDCD
        assert type(mine.__smartptr__()) == cppjit.gbl.std.shared_ptr(Countable)
        assert mine.__smartptr__().get().m_check == 0xCDCDCDCD
        assert mine.say_hi() == "Hi!"

    @mark.xfail(
        condition=IS_VALGRIND and IS_LINUX_ARM and IS_CLANG_REPL,
        run=False,
        reason="Crashes on Valgind Clang-Repl-ARM",
    )
    def test05_converters(self):
        """Smart pointer argument passing"""

        import cppjit

        pz = cppjit.gbl.pyzables
        mine = pz.mine

        assert 0xCDCDCDCD == pz.pass_mine_rp_ptr(mine)
        assert 0xCDCDCDCD == pz.pass_mine_rp_ref(mine)
        assert 0xCDCDCDCD == pz.pass_mine_rp(mine)

        assert 0xCDCDCDCD == pz.pass_mine_sp_ptr(mine)
        assert 0xCDCDCDCD == pz.pass_mine_sp_ref(mine)

        assert 0xCDCDCDCD == pz.pass_mine_sp_ptr(mine.__smartptr__())
        assert 0xCDCDCDCD == pz.pass_mine_sp_ref(mine.__smartptr__())

        assert 0xCDCDCDCD == pz.pass_mine_sp(mine)
        assert 0xCDCDCDCD == pz.pass_mine_sp(mine.__smartptr__())

        # TODO:
        # cppjit.gbl.mine = mine
        pz.renew_mine()

    @mark.xfail(
        condition=IS_VALGRIND and IS_LINUX_ARM and IS_CLANG_REPL,
        run=False,
        reason="Fails with Valgrind with Clang-Repl ARM",
    )
    def test06_executors(self):
        """Smart pointer return types"""

        import cppjit

        pz = cppjit.gbl.pyzables
        Countable = pz.Countable

        mine = pz.gime_mine_ptr()
        assert type(mine) == Countable
        assert mine.m_check == 0xCDCDCDCD
        assert type(mine.__smartptr__()) == cppjit.gbl.std.shared_ptr(Countable)
        assert mine.__smartptr__().get().m_check == 0xCDCDCDCD
        assert mine.say_hi() == "Hi!"

        mine = pz.gime_mine_ref()
        assert type(mine) == Countable
        assert mine.m_check == 0xCDCDCDCD
        assert type(mine.__smartptr__()) == cppjit.gbl.std.shared_ptr(Countable)
        assert mine.__smartptr__().get().m_check == 0xCDCDCDCD
        assert mine.say_hi() == "Hi!"

        mine = pz.gime_mine()
        assert type(mine) == Countable
        assert mine.m_check == 0xCDCDCDCD
        assert type(mine.__smartptr__()) == cppjit.gbl.std.shared_ptr(Countable)
        assert mine.__smartptr__().get().m_check == 0xCDCDCDCD
        assert mine.say_hi() == "Hi!"

    def test07_creates_flag(self):
        """Effect of creates flag on return type"""

        import gc

        import cppjit

        pz = cppjit.gbl.pyzables
        Countable = pz.Countable

        gc.collect()
        oldcount = Countable.sInstances  # there's eg. one global variable

        pz.gime_naked_countable.__creates__ = True
        for i in range(10):
            cnt = pz.gime_naked_countable()
            gc.collect()
            assert Countable.sInstances == oldcount + 1
            del cnt
        gc.collect()

        assert Countable.sInstances == oldcount

    def test08_base_class_pythonization(self):
        """Derived class should not re-pythonize base class pythonization"""

        import cppjit

        d = cppjit.gbl.pyzables.IndexableDerived()

        assert d[0] == 42
        assert d[-1] == 42
        # skip the IndexErorr test: pythonization for __getitem__[index] < size()
        # can not be applied strict enough (instead of an index, this could be an
        # associative  container, with 'index' a key, not a counter
        # raises(IndexError, d.__getitem__, 1)

    @mark.xfail(condition=IS_MAC, reason="Fails on OS X")
    def test09_cpp_side_pythonization(self):
        """Use of C++ side pythonizations"""

        import cppjit

        # explicit pythonization
        for kls in [
            cppjit.gbl.pyzables.WithCallback1,
            cppjit.gbl.pyzables.WithCallback2,
        ]:
            w = kls(42)
            assert hasattr(w, "GetInt")
            assert not hasattr(w, "get_int")
            assert w.GetInt() == 42

            assert hasattr(w, "SetInt")
            assert not hasattr(w, "set_int")
            w.SetInt(17)
            assert w.GetInt() == 17

            assert kls.klass_name == kls.__cpp_name__

        # up-the-hierarchy pythonization
        w = cppjit.gbl.pyzables.WithCallback3(42)
        assert hasattr(w, "GetInt")
        assert not hasattr(w, "get_int")
        assert w.GetInt() == 2 * 42

        assert hasattr(w, "SetInt")
        assert not hasattr(w, "set_int")
        w.SetInt(17)
        assert w.GetInt() == 4 * 17

        assert cppjit.gbl.pyzables.WithCallback2.klass_name == "pyzables::WithCallback3"

    @mark.xfail(condition=IS_MAC, reason="Fails on OS X")
    def test10_shared_ptr_reset(self):
        """Checks that smart pointer types are Pythonized with the special
        __smartptr__ member that can also be used to reset the underlying smart
        pointer."""

        import cppjit

        optr = cppjit.gbl.std.make_shared["std::string"]("hello smart pointer")
        o2 = cppjit.gbl.std.string()
        cppjit._backend.SetOwnership(
            o2, False
        )  # This object will be owned by the smart pointer
        optr.__smartptr__().reset(o2)
        assert optr == o2


## actual test run
if __name__ == "__main__":
    result = run_pytest(__file__)  # noqa: F821
    sys.exit(result)
