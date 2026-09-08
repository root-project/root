import shutil
import tempfile

import py
from pytest import raises
from support import setup_make

# reuse the example01
currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/example01Dict"))


def setup_module(mod):
    setup_make("example01")


class TestBASICAPI:
    def test01_evaluate(self):
        import cppjit

        # we use evaluate as a way to set these flags in support, for example
        # for IS_CLANG_REPL, etc. let's make sure basic uses work
        out = cppjit.evaluate("""#define TEST01_EVALUATE
                                #ifdef TEST01_EVALUATE
                                    true
                                #else
                                    false
                                #endif""")
        assert out

        cppjit.evaluate("__cplusplus") >= 201100

        x = 42
        assert cppjit.evaluate(str(x)) == x

    def test02_cppdef(self):
        import cppjit

        assert cppjit.cppdef("namespace test02_NS { int x = 42; }")
        assert cppjit.gbl.test02_NS.x == 42

    def test03_add_library_path(self):
        import cppjit

        with raises(OSError, match="No such directory"):
            cppjit.add_library_path("not/a/real/path")

        with tempfile.TemporaryDirectory() as tpath:
            cppjit.add_library_path(tpath)

            # now we should actually see if load library can follow this...
            # first, try to load without moving to directory...
            with raises(RuntimeError, match="Could not load library"):
                cppjit.load_library("test.so")

            # then copy to our rpath, and make sure it can be loaded now
            shutil.copyfile(test_dct + ".so", tpath + "/test.so")
            cppjit.load_library("test.so")

    def test04_add_include_path(self):
        import cppjit

        with tempfile.TemporaryDirectory() as tpath:
            cppjit.add_include_path(tpath)

            header = "test_add_include_path.h"
            with open(tpath + "/" + header, "w") as out:
                print("int test04f() { return 42; }", file=out)

            assert cppjit.include(header)
            assert cppjit.gbl.test04f() == 42
