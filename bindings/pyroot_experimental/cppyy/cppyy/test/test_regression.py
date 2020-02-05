import py, os, sys
from pytest import raises
from .support import setup_make, IS_WINDOWS


class TestREGRESSION:
    helpout = []

    def setup_class(cls):
        import cppyy

        def stringpager(text, cls=cls):
            cls.helpout.append(text)

        import pydoc
        pydoc.pager = stringpager

    def test01_kdcraw(self):
        """Doc strings for KDcrawIface (used to crash)."""

        import cppyy, pydoc

        # TODO: run a find for these paths
        qtpath = "/usr/include/qt5"
        kdcraw_h = "/usr/include/KF5/KDCRAW/kdcraw/kdcraw.h"
        if not os.path.isdir(qtpath) or not os.path.exists(kdcraw_h):
            py.test.skip("no KDE/Qt found, skipping test01_kdcraw")

        # need to resolve qt_version_tag for the incremental compiler; since
        # it's not otherwise used, just make something up
        cppyy.cppdef("int qt_version_tag = 42;")
        cppyy.add_include_path(qtpath)
        cppyy.include(kdcraw_h)

        # bring in some symbols to resolve the class
        cppyy.load_library("libQt5Core.so")
        cppyy.load_library("libKF5KDcraw.so")

        from cppyy.gbl import KDcrawIface

        self.__class__.helpout = []
        pydoc.doc(KDcrawIface.KDcraw)
        helptext  = ''.join(self.__class__.helpout)
        assert 'KDcraw' in helptext
        assert 'CPPInstance' in helptext

    def test02_dir(self):
        """For the same reasons as test01_kdcraw, this used to crash."""

        import cppyy, pydoc

        assert not '__abstractmethods__' in dir(cppyy.gbl.gInterpreter)
        assert '__class__' in dir(cppyy.gbl.gInterpreter)

        self.__class__.helpout = []
        pydoc.doc(cppyy.gbl.gInterpreter)
        helptext = ''.join(self.__class__.helpout)
        assert 'TInterpreter' in helptext
        assert 'CPPInstance' in helptext
        assert 'AddIncludePath' in helptext

        cppyy.cppdef("namespace cppyy_regression_test { void iii() {}; }")

        assert not 'iii' in cppyy.gbl.cppyy_regression_test.__dict__
        assert not '__abstractmethods__' in dir(cppyy.gbl.cppyy_regression_test)
        assert '__class__' in dir(cppyy.gbl.cppyy_regression_test)
        assert 'iii' in dir(cppyy.gbl.cppyy_regression_test)

        assert not 'iii' in cppyy.gbl.cppyy_regression_test.__dict__
        assert cppyy.gbl.cppyy_regression_test.iii
        assert 'iii' in cppyy.gbl.cppyy_regression_test.__dict__

        self.__class__.helpout = []
        pydoc.doc(cppyy.gbl.cppyy_regression_test)
        helptext = ''.join(self.__class__.helpout)
        # TODO: it's deeply silly that namespaces inherit from CPPInstance (in CPyCppyy)
        assert ('CPPInstance' in helptext or 'CPPNamespace' in helptext)

    def test03_pyfunc_doc(self):
        """Help on a generated pyfunc used to crash."""

        import cppyy, distutils, pydoc, sys
        import distutils.sysconfig as sc

        cppyy.add_include_path(sc.get_python_inc())
        if sys.hexversion < 0x3000000:
            cppyy.cppdef("#undef _POSIX_C_SOURCE")
            cppyy.cppdef("#undef _XOPEN_SOURCE")
        else:
            cppyy.cppdef("#undef slots")     # potentially pulled in by Qt/xapian.h

        cppyy.cppdef("""#include "Python.h"
           long py2long(PyObject* obj) { return PyLong_AsLong(obj); }""")

        pydoc.doc(cppyy.gbl.py2long)

        assert 1 == cppyy.gbl.py2long(1)

    def test04_avx(self):
        """Test usability of AVX by default."""

        import cppyy, subprocess

        has_avx = False
        try:
            for line in open('/proc/cpuinfo', 'r'):
                if 'avx' in line:
                    has_avx = True
                    break
        except Exception:
            try:
                cli_arg = subprocess.check_output(['sysctl', 'machdep.cpu.features'])
                has_avx = 'avx' in cli_arg.decode("utf-8").strip().lower()
            except Exception:
                pass

        if has_avx:
            assert cppyy.cppdef('int check_avx() { return (int) __AVX__; }')
            assert cppyy.gbl.check_avx()   # attribute error if compilation failed

    def test05_default_template_arguments(self):
        """Calling a templated method on a templated class with all defaults used to crash."""

        import cppyy

        cppyy.cppdef("""
        template<typename T>
        class AllDefault {
        public:
           AllDefault(int val) : m_t(val) {}
           template<int aap=1, int noot=2>
              int do_stuff() { return m_t+aap+noot; }

        public:
           T m_t;
        };""")

        a = cppyy.gbl.AllDefault[int](24)
        a.m_t = 21;
        assert a.do_stuff() == 24

    def test06_class_refcounting(self):
        """The memory regulator would leave an additional refcount on classes"""

        import cppyy, gc, sys

        x = cppyy.gbl.std.vector['float']
        old_refcnt = sys.getrefcount(x)

        y = x()
        del y
        gc.collect()

        assert sys.getrefcount(x) == old_refcnt

    def test07_typedef_identity(self):
        """Nested typedefs should retain identity"""

        import cppyy

        cppyy.cppdef("""namespace PyABC {
            struct S1 {};
            struct S2 {
                typedef std::vector<const PyABC::S1*> S1_coll;
            };
        }""")

        from cppyy.gbl import PyABC

        assert PyABC.S2.S1_coll
        assert 'S1_coll' in dir(PyABC.S2)
        assert not 'vector<const PyABC::S1*>' in dir(PyABC.S2)
        assert PyABC.S2.S1_coll is cppyy.gbl.std.vector('const PyABC::S1*')

    def test08_gil_not_released(self):
        """GIL was released by accident for by-value returns"""

        import cppyy

        something = 5.0

        code = """
#include "Python.h"

std::vector<float> some_foo_calling_python() {
   auto pyobj = reinterpret_cast<PyObject*>(ADDRESS);
   float f = (float)PyFloat_AsDouble(pyobj);
   std::vector<float> v;
   v.push_back(f);
   return v;
}
""".replace("ADDRESS", str(id(something)))

        cppyy.cppdef(code)
        cppyy.gbl.some_foo_calling_python()

    def test09_enum_in_global_space(self):
        """Enum declared in search.h did not appear in global space"""

        if IS_WINDOWS:
            return           # no such enum in MSVC's search.h

        import cppyy

        cppyy.include('search.h')

        assert cppyy.gbl.ACTION
        assert hasattr(cppyy.gbl, 'ENTER')
        assert hasattr(cppyy.gbl, 'FIND')

    def test10_cobject_addressing(self):
        """AsCObject (now as_cobject) had a deref too many"""

        import cppyy
        import cppyy.ll

        cppyy.cppdef('struct CObjA { CObjA() : m_int(42) {} int m_int; };')
        a = cppyy.gbl.CObjA()
        co = cppyy.ll.as_cobject(a)

        assert a == cppyy.bind_object(co, 'CObjA')
        assert a.m_int == 42
        assert cppyy.bind_object(co, 'CObjA').m_int == 42

    def test11_exception_while_exception(self):
        """Exception from SetDetailedException during exception handling used to crash"""

        import cppyy

        cppyy.cppdef("namespace AnExceptionNamespace { }")

        try:
            cppyy.gbl.blabla
        except AttributeError:
            try:
                cppyy.gbl.AnExceptionNamespace.blabla
            except AttributeError:
                pass

    def test12_char_star_over_char(self):
        """Map str to const char* over char"""

      # This is debatable, but although a single character string passes through char,
      # it is more consistent to prefer const char* or std::string in all cases. The
      # original bug report is here:
      #    https://bitbucket.org/wlav/cppyy/issues/127/string-argument-resolves-incorrectly

        import cppyy

        cppyy.cppdef("""
        namespace csoc1 {
            std::string call(char) { return "char"; }
        }

        namespace csoc2 {
            std::string call(char) { return "char"; }
            std::string call(const char*) { return "const char*"; }
        }

        namespace csoc3 {
            std::string call(char) { return "char"; }
            std::string call(const std::string&) { return "string"; }
        }
        """)

        assert cppyy.gbl.csoc1.call('0') == 'char'
        raises(ValueError, cppyy.gbl.csoc1.call, '00')

        assert cppyy.gbl.csoc2.call('0')  == 'const char*'
        assert cppyy.gbl.csoc2.call('00') == 'const char*'

        assert cppyy.gbl.csoc3.call('0')  == 'string'
        assert cppyy.gbl.csoc3.call('00') == 'string'

    def test13_struct_direct_definition(self):
        """Struct defined directly in a scope miseed scope in renormalized name"""

        import cppyy

        cppyy.cppdef("""
        namespace struct_direct_definition {
        struct Bar {
            struct Baz {
                std::vector<double> data;
            } baz[2];

            Bar() {
                baz[0].data.push_back(3.14);
                baz[1].data.push_back(2.73);
            }
        }; }""")

        from cppyy.gbl import struct_direct_definition as sds

        f = sds.Bar()

        assert len(f.baz) == 2
        assert len(f.baz[0].data) == 1
        assert f.baz[0].data[0]   == 3.14
        assert len(f.baz[1].data) == 1
        assert f.baz[1].data[0]   == 2.73
