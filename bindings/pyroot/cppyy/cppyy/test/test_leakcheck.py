import py, os, sys
from pytest import mark, skip
from .support import setup_make, pylong, pyunicode

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("datatypesDict"))

def setup_module(mod):
    setup_make("datatypes")

nopsutil = False
try:
    import psutil
except ImportError:
    nopsutil = True


@mark.skipif(nopsutil == True, reason="module psutil not installed")
class TestLEAKCHECK:
    def setup_class(cls):
        import cppyy, psutil

        cls.test_dct = test_dct
        cls.memory = cppyy.load_reflection_info(cls.test_dct)

        cls.process = psutil.Process(os.getpid())

    def runit(self, N, scope, func, *args, **kwds):
        i = 0
        while i < N:
            getattr(scope, func)(*args, **kwds)
            i += 1

    def runit_template(self, N, scope, func, tmpl_args, *args, **kwds):
        i = 0
        while i < N:
            getattr(scope, func)[tmpl_args](*args, **kwds)
            i += 1

    def check_func(self, scope, func, *args, **kwds):
        """Leak-check 'func', given args and kwds"""

        import gc

      # if tmpl_args is provided as a keyword, then this is a templated
      # function that is to be found on each call python-side
        tmpl_args = kwds.pop('tmpl_args', None)

      # warmup function (TOOD: why doesn't once suffice?)
        for i in range(8):   # actually, 2 seems to be enough
            if tmpl_args is None:
                getattr(scope, func)(*args, **kwds)
            else:
                getattr(scope, func)[tmpl_args](*args, **kwds)

      # number of iterations
        N = 100000

      # leak check
        gc.collect()
        last = self.process.memory_info().rss

        if tmpl_args is None:
            self.runit(N, scope, func, *args, **kwds)
        else:
            self.runit_template(N, scope, func, tmpl_args, *args, **kwds)

        gc.collect()
        assert last == self.process.memory_info().rss

    def test01_free_functions(self):
        """Leak test of free functions"""

        import cppyy

        cppyy.cppdef("""\
        namespace LeakCheck {
        void free_f() {}
        void free_f_ol(int) {}
        void free_f_ol(std::string s) {}
        template<class T> void free_f_ol(T) {}
        int free_f_ret1() { return 27; }
        std::string free_f_ret2() { return "aap"; }
        }""")

        ns = cppyy.gbl.LeakCheck

        self.check_func(ns, 'free_f')
        self.check_func(ns, 'free_f_ol', 42)
        self.check_func(ns, 'free_f_ol', '42')
        self.check_func(ns, 'free_f_ol', 42.)    # template
        self.check_func(ns, 'free_f_ol', 42., tmpl_args='float')
        self.check_func(ns, 'free_f_ret1')
        self.check_func(ns, 'free_f_ret1')

    def test02_test_static_methods(self):
        """Leak test of static methods"""

        import cppyy

        cppyy.cppdef("""\
        namespace LeakCheck {
        class MyClass02 {
        public:
            static void static_method() {}
            static void static_method_ol(int) {}
            static void static_method_ol(std::string s) {}
            template<class T> static void static_method_ol(T) {}
            static std::string static_method_ret() { return "aap"; }
        }; }""")

        ns = cppyy.gbl.LeakCheck

        for m in [ns.MyClass02, ns.MyClass02()]:
            self.check_func(m, 'static_method')
            self.check_func(m, 'static_method_ol', 42)
            self.check_func(m, 'static_method_ol', '42')
            self.check_func(m, 'static_method_ol', 42.)    # template
            self.check_func(m, 'static_method_ol', 42., tmpl_args='float')
            self.check_func(m, 'static_method_ret')

    def test03_test_methods(self):
        """Leak test of methods"""

        import cppyy

        cppyy.cppdef("""\
        namespace LeakCheck {
        class MyClass03 {
        public:
            void method() {}
            void method_ol(int) {}
            void method_ol(std::string s) {}
            std::string method_ret() { return "aap"; }
            template<class T> void method_ol(T) {}
        }; }""")

        ns = cppyy.gbl.LeakCheck

        m = ns.MyClass03()
        self.check_func(m, 'method')
        self.check_func(m, 'method_ol', 42)
        self.check_func(m, 'method_ol', '42')
        self.check_func(m, 'method_ol', 42.)     # template
        self.check_func(m, 'method_ol', 42., tmpl_args='float')
        self.check_func(m, 'method_ret')

    def test04_default_arguments(self):
        """Leak test for functions with default arguments"""

        import cppyy

        cppyy.cppdef("""\
        namespace LeakCheck {
        double free_default(int a=11, float b=22.f, double c=33.) {
            return a*b*c;
        }

        class MyClass04 {
        public:
            static double static_default(int a=11, float b=22.f, double c=33.) {
                return free_default(a, b, c);
            }
            double method_default(int a=11, float b=22.f, double c=33.) {
                return free_default(a, b, c);
            }
        }; }""")

        ns = cppyy.gbl.LeakCheck

        self.check_func(ns, 'free_default')
        self.check_func(ns, 'free_default', a=-99)
        self.check_func(ns, 'free_default', b=-99)
        self.check_func(ns, 'free_default', c=-99)

        # TODO: no keyword arguments for static methods yet
        #for m in [ns.MyClass04, ns.MyClass04()]:
        #    self.check_func(m, 'static_default')
        #    self.check_func(m, 'static_default', a=-99)
        #    self.check_func(m, 'static_default', b=-99)
        #    self.check_func(m, 'static_default', c=-99)

        m = ns.MyClass04()
        self.check_func(m, 'method_default')
        self.check_func(m, 'method_default', a=-99)
        self.check_func(m, 'method_default', b=-99)
        self.check_func(m, 'method_default', c=-99)

    def test05_aggregates(self):
        """Leak test of aggregate creation"""

        import cppyy

        cppyy.cppdef("""\
        namespace LeakCheck {
        typedef enum _TYPE { DATA=0, SHAPE } TYPE;

        struct SomePOD {
            int    fInt;
            double fDouble;
        };

        struct SomeBuf {
            int val;
            const char *name;
            TYPE buf_type;
        }; }""")

        ns = cppyy.gbl.LeakCheck

        self.check_func(ns, 'SomePOD')
        self.check_func(ns, 'SomePOD', fInt=42)
        self.check_func(ns, 'SomePOD', fDouble=42.)
        self.check_func(ns, 'SomePOD', fInt=42, fDouble=42.)
        self.check_func(ns, 'SomePOD', fDouble=42., fInt=42)

        self.check_func(ns, 'SomeBuf')
        self.check_func(ns, 'SomeBuf', val=10, name="aap", buf_type=ns.SHAPE)

    def test06_dir(self):
        """Global function uploads used to cause more function generation"""

        if sys.hexversion < 0x03000000:
            skip("too slow on py2 and won't be fixed as py2 has reached eol")

        import cppyy

        self.check_func(cppyy.gbl, '__dir__', cppyy.gbl)
