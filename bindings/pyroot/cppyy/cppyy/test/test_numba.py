import os
import math, time
from pytest import mark, raises
from .support import setup_make

try:
    import numba
    has_numba = True
except ImportError:
    has_numba = False


class TestREFLEX:
    def setup_class(cls):
        import cppyy
        import cppyy.reflex

    def test01_instance_box_unbox(self):
        """Access to box/unbox methods"""

        import cppyy

        assert cppyy.addressof('Instance_AsVoidPtr')
        assert cppyy.addressof('Instance_FromVoidPtr')

        with raises(TypeError):
            cppyy.addressof('doesnotexist')

    def test02_method_reflection(self):
        """Method reflection tooling"""

        import cppyy
        import cppyy.reflex as r

        cppyy.cppdef("""\
        namespace ReflexTest {
        int free1() { return 42; }
        double free2() { return 42.; }

        class MyData_m1 {};
        }""")

        ns = cppyy.gbl.ReflexTest

        assert ns.free1.__cpp_reflex__(r.RETURN_TYPE) == 'int'
        assert ns.free2.__cpp_reflex__(r.RETURN_TYPE) == 'double'

        assert ns.MyData_m1.__init__.__cpp_reflex__(r.RETURN_TYPE)              == ns.MyData_m1
        assert ns.MyData_m1.__init__.__cpp_reflex__(r.RETURN_TYPE, r.OPTIMAL)   == ns.MyData_m1
        assert ns.MyData_m1.__init__.__cpp_reflex__(r.RETURN_TYPE, r.AS_TYPE)   == ns.MyData_m1
        assert ns.MyData_m1.__init__.__cpp_reflex__(r.RETURN_TYPE, r.AS_STRING) == 'ReflexTest::MyData_m1'

    def test03_datamember_reflection(self):
        """Data member reflection tooling"""

        import cppyy
        import cppyy.reflex as r

        cppyy.cppdef("""\
        namespace ReflexTest {
        class MyData_d1 {
        public:
           int    m_int;
           double m_double;
        }; }""")

        ns = cppyy.gbl.ReflexTest

        assert ns.MyData_d1.__dict__['m_int'].__cpp_reflex__(r.TYPE)    == 'int'
        assert ns.MyData_d1.__dict__['m_double'].__cpp_reflex__(r.TYPE) == 'double'

        d = ns.MyData_d1(); daddr = cppyy.addressof(d)
        assert ns.MyData_d1.__dict__['m_int'].__cpp_reflex__(r.OFFSET)    == 0
        assert ns.MyData_d1.__dict__['m_double'].__cpp_reflex__(r.OFFSET) == cppyy.addressof(d, 'm_double') - daddr


@mark.skipif(has_numba == False, reason="numba not found")
class TestNUMBA:
    def setup_class(cls):
        import cppyy
        import cppyy.numba_ext

    def compare(self, go_slow, go_fast, N, *args):
        t0 = time.time()
        for i in range(N):
            go_slow(*args)
        slow_time = time.time() - t0

        t0 = time.time()
        for i in range(N):
            go_fast(*args)
        fast_time = time.time() - t0

        return fast_time < slow_time

    def test01_compiled_free_func(self):
        """Numba-JITing of a compiled free function"""

        import cppyy
        import numpy as np

        def go_slow(a):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += math.tanh(a[i, i])
            return a + trace

        @numba.jit(nopython=True)
        def go_fast(a):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += cppyy.gbl.tanh(a[i, i])
            return a + trace

        x = np.arange(100, dtype=np.float64).reshape(10, 10)

        assert (go_fast(x) == go_slow(x)).all()
        assert self.compare(go_slow, go_fast, 300000, x)

    def test02_JITed_template_free_func(self):
        """Numba-JITing of Cling-JITed templated free function"""

        import cppyy
        import numpy as np

        cppyy.cppdef(r"""\
        template<class T>
        T add42(T t) {
            return T(t+42);
        }""")

        def add42(t):
            return type(t)(t+42)

        def go_slow(a):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += add42(a[i, i]) + add42(int(a[i, i]))
            return a + trace

        @numba.jit(nopython=True)
        def go_fast(a):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += cppyy.gbl.add42(a[i, i]) + cppyy.gbl.add42(int(a[i, i]))
            return a + trace

        x = np.arange(100, dtype=np.float64).reshape(10, 10)

        assert (go_fast(x) == go_slow(x)).all()
        assert self.compare(go_slow, go_fast, 100000, x)

    def test03_proxy_argument_for_field(self):
        """Numba-JITing of a free function taking a proxy argument for field access"""

        import cppyy
        import numpy as np

        cppyy.cppdef(r"""\
        struct MyNumbaData03 {
            MyNumbaData03(int64_t i1, int64_t i2) : fField1(i1), fField2(i2) {}
            int64_t fField1;
            int64_t fField2;
        };""")

        def go_slow(a, d):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += d.fField1 + d.fField2
            return a + trace

        @numba.jit(nopython=True)
        def go_fast(a, d):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += d.fField1 + d.fField2
            return a + trace

      # note: need a sizable array to outperform given the unboxing overhead
        x = np.arange(10000, dtype=np.float64).reshape(100, 100)
        d = cppyy.gbl.MyNumbaData03(42, 27)

        assert((go_fast(x, d) == go_slow(x, d)).all())
        assert self.compare(go_slow, go_fast, 10000, x, d)

    def test04_proxy_argument_for_method(self):
        """Numba-JITing of a free function taking a proxy argument for method access"""

        import cppyy
        import numpy as np

        cppyy.cppdef(r"""\
        struct MyNumbaData04 {
            MyNumbaData04(int64_t i) : fField(i) {}
            int64_t get_field() { return fField; }
            int64_t fField;
        };""")

        def go_slow(a, d):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += d.get_field()
            return a + trace

        @numba.jit(nopython=True)
        def go_fast(a, d):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += d.get_field()
            return a + trace

      # note: need a sizable array to outperform given the unboxing overhead
        x = np.arange(10000, dtype=np.float64).reshape(100, 100)
        d = cppyy.gbl.MyNumbaData04(42)

        assert((go_fast(x, d) == go_slow(x, d)).all())
        assert self.compare(go_slow, go_fast, 10000, x, d)

    def test05_multiple_arguments_function(self):
        """Numba-JITing of functions with multiple arguments"""

        import cppyy
        import numpy as np

        cppyy.cppdef("""
               double add_double(double a, double b, double c) {
                   double d = a + b + c;
                   return d;
                   }
               """)
        @numba.njit()
        def loop_add(x):
            sum = 0
            for row in x:
                sum += cppyy.gbl.add_double(row[0], row[1], row[2])
            return sum

        x = np.arange(3000, dtype=np.float64).reshape(1000, 3)
        sum = 0
        for row in x:
            sum += row[0] + row[1] + row[2]

        assert sum == loop_add(x)

    def test06_multiple_arguments_template_freefunction(self):
        """Numba-JITing of a free template function that recieves more than one template arg"""

        import cppyy
        import numpy as np
        cppyy.cppdef("""
                namespace NumbaSupportExample {
                template<typename T1>
                T1 add(T1 a, T1 b) { return a + b; }
                }""")

        @numba.jit(nopython=True)
        def tma(x):
            sum = 0
            for row in x:
                sum += cppyy.gbl.NumbaSupportExample.add(row[0], row[1])
            return sum

        x = np.arange(2000, dtype=np.float64).reshape(1000, 2)
        sum = 0
        for row in x:
            sum += row[0] + row[1]

        assert sum == tma(x)

    def test07_datatype_mapping(self):
        """Numba-JITing of various data types"""

        import cppyy

        @numba.jit(nopython=True)
        def access_field(d):
            return d.fField

        code = """\
        namespace NumbaDTT {
        struct M%d { M%d(%s f) : fField(f) {};
             %s buf, fField;
        }; }"""

        cppyy.cppdef("namespace NumbaDTT { }")
        ns = cppyy.gbl.NumbaDTT

        types = (
            # 'int8_t', 'uint8_t',     # TODO b/c check using return type fails
            'short', 'unsigned short', 'int', 'unsigned int',
            'int32_t', 'uint32_t', 'int64_t', 'uint64_t',
            'long', 'unsigned long', 'long long', 'unsigned long long',
            'float', 'double',
        )

        nl = cppyy.gbl.std.numeric_limits
        for i, ntype in enumerate(types):
            cppyy.cppdef(code % (i, i, ntype, ntype))
            for m in ('min', 'max'):
                val = getattr(nl[ntype], m)()
                assert access_field(getattr(ns, 'M%d'%i)(val)) == val

    def test08_object_returns(self):
        """Numba-JITing of a function that returns an object"""

        import cppyy
        import numpy as np

        cppyy.cppdef(r"""\
        struct MyNumbaData06 {
            MyNumbaData06(int64_t i1) : fField(i1) {}
            int64_t fField;
        };

        MyNumbaData06 get_numba_data_06() { return MyNumbaData06(42); }
        """)

        def go_slow(a):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += cppyy.gbl.get_numba_data_06().fField
            return a + trace

        @numba.jit(nopython=True)
        def go_fast(a):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += cppyy.gbl.get_numba_data_06().fField
            return a + trace

        x = np.arange(100, dtype=np.float64).reshape(10, 10)

        assert((go_fast(x) == go_slow(x)).all())
        assert self.compare(go_slow, go_fast, 100000, x)

    def test09_non_typed_templates(self):
        """Numba-JITing of a free template function that recieves multiple template args with non types"""

        import cppyy
        import numpy as np
        cppyy.cppdef("""
                namespace NumbaSupportExample {
                template<typename T1, typename T2>
                double add(double a, T1 b, T2 c) { return a + b + c; }
                }""")

        @numba.jit(nopython=True)
        def tma(x):
            sum = 0
            for row in x:
                sum += cppyy.gbl.NumbaSupportExample.add(row[0], row[1], row[2])
            return sum

        x = np.arange(3000, dtype=np.float64).reshape(1000, 3)
        sum = 0
        for row in x:
            sum += row[0] + row[1] + row[2]

        assert sum == tma(x)

    def test10_returning_a_reference(self):
        import cppyy
        import numpy as np
        import numba

        cppyy.cppdef("""
        int64_t& ref_add(int64_t x, int64_t y) {
        int64_t c = x + y;
        static int64_t result = 0;
        result = c;
        return result;
        }
        """)

        def slow_add(X):
            i = 0
            k = []
            for row in X:
                k.append(row[0] + row[1])
                i = i + 1
            return k


        @numba.njit()
        def fast_add(X):
            res = []
            for row in X:
                a = row[0]
                b = row[1]
                k = cppyy.gbl.ref_add(a, b)
                res.append(k[0])
            return res

        X = np.arange(100, dtype=np.int64).reshape(50, 2)
        assert fast_add(X) == slow_add(X)

    def test11_ptr_ref_support(self):
        """Numba-JITing of a increment method belonging to a class, and also swaps the pointers and reflects the change on the python ctypes variables"""
        import cppyy
        import ctypes
        import random

        cppyy.cppdef("""
           namespace RefTest {
               class Box{
                   public:
                       long a;
                       long *b;
                       long *c;
                       Box(long i, long& j, long& k){
                       a = i;
                       b = &j;
                       c = &k;
                       }

                       void swap_ref(long &a, long &b) {
                            long temp = a;
                            a = b;
                            b = temp;
                        }

                       void inc(long* value) {
                        (*value)++;
                        }
                   };
               }
           """)

        ns = cppyy.gbl.RefTest
        assert ns.Box.__dict__['a'].__cpp_reflex__(cppyy.reflex.TYPE) == 'long'
        assert ns.Box.__dict__['b'].__cpp_reflex__(cppyy.reflex.TYPE) == 'long*'

        @numba.njit()
        def inc_b(d, k):
            for i in range(k):
                d.inc(d.b)

        @numba.njit()
        def inc_c(d, k):
            for i in range(k):
                d.inc(d.c)

        x = random.randint(1, 5000)
        y = random.randint(1, 5000)
        z = random.randint(1, 5000)
        b = ctypes.c_long(y)
        c = ctypes.c_long(z)

        d = ns.Box(x, b, c)
        k = 5000

        inc_b(d, k)
        inc_c(d, k)

        assert b.value == y + k
        assert c.value == z + k

        d.swap_ref(d.b, d.c)

        assert b.value == z + k
        assert c.value == y + k

    def test12_std_vector_pass_by_ref(self):
        """Numba-JITing of a method that performs scalar addition to a std::vector initialised through pointers """
        import cppyy
        import ctypes
        import numba
        import numpy as np

        cppyy.cppdef("""
        template<typename T>
        std::vector<T> make_vector(const std::vector<T>& v, std::vector<T> l) {
           std::vector<T> u(l);
           u.insert(u.end(), v.begin(), v.end());
           return u;
        }
           namespace RefTest {
               class BoxVector{
                    public:
                        std::vector<long>* a;

                        BoxVector() : a(new std::vector<long>()) {}
                        BoxVector(std::vector<long>* i) : a(i){}


                           void square_vec(){
                           for (auto& num : *a) {
                                num = num * num;
                            }
                        }

                            void add_2_vec(long k){
                           for (auto& num : *a) {
                                num = num + k;
                            }
                        }

                            void append_vector(const std::vector<long>& value) {
                                *a = make_vector(value, *a);
                            }
                       };
                   }
           """)
        ns = cppyy.gbl.RefTest
        @numba.njit()
        def add_vec_fast(d):
            for i in range(10000):
                d.add_2_vec(i)

        @numba.njit()
        def add_vec_slow(x):
            for i in range(10000):
                x = x + i
            return x

        @numba.njit()
        def square_vec_fast(d):
            for i in range(5):
                d.square_vec()

        @numba.njit()
        def square_vec_slow(x):
            for i in range(5):
                x = np.square(x)
            return x

        assert ns.BoxVector.__dict__['a'].__cpp_reflex__(cppyy.reflex.TYPE) == 'std::vector<long>*'

        add_vec_fast(ns.BoxVector())
        square_vec_fast(ns.BoxVector())

        # We use b to run square_vec where the values must be < 4 to avoid exceeding longs max value
        a = np.random.randint(1, 100, size=10000, dtype=np.int64)
        b = np.random.randint(1, 4, size=10, dtype=np.int64)

        x = cppyy.gbl.std.vector['long'](a.flatten())
        y = cppyy.gbl.std.vector['long'](b.flatten())

        t0 = time.time()
        add_vec_fast(ns.BoxVector(x))
        time_add_njit = time.time() - t0

        t0 = time.time()
        square_vec_fast(ns.BoxVector(y))
        time_square_njit = time.time() - t0

        t0 = time.time()
        np_add_res = add_vec_slow(a)
        time_add_normal = time.time() - t0

        t0 = time.time()
        np_square_res = square_vec_slow(b)
        time_square_normal = time.time() - t0

        assert (np.array(y) == np_square_res).all()
        assert (np.array(x) == np_add_res).all()

    def test13_std_vector_dot_product(self):
        """Numba-JITing of a dot_product method of a class that stores pointers to std::vectors on the python side"""
        import cppyy, cppyy.ll
        import ctypes
        import cppyy.numba_ext
        import numba
        import numpy as np

        cppyy.cppdef("""
        namespace RefTest {
            class DotVector{
                private:
                    std::vector<long>* a;
                    std::vector<long>* b;

                public:
                    long g = 0;
                    long *res = &g;
                    DotVector(std::vector<long>* i, std::vector<long>* j) : a(i), b(j) {}

                    long self_dot_product() {
                        long result = 0;
                        size_t size = a->size();  // Cache the vector size
                        const long* data_a = a->data();
                        const long* data_b = b->data();

                        for (size_t i = 0; i < size; ++i) {
                            result += data_a[i] * data_b[i];
                        }
                        return result;
                    }

                    long dot_product(const std::vector<long>& vec1, const std::vector<long>& vec2) {
                                        long result = 0;
                                        for (size_t i = 0; i < vec1.size(); ++i) {
                                            result += vec1[i] * vec2[i];
                                        }
                                        return result;
                    }
                };
            }""")

        @numba.njit()
        def dot_product_fast(d):
            res = 0
            for i in range(10000):
                res += d.self_dot_product()
            return res
        def np_dot_product(x, y):
            res = 0
            for i in range(10000):
                res += np.dot(x, y)
            return res

        ns = cppyy.gbl.RefTest

        a = np.arange(20000, dtype=np.int64)
        b = np.arange(20000, dtype=np.int64)

        # TODO : Interestingly njit fails while passing a list of std.vectors because it
        #  cannot reflect element of reflected container: reflected list(reflected list(CppClass(std::vector<long>))<iv=None>)<iv=None>
        # vec_list = []
        # for i in a:
        #     vec_list.append([vector['long'](i[0]), vector['long'](i[1])])

        x = cppyy.gbl.std.vector['long'](a.flatten())
        y = cppyy.gbl.std.vector['long'](b.flatten())
        d = ns.DotVector(x, y)
        dot_product_fast(d)
        res = 0

        t0 = time.time()
        njit_res = dot_product_fast(d)
        time_njit = time.time() - t0

        res = 0
        t0 = time.time()
        res = np_dot_product(x, y)
        time_np = time.time() - t0

        assert (njit_res == res)
        assert (time_njit < time_np)

    @mark.skip(reason="Fails at ImplCLassType Boxing call in lowering")
    def test14_eigen_numba(self):
        """Numba-JITing of a function that uses a cppyy declared Eigen Vector"""

        import numpy as np
        import time
        import cppyy, numba, warnings
        import cppyy.numba_ext
        import os

        inc_paths = [os.path.join(os.path.sep, 'usr', 'include'),
                     os.path.join(os.path.sep, 'usr', 'local', 'include')]

        eigen_path = None
        for p in inc_paths:
            p = os.path.join(p, 'eigen3')
            if os.path.exists(p):
                eigen_path = p

        cppyy.add_include_path(eigen_path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cppyy.include('Eigen/Dense')

        # Define the templated function that takes Eigen objects
        cppyy.cppdef('''
        template<typename T>
        T multiply_scalar(T value, int64_t scalar) {
            return value * scalar;
        }
        ''')

        cppyy.cppdef('''
        #include <iostream>
        #include <vector>
        namespace EigenFake {
        template <typename T, int Rows, int Cols>
        class Matrix {
        public:
            std::vector<T> data_;
            Matrix() {
                data_.resize(Rows * Cols);
            }
            Matrix(std::initializer_list<T> values) {
                if (values.size() != Rows * Cols) {
                throw std::runtime_error("Initializer list size does not match matrix dimensions.");
                }
                std::copy(values.begin(), values.end(), data_.begin());
            }

            T& operator()(int row, int col) {
                return data_[row * Cols + col];
            }

            const T& operator()(int row, int col) const {
                return data_[row * Cols + col];
            }
        };
        }
        ''')

        @numba.jit(nopython=True)
        def mul_njit(m, x):
            matrix = cppyy.gbl.multiply_scalar(m, x)
            return matrix

        mat = cppyy.gbl.EigenFake.Matrix(int, 2, 2)
        mat = {1.0, 2.0, 3.0, 4.0}

        vector = cppyy.gbl.Eigen.VectorXd(2)
        vector[0] = 4.0
        vector[1] = 2.0
        vector[2] = 3.0
        matrix2 = cppyy.gbl.multiply_scalar(vector, 5)
        result = mul_njit(vector, 5)
        assert(result == matrix2)


@mark.skipif(has_numba == False, reason="numba not found")
class TestNUMBA_DOC:
    def setup_class(cls):
        import cppyy
        import cppyy.numba_ext

    def test01_templated_freefunction(self):
        """Numba support documentation example: free templated function"""

        import cppyy
        import numba
        import numpy as np

        cppyy.cppdef("""
        namespace NumbaSupportExample {
        template<typename T>
        T square(T t) { return t*t; }
        }""")

        @numba.jit(nopython=True)
        def tsa(a):
            total = type(a[0])(0)
            for i in range(len(a)):
                total += cppyy.gbl.NumbaSupportExample.square(a[i])
            return total

        a = np.array(range(10), dtype=np.float32)
        assert type(tsa(a)) == float
        assert tsa(a) == 285.0

        a = np.array(range(10), dtype=np.int64)
        assert type(tsa(a)) == int
        assert tsa(a) == 285

    def test02_class_features(self):
        """Numba support documentation example: class features"""

        import cppyy
        import numba
        import numpy as np

        cppyy.cppdef("""\
        namespace NumbaSupportExample {
        class MyData {
        public:
            MyData(int i, int j) : fField1(i), fField2(j) {}
            virtual ~MyData() {}

        public:
            int get_field1() { return fField1; }
            int get_field2() { return fField2; }

            MyData copy() { return *this; }

        public:
            int fField1;
            int fField2;
        }; }""")

        @numba.jit(nopython=True)
        def tsdf(a, d):
            total = type(a[0])(0)
            for i in range(len(a)):
                total += a[i] + d.fField1 + d.fField2
            return total

        d = cppyy.gbl.NumbaSupportExample.MyData(5, 6)
        a = np.array(range(10), dtype=np.int32)

        assert tsdf(a, d) == 155

        @numba.jit(nopython=True)
        def tsdm(a, d):
            total = type(a[0])(0)
            for i in range(len(a)):
                total += a[i] +  d.get_field1() + d.get_field2()
            return total

        assert tsdm(a, d) == 155

        @numba.jit(nopython=True)
        def tsdcm(a, d):
            total = type(a[0])(0)
            for i in range(len(a)):
                total += a[i] + d.copy().fField1 + d.get_field2()
            return total

        assert tsdcm(a, d) == 155

        @numba.jit(nopython=True)
        def tsdcmm(a, d):
            total = type(a[0])(0)
            for i in range(len(a)):
                total += a[i] + d.copy().fField1 + d.copy().fField2
            return total

        assert tsdcmm(a, d) == 155
