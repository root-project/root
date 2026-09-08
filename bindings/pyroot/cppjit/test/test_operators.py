import py
from pytest import raises, skip
from support import (
    IS_WINDOWS,
    maxvalue,
    pylong,
    setup_make,
)

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/operatorsDict"))


def setup_module(mod):
    setup_make("operators")


class TestOPERATORS:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        cls.operators = cppjit.load_reflection_info(cls.test_dct)

    def teardown_method(self, meth):
        import gc

        gc.collect()

    def test01_math_operators(self):
        """Test overloading of math operators"""

        import cppjit

        number = cppjit.gbl.number

        assert (number(20) + number(10)) == number(30)
        assert (number(20) + 10) == number(30)
        assert (number(20) - number(10)) == number(10)
        assert (number(20) - 10) == number(10)
        assert (number(20) / number(10)) == number(2)
        assert (number(20) / 10) == number(2)
        assert (number(20) * number(10)) == number(200)
        assert (number(20) * 10) == number(200)
        assert (number(20) % 10) == number(0)
        assert (number(20) % number(10)) == number(0)
        assert (number(5) & number(14)) == number(4)
        assert (number(5) | number(14)) == number(15)
        assert (number(5) ^ number(14)) == number(11)
        assert (number(5) << 2) == number(20)
        assert (number(20) >> 2) == number(5)

    def test02_unary_math_operators(self):
        """Test overloading of unary math operators"""

        import cppjit

        number = cppjit.gbl.number

        n = number(20)
        n += number(10)
        n -= number(10)
        n *= number(10)
        n /= number(2)
        assert n == number(100)

        nn = -n
        assert nn == number(-100)

    def test03_comparison_operators(self):
        """Test overloading of comparison operators"""

        import cppjit

        number = cppjit.gbl.number

        assert (number(20) > number(10)) == True
        assert (number(20) < number(10)) == False
        assert (number(20) >= number(20)) == True
        assert (number(20) <= number(10)) == False
        assert (number(20) != number(10)) == True
        assert (number(20) == number(10)) == False

    def test04_boolean_operator(self):
        """Test implementation of operator bool"""

        import cppjit

        number = cppjit.gbl.number

        n = number(20)
        assert n

        n = number(0)
        assert not n

    def test05_exact_types(self):
        """Test converter operators of exact types"""

        import cppjit

        gbl = cppjit.gbl

        o = gbl.operator_char_star()
        assert o.m_str == "operator_char_star"
        assert str(o) == "operator_char_star"

        o = gbl.operator_const_char_star()
        assert o.m_str == "operator_const_char_star"
        assert str(o) == "operator_const_char_star"

        o = gbl.operator_int()
        o.m_int = -13
        assert o.m_int == -13
        assert int(o) == -13

        o = gbl.operator_long()
        o.m_long = 42
        assert o.m_long == 42
        assert pylong(o) == 42

        o = gbl.operator_double()
        o.m_double = 3.1415
        assert o.m_double == 3.1415
        assert float(o) == 3.1415

    def test06_approximate_types(self):
        """Test converter operators of approximate types"""

        import cppjit

        gbl = cppjit.gbl

        o = gbl.operator_short()
        o.m_short = 256
        assert o.m_short == 256
        assert int(o) == 256

        o = gbl.operator_unsigned_int()
        o.m_uint = 2147483647 + 32
        assert o.m_uint == 2147483647 + 32
        assert pylong(o) == 2147483647 + 32

        o = gbl.operator_unsigned_long()
        o.m_ulong = maxvalue + 128
        assert o.m_ulong == maxvalue + 128
        assert pylong(o) == maxvalue + 128

        o = gbl.operator_float()
        o.m_float = 3.14
        assert round(o.m_float - 3.14, 5) == 0.0
        assert round(float(o) - 3.14, 5) == 0.0

    def test07_virtual_operator_eq(self):
        """Test use of virtual bool operator=="""

        import cppjit

        b1 = cppjit.gbl.v_opeq_base(1)
        b1a = cppjit.gbl.v_opeq_base(1)
        b2 = cppjit.gbl.v_opeq_base(2)
        b2a = cppjit.gbl.v_opeq_base(2)

        assert b1 == b1
        assert b1 == b1a
        assert not b1 == b2
        assert not b1 == b2a
        assert b2 == b2
        assert b2 == b2a

        d1 = cppjit.gbl.v_opeq_derived(1)
        d1a = cppjit.gbl.v_opeq_derived(1)
        d2 = cppjit.gbl.v_opeq_derived(2)
        d2a = cppjit.gbl.v_opeq_derived(2)

        # derived operator== returns opposite
        assert not d1 == d1
        assert not d1 == d1a
        assert d1 == d2
        assert d1 == d2a
        assert not d2 == d2
        assert not d2 == d2a

        # the following is a wee bit interesting due to python resolution
        # rules on the one hand, and C++ inheritance on the other: python
        # will never select the derived comparison b/c the call will fail
        # to pass a base through a const derived&
        assert b1 == d1
        assert d1 == b1
        assert not b1 == d2
        assert not d2 == b1

    def test08_call_to_getsetitem_mapping(self):
        """Map () to []"""

        import cppjit

        m = cppjit.gbl.YAMatrix1()
        assert m.m_val == 42
        assert m[1, 2] == 42
        assert m(1, 2) == 42
        m[1, 2] = 27
        assert m.m_val == 27
        assert m[1, 2] == 27
        assert m(1, 2) == 27

        m = cppjit.gbl.YAMatrix2()
        assert m.m_val == 42
        assert m[1] == 42
        m[1] = 27
        assert m.m_val == 27
        assert m[1] == 27

        for cls in [
            cppjit.gbl.YAMatrix3,
            cppjit.gbl.YAMatrix4,
            cppjit.gbl.YAMatrix5,
            cppjit.gbl.YAMatrix6,
            cppjit.gbl.YAMatrix7,
        ]:
            m = cls()
            assert m.m_val == 42
            assert m[1, 2] == 42
            assert m[1] == 42
            assert m(1, 2) == 42

            m[1, 2] = 27
            assert m.m_val == 27
            assert m[1, 2] == 27
            assert m[1] == 27
            assert m(1, 2) == 27

            m[1] = 83
            assert m.m_val == 83
            assert m[1, 2] == 83
            assert m[1] == 83
            assert m(1, 2) == 83

            m.m_val = 74
            assert m.m_val == 74
            assert m[1, 2] == 74
            assert m[1] == 74
            assert m(1, 2) == 74

    def test09_templated_operator(self):
        """Templated operator<()"""

        from cppjit.gbl import TOIClass

        assert TOIClass() < 1

    def test10_r_non_associative(self):
        """Use of radd/rmul with non-associative types"""

        import cppjit

        # Note: calls are repeated to test caching, if any

        a = cppjit.gbl.AssocADD(5.0)
        assert 5 + a == 10.0
        assert a + 5 == 10.0
        assert 5 + a == 10.0
        assert a + 5 == 10.0

        a = cppjit.gbl.NonAssocRADD(5.0)
        assert 5 + a == 10.0
        assert 5 + a == 10.0
        with raises(NotImplementedError):
            v = a + 5

        a = cppjit.gbl.AssocMUL(5.0)
        assert 2 * a == 10.0
        assert a * 2 == 10.0
        assert 2 * a == 10.0
        assert a * 2 == 10.0

        m = cppjit.gbl.NonAssocRMUL(5.0)
        assert 2 * m == 10.0
        assert 2 * m == 10.0
        with raises(NotImplementedError):
            v = m * 2

    def test11_overloaded_operators(self):
        """Overloaded operator*/+-"""

        import cppjit

        v = cppjit.gbl.MultiLookup.Vector2(1, 2)
        w = cppjit.gbl.MultiLookup.Vector2(3, 4)

        u = v * 2
        assert u.x == 2.0
        assert u.y == 4.0

        assert v * w == 1 * 3 + 2 * 4

        u = v / 2
        assert u.x == 0.5
        assert u.y == 1.0

        assert round(v / w - (1.0 / 3.0 + 2.0 / 4.0), 8) == 0.0

        u = v + 2
        assert u.x == 3.0
        assert u.y == 4.0

        assert v + w == 1 + 3 + 2 + 4

        u = v - 2
        assert u.x == -1.0
        assert u.y == 0.0

        assert v - w == 1 - 3 + 2 - 4

    def test12_unary_operators(self):
        """Unary operator-+~"""

        import cppjit

        for cls in [cppjit.gbl.SomeGlobalNumber, cppjit.gbl.Unary.SomeNumber]:
            n = cls(42)

            assert (-n).i == -42
            assert (+n).i == 42
            # assert (~n).i == ~42

    def test13_comma_operator(self):
        """Comma operator"""

        import cppjit

        c = cppjit.gbl.CommaOperator(1)
        assert c.__comma__(2).__comma__(3).fInt == 6

    def test14_single_argument_call(self):
        """Non-reference, single-argument, call not mapped to getitem"""

        import cppjit

        cppjit.cppdef("""\
        namespace IndexingOperators {
        struct Foo {
            float operator[] (float x) { return x; }
        };

        struct Bar : public Foo {
            float operator() (float x) { return 5.f; }
        }; }""")

        ns = cppjit.gbl.IndexingOperators

        f = ns.Foo()
        assert f[42] == 42
        b = ns.Bar()
        assert b[42] == 42

    def test15_class_and_global_mix(self):
        """Iterator methods have both class and global overloads"""

        if IS_WINDOWS:
            skip("missing symbol __std_max_element_4")

        from cppjit.gbl import std

        x = std.vector[int]([1, 2, 3])
        assert (x.end() - 1).__deref__() == 3
        assert std.max_element(x.begin(), x.end()) - x.begin() == 2
        assert (x.end() - 3).__deref__() == 1

    def test16_global_ordered_operators(self):
        """Globally defined ordered oeprators"""

        import cppjit

        cppjit.cppdef("""\
        namespace FriendOperator {

        struct ALt { ALt(int d) : data(d) {} int data; };
        bool operator< (const ALt& a1, const ALt& a2) { return a1.data <  a2.data; }

        struct ALe { ALe(int d) : data(d) {} int data; };
        bool operator<=(const ALe& a1, const ALe& a2) { return a1.data <= a2.data; }

        struct AGt { AGt(int d) : data(d) {} int data; };
        bool operator> (const AGt& a1, const AGt& a2) { return a1.data >  a2.data; }

        struct AGe { AGe(int d) : data(d) {} int data; };
        bool operator>=(const AGe& a1, const AGe& a2) { return a1.data >= a2.data; }

        }""")

        ns = cppjit.gbl.FriendOperator

        assert ns.ALt(4) < ns.ALt(5)
        assert not ns.ALt(5) < ns.ALt(4)

        assert ns.ALe(4) <= ns.ALe(5)
        assert not ns.ALe(5) <= ns.ALe(4)

        assert ns.AGt(5) > ns.AGt(4)
        assert not ns.AGt(4) > ns.AGt(5)

        assert ns.AGe(5) >= ns.AGe(4)
        assert not ns.AGe(4) >= ns.AGe(5)

    def test17_arrow_operator_recursion(self):
        """operator->() returning same type should not recurse"""

        import cppjit

        cppjit.cppdef(r"""\
        namespace Recursion {
        class MCPCollection {
        public:
          MCPCollection() = default;
          MCPCollection* operator->() { return this; }
        }; }""")

        ns = cppjit.gbl.Recursion

        coll = ns.MCPCollection()
        with raises(AttributeError):
            coll.non_existing_method
