import py, os, sys
from pytest import raises


class TestDOCFEATURES:
    def setup_class(cls):
        import cppyy

        # touch __version__ as a test
        assert hasattr(cppyy, '__version__')

        cppyy.cppdef("""
#include <cmath>
#include <iostream>
#include <vector>


//-----
unsigned int gUint = 0;

//-----
class Abstract {
public:
    virtual ~Abstract() {}
    virtual std::string abstract_method() = 0;
    virtual void concrete_method() = 0;
};

void Abstract::concrete_method() {
    std::cout << "called Abstract::concrete_method" << std::endl;
}

//-----
class Concrete : Abstract {
public:
    Concrete(int n=42) : m_int(n), m_const_int(17) {}
    ~Concrete() {}

    virtual std::string abstract_method() {
        return "called Concrete::abstract_method";
    }

    virtual void concrete_method() {
        std::cout << "called Concrete::concrete_method" << std::endl;
    }

    void array_method(int* ad, int size) {
        for (int i=0; i < size; ++i)
            std::cerr << ad[i] << ' ';
        std::cerr << std::endl;
    }

    void array_method(double* ad, int size) {
        for (int i=0; i < size; ++i)
            std::cerr << ad[i] << ' ';
        std::cerr << std::endl;
    }

    void uint_ref_assign(unsigned int& target, unsigned int value) {
        target = value;
    }

    Abstract* show_autocast() {
        return this;
    }

    operator const char*() {
        return "Hello operator const char*!";
    }

public:
    double m_data[4];
    int m_int;
    const int m_const_int;

    static int s_int;
};

typedef Concrete Concrete_t;

int Concrete::s_int = 321;

std::string call_abstract_method(Abstract* a) {
    return a->abstract_method();
}

//-----
int global_function(int) {
    return 42;
}

double global_function(double) {
    return std::exp(1);
}

int call_int_int(int (*f)(int, int), int i1, int i2) {
    return f(i1, i2);
}

//-----
namespace Namespace {

    class Concrete {
    public:
        class NestedClass {
        public:
            std::vector<int> m_v;
        };

    };

    int global_function(int i) {
        return 2*::global_function(i);
    }

    double global_function(double d) {
        return 2*::global_function(d);
    }

} // namespace Namespace

""")

    def test_abstract_class(self):
        import cppyy
        from cppyy.gbl import Abstract, Concrete

        raises(TypeError, Abstract)
        assert issubclass(Concrete, Abstract)
        c = Concrete()
        assert isinstance(c, Abstract)

    def test_array(self):
        import cppyy
        from cppyy.gbl import Concrete
        from array import array

        c = Concrete()
        c.array_method(array('d', [1., 2., 3., 4.]), 4)
        raises(IndexError, c.m_data.__getitem__, 4)

    def test_builtin_data(self):
        import cppyy

        assert cppyy.gbl.gUint == 0
        raises(ValueError, setattr, cppyy.gbl, 'gUint', -1)
        
    def test_casting(self):
        import cppyy
        from cppyy.gbl import Abstract, Concrete

        c = Concrete()
        assert 'Abstract' in Concrete.show_autocast.__doc__
        d = c.show_autocast()
        assert type(d) == cppyy.gbl.Concrete

        from cppyy import addressof, bind_object
        e = bind_object(addressof(d), Abstract)
        assert type(e) == cppyy.gbl.Abstract

    def test_classes_and_structs(self):
        import cppyy
        from cppyy.gbl import Concrete, Namespace

        assert Concrete != Namespace.Concrete
        n = Namespace.Concrete.NestedClass()
        assert 'Namespace.Concrete.NestedClass' in str(type(n))
        assert 'NestedClass' == type(n).__name__
        assert 'cppyy.gbl.Namespace.Concrete' == type(n).__module__
        assert 'Namespace::Concrete::NestedClass' == type(n).__cppname__

    def test_data_members(self):
        import cppyy
        from cppyy.gbl import Concrete

        c = Concrete()
        assert c.m_int == 42
        raises(TypeError, setattr, c, 'm_const_int', 71)

    def test_default_arguments(self):
        import cppyy
        from cppyy.gbl import Concrete

        c = Concrete()
        assert c.m_int == 42
        c = Concrete(13)
        assert c.m_int == 13
        args = (27,)
        c = Concrete(*args)
        assert c.m_int == 27

    def test_doc_strings(self):
        import cppyy
        from cppyy.gbl import Concrete
        assert 'void Concrete::array_method(int* ad, int size)' in Concrete.array_method.__doc__
        assert 'void Concrete::array_method(double* ad, int size)' in Concrete.array_method.__doc__

    def test_enums(self):
        import cppyy

        pass

    def test_functions(self):
        import cppyy

        from cppyy.gbl import global_function, call_int_int, Namespace
        assert not(global_function == Namespace.global_function)

        assert round(global_function(1.)-2.718281828459045, 8) == 0.
        assert global_function(1) == 42

        assert Namespace.global_function(1) == 42*2
        assert round(Namespace.global_function(1.)-2.718281828459045*2, 8) == 0.

        assert round(global_function.__overload__('double')(1)-2.718281828459045, 8) == 0.

        def add(a, b):
            return a+b
        assert call_int_int(add, 3, 4) == 7
        assert call_int_int(lambda x, y: x*y, 3, 7) == 21

    def test_inheritance(self):
        import cppyy

        pass

    def test_memory(self):
        import cppyy
        from cppyy.gbl import Concrete

        c = Concrete()
        assert c.__python_owns__ == True

    def test_methods(self):
        import cppyy

        pass

    def test_namespaces(self):
        import cppyy

        pass

    def test_null(self):
        import cppyy

        assert hasattr(cppyy, 'nullptr')
        assert not cppyy.nullptr

    def test_operator_conversions(self):
        import cppyy
        from cppyy.gbl import Concrete

        assert str(Concrete()) == 'Hello operator const char*!'

    def test_operator_overloads(self):
        import cppyy
        
        pass
        
    def test_pointers(self):
        import cppyy

        pass

    def test_pyobject(self):
        import cppyy

        pass

    def test_ref(self):
        import cppyy
        from cppyy.gbl import Concrete
        from ctypes import c_uint

        c = Concrete()
        u = c_uint(0)
        c.uint_ref_assign(u, 42)
        assert u.value == 42

    def test_static_data_members(self):
        import cppyy
        from cppyy.gbl import Concrete

        assert Concrete.s_int == 321
        Concrete().s_int = 123
        assert Concrete.s_int == 123

    def test_static_methods(self):
        import cppyy

        pass

    def test_strings(self):
        import cppyy

        pass

    def test_templated_classes(self):
        import cppyy

        assert cppyy.gbl.std.vector
        assert isinstance(cppyy.gbl.std.vector(int), type)
        assert type(cppyy.gbl.std.vector(int)()) == cppyy.gbl.std.vector(int)

    def test_templated_functions(self):
        import cppyy

        pass

    def test_templated_methods(self):
        import cppyy

        pass

    def test_typedefs(self):
        import cppyy
        from cppyy.gbl import Concrete, Concrete_t

        assert Concrete is Concrete_t

    def test_unary_operators(sef):
        import cppyy

        pass

    def test_x_inheritance(self):
        import cppyy
        from cppyy.gbl import Abstract, Concrete, call_abstract_method

        class PyConcrete1(Abstract):
            def abstract_method(self):
                return cppyy.gbl.std.string("Hello, Python World! (1)")

            def concrete_method(self):
                pass

        pc = PyConcrete1()
        assert call_abstract_method(pc) == "Hello, Python World! (1)"

        class PyConcrete2(Abstract):
            def abstract_method(self):
                return "Hello, Python World! (2)"

            def concrete_method(self):
                pass

        pc = PyConcrete2()
        assert call_abstract_method(pc) == "Hello, Python World! (2)"

        class PyConcrete3(Abstract):
            def __init__(self):
                super(PyConcrete3, self).__init__()

            def abstract_method(self):
                return "Hello, Python World! (3)"

            def concrete_method(self):
                pass

        pc = PyConcrete3()
        assert call_abstract_method(pc) == "Hello, Python World! (3)"

        class PyConcrete4(Concrete):
            def __init__(self):
                super(PyConcrete4, self).__init__()

            def abstract_method(self):
                return "Hello, Python World! (4)"

        pc = PyConcrete4()
        assert call_abstract_method(pc) == "Hello, Python World! (4)"
