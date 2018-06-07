import py, os, sys
from pytest import raises


class TestDOCFEATURES:
    def setup_class(cls):
        import cppyy

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
    virtual void abstract_method() = 0;
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

    virtual void abstract_method() {
        std::cout << "called Concrete::abstract_method" << std::endl;
    }

    virtual void concrete_method() {
        std::cout << "called Concrete::concrete_method" << std::endl;
    }

    void array_method(int* ad, int size) {
        for (int i=0; i < size; ++i)
            std::cout << ad[i] << ' ';
        std::cout << std::endl;
    }

    void array_method(double* ad, int size) {
        for (int i=0; i < size; ++i)
            std::cout << ad[i] << ' ';
        std::cout << std::endl;
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
};

//-----
int global_function(int) {
    return 42;
}

double global_function(double) {
    return std::exp(1);
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

        from cppyy.gbl import global_function, Namespace
        assert not(global_function == Namespace.global_function)

        assert round(global_function(1.)-2.718281828459045, 8) == 0.
        assert global_function(1) == 42

        assert Namespace.global_function(1) == 42*2
        assert round(Namespace.global_function(1.)-2.718281828459045*2, 8) == 0.

        assert round(global_function.__overload__('double')(1)-2.718281828459045, 8) == 0.

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

    def test_static_data_members(self):
        import cppyy

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

        pass

    def test_unary_operators(sef):
        import cppyy

        pass
