import sys

import py
from pytest import mark, raises, skip
from support import (
    IS_CLANG_REPL,
    IS_CLING,
    IS_LINUX_ARM,
    IS_MAC,
    IS_MAC_ARM,
    IS_VALGRIND,
    IS_WINDOWS,
    ispypy,
    setup_make,
)

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/doc_helperDict"))


def setup_module(mod):
    setup_make("doc_helper")


@mark.skipif(IS_MAC and IS_CLING, reason="setup class fails with OS X cling")
class TestDOCFEATURES:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        # touch __version__ as a test
        assert hasattr(cppjit, "__version__")

        cls.doc_helper = cppjit.load_reflection_info(cls.test_dct)

        cppjit.cppdef("""
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
class Abstract1 {
public:
    virtual ~Abstract1() {}
    virtual std::string abstract_method1() = 0;
};

class Abstract2 {
public:
    virtual ~Abstract2() {}
    virtual std::string abstract_method2() = 0;
};

std::string call_abstract_method1(Abstract1* a) {
    return a->abstract_method1();
}

std::string call_abstract_method2(Abstract2* a) {
    return a->abstract_method2();
}

//-----
int global_function(int) {
    return 42;
}

double global_function(double) {
    return std::exp(1);
}

int call_int_int_function(int (*f)(int, int), int i1, int i2) {
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

    //-----
    enum EFruit {kApple=78, kBanana=29, kCitrus=34};
    enum class NamedClassEnum { E1 = 42 };

} // namespace Namespace

""")

    def test_abstract_class(self):
        from cppjit.gbl import Abstract, Concrete

        raises(TypeError, Abstract)
        assert issubclass(Concrete, Abstract)
        c = Concrete()
        assert isinstance(c, Abstract)

    def test_array(self):
        from array import array

        from cppjit.gbl import Concrete

        c = Concrete()
        c.array_method(array("d", [1.0, 2.0, 3.0, 4.0]), 4)
        raises(IndexError, c.m_data.__getitem__, 4)

    def test_builtin_data(self):
        import cppjit

        assert cppjit.gbl.gUint == 0
        raises(ValueError, setattr, cppjit.gbl, "gUint", -1)

    def test_casting(self):
        import cppjit
        from cppjit.gbl import Abstract, Concrete

        c = Concrete()
        assert "Abstract" in Concrete.show_autocast.__doc__
        d = c.show_autocast()
        assert type(d) == cppjit.gbl.Concrete

        from cppjit import addressof, bind_object

        e = bind_object(addressof(d), Abstract)
        assert type(e) == cppjit.gbl.Abstract

    def test_classes_and_structs(self):
        from cppjit.gbl import Concrete, Namespace

        assert Concrete != Namespace.Concrete
        n = Namespace.Concrete.NestedClass()
        assert "Namespace.Concrete.NestedClass" in str(type(n))
        assert "NestedClass" == type(n).__name__
        assert "cppjit.gbl.Namespace.Concrete" == type(n).__module__
        assert "Namespace::Concrete::NestedClass" == type(n).__cpp_name__

    def test_data_members(self):
        from cppjit.gbl import Concrete

        c = Concrete()
        assert c.m_int == 42
        raises(TypeError, setattr, c, "m_const_int", 71)

    def test_default_arguments(self):
        from cppjit.gbl import Concrete

        c = Concrete()
        assert c.m_int == 42
        c = Concrete(13)
        assert c.m_int == 13
        args = (27,)
        c = Concrete(*args)
        assert c.m_int == 27

    def test_keyword_arguments(self):
        from cppjit.gbl import Concrete

        c = Concrete(n=17)
        assert c.m_int == 17

        caught = False
        try:
            c = Concrete(m=18)  # unknown keyword
        except TypeError as e:
            assert "unexpected keyword argument" in str(e)
            caught = True
        assert caught == True

        kwds = {"n": 18}
        c = Concrete(**kwds)
        assert c.m_int == 18

    def test_doc_strings(self):
        from cppjit.gbl import Concrete

        assert "void Concrete::array_method(int * ad, int size)".replace(
            " ", ""
        ) in Concrete.array_method.__doc__.replace(" ", "")
        assert "void Concrete::array_method(double * ad, int size)".replace(
            " ", ""
        ) in Concrete.array_method.__doc__.replace(" ", "")

    def test_enums(self):

        pass

    @mark.xfail(condition=IS_MAC, run=False, reason="Seg Fault")
    def test_functions(self):

        from cppjit.gbl import Namespace, call_int_int_function, global_function

        assert not (global_function == Namespace.global_function)

        assert round(global_function(1.0) - 2.718281828459045, 8) == 0.0
        assert global_function(1) == 42

        assert Namespace.global_function(1) == 42 * 2
        assert round(Namespace.global_function(1.0) - 2.718281828459045 * 2, 8) == 0.0

        assert (
            round(global_function.__overload__("double")(1) - 2.718281828459045, 8)
            == 0.0
        )

        def add(a, b):
            return a + b

        assert call_int_int_function(add, 3, 4) == 7
        assert call_int_int_function(lambda x, y: x * y, 3, 7) == 21

    def test_inheritance(self):

        pass

    def test_memory(self):
        from cppjit.gbl import Concrete

        c = Concrete()
        assert c.__python_owns__ == True

    def test_methods(self):

        pass

    def test_namespaces(self):

        pass

    def test_null(self):
        import cppjit

        assert hasattr(cppjit, "nullptr")
        assert not cppjit.nullptr

    def test_operator_conversions(self):
        from cppjit.gbl import Concrete

        assert str(Concrete()) == "Hello operator const char*!"

    def test_operator_overloads(self):

        pass

    def test_pointers(self):

        pass

    def test_pyobject(self):

        pass

    def test_ref(self):
        from ctypes import c_uint

        from cppjit.gbl import Concrete

        c = Concrete()
        u = c_uint(0)
        c.uint_ref_assign(u, 42)
        assert u.value == 42

    def test_static_data_members(self):
        from cppjit.gbl import Concrete

        assert Concrete.s_int == 321
        Concrete().s_int = 123
        assert Concrete.s_int == 123

    def test_static_methods(self):

        pass

    def test_strings(self):

        pass

    def test_templated_classes(self):
        import cppjit

        assert cppjit.gbl.std.vector
        assert isinstance(cppjit.gbl.std.vector(int), type)
        assert type(cppjit.gbl.std.vector(int)()) == cppjit.gbl.std.vector(int)

    def test_templated_functions(self):

        pass

    def test_templated_methods(self):

        pass

    def test_typedefs(self):
        from cppjit.gbl import Concrete, Concrete_t

        assert Concrete is Concrete_t

    def test_unary_operators(sef):

        pass

    @mark.xfail(
        condition=IS_MAC,
        run=not IS_MAC_ARM,
        reason="Crashes with exception not being caught on Apple Silicon",
    )
    def test_x_inheritance(self):
        import cppjit
        from cppjit.gbl import Abstract, Concrete, call_abstract_method

        class PyConcrete1(Abstract):
            def abstract_method(self):
                return cppjit.gbl.std.string("Hello, Python World! (1)")

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

    def test_multi_x_inheritance(self):
        """Multiple cross-inheritance"""

        import cppjit

        class PyConcrete(cppjit.multi(cppjit.gbl.Abstract1, cppjit.gbl.Abstract2)):
            def abstract_method1(self):
                return "first message"

            def abstract_method2(self):
                return "second message"

        pc = PyConcrete()

        assert cppjit.gbl.call_abstract_method1(pc) == "first message"
        assert cppjit.gbl.call_abstract_method2(pc) == "second message"

    @mark.xfail(
        condition=IS_MAC_ARM,
        run=False,
        reason="Crashes with exception not being caught on Apple Silicon",
    )
    def test_exceptions(self):
        """Exception throwing and catching"""

        if ispypy:
            skip("throwing exceptions terminates the process")

        import cppjit

        caught = False
        try:
            cppjit.gbl.DocHelper.throw_an_error(1)
        except cppjit.gbl.SomeError as e:
            assert "this is an error" in str(e)
            assert e.what() == "this is an error"
            caught = True
        assert caught == True

        caught = False
        for exc_type in (
            cppjit.gbl.SomeOtherError,
            cppjit.gbl.SomeError,
            cppjit.gbl.std.exception,
            Exception,
            BaseException,
        ):
            caught = False
            try:
                cppjit.gbl.DocHelper.throw_an_error(0)
            except exc_type as e:
                caught = True
            assert caught == True
        assert caught == True


@mark.skipif(IS_MAC and IS_CLING, reason="setup class fails with OS X cling")
class TestTUTORIALFEATURES:
    def setup_class(cls):
        import cppjit

        cppjit.cppdef("""
class Integer1 {
public:
    Integer1(int i) : m_data(i) {}
    int m_data;
};""")

        cppjit.cppdef("""
namespace Math {
    class Integer2 : public Integer1 {
    public:
        using Integer1::Integer1;
        operator int() { return m_data; }
    };
}""")

        cppjit.cppdef("""
namespace Zoo {

    enum EAnimal { eLion, eMouse };

    class Animal {
    public:
        virtual ~Animal() {}
        virtual std::string make_sound() = 0;
    };

    class Lion : public Animal {
    public:
        virtual std::string make_sound() { return s_lion_sound; }
        static std::string s_lion_sound;
    };
    std::string Lion::s_lion_sound = "growl!";

    class Mouse : public Animal {
    public:
        virtual std::string make_sound() { return "peep!"; }
    };

    Animal* release_animal(EAnimal animal) {
        if (animal == eLion) return new Lion{};
        if (animal == eMouse) return new Mouse{};
        return nullptr;
    }

    std::string identify_animal(Lion*) {
        return "the animal is a lion";
    }

    std::string identify_animal(Mouse*) {
        return "the animal is a mouse";
    }

}
""")

        # pythonize the animal release function to take ownership on return
        cppjit.gbl.Zoo.release_animal.__creates__ = True

    def test01_class_existence(self):
        """Existence and importability of created class"""

        import cppjit

        assert cppjit.gbl.Integer1
        assert issubclass(cppjit.gbl.Integer1, object)

        from cppjit.gbl import Integer1

        assert cppjit.gbl.Integer1 is Integer1

        i = Integer1(42)
        assert isinstance(i, Integer1)

    def test02_python_introspection(self):
        """Introspection of newly created class/instances"""

        from cppjit.gbl import Integer1

        i = Integer1(42)
        assert hasattr(i, "m_data")
        assert not isinstance(i, int)
        assert isinstance(i, Integer1)

    @mark.xfail(
        condition=IS_MAC and IS_CLING, run=False, reason="Crashes on OS X Cling"
    )
    def test03_STL_containers(self):
        """Instantiate STL containers with new class"""

        from cppjit.gbl import Integer1
        from cppjit.gbl.std import vector

        v = vector[Integer1]()

        v += [Integer1(j) for j in range(10)]

    def test04_pretty_repr(self):
        """Create a pretty repr for the new class"""

        from cppjit.gbl import Integer1

        Integer1.__repr__ = lambda self: repr(self.m_data)
        assert str([Integer1(j) for j in range(10)]) == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"

    def test05_pythonizer(self):
        """Implement and test a pythonizor"""

        import cppjit

        def pythonizor(klass, name):
            if name == "Integer2":
                klass.__repr__ = lambda self: repr(self.m_data)

        cppjit.py.add_pythonization(pythonizor, "Math")

        Integer2 = (
            cppjit.gbl.Math.Integer2
        )  # first time a new namespace is used, it can not be imported from
        v = [Integer2(j) for j in range(10)]

        assert str(v) == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"

        i2 = Integer2(13)
        assert int(i2) == 13

    @mark.xfail(condition=IS_MAC and IS_CLING, reason="Fails on OSX Cling")
    def test06_add_operator(self):
        """Add operator+"""

        import cppjit

        cppjit.cppdef("""
namespace Math {
    Integer2 operator+(const Integer2& left, const Integer1& right) {
        return left.m_data + right.m_data;
    }
}""")

        from cppjit.gbl import Integer1
        from cppjit.gbl.Math import Integer2

        i = Integer1(42)
        i2 = Integer2(13)

        k = i2 + i
        assert int(k) == i2.m_data + i.m_data

    def test07_run_zoo(self):
        """Bunch of zoo animals running around"""

        from cppjit.gbl import Zoo

        assert raises(TypeError, Zoo.Animal)

        assert issubclass(Zoo.Lion, Zoo.Animal)

        mouse = Zoo.release_animal(Zoo.eMouse)
        assert type(mouse) == Zoo.Mouse
        lion = Zoo.release_animal(Zoo.eLion)
        assert type(lion) == Zoo.Lion

        assert lion.__python_owns__
        assert mouse.__python_owns__

        assert mouse.make_sound() == "peep!"
        assert lion.make_sound() == "growl!"

        Zoo.Lion.s_lion_sound = "mooh!"
        assert lion.make_sound() == "mooh!"

        assert Zoo.identify_animal(mouse) == "the animal is a mouse"
        assert Zoo.identify_animal(lion) == "the animal is a lion"

    def test08_shared_ptr(self):
        """Shared pointer transparency"""

        import cppjit

        cppjit.cppdef("""
namespace Zoo {
   std::shared_ptr<Lion> free_lion{new Lion{}};

   std::string identify_animal_smart(std::shared_ptr<Lion>& smart) {
       return "the animal is a lion";
   }
}
""")

        from cppjit.gbl import Zoo

        assert type(Zoo.free_lion).__name__ == "Lion"

        smart_lion = Zoo.free_lion.__smartptr__()
        assert type(smart_lion).__name__ in [
            "shared_ptr<Zoo::Lion>",
            "std::shared_ptr<Zoo::Lion>",
            "shared_ptr<Lion>",
            "std::shared_ptr<Lion>",
        ]

        assert Zoo.identify_animal(Zoo.free_lion) == "the animal is a lion"
        assert Zoo.identify_animal_smart(Zoo.free_lion) == "the animal is a lion"

    def test09_templated_function(self):
        """Templated free function"""

        import cppjit

        cppjit.cppdef("""

template<class A, class B, class C = A>
C multiply(A a, B b) {
return static_cast<C>(a * b);
}

""")
        mul = cppjit.gbl.multiply

        assert "multiply" in cppjit.gbl.__dict__

        assert mul(1, 2) == 2
        assert "multiply<int,int,int>" in map(
            lambda x: x.replace(" ", ""), cppjit.gbl.__dict__
        )
        assert mul(1.0, 5) == 5.0
        assert "multiply<double,int,double>" in map(
            lambda x: x.replace(" ", ""), cppjit.gbl.__dict__
        )

        assert mul[int](1, 1) == 1
        assert "multiply<int>" in map(lambda x: x.replace(" ", ""), cppjit.gbl.__dict__)
        assert mul[int, int](1, 1) == 1
        assert "multiply<int,int>" in map(
            lambda x: x.replace(" ", ""), cppjit.gbl.__dict__
        )

        # make sure cached values are actually looked up
        old = getattr(cppjit.gbl, "multiply<int,int>")
        setattr(cppjit.gbl, "multiply<int,int>", staticmethod(lambda x, y: 2 * x * y))
        assert mul[int, int](2, 2) == 8
        setattr(cppjit.gbl, "multiply<int,int>", old)
        assert mul[int, int](2, 2) == 4

        assert raises(TypeError, mul[int, int, int, int], 1, 1)
        assert raises(TypeError, mul[int, int], 1, 1.0)
        assert type(mul[int, int, float](1, 1)) == float
        # TODO: the following error message is rather confusing :(
        assert raises(TypeError, mul[int, int], 1, "a")

        assert mul["double, double, double"](1.0, 5) == 5.0

    @mark.xfail(condition=IS_MAC, reason="Fails on OSX")
    def test10_stl_algorithm(self):
        """STL algorithm on std::string"""

        import cppjit

        cppstr = cppjit.gbl.std.string
        n = cppstr("this is a C++ string")
        assert n == "this is a C++ string"
        n.erase(cppjit.gbl.std.remove(n.begin(), n.end(), cppstr.value_type(" ")))
        assert n == "thisisaC++stringing"


@mark.skipif(IS_MAC and IS_CLING, reason="setup class fails with OS X cling")
class TestADVERTISED:
    def setup_class(cls):
        pass

    def test01_reduction_of_overloads(self):
        """Reduce available overloads to 1"""

        import cppjit

        cppjit.cppdef("""namespace Advert01 {
        class A {
        public:
            A(int) {}
            A(double) {}
        }; }""")

        def pythonize_A(klass, name):
            if name == "A":
                klass.__init__ = klass.__init__.__overload__("int")

        cppjit.py.add_pythonization(pythonize_A, "Advert01")

        from cppjit.gbl import Advert01

        assert Advert01.A(1)
        raises(TypeError, Advert01.A, 1.0)

    def test02_use_c_void_p(self):
        """Use of opaque handles and ctypes.c_void_p"""

        import ctypes

        import cppjit

        ### void pointer as opaque handle
        cppjit.cppdef("""namespace Advert02 {
            typedef void* PicamHandle;
            void Picam_OpenFirstCamera(PicamHandle* cam) {
                *cam = new int(42);
            }

            bool Picam_CloseCamera(PicamHandle cam) {
                bool ret = false;
                if (*((int*)cam) == 42) ret = true;
                delete (int*)cam;
                return ret;
            }
        }""")

        from cppjit.gbl import Advert02

        assert Advert02.PicamHandle

        # first approach
        cam = Advert02.PicamHandle(cppjit.nullptr)
        Advert02.Picam_OpenFirstCamera(cam)
        assert Advert02.Picam_CloseCamera(cam)

        # second approch
        cam = ctypes.c_void_p()
        Advert02.Picam_OpenFirstCamera(cam)
        assert Advert02.Picam_CloseCamera(cam)

    def test03_use_of_ctypes_and_enum(self):
        """Use of (opaque) enum through ctypes.c_void_p"""

        import ctypes

        import cppjit

        cppjit.cppdef("""namespace Advert03 {
        enum SomeEnum1 { AA = -1, BB = 42 };
        void build_enum_array1(SomeEnum1** ptr, int* sz) {
            *ptr = (SomeEnum1*)malloc(sizeof(SomeEnum1)*4);
            *sz = 4;
            (*ptr)[0] = AA; (*ptr)[1] = BB; (*ptr)[2] = AA; (*ptr)[3] = BB;
        }

        enum SomeEnum2 { CC = 1, DD = 42 };
        void build_enum_array2(SomeEnum2** ptr, int* sz) {
            *ptr = (SomeEnum2*)malloc(sizeof(SomeEnum2)*4);
            *sz = 4;
            (*ptr)[0] = CC; (*ptr)[1] = DD; (*ptr)[2] = CC; (*ptr)[3] = DD;
        } }""")

        # enum through void pointer (b/c underlying type unknown)
        vp = ctypes.c_void_p(0)
        cnt = ctypes.c_int(0)
        cppjit.gbl.Advert03.build_enum_array2(vp, ctypes.pointer(cnt))
        assert cnt.value == 4

        vp = ctypes.c_void_p(0)
        cnt = ctypes.c_int(0)
        cppjit.gbl.Advert03.build_enum_array1(vp, ctypes.pointer(cnt))
        assert cnt.value == 4

        # helper to convert the enum array pointer & size to something packaged
        cppjit.cppdef("""namespace Advert03 {
        std::vector<SomeEnum1> ptr2vec(intptr_t ptr, int sz) {
            std::vector<SomeEnum1> result{(SomeEnum1*)ptr, (SomeEnum1*)ptr+sz};
            free((void*)ptr);
            return result;
        } }""")

        assert list(cppjit.gbl.Advert03.ptr2vec(vp.value, cnt.value)) == [
            -1,
            42,
            -1,
            42,
        ]

        # 2nd approach through low level cast
        vp = ctypes.pointer(cppjit.gbl.Advert03.SomeEnum2.__ctype__(0))
        cnt = ctypes.c_int(0)
        cppjit.gbl.Advert03.build_enum_array2(vp, ctypes.pointer(cnt))
        assert cnt.value == 4

        import cppjit.ll

        arr = cppjit.ll.cast["Advert03::SomeEnum2*"](vp)
        arr.reshape((cnt.value,))

        assert list(arr) == [1, 42, 1, 42]
        cppjit.gbl.free(vp)

    @mark.xfail(
        condition=IS_MAC and IS_CLING, run=False, reason="Crashes on OS X Cling"
    )
    def test04_ptr_ptr_python_owns(self):
        """Example of ptr-ptr use where python owns"""

        import cppjit

        cppjit.cppdef("""namespace Advert04 {
        struct SomeStruct {
            SomeStruct(int i) : i(i) {}
            int i;
        };

        int count_them(SomeStruct** them, int sz) {
            int total = 0;
            for (int i = 0; i < sz; ++i) total += them[i]->i;
            return total;
        } }""")

        cppjit.gbl.Advert04
        from cppjit.gbl.Advert04 import SomeStruct, count_them

        # initialization on python side
        v = cppjit.gbl.std.vector["Advert04::SomeStruct*"]()
        v._l = [SomeStruct(i) for i in range(10)]
        for s in v._l:
            v.push_back(s)
        assert count_them(v.data(), v.size()) == sum(range(10))

        # initialization on C++ side
        cppjit.cppdef("""namespace Advert04 {
        void ptr2ptr_init(SomeStruct** ref) {
            *ref = new SomeStruct(42);
        } }""")

        s = cppjit.bind_object(cppjit.nullptr, SomeStruct)
        cppjit.gbl.Advert04.ptr2ptr_init(s)
        assert s.i == 42

    def test05_ptr_ptr_with_array(self):
        """Example of ptr-ptr with array"""

        import ctypes

        import cppjit

        cppjit.cppdef("""namespace Advert05 {
        struct SomeStruct { int i; };

        void create_them(SomeStruct** them, int* sz) {
            *sz = 4;
            *them = new SomeStruct[*sz];
            for (int i = 0; i < *sz; ++i) (*them)[i].i = i*i;
        } }""")

        cppjit.gbl.Advert05
        from cppjit.gbl.Advert05 import SomeStruct, create_them

        ptr = cppjit.bind_object(cppjit.nullptr, SomeStruct)
        sz = ctypes.c_int(0)
        create_them(ptr, sz)

        arr = cppjit.bind_object(
            cppjit.addressof(ptr), cppjit.gbl.std.array[SomeStruct, sz.value]
        )
        total = 0
        for s in arr:
            total += s.i
        assert total == 14

    def test06_c_char_p(self):
        """Example of ctypes.c_char_p usage"""

        import ctypes

        import cppjit

        cppjit.cppdef("""namespace Advert06 {
        intptr_t createit(const char** out) {
            *out = (char*)malloc(4);
            return (intptr_t)*out;
        }
        intptr_t destroyit(const char* in) {
            intptr_t out = (intptr_t)in;
            free((void*)in);
            return out;
        } }""")

        cppjit.gbl.Advert06
        from cppjit.gbl.Advert06 import createit, destroyit

        ptr = ctypes.c_char_p()
        val = createit(ptr)
        assert destroyit(ptr) == val

    def test07_array_of_arrays(self):
        """Example of array of array usage"""

        import cppjit
        import cppjit.ll

        NREADOUTS = 4
        NPIXELS = 16

        cppjit.cppdef(
            """
        #define NREADOUTS %d
        #define NPIXELS %d
        namespace Advert07 {
        struct S {
            S() {
                uint16_t** readout = new uint16_t*[NREADOUTS];
                for (int i=0; i<4; ++i) {
                    readout[i] = new uint16_t[NPIXELS];
                    for (int j=0; j<NPIXELS; ++j) readout[i][j] = i*NPIXELS+j;
                }
                fField = (void*)readout;
            }
            ~S() {
                for (int i = 0; i < 4; ++i) delete[] ((uint16_t**)fField)[i];
                delete [] (uint16_t**)fField;
            }
            void* fField;
        }; }"""
            % (NREADOUTS, NPIXELS)
        )

        s = cppjit.gbl.Advert07.S()

        for i in range(NREADOUTS):
            image_array = cppjit.ll.cast["uint16_t*"](s.fField[i])
            for j in range(NPIXELS):
                assert image_array[j] == i * NPIXELS + j

    def test08_voidptr_array(self):
        """Example of access to array of void ptrs"""

        import cppjit

        cppjit.cppdef("""
        namespace VoidPtrArray {
            typedef struct _name {
                _name() { p[0] = (void*)0x1; p[1] = (void*)0x2; p[2] = (void*)0x3; }
                void* p[3];
            } name;
        }""")

        n = cppjit.gbl.VoidPtrArray.name()
        assert n.p[0] == 0x1
        assert n.p[1] == 0x2
        assert n.p[2] == 0x3
        assert len(n.p) == 3

    @mark.xfail(
        condition=IS_CLANG_REPL and IS_MAC,
        run=False,
        reason="Crashes with ClangRepl with 'toString not implemented'",
    )
    def test09_custom_str(self):
        """Example of customized str"""

        import cppjit

        cppjit.cppdef("""\
        namespace TopologicCore {

        class Shell {
        public:
            virtual std::string GetTypeAsString() const {
                return "hi there!";
            }
        }; }""")

        def pythonize_topologic_printing(klass, name):
            if "GetTypeAsString" in klass.__dict__:
                klass.__str__ = lambda self: str(self.GetTypeAsString())

        cppjit.py.add_pythonization(pythonize_topologic_printing, "TopologicCore")

        s = cppjit.gbl.TopologicCore.Shell()

        assert str(s) == "hi there!"

    @mark.xfail(condition=IS_MAC, reason="Fails on OS X")
    def test10_llvm_blog(self):
        """Test code posted in the LLVM blog posting"""

        import cppjit

        cppjit.cppdef(r"""\
        namespace LLVMBlog {
        template<typename T> class Producer {
        public:
            Producer(const T& value) : m_value(value) {}
            virtual ~Producer() {}

            T produce_total() { return m_value + produce_imp(); }

        protected:
            virtual T produce_imp() = 0;

        private:
            T m_value;
        };

        class Consumer {
        public:
            template<typename T>
            std::string consume(Producer<T>& p) {
                std::ostringstream s;
                s << "received: \"" << p.produce_total() << "\"";
                return s.str();
            }
        }; } """)

        ns = cppjit.gbl.LLVMBlog

        def factory(base_v, *derived_v):
            class _F(ns.Producer[type(base_v)]):
                def __init__(self, base_v, *derived_v):
                    if sys.hexversion < 0x3000000:
                        super(type(self), self).__init__(base_v)
                    else:
                        super().__init__(base_v)
                    self._values = derived_v

                def produce_imp(self):
                    return type(base_v)(sum(self._values))

            return _F(base_v, *derived_v)

        consumer = ns.Consumer()

        assert consumer.consume(factory("hello ", 42)) == 'received: "hello 42"'
        assert consumer.consume(factory(3.0, 0.14, 0.0015)) == 'received: "3.1415"'


# The series of tests below mostly exists already in other places, but these
# were used as examples for the CaaS' cppjit presentation and are preserved here.
@mark.skipif(IS_MAC and IS_CLING, reason="setup class fails with OS X cling")
class TestTALKEXAMPLES:
    def setup_class(cls):
        import cppjit

        cppjit.cppdef("""\
        namespace talk_examples {
        struct MyClass {
            MyClass(int i) : fData(i) {}
            virtual ~MyClass() {}
            virtual int add(int i) {
                return fData + i;
            }
            int fData;
        };}""")

        cppjit.gbl.talk_examples

    # @mark.skip
    def test_template_instantiation(self):
        """Run-time template instantiation example"""

        import cppjit
        import cppjit.gbl.talk_examples as CC

        v = cppjit.gbl.std.vector[CC.MyClass]()

        for i in range(10):
            v.emplace_back(i)

        assert len(v) == 10
        assert [m.fData for m in v] == list(range(10))

    @mark.xfail(condition=IS_LINUX_ARM, run=False, reason="Crashes pytest on Linux ARM")
    def test_cross_inheritance(self):
        """Cross-inheritance example"""

        import cppjit
        import cppjit.gbl.talk_examples as CC

        cppjit.cppdef("""\
        namespace talk_examples {
        int callb(MyClass* m, int i) {
          return m->add(i);
        }}""")

        class PyMyClass(CC.MyClass):
            def add(self, i):
                return self.fData + 2 * i

        m = PyMyClass(1)
        assert CC.callb(m, 2) == 5

    @mark.xfail(condition=IS_MAC_ARM, run=False, reason="Crashes on OS X arm")
    def test_cross_and_templates(self):
        """Template instantiation with cross-inheritance example"""

        import cppjit
        import cppjit.gbl.talk_examples as CC

        class PyMyClass(CC.MyClass):
            def __init__(self, data, extra):
                super(PyMyClass, self).__init__(data)
                self.extra = extra

            def add(self, i):
                return self.fData + self.extra + 2 * i

        v = cppjit.gbl.std.vector[PyMyClass]()
        v.push_back(PyMyClass(4, 42))

        assert v.back().add(17) == 4 + 42 + 2 * 17

    @mark.xfail(condition=IS_LINUX_ARM, run=False, reason="Crashes pytest on Linux ARM")
    def test_fallbacks(self):
        """Template instantation switches based on value sizes"""

        import cppjit
        import cppjit.gbl.talk_examples as CC

        cppjit.cppdef("""\
        namespace talk_examples {
        template<typename T>
        T passT(T t) {
            return t;
        }}""")

        assert CC.passT(1) == 1
        assert "int" in CC.passT.__doc__
        assert CC.passT(2**64 - 1) == 2**64 - 1
        assert "unsigned long long" in CC.passT.__doc__

    @mark.xfail(condition=IS_LINUX_ARM, run=False, reason="Crashes pytest on Linux ARM")
    def test_callbacks(self):
        """Function callback example"""

        import cppjit
        import cppjit.gbl.talk_examples as CC

        cppjit.cppdef("""\
        namespace talk_examples {
        typedef int (*P)(int);
        int callPtr(P f, int i) {
            return f(i);
        }

        typedef std::function<int(int)> F;
        int callFun(const F& f, int i) {
            return f(i);
        }}""")

        def f(val):
            return 2 * val

        assert CC.callPtr(f, 2) == 4
        assert CC.callFun(f, 3) == 6
        assert CC.callPtr(lambda i: 5 * i, 4) == 20
        assert CC.callFun(lambda i: 6 * i, 4) == 24

    @mark.xfail(
        condition=IS_VALGRIND and IS_LINUX_ARM,
        run=False,
        reason="Crashes on Valgrind-ARM",
    )
    def test_templated_callback(self):
        """Templated callback example"""

        import cppjit
        import cppjit.gbl.talk_examples as CC

        cppjit.cppdef("""\
        namespace talk_examples {
        template<typename R, typename... U, typename... A>
        R callT(R (*f)(U...), A&&... args) {
            return f(args...);
        }}""")

        def ann_f1(arg: "int") -> "double":  # noqa: F821
            return 3.1415 * arg

        def ann_f2(arg1: "int", arg2: "int") -> "int":
            return 3 * arg1 * arg2

        assert round(CC.callT(ann_f1, 2) - 2 * 3.1415, 5) == 0.0

        assert CC.callT(ann_f2, 6, 7) == 3 * 6 * 7
        assert round(CC.callT(ann_f1, 2) - 2 * 3.1415, 5) == 0.0

    def test_autocast_and_identiy(self):
        """Auto-cast and identiy preservation example"""

        import cppjit
        import cppjit.gbl.talk_examples as CC

        cppjit.cppdef("""\
        namespace talk_examples {
        struct Base {
            virtual ~Base() {}
        };
        struct Derived : public Base {};
        Base* passB(Base* b) { return b; }
        }""")

        d = CC.Derived()
        b = CC.passB(d)

        assert type(b) == CC.Derived
        assert d is b

    @mark.xfail(condition=IS_MAC_ARM, run=False, reason="Seg Faults")
    def test_exceptions(self):
        """Exceptions example"""

        if ispypy or IS_WINDOWS:
            skip("throwing exceptions from the JIT terminates the process")

        import cppjit
        import cppjit.gbl.talk_examples as CC

        cppjit.cppdef("""\
        namespace talk_examples {
        class MyException : public std::exception {
        public:
           const char* what() const throw() {
               return "C++ failed";
           }
        };
        void throw_error() {
            throw MyException{};
        }}""")

        with raises(CC.MyException):
            CC.throw_error()

    def test_unicode(self):
        """Unicode non-UTF-8 example"""

        import cppjit
        import cppjit.gbl.talk_examples as CC

        cppjit.cppdef("""\
        namespace talk_examples {
        template<class T>
        std::string to_str(const T& chars) {
            char buf[12]; int n = 0;
            for (auto c : chars) buf[n++] = char(c);
            return std::string(buf, n-1);
        }
        std::string utf8_chinese() {
            auto chars = {0xe4, 0xb8, 0xad, 0xe6, 0x96, 0x87, 0};
            return to_str(chars);
        }
        std::string gbk_chinese() {
            auto chars = {0xd6, 0xd0, 0xce, 0xc4, 0};
            return to_str(chars);
        }}""")

        with raises(Exception) as exc_info:
            CC.gbk_chinese().decode("utf-8")
        assert isinstance(exc_info.value, (UnicodeDecodeError, LookupError))

        assert CC.gbk_chinese() == "\u4e2d\u6587".encode("gbk")
        if 0x3000000 <= sys.hexversion:
            assert CC.utf8_chinese() == "\u4e2d\u6587"
        else:
            assert CC.utf8_chinese() == b"\xe4\xb8\xad\xe6\x96\x87"
