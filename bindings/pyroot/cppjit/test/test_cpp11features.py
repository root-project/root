import sys

import py
from pytest import mark, raises
from support import (
    IS_LINUX,
    IS_LINUX_ARM,
    IS_MAC,
    IS_VALGRIND,
    ispypy,
    setup_make,
)

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/cpp11featuresDict"))


def setup_module(mod):
    setup_make("cpp11features")


class TestCPP11FEATURES:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        cls.cpp11features = cppjit.load_reflection_info(cls.test_dct)

    @mark.xfail(condition=IS_LINUX_ARM, run=False, reason="Crashes pytest on Linux ARM")
    def test01_smart_ptr(self):
        """Usage and access of std::shared/unique_ptr<>"""

        import gc

        from cppjit.gbl import (
            TestSmartPtr,
            create_shared_ptr_instance,
            create_unique_ptr_instance,
        )

        for cf in [create_shared_ptr_instance, create_unique_ptr_instance]:
            assert TestSmartPtr.s_counter == 0

            ptr1 = cf()
            assert ptr1
            assert not not ptr1
            assert TestSmartPtr.s_counter == 1

            ptr2 = cf()
            assert ptr2
            assert not not ptr2
            assert TestSmartPtr.s_counter == 2

            del ptr2
            gc.collect()
            assert TestSmartPtr.s_counter == 1

            del ptr1
            gc.collect()
            assert TestSmartPtr.s_counter == 0

    @mark.xfail(
        condition=IS_LINUX_ARM and IS_VALGRIND,
        run=False,
        reason="Valgrind issues on ARM",
    )
    def test02_smart_ptr_construction(self):
        """Shared/Unique pointer ctor is templated, requiring special care"""

        import gc

        from cppjit.gbl import TestSmartPtr, std

        class C(TestSmartPtr):
            pass

        for cls in [std.shared_ptr, std.unique_ptr]:
            assert TestSmartPtr.s_counter == 0

            c = C()
            assert TestSmartPtr.s_counter == 1
            c.__python_owns__ = False
            cc = cls[TestSmartPtr](c)
            assert cc.__python_owns__

            del c

            assert cc
            assert TestSmartPtr.s_counter == 1

            del cc
            gc.collect()
            assert TestSmartPtr.s_counter == 0

    @mark.xfail(condition=IS_LINUX and IS_VALGRIND, run=False, reason="Valgrind issue")
    def test03_smart_ptr_memory_handling(self):
        """Test shared/unique pointer memory ownership"""

        import gc

        from cppjit.gbl import TestSmartPtr, std

        class C(TestSmartPtr):
            pass

        for cls in [std.shared_ptr, std.unique_ptr]:
            assert TestSmartPtr.s_counter == 0

            t = TestSmartPtr()
            assert TestSmartPtr.s_counter == 1
            assert t.__python_owns__

            tt = std.shared_ptr[TestSmartPtr](t)
            assert not t.__python_owns__

            c = C()
            assert TestSmartPtr.s_counter == 2
            assert c.__python_owns__

            cc = std.shared_ptr[TestSmartPtr](c)
            assert not c.__python_owns__

            del cc, tt
            gc.collect()
            assert TestSmartPtr.s_counter == 0

    @mark.xfail(condition=IS_VALGRIND, run=False, reason="Crashes on Valgrind")
    def test04_shared_ptr_passing(self):
        """Ability to pass shared_ptr<Derived> through shared_ptr<Base>"""

        import gc

        from cppjit.gbl import (
            DerivedTestSmartPtr,
            TestSmartPtr,
            create_TestSmartPtr_by_value,
            move_shared_ptr,
            pass_shared_ptr,
            std,
        )

        for ff, mv in [(pass_shared_ptr, lambda x: x), (move_shared_ptr, std.move)]:
            assert TestSmartPtr.s_counter == 0

            dd = std.make_shared[DerivedTestSmartPtr](DerivedTestSmartPtr(24))
            assert TestSmartPtr.s_counter == 1
            assert ff(mv(dd)) == 100

            del dd
            gc.collect()
            assert TestSmartPtr.s_counter == 0

        # ability to take over by-value python-owned objects
        tsp = create_TestSmartPtr_by_value()
        assert TestSmartPtr.s_counter == 1
        assert tsp.__python_owns__

        shared_stp = std.make_shared[TestSmartPtr](tsp)
        assert TestSmartPtr.s_counter == 1
        assert not tsp.__python_owns__

        del shared_stp
        gc.collect()
        assert TestSmartPtr.s_counter == 0

        # alternative make_shared with type taken from pointer
        tsp = create_TestSmartPtr_by_value()
        shared_stp = std.make_shared(tsp)
        assert TestSmartPtr.s_counter == 1
        del shared_stp
        gc.collect()
        assert TestSmartPtr.s_counter == 0

    def test05_unique_ptr_passing(self):
        """Ability to pass unique_ptr<Derived> through unique_ptr<Base>"""

        import gc

        from cppjit.gbl import (
            DerivedTestSmartPtr,
            TestSmartPtr,
            create_TestSmartPtr_by_value,
            move_unique_ptr,  # noqa: F401
            move_unique_ptr_derived,
            std,
        )

        assert TestSmartPtr.s_counter == 0

        # move matching unique_ptr
        dd = std.make_unique[DerivedTestSmartPtr](DerivedTestSmartPtr(24))
        assert TestSmartPtr.s_counter == 1
        assert move_unique_ptr_derived(std.move(dd)) == 100
        assert dd.__python_owns__

        del dd
        gc.collect()
        assert TestSmartPtr.s_counter == 0

        # move with conversion
        dd = std.make_unique[DerivedTestSmartPtr](DerivedTestSmartPtr(24))
        assert TestSmartPtr.s_counter == 1
        # TODO: why does the following fail, but succeed for shared_ptr??
        # assert move_unique_ptr(std.move(dd)) == 100
        assert dd.__python_owns__

        del dd
        gc.collect()
        assert TestSmartPtr.s_counter == 0

        # ability to take over by-value python-owned objects
        tsp = create_TestSmartPtr_by_value()
        assert TestSmartPtr.s_counter == 1
        assert tsp.__python_owns__

        unique_stp = std.make_unique[TestSmartPtr](tsp)
        assert TestSmartPtr.s_counter == 1
        assert not tsp.__python_owns__

        del unique_stp
        gc.collect()
        assert TestSmartPtr.s_counter == 0

        # alternative make_unique with type taken from pointer
        tsp = create_TestSmartPtr_by_value()
        unique_stp = std.make_unique(tsp)
        assert TestSmartPtr.s_counter == 1

        del unique_stp
        gc.collect()
        assert TestSmartPtr.s_counter == 0

    def test06_nullptr(self):
        """Allow the programmer to pass NULL in certain cases"""

        import cppjit

        # test existence
        nullptr = cppjit.nullptr
        # assert not hasattr(cppjit.gbl, 'nullptr')

        assert cppjit.bind_object(cppjit.nullptr, "std::vector<int>") == cppjit.nullptr
        assert (
            not cppjit.bind_object(cppjit.nullptr, "std::vector<int>") != cppjit.nullptr
        )

    # further usage is tested in datatypes.py:test15_nullptr_passing

    def test07_move(self):
        """Move construction, assignment, and methods"""

        import gc

        import cppjit

        def moveit(T):
            assert T.s_instance_counter == 0

            from cppjit.gbl import std

            # move constructor
            i1 = T()
            assert T.s_move_counter == 0

            i2 = T(i1)  # cctor
            assert T.s_move_counter == 0

            if ispypy or 0x3000000 <= sys.hexversion:
                i3 = T(std.move(T()))  # can't check ref-count
            else:
                i3 = T(T())  # should call move, not memoized cctor
            assert T.s_move_counter == 1

            i3 = T(std.move(T()))  # both move and ref-count
            assert T.s_move_counter == 2

            i4 = T(std.move(i1))
            assert T.s_move_counter == 3

            # move assignment
            i4.__assign__(i2)
            assert T.s_move_counter == 3

            if ispypy or 0x3000000 <= sys.hexversion:
                i4.__assign__(std.move(T()))  # can't check ref-count
            else:
                i4.__assign__(T())
            assert T.s_move_counter == 4

            i4.__assign__(std.move(i2))
            assert T.s_move_counter == 5

            del i1, i2, i3, i4
            gc.collect()
            assert T.s_instance_counter == 0

        # order of moving and normal functions are reversed in 1, 2, for
        # overload resolution testing
        moveit(cppjit.gbl.TestMoving1)
        moveit(cppjit.gbl.TestMoving2)

        # implicit conversion and move
        assert cppjit.gbl.TestMoving1.s_instance_counter == 0
        assert cppjit.gbl.TestMoving2.s_instance_counter == 0
        cppjit.gbl.implicit_converion_move(cppjit.gbl.TestMoving1())
        cppjit.gbl.implicit_converion_move(cppjit.gbl.TestMoving2())
        gc.collect()
        assert cppjit.gbl.TestMoving1.s_instance_counter == 0
        assert cppjit.gbl.TestMoving2.s_instance_counter == 0

    @mark.xfail(condition=IS_MAC, reason="Fails on OSX")
    def test08_initializer_list(self):
        """Initializer list construction"""

        from cppjit.gbl import TestData, TestData2, WithInitList, std

        v = std.vector[int]((1, 2, 3, 4))
        assert list(v) == [1, 2, 3, 4]

        v = std.vector["double"]((1, 2, 3, 4))
        assert list(v) == [1.0, 2.0, 3.0, 4.0]

        raises(TypeError, std.vector[int], [1.0, 2.0, 3.0, 4.0])

        for cls in [std.vector, WithInitList]:
            for cls_arg in [TestData, TestData2]:
                l = list()
                for i in range(10):
                    l.append(cls_arg(i))

                v = cls[cls_arg](l)
                assert len(v) == len(l)
                for i in range(len(l)):
                    assert v[i].m_int == l[i].m_int

        import cppjit

        cppjit.cppdef(r"""
        namespace InitializerListTest {
        std::vector<std::string> foo(const std::initializer_list<std::string>& vals) {
            return std::vector<std::string>{vals};
        } }""")

        ns = cppjit.gbl.InitializerListTest

        for l in (["x"], ["x", "y", "z"]):
            assert ns.foo(l) == std.vector["std::string"](l)

    def test09_lambda_calls(self):
        """Call (global) lambdas"""

        import cppjit

        cppjit.cppdef("auto gMyLambda = [](int a) { return 40 + a; };")

        assert cppjit.gbl.gMyLambda
        assert cppjit.gbl.gMyLambda(2) == 42
        assert cppjit.gbl.gMyLambda(40) == 80

        cppjit.cppdef("auto gime_a_lambda1() { return []() { return 42; }; }")
        l1 = cppjit.gbl.gime_a_lambda1()
        assert l1
        assert l1() == 42

        cppjit.cppdef(
            "auto gime_a_lambda2() { int a = 4; return [a](int b) { return 42+a+b; }; }"
        )
        l2 = cppjit.gbl.gime_a_lambda2()
        assert l2
        assert l2(2) == 48

        cppjit.cppdef(
            "auto gime_a_lambda3(int a ) { return [a](int b) { return 42+a+b; }; }"
        )
        l3 = cppjit.gbl.gime_a_lambda3(4)
        assert l3
        assert l3(2) == 48

    def test10_optional(self):
        """Use of optional and nullopt"""

        import cppjit

        assert cppjit.gbl.std.optional
        assert cppjit.gbl.std.nullopt

        cppjit.cppdef("""
        enum Enum { A = -1 };
        bool callopt(std::optional<Enum>) { return true; }
        """)

        a = cppjit.gbl.std.optional[cppjit.gbl.Enum]()
        assert cppjit.gbl.callopt(a)

        c = cppjit.gbl.std.nullopt
        assert cppjit.gbl.callopt(c)

    def test11_chrono(self):
        """Use of chrono and overloaded operator+"""

        from cppjit.gbl import std

        t = std.chrono.system_clock.now() - std.chrono.seconds(1)
        # following used to fail with compilation error
        t = std.chrono.system_clock.now() + std.chrono.seconds(1)

    def test12_stdfunction(self):
        """Use of std::function with arguments in a namespace"""

        import cppjit
        from cppjit.gbl import FNCreateTestStructFunc, FNTestStruct, FunctionNS

        t = FNTestStruct(42)
        f = FNCreateTestStructFunc()
        assert f(t) == 42

        t = FunctionNS.FNTestStruct(13)
        f = FunctionNS.FNCreateTestStructFunc()
        assert f(t) == 13

        # and for good measure, inline
        cppjit.cppdef("""namespace FunctionNS2 {
        struct FNTestStruct { FNTestStruct(int i) : t(i) {} int t; };
        std::function<int(const FNTestStruct& t)> FNCreateTestStructFunc() { return [](const FNTestStruct& t) { return t.t; }; }
        }""")

        from cppjit.gbl import FunctionNS2  # noqa: F401

        t = FunctionNS.FNTestStruct(27)
        f = FunctionNS.FNCreateTestStructFunc()
        assert f(t) == 27

    def test13_stdhash(self):
        """Use of std::hash"""

        from cppjit.gbl import StructWithHash, StructWithoutHash

        for i in range(3):  # to test effect of caching
            swo = StructWithoutHash()
            assert hash(swo) == object.__hash__(swo)
            assert hash(swo) == object.__hash__(swo)

            sw = StructWithHash()
            assert hash(sw) == 17
            assert hash(sw) == 17

    @mark.xfail(reason="plain pointer does not convert to a shared_ptr argument")
    def test14_shared_ptr_passing(self):
        """Ability to pass normal pointers through shared_ptr by value"""

        import gc

        from cppjit.gbl import (  # noqa: F401
            DerivedTestSmartPtr,
            TestSmartPtr,
            pass_shared_ptr,
            std,
        )

        for cls, val in [
            (lambda: TestSmartPtr(), 17),
            (lambda: DerivedTestSmartPtr(24), 100),
        ]:
            assert TestSmartPtr.s_counter == 0

            obj = cls()

            assert TestSmartPtr.s_counter == 1
            assert not obj.__smartptr__()
            assert pass_shared_ptr(obj) == val
            assert obj.__smartptr__()
            assert obj.__python_owns__
            assert TestSmartPtr.s_counter == 1

            assert not not obj  # pass was by shared copy

            del obj
            gc.collect()
            assert TestSmartPtr.s_counter == 0

    @mark.xfail(condition=IS_MAC, reason="Fails for MacOS 26")
    def test15_unique_ptr_template_deduction(self):
        """Argument type deduction with std::unique_ptr"""

        import cppjit

        cppjit.cppdef("""namespace UniqueTempl {
        template <typename T>
        std::unique_ptr<T> returnptr(std::unique_ptr<T>&& a) {
          return std::move(a);
        } }""")

        uptr_in = cppjit.gbl.std.make_unique[int]()
        uptr_out = cppjit.gbl.UniqueTempl.returnptr["int"](cppjit.gbl.std.move(uptr_in))
        assert not not uptr_out

        uptr_in = cppjit.gbl.std.make_unique["int"]()
        with raises(ValueError):  # not an RValue
            cppjit.gbl.UniqueTempl.returnptr[int](uptr_in)

    @mark.xfail(condition=IS_MAC, reason="Fails on Mac platforms")
    def test16_unique_ptr_moves(self):
        """std::unique_ptr requires moves"""

        import cppjit

        cppjit.cppdef("""namespace unique_ptr_moves {
        template <typename T>
        std::unique_ptr<T> returnptr_value(std::unique_ptr<T> a) {
          return std::move(a);
        }
        template <typename T>
        std::unique_ptr<T> returnptr_move(std::unique_ptr<T>&& a) {
          return std::move(a);
        } }""")

        up = cppjit.gbl.std.make_unique[int](42)

        ns = cppjit.gbl.unique_ptr_moves
        up = ns.returnptr_value(up)
        assert up and up.get()[0] == 42
        up = ns.returnptr_value(cppjit.gbl.std.move(up))
        assert up and up.get()[0] == 42
        up = ns.returnptr_move(cppjit.gbl.std.move(up))
        assert up and up.get()[0] == 42

        with raises(TypeError):
            ns.returnptr_move(up)

    def test17_unique_ptr_data(self):
        """std::unique_ptr as data means implicitly no copy ctor"""

        import cppjit

        cppjit.cppdef("""namespace unique_ptr_data{
        class Example {
        private:
          std::unique_ptr<double> x;
        public:
          Example() {}
          virtual ~Example() = default;
          double y = 66.;
        }; }""")

        class Inherit(cppjit.gbl.unique_ptr_data.Example):
            pass

        a = Inherit()
        # Test whether this attribute was inherited
        assert a.y == 66.0

    def test18_unique_ptr_identity(self):
        """std::unique_ptr identity preservation"""

        import cppjit

        cppjit.cppdef("""\
        namespace UniqueIdentity {
        struct A {
            A(int _a) : a(_a) {}
            int a;
        };

        std::unique_ptr<A> create() { return std::make_unique<A>(37); }

        struct Consumer {
        public:
            Consumer(std::unique_ptr<A> & ptr) : fPtr{std::move(ptr)} {
                ptr.reset();
            }

            const A& get() const { return *fPtr; }
            const std::unique_ptr<A>& pget() const { return fPtr; }

        private:
            std::unique_ptr<A> fPtr;
        }; }""")

        ns = cppjit.gbl.UniqueIdentity

        x = ns.create()
        assert x.a == 37

        c = ns.Consumer(x)
        x = c.get()
        assert x.a == 37

        p1 = c.pget()
        p2 = c.pget()
        assert p1 is p2

    @mark.xfail(
        condition=IS_LINUX_ARM and IS_VALGRIND,
        run=False,
        reason="Valgrind issues on ARM",
    )
    def test19_smartptr_from_callback(self):
        """Return a smart pointer from a callback"""

        import cppjit

        cppjit.cppdef(r"""\
        namespace SmartPtrCallback {
        struct Dummy {
            virtual ~Dummy() = default;
        };

        std::shared_ptr<Dummy> dummy_create() {
            return std::make_shared<Dummy>();
        }

        typedef std::shared_ptr<Dummy> (*fff)();

        std::shared_ptr<Dummy> call_creator(fff func) {
            return func();
        }}""")

        std = cppjit.gbl.std
        ns = cppjit.gbl.SmartPtrCallback

        def pyfunc() -> std.shared_ptr[ns.Dummy]:
            return ns.dummy_create()

        assert ns.call_creator(pyfunc)

    def test21_smart_ptr_downcast(self):
        """Object returned through a smart pointer is auto-downcast"""

        import cppjit

        gbl = cppjit.gbl

        # unique_ptr<Base> holding a Derived comes back as Derived, with the
        # derived-only method callable, just like a raw pointer return
        for cf in [gbl.create_unique_ptr_to_derived, gbl.create_shared_ptr_to_derived]:
            obj = cf()
            assert type(obj) == gbl.PubDerivedTestSmartPtr
            assert obj.only_in_derived() == 27
            assert obj.__smartptr__()  # smart-pointer semantics preserved

        # an object that really is of the declared type stays that type
        obj = gbl.create_unique_ptr_instance()
        assert type(obj) == gbl.TestSmartPtr

        # the most derived type sits at a non-zero offset from the declared
        # interface, which the dereferencer can not apply: stay the declared
        # type and keep behaving correctly
        obj = gbl.create_unique_ptr_to_offset_derived()
        assert type(obj) == gbl.TestSmartPtrIface
        assert obj.only_in_iface() == 37

        # the auto-down-cast must not enable C++-invalid conversions: the proxy
        # still embeds a smart pointer to the *base* type, which does not convert
        # to a smart pointer to the derived type (no implicit down-conversion of
        # smart pointers in C++), so passing it to such a sink must be rejected
        raises(
            TypeError,
            gbl.pass_unique_ptr_to_derived,
            gbl.create_unique_ptr_to_derived(),
        )
        raises(
            TypeError,
            gbl.pass_shared_ptr_to_derived,
            gbl.create_shared_ptr_to_derived(),
        )

        # passing it where the matching base smart pointer is expected still works
        assert gbl.pass_shared_ptr(gbl.create_shared_ptr_to_derived()) == 17

        # calling function with overloads for both the base class and the
        # derived class should resolve to the downcasted type overload,
        # no matter if the Python proxy is a regular proxy or wraps a smart pointer
        # (should hold for pointer, reference, and value types)
        assert (
            gbl.pass_ptr_overloaded(gbl.PubDerivedTestSmartPtr())
            == "PubDerivedTestSmartPtr"
        )
        assert (
            gbl.pass_ptr_overloaded(gbl.create_unique_ptr_to_derived())
            == "PubDerivedTestSmartPtr"
        )
        assert (
            gbl.pass_ref_overloaded(gbl.PubDerivedTestSmartPtr())
            == "PubDerivedTestSmartPtr"
        )
        assert (
            gbl.pass_ref_overloaded(gbl.create_unique_ptr_to_derived())
            == "PubDerivedTestSmartPtr"
        )
        assert (
            gbl.pass_val_overloaded(gbl.PubDerivedTestSmartPtr())
            == "PubDerivedTestSmartPtr"
        )
        assert (
            gbl.pass_val_overloaded(gbl.create_unique_ptr_to_derived())
            == "PubDerivedTestSmartPtr"
        )
