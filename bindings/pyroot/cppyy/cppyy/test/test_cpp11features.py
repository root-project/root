import py, sys, pytest, os
from pytest import mark, raises
from support import setup_make, ispypy, IS_MAC_ARM


currpath = os.getcwd()
test_dct = currpath + "/libcpp11featuresDict"


class TestCPP11FEATURES:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.cpp11features = cppyy.load_reflection_info(cls.test_dct)

    def test01_smart_ptr(self):
        """Usage and access of std::shared/unique_ptr<>"""

        from cppyy.gbl import TestSmartPtr
        from cppyy.gbl import create_shared_ptr_instance, create_unique_ptr_instance
        import gc

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

    def test02_smart_ptr_construction(self):
        """Shared/Unique pointer ctor is templated, requiring special care"""

        from cppyy.gbl import std, TestSmartPtr
        import gc

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

    def test03_smart_ptr_memory_handling(self):
        """Test shared/unique pointer memory ownership"""

        from cppyy.gbl import std, TestSmartPtr
        import gc

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

    def test04_shared_ptr_passing(self):
        """Ability to pass shared_ptr<Derived> through shared_ptr<Base>"""

        from cppyy.gbl import std, TestSmartPtr, DerivedTestSmartPtr
        from cppyy.gbl import pass_shared_ptr, move_shared_ptr, create_TestSmartPtr_by_value
        import gc

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

        from cppyy.gbl import std, TestSmartPtr, DerivedTestSmartPtr
        from cppyy.gbl import move_unique_ptr, move_unique_ptr_derived
        from cppyy.gbl import create_TestSmartPtr_by_value
        import gc

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

        import cppyy

      # test existence
        nullptr = cppyy.nullptr
      # assert not hasattr(cppyy.gbl, 'nullptr')

        assert     cppyy.bind_object(cppyy.nullptr, 'std::vector<int>') == cppyy.nullptr
        assert not cppyy.bind_object(cppyy.nullptr, 'std::vector<int>') != cppyy.nullptr

      # further usage is tested in datatypes.py:test15_nullptr_passing

    def test07_move(self):
        """Move construction, assignment, and methods"""

        import cppyy, gc

        def moveit(T):
            assert T.s_instance_counter == 0

            from cppyy.gbl import std

          # move constructor
            i1 = T()
            assert T.s_move_counter == 0

            i2 = T(i1)  # cctor
            assert T.s_move_counter == 0

            if ispypy or 0x3000000 <= sys.hexversion:
                i3 = T(std.move(T()))            # can't check ref-count
            else:
                i3 = T(T()) # should call move, not memoized cctor
            assert T.s_move_counter == 1

            i3 = T(std.move(T()))                # both move and ref-count
            assert T.s_move_counter == 2

            i4 = T(std.move(i1))
            assert T.s_move_counter == 3

          # move assignment
            i4.__assign__(i2)
            assert T.s_move_counter == 3

            if ispypy or 0x3000000 <= sys.hexversion:
                i4.__assign__(std.move(T()))     # can't check ref-count
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
        moveit(cppyy.gbl.TestMoving1)
        moveit(cppyy.gbl.TestMoving2)

      # implicit conversion and move
        assert cppyy.gbl.TestMoving1.s_instance_counter == 0
        assert cppyy.gbl.TestMoving2.s_instance_counter == 0
        cppyy.gbl.implicit_converion_move(cppyy.gbl.TestMoving1())
        cppyy.gbl.implicit_converion_move(cppyy.gbl.TestMoving2())
        gc.collect()
        assert cppyy.gbl.TestMoving1.s_instance_counter == 0
        assert cppyy.gbl.TestMoving2.s_instance_counter == 0

    def test08_initializer_list(self):
        """Initializer list construction"""

        from cppyy.gbl import std, TestData, TestData2, WithInitList

        v = std.vector[int]((1, 2, 3, 4))
        assert list(v) == [1, 2, 3, 4]

        v = std.vector['double']((1, 2, 3, 4))
        assert list(v) == [1., 2., 3., 4.]

        raises(TypeError, std.vector[int], [1., 2., 3., 4.])

        for cls in [std.vector, WithInitList]:
            for cls_arg in [TestData, TestData2]:
                l = list()
                for i in range(10):
                    l.append(cls_arg(i))

                v = cls[cls_arg](l)
                assert len(v) == len(l)
                for i in range(len(l)):
                    assert v[i].m_int == l[i].m_int

        import cppyy

        cppyy.cppdef(r"""
        namespace InitializerListTest {
        std::vector<std::string> foo(const std::initializer_list<std::string>& vals) {
            return std::vector<std::string>{vals};
        } }""")

        ns = cppyy.gbl.InitializerListTest

        for l in (['x'], ['x', 'y', 'z']):
            assert ns.foo(l) == std.vector['std::string'](l)

    @mark.xfail()
    def test09_lambda_calls(self):
        """Call (global) lambdas"""

        import cppyy

        cppyy.cppdef("auto gMyLambda = [](int a) { return 40 + a; };")

        assert cppyy.gbl.gMyLambda
        assert cppyy.gbl.gMyLambda(2)  == 42
        assert cppyy.gbl.gMyLambda(40) == 80

        if cppyy.gbl.gInterpreter.ProcessLine("__cplusplus;") >= 201402:
            cppyy.cppdef("auto gime_a_lambda1() { return []() { return 42; }; }")
            l1 = cppyy.gbl.gime_a_lambda1()
            assert l1
            assert l1() == 42

            cppyy.cppdef("auto gime_a_lambda2() { int a = 4; return [a](int b) { return 42+a+b; }; }")
            l2 = cppyy.gbl.gime_a_lambda2()
            assert l2
            assert l2(2) == 48

            cppyy.cppdef("auto gime_a_lambda3(int a ) { return [a](int b) { return 42+a+b; }; }")
            l3 = cppyy.gbl.gime_a_lambda3(4)
            assert l3
            assert l3(2) == 48

    def test10_optional(self):
        """Use of optional and nullopt"""

        import cppyy

        if 201703 <= cppyy.gbl.gInterpreter.ProcessLine("__cplusplus;"):
            assert cppyy.gbl.std.optional
            assert cppyy.gbl.std.nullopt

            cppyy.cppdef("""
            enum Enum { A = -1 };
            bool callopt(std::optional<Enum>) { return true; }
            """)

            a = cppyy.gbl.std.optional[cppyy.gbl.Enum]()
            assert cppyy.gbl.callopt(a)

            c = cppyy.gbl.std.nullopt
            assert cppyy.gbl.callopt(c)

    @mark.xfail(run = False, reason = "Crashes")
    def test11_chrono(self):
        """Use of chrono and overloaded operator+"""

        import cppyy
        from cppyy.gbl import std

        t = std.chrono.system_clock.now() - std.chrono.seconds(1)
        # following used to fail with compilation error
        t = std.chrono.system_clock.now() + std.chrono.seconds(1)

    @mark.xfail()
    def test12_stdfunction(self):
        """Use of std::function with arguments in a namespace"""

        import cppyy
        from cppyy.gbl import FunctionNS, FNTestStruct, FNCreateTestStructFunc

        t = FNTestStruct(42)
        f = FNCreateTestStructFunc()
        assert f(t) == 42

        t = FunctionNS.FNTestStruct(13)
        f = FunctionNS.FNCreateTestStructFunc()
        assert f(t) == 13

      # and for good measure, inline
        cppyy.cppdef("""namespace FunctionNS2 {
        struct FNTestStruct { FNTestStruct(int i) : t(i) {} int t; };
        std::function<int(const FNTestStruct& t)> FNCreateTestStructFunc() { return [](const FNTestStruct& t) { return t.t; }; }
        }""")

        from cppyy.gbl import FunctionNS2

        t = FunctionNS.FNTestStruct(27)
        f = FunctionNS.FNCreateTestStructFunc()
        assert f(t) == 27

    def test13_stdhash(self):
        """Use of std::hash"""

        import cppyy
        from cppyy.gbl import StructWithHash, StructWithoutHash

        for i in range(3):   # to test effect of caching
            swo = StructWithoutHash()
            assert hash(swo) == object.__hash__(swo)
            assert hash(swo) == object.__hash__(swo)

            sw = StructWithHash()
            assert hash(sw)  == 17
            assert hash(sw)  == 17

    @mark.xfail()
    def test14_shared_ptr_passing(self):
        """Ability to pass normal pointers through shared_ptr by value"""

        from cppyy.gbl import std, TestSmartPtr, DerivedTestSmartPtr
        from cppyy.gbl import pass_shared_ptr
        import gc

        for cls, val in [(lambda: TestSmartPtr(), 17), (lambda: DerivedTestSmartPtr(24), 100)]:
            assert TestSmartPtr.s_counter == 0

            obj = cls()

            assert TestSmartPtr.s_counter == 1
            assert not obj.__smartptr__()
            assert pass_shared_ptr(obj) == val
            assert obj.__smartptr__()
            assert obj.__python_owns__
            assert TestSmartPtr.s_counter == 1

            assert not not obj    # pass was by shared copy

            del obj
            gc.collect()
            assert TestSmartPtr.s_counter == 0

    def test15_unique_ptr_template_deduction(self):
        """Argument type deduction with std::unique_ptr"""

        import cppyy

        cppyy.cppdef("""namespace UniqueTempl {
        template <typename T>
        std::unique_ptr<T> returnptr(std::unique_ptr<T>&& a) {
          return std::move(a);
        } }""")

        uptr_in = cppyy.gbl.std.make_unique[int]()
        uptr_out = cppyy.gbl.UniqueTempl.returnptr["int"](cppyy.gbl.std.move(uptr_in))
        assert not not uptr_out

        uptr_in = cppyy.gbl.std.make_unique['int']()
        with raises(ValueError):  # not an RValue
            cppyy.gbl.UniqueTempl.returnptr[int](uptr_in)

    def test16_unique_ptr_moves(self):
        """std::unique_ptr requires moves"""

        import cppyy
        cppyy.cppdef("""namespace unique_ptr_moves {
        template <typename T>
        std::unique_ptr<T> returnptr_value(std::unique_ptr<T> a) {
          return std::move(a);
        }
        template <typename T>
        std::unique_ptr<T> returnptr_move(std::unique_ptr<T>&& a) {
          return std::move(a);
        } }""")

        up = cppyy.gbl.std.make_unique[int](42)

        ns = cppyy.gbl.unique_ptr_moves
        up = ns.returnptr_value(up)                    ; assert up and up.get()[0] == 42
        up = ns.returnptr_value(cppyy.gbl.std.move(up)); assert up and up.get()[0] == 42
        up = ns.returnptr_move(cppyy.gbl.std.move(up)) ; assert up and up.get()[0] == 42

        with raises(TypeError):
            ns.returnptr_move(up)

    def test17_unique_ptr_data(self):
        """std::unique_ptr as data means implicitly no copy ctor"""

        import cppyy

        cppyy.cppdef("""namespace unique_ptr_data{
        class Example {
        private:
          std::unique_ptr<double> x;
        public:
          Example() {}
          virtual ~Example() = default;
          double y = 66.;
        }; }""")

        class Inherit(cppyy.gbl.unique_ptr_data.Example):
            pass

        a = Inherit()
      # Test whether this attribute was inherited
        assert a.y == 66.

    def test18_unique_ptr_identity(self):
        """std::unique_ptr identity preservation"""

        import cppyy

        cppyy.cppdef("""\
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

        ns = cppyy.gbl.UniqueIdentity

        x = ns.create()
        assert x.a == 37

        c = ns.Consumer(x)
        x = c.get()
        assert x.a == 37

        p1 = c.pget()
        p2 = c.pget()
        assert p1 is p2

    @mark.xfail()
    def test19_smartptr_from_callback(self):
        """Return a smart pointer from a callback"""

        import cppyy

        cppyy.cppdef(r"""\
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

        std = cppyy.gbl.std
        ns = cppyy.gbl.SmartPtrCallback

        def pyfunc() -> std.shared_ptr[ns.Dummy]:
             return ns.dummy_create()

        assert ns.call_creator(pyfunc)


if __name__ == "__main__":
    exit(pytest.main(args=['-sv', '-ra', __file__]))
