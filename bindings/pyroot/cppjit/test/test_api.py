from pytest import mark, skip
from support import IS_LINUX_ARM, IS_MAC, ispypy


class TestAPI:
    def setup_class(cls):
        if ispypy:
            skip("C++ API only available on CPython")

        import cppjit

        cppjit.include("cpyrt/API.h")

    def test01_type_checking(self):
        """Python class type checks"""

        import cppjit

        cpp = cppjit.gbl
        API = cpp.cppjit.cpyrt

        cppjit.cppdef("""
        class APICheck {
        public:
          void some_method() {}
        };""")

        assert API.Scope_Check(cpp.APICheck)
        assert not API.Scope_CheckExact(cpp.APICheck)

        a = cpp.APICheck()
        assert API.Instance_Check(a)
        assert not API.Instance_CheckExact(a)

        m = a.some_method
        assert API.Overload_Check(m)
        assert API.Overload_CheckExact(m)

    @mark.xfail(condition=IS_MAC, reason="Fails on OS X")
    def test02_interpreter_access(self):
        """Access to the python interpreter"""

        import cppjit

        API = cppjit.gbl.cppjit.cpyrt

        assert API.Exec("import sys")

    @mark.xfail(condition=IS_MAC, reason="Fails on OS X")
    def test03_instance_conversion(self):
        """Proxy object conversions"""

        import cppjit

        cpp = cppjit.gbl
        API = cpp.cppjit.cpyrt

        cppjit.cppdef("""
        class APICheck2 {
        public:
          virtual ~APICheck2() {}
        };""")

        m = cpp.APICheck2()

        voidp = API.Instance_AsVoidPtr(m)
        m2 = API.Instance_FromVoidPtr(voidp, "APICheck2")
        assert m is m2

    @mark.xfail(condition=IS_LINUX_ARM, run=False, reason="Crashes pytest on Linux ARM")
    def test04_custom_converter(self):
        """Custom type converter"""

        import cppjit

        cppjit.cppdef("""
        #include "cpyrt/API.h"

        class APICheck3 {
            int fFlags;
        public:
            APICheck3() : fFlags(0) {}
            virtual ~APICheck3() {}

            void setSetArgCalled()     { fFlags |= 0x01; }
            bool wasSetArgCalled()     { return fFlags & 0x01; }
            void setFromMemoryCalled() { fFlags |= 0x02; }
            bool wasFromMemoryCalled() { return fFlags & 0x02; }
            void setToMemoryCalled()   { fFlags |= 0x04; }
            bool wasToMemoryCalled()   { return fFlags & 0x04; }
        };

        class APICheck3Converter : public cppjit::cpyrt::Converter {
        public:
            virtual bool SetArg(PyObject* pyobject, cppjit::cpyrt::Parameter& para, cppjit::cpyrt::CallContext* = nullptr) {
                APICheck3* a3 = (APICheck3*)cppjit::cpyrt::Instance_AsVoidPtr(pyobject);
                a3->setSetArgCalled();
                para.fValue.fVoidp = a3;
                para.fTypeCode = 'V';
                return true;
            }

            virtual PyObject* FromMemory(void* address) {
                APICheck3* a3 = (APICheck3*)address;
                a3->setFromMemoryCalled();
                return cppjit::cpyrt::Instance_FromVoidPtr(a3, "APICheck3");
            }

            virtual bool ToMemory(PyObject* value, void* address) {
                APICheck3* a3 = (APICheck3*)address;
                a3->setToMemoryCalled();
                *a3 = *(APICheck3*)cppjit::cpyrt::Instance_AsVoidPtr(value);
                return true;
            }
        };

        typedef cppjit::cpyrt::ConverterFactory_t cf_t;
        void register_a3() {
            cppjit::cpyrt::RegisterConverter("APICheck3",  (cf_t)+[](cppjit::cpyrt::cdims_t) { static APICheck3Converter c{}; return &c; });
            cppjit::cpyrt::RegisterConverter("APICheck3&", (cf_t)+[](cppjit::cpyrt::cdims_t) { static APICheck3Converter c{}; return &c; });
        }
        void unregister_a3() {
            cppjit::cpyrt::UnregisterConverter("APICheck3");
            cppjit::cpyrt::UnregisterConverter("APICheck3&");
        }

        APICheck3 gA3a, gA3b;
        void CallWithAPICheck3(APICheck3&) {}
        """)

        cppjit.gbl.register_a3()

        gA3a = cppjit.gbl.gA3a
        assert gA3a
        assert type(gA3a) == cppjit.gbl.APICheck3
        assert gA3a.wasFromMemoryCalled()

        assert not gA3a.wasSetArgCalled()
        cppjit.gbl.CallWithAPICheck3(gA3a)
        assert gA3a.wasSetArgCalled()

        cppjit.gbl.unregister_a3()

        gA3b = cppjit.gbl.gA3b
        assert gA3b
        assert type(gA3b) == cppjit.gbl.APICheck3
        assert not gA3b.wasFromMemoryCalled()

    @mark.xfail(condition=IS_LINUX_ARM, run=False, reason="Crashes pytest on Linux ARM")
    def test05_custom_executor(self):
        """Custom type executor"""

        import cppjit

        cppjit.cppdef("""
        #include "cpyrt/API.h"

        class APICheck4 {
            int fFlags;
        public:
            APICheck4() : fFlags(0) {}
            virtual ~APICheck4() {}

            void setExecutorCalled() { fFlags |= 0x01; }
            bool wasExecutorCalled() { return fFlags & 0x01; }
        };

        class APICheck4Executor : public cppjit::cpyrt::Executor {
        public:
             virtual PyObject* Execute(cppjit::interop::TCppMethod_t meth, cppjit::interop::TCppObject_t obj, cppjit::cpyrt::CallContext* ctxt) {
                 APICheck4* a4 = (APICheck4*)cppjit::cpyrt::CallVoidP(meth, obj, ctxt);
                 a4->setExecutorCalled();
                 return cppjit::cpyrt::Instance_FromVoidPtr(a4, "APICheck4", true);
             }
        };

        typedef cppjit::cpyrt::ExecutorFactory_t ef_t;
        void register_a4() {
            cppjit::cpyrt::RegisterExecutor("APICheck4*", (ef_t)+[](cppjit::cpyrt::cdims_t) { static APICheck4Executor c{}; return &c; });
        }
        void unregister_a4() {
            cppjit::cpyrt::UnregisterExecutor("APICheck4*");
        }

        APICheck4* CreateAPICheck4() { return new APICheck4{}; }
        APICheck4* CreateAPICheck4b() { return new APICheck4{}; }
        """)

        cppjit.gbl.register_a4()

        a4 = cppjit.gbl.CreateAPICheck4()
        assert a4
        assert type(a4) == cppjit.gbl.APICheck4
        assert a4.wasExecutorCalled()
        del a4

        cppjit.gbl.unregister_a4()

        a4 = cppjit.gbl.CreateAPICheck4b()
        assert a4
        assert type(a4) == cppjit.gbl.APICheck4
        assert not a4.wasExecutorCalled()

    def test06_custom_executor(self):
        """Custom type executor"""

        import cppjit

        cppjit.cppdef("""
        #include "cpyrt/API.h"

        namespace ArrayLike {
        class MyClass{};
        MyClass* my = nullptr;
        MyClass  myA[5];

        class MyArray {
        public:
            int operator[](int) { return 42; }
        }; }""")

        ns = cppjit.gbl.ArrayLike
        Sequence_Check = cppjit.gbl.cppjit.cpyrt.Sequence_Check

        assert not Sequence_Check(ns.my)
        assert Sequence_Check(ns.myA)
        assert not Sequence_Check(ns.MyClass())
        assert Sequence_Check(ns.MyArray())
        assert Sequence_Check(tuple())
        assert Sequence_Check(cppjit.gbl.std.vector[ns.MyClass]())
        assert not Sequence_Check(cppjit.gbl.std.list[ns.MyClass]())
