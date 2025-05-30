#ifndef CPYCPPYY_DECLAREEXECUTORS_H
#define CPYCPPYY_DECLAREEXECUTORS_H

// Bindings
#include "Executors.h"
#include "CallContext.h"
#include "Dimensions.h"

// Standard
#if (__cplusplus > 201402L) || (defined(_MSC_VER) && _MSVC_LANG > 201402L)
#include <cstddef>
#endif


namespace CPyCppyy {

namespace {

#define CPPYY_DECL_EXEC(name)                                                \
class name##Executor : public Executor {                                     \
public:                                                                      \
    PyObject* Execute(                                                       \
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) override;    \
}

// executors for built-ins
CPPYY_DECL_EXEC(Bool);
CPPYY_DECL_EXEC(BoolConstRef);
CPPYY_DECL_EXEC(Char);
CPPYY_DECL_EXEC(CharConstRef);
CPPYY_DECL_EXEC(UChar);
CPPYY_DECL_EXEC(UCharConstRef);
CPPYY_DECL_EXEC(WChar);
CPPYY_DECL_EXEC(Char16);
CPPYY_DECL_EXEC(Char32);
CPPYY_DECL_EXEC(Int8);
CPPYY_DECL_EXEC(UInt8);
CPPYY_DECL_EXEC(Short);
CPPYY_DECL_EXEC(Int);
CPPYY_DECL_EXEC(Long);
CPPYY_DECL_EXEC(ULong);
CPPYY_DECL_EXEC(LongLong);
CPPYY_DECL_EXEC(ULongLong);
CPPYY_DECL_EXEC(Float);
CPPYY_DECL_EXEC(Double);
CPPYY_DECL_EXEC(LongDouble);
CPPYY_DECL_EXEC(Void);
CPPYY_DECL_EXEC(CString);
CPPYY_DECL_EXEC(CStringRef);
CPPYY_DECL_EXEC(WCString);
CPPYY_DECL_EXEC(CString16);
CPPYY_DECL_EXEC(CString32);

// pointer/array executors
#define CPPYY_ARRAY_DECL_EXEC(name)                                          \
class name##ArrayExecutor : public Executor {                                \
    dims_t fShape;                                                           \
public:                                                                      \
    name##ArrayExecutor(dims_t dims) : fShape(dims) {}                       \
    PyObject* Execute(                                                       \
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) override;    \
    bool HasState() override { return true; }                                \
}
CPPYY_ARRAY_DECL_EXEC(Void);
CPPYY_ARRAY_DECL_EXEC(Bool);
CPPYY_ARRAY_DECL_EXEC(SChar);
CPPYY_ARRAY_DECL_EXEC(UChar);
#if (__cplusplus > 201402L) || (defined(_MSC_VER) && _MSVC_LANG > 201402L)
CPPYY_ARRAY_DECL_EXEC(Byte);
#endif
CPPYY_ARRAY_DECL_EXEC(Int8);
CPPYY_ARRAY_DECL_EXEC(UInt8);
CPPYY_ARRAY_DECL_EXEC(Short);
CPPYY_ARRAY_DECL_EXEC(UShort);
CPPYY_ARRAY_DECL_EXEC(Int);
CPPYY_ARRAY_DECL_EXEC(UInt);
CPPYY_ARRAY_DECL_EXEC(Long);
CPPYY_ARRAY_DECL_EXEC(ULong);
CPPYY_ARRAY_DECL_EXEC(LLong);
CPPYY_ARRAY_DECL_EXEC(ULLong);
CPPYY_ARRAY_DECL_EXEC(Float);
CPPYY_ARRAY_DECL_EXEC(Double);
CPPYY_ARRAY_DECL_EXEC(ComplexF);
CPPYY_ARRAY_DECL_EXEC(ComplexD);
CPPYY_ARRAY_DECL_EXEC(ComplexI);
CPPYY_ARRAY_DECL_EXEC(ComplexL);

// special cases
CPPYY_DECL_EXEC(ComplexD);
CPPYY_DECL_EXEC(STLString);
CPPYY_DECL_EXEC(STLWString);

class InstancePtrExecutor : public Executor {
public:
    InstancePtrExecutor(Cppyy::TCppType_t klass) : fClass(klass) {}
    PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) override;
    bool HasState() override { return true; }

protected:
    Cppyy::TCppType_t fClass;
};

class InstanceExecutor : public Executor {
public:
    InstanceExecutor(Cppyy::TCppType_t klass);
    PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) override;
    bool HasState() override { return true; }

protected:
    Cppyy::TCppType_t fClass;
    uint32_t          fFlags;
};

class IteratorExecutor : public InstanceExecutor {
public:
    IteratorExecutor(Cppyy::TCppType_t klass);
};

CPPYY_DECL_EXEC(Constructor);
CPPYY_DECL_EXEC(PyObject);

#define CPPYY_DECL_REFEXEC(name)                                             \
class name##RefExecutor : public RefExecutor {                               \
public:                                                                      \
    PyObject* Execute(                                                       \
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) override;    \
}

CPPYY_DECL_REFEXEC(Bool);
CPPYY_DECL_REFEXEC(Char);
CPPYY_DECL_REFEXEC(UChar);
CPPYY_DECL_REFEXEC(Int8);
CPPYY_DECL_REFEXEC(UInt8);
CPPYY_DECL_REFEXEC(Short);
CPPYY_DECL_REFEXEC(UShort);
CPPYY_DECL_REFEXEC(Int);
CPPYY_DECL_REFEXEC(UInt);
CPPYY_DECL_REFEXEC(Long);
CPPYY_DECL_REFEXEC(ULong);
CPPYY_DECL_REFEXEC(LongLong);
CPPYY_DECL_REFEXEC(ULongLong);
CPPYY_DECL_REFEXEC(Float);
CPPYY_DECL_REFEXEC(Double);
CPPYY_DECL_REFEXEC(LongDouble);
CPPYY_DECL_REFEXEC(ComplexD);
CPPYY_DECL_REFEXEC(STLString);

// special cases
class InstanceRefExecutor : public RefExecutor {
public:
    InstanceRefExecutor(Cppyy::TCppType_t klass) : fClass(klass) {}
    PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) override;

protected:
    Cppyy::TCppType_t fClass;
};

class InstancePtrPtrExecutor : public InstanceRefExecutor {
public:
    using InstanceRefExecutor::InstanceRefExecutor;
    PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) override;
};

class InstancePtrRefExecutor : public InstanceRefExecutor {
public:
    using InstanceRefExecutor::InstanceRefExecutor;
    PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) override;
};

class InstanceArrayExecutor : public InstancePtrExecutor {
public:
    InstanceArrayExecutor(Cppyy::TCppType_t klass, dim_t array_size)
        : InstancePtrExecutor(klass), fSize(array_size) {}
    PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) override;

protected:
    dim_t fSize;
};

class FunctionPointerExecutor : public Executor {
public:
    FunctionPointerExecutor(const std::string& ret, const std::string& sig) :
        fRetType(ret), fSignature(sig) {}
    PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) override;

protected:
    std::string fRetType;
    std::string fSignature;
};

} // unnamed namespace

} // namespace CPyCppyy

#endif // !CPYCPPYY_DECLAREEXECUTORS_H
