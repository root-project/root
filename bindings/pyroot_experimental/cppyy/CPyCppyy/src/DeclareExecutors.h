#ifndef CPYCPPYY_DECLAREEXECUTORS_H
#define CPYCPPYY_DECLAREEXECUTORS_H

// Bindings
#include "Executors.h"
#include "CallContext.h"


namespace CPyCppyy {

namespace {

#define CPPYY_DECL_EXEC(name)                                                \
class name##Executor : public Executor {                                     \
public:                                                                      \
    virtual PyObject* Execute(                                               \
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);             \
}

// executors for built-ins
CPPYY_DECL_EXEC(Bool);
CPPYY_DECL_EXEC(BoolConstRef);
CPPYY_DECL_EXEC(Char);
CPPYY_DECL_EXEC(CharConstRef);
CPPYY_DECL_EXEC(UChar);
CPPYY_DECL_EXEC(UCharConstRef);
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

// pointer/array executors
CPPYY_DECL_EXEC(VoidArray);
CPPYY_DECL_EXEC(BoolArray);
CPPYY_DECL_EXEC(UCharArray);
CPPYY_DECL_EXEC(ShortArray);
CPPYY_DECL_EXEC(UShortArray);
CPPYY_DECL_EXEC(IntArray);
CPPYY_DECL_EXEC(UIntArray);
CPPYY_DECL_EXEC(LongArray);
CPPYY_DECL_EXEC(ULongArray);
CPPYY_DECL_EXEC(LLongArray);
CPPYY_DECL_EXEC(ULLongArray);
CPPYY_DECL_EXEC(FloatArray);
CPPYY_DECL_EXEC(DoubleArray);

// special cases
CPPYY_DECL_EXEC(STLString);

class CppObjectExecutor : public Executor {
public:
    CppObjectExecutor(Cppyy::TCppType_t klass) : fClass(klass) {}
    virtual PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);

protected:
    Cppyy::TCppType_t fClass;
};

class CppObjectByValueExecutor : public CppObjectExecutor {
public:
    using CppObjectExecutor::CppObjectExecutor;
    virtual PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);
};

CPPYY_DECL_EXEC(Constructor);
CPPYY_DECL_EXEC(PyObject);

#define CPPYY_DECL_REFEXEC(name)                                             \
class name##RefExecutor : public RefExecutor {                               \
public:                                                                      \
    virtual PyObject* Execute(                                               \
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);             \
}

CPPYY_DECL_REFEXEC(Bool);
CPPYY_DECL_REFEXEC(Char);
CPPYY_DECL_REFEXEC(UChar);
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
CPPYY_DECL_REFEXEC(STLString);

// special cases
class CppObjectRefExecutor : public RefExecutor {
public:
    CppObjectRefExecutor(Cppyy::TCppType_t klass) : fClass(klass) {}
    virtual PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);

protected:
    Cppyy::TCppType_t fClass;
};

class CppObjectPtrPtrExecutor : public CppObjectRefExecutor {
public:
    using CppObjectRefExecutor::CppObjectRefExecutor;
    virtual PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);
};

class CppObjectPtrRefExecutor : public CppObjectRefExecutor {
public:
    using CppObjectRefExecutor::CppObjectRefExecutor;
    virtual PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);
};

class CppObjectArrayExecutor : public CppObjectExecutor {
public:
    CppObjectArrayExecutor(Cppyy::TCppType_t klass, Py_ssize_t array_size)
        : CppObjectExecutor(klass), fArraySize(array_size) {}
    virtual PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);

protected:
    Py_ssize_t fArraySize;
};

// smart pointer executors
class CppObjectBySmartPtrExecutor : public Executor {
public:
    CppObjectBySmartPtrExecutor(Cppyy::TCppType_t smart,
            Cppyy::TCppType_t raw, Cppyy::TCppMethod_t deref)
        : fSmartPtrType(smart), fRawPtrType(raw), fDereferencer(deref) {}

    virtual PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);

protected:
    Cppyy::TCppType_t   fSmartPtrType;
    Cppyy::TCppType_t   fRawPtrType;
    Cppyy::TCppMethod_t fDereferencer;
};

class CppObjectBySmartPtrPtrExecutor : public CppObjectBySmartPtrExecutor {
public:
    using CppObjectBySmartPtrExecutor::CppObjectBySmartPtrExecutor;
    virtual PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);
};

class CppObjectBySmartPtrRefExecutor : public RefExecutor {
public:
    CppObjectBySmartPtrRefExecutor(Cppyy::TCppType_t smart,
            Cppyy::TCppType_t raw, Cppyy::TCppMethod_t deref)
        : fSmartPtrType(smart), fRawPtrType(raw), fDereferencer(deref) {}

    virtual PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);

protected:
    Cppyy::TCppType_t fSmartPtrType;
    Cppyy::TCppType_t fRawPtrType;
    Cppyy::TCppMethod_t fDereferencer;
};

} // unnamed namespace

} // namespace CPyCppyy

#endif // !CPYCPPYY_DECLAREEXECUTORS_H
