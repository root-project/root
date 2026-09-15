#ifndef CPYRT_DECLAREEXECUTORS_H
#define CPYRT_DECLAREEXECUTORS_H

#include "Python.h"
#include "cppjit_interop.h"

// Bindings
#include "CallContext.h"
#include "Dimensions.h"
#include "Executors.h"

// Standard
#include <cstddef>

namespace cppjit::cpyrt {

namespace {

#define CPPJIT_DECL_EXEC(name)                                                 \
  class name##Executor : public Executor {                                     \
  public:                                                                      \
    PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,            \
                      CallContext*) override;                                  \
  }

// executors for built-ins
CPPJIT_DECL_EXEC(Bool);
CPPJIT_DECL_EXEC(BoolConstRef);
CPPJIT_DECL_EXEC(Char);
CPPJIT_DECL_EXEC(CharConstRef);
CPPJIT_DECL_EXEC(UChar);
CPPJIT_DECL_EXEC(UCharConstRef);
CPPJIT_DECL_EXEC(WChar);
CPPJIT_DECL_EXEC(Char16);
CPPJIT_DECL_EXEC(Char32);
CPPJIT_DECL_EXEC(Int8);
CPPJIT_DECL_EXEC(Int8ConstRef);
CPPJIT_DECL_EXEC(UInt8);
CPPJIT_DECL_EXEC(UInt8ConstRef);
CPPJIT_DECL_EXEC(Short);
CPPJIT_DECL_EXEC(Int);
CPPJIT_DECL_EXEC(Long);
CPPJIT_DECL_EXEC(ULong);
CPPJIT_DECL_EXEC(LongLong);
CPPJIT_DECL_EXEC(ULongLong);
CPPJIT_DECL_EXEC(Float);
CPPJIT_DECL_EXEC(Double);
CPPJIT_DECL_EXEC(LongDouble);
CPPJIT_DECL_EXEC(Void);
CPPJIT_DECL_EXEC(CString);
CPPJIT_DECL_EXEC(CStringRef);
CPPJIT_DECL_EXEC(WCString);
CPPJIT_DECL_EXEC(CString16);
CPPJIT_DECL_EXEC(CString32);

// pointer/array executors
#define CPPJIT_ARRAY_DECL_EXEC(name)                                           \
  class name##ArrayExecutor : public Executor {                                \
    dims_t fShape;                                                             \
                                                                               \
  public:                                                                      \
    name##ArrayExecutor(dims_t dims) : fShape(dims) {}                         \
    PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,            \
                      CallContext*) override;                                  \
    bool HasState() override { return true; }                                  \
  }
CPPJIT_ARRAY_DECL_EXEC(Void);
CPPJIT_ARRAY_DECL_EXEC(Bool);
CPPJIT_ARRAY_DECL_EXEC(SChar);
CPPJIT_ARRAY_DECL_EXEC(UChar);
CPPJIT_ARRAY_DECL_EXEC(Byte);
CPPJIT_ARRAY_DECL_EXEC(Int8);
CPPJIT_ARRAY_DECL_EXEC(UInt8);
CPPJIT_ARRAY_DECL_EXEC(Short);
CPPJIT_ARRAY_DECL_EXEC(UShort);
CPPJIT_ARRAY_DECL_EXEC(Int);
CPPJIT_ARRAY_DECL_EXEC(UInt);
CPPJIT_ARRAY_DECL_EXEC(Long);
CPPJIT_ARRAY_DECL_EXEC(ULong);
CPPJIT_ARRAY_DECL_EXEC(LLong);
CPPJIT_ARRAY_DECL_EXEC(ULLong);
CPPJIT_ARRAY_DECL_EXEC(Float);
CPPJIT_ARRAY_DECL_EXEC(Double);
CPPJIT_ARRAY_DECL_EXEC(LDouble);
CPPJIT_ARRAY_DECL_EXEC(ComplexF);
CPPJIT_ARRAY_DECL_EXEC(ComplexD);
CPPJIT_ARRAY_DECL_EXEC(ComplexI);
CPPJIT_ARRAY_DECL_EXEC(ComplexL);

// special cases
CPPJIT_DECL_EXEC(ComplexD);
CPPJIT_DECL_EXEC(STLString);
CPPJIT_DECL_EXEC(STLWString);

class InstancePtrExecutor : public Executor {
public:
  InstancePtrExecutor(interop::TCppScope_t klass) : fClass(klass) {}
  PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,
                    CallContext*) override;
  bool HasState() override { return true; }

protected:
  interop::TCppScope_t fClass;
};

class InstanceExecutor : public Executor {
public:
  InstanceExecutor(interop::TCppScope_t klass);
  PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,
                    CallContext*) override;
  bool HasState() override { return true; }

protected:
  interop::TCppScope_t fClass;
  uint32_t fFlags;
};

class IteratorExecutor : public InstanceExecutor {
public:
  IteratorExecutor(interop::TCppScope_t klass);
};

CPPJIT_DECL_EXEC(Constructor);
CPPJIT_DECL_EXEC(PyObject);

#define CPPJIT_DECL_REFEXEC(name)                                              \
  class name##RefExecutor : public RefExecutor {                               \
  public:                                                                      \
    PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,            \
                      CallContext*) override;                                  \
  }

CPPJIT_DECL_REFEXEC(Bool);
CPPJIT_DECL_REFEXEC(Char);
CPPJIT_DECL_REFEXEC(UChar);
CPPJIT_DECL_REFEXEC(Int8);
CPPJIT_DECL_REFEXEC(UInt8);
CPPJIT_DECL_REFEXEC(Short);
CPPJIT_DECL_REFEXEC(UShort);
CPPJIT_DECL_REFEXEC(Int);
CPPJIT_DECL_REFEXEC(UInt);
CPPJIT_DECL_REFEXEC(Long);
CPPJIT_DECL_REFEXEC(ULong);
CPPJIT_DECL_REFEXEC(LongLong);
CPPJIT_DECL_REFEXEC(ULongLong);
CPPJIT_DECL_REFEXEC(Float);
CPPJIT_DECL_REFEXEC(Double);
CPPJIT_DECL_REFEXEC(LongDouble);
CPPJIT_DECL_REFEXEC(ComplexD);
CPPJIT_DECL_REFEXEC(STLString);

// special cases
class InstanceRefExecutor : public RefExecutor {
public:
  InstanceRefExecutor(interop::TCppScope_t klass) : fClass(klass) {}
  PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,
                    CallContext*) override;

protected:
  interop::TCppScope_t fClass;
};

class InstancePtrPtrExecutor : public InstanceRefExecutor {
public:
  using InstanceRefExecutor::InstanceRefExecutor;
  PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,
                    CallContext*) override;
};

class InstancePtrRefExecutor : public InstanceRefExecutor {
public:
  using InstanceRefExecutor::InstanceRefExecutor;
  PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,
                    CallContext*) override;
};

class InstanceArrayExecutor : public InstancePtrExecutor {
public:
  InstanceArrayExecutor(interop::TCppScope_t klass, dim_t array_size)
      : InstancePtrExecutor(klass), fSize(array_size) {}
  PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,
                    CallContext*) override;

protected:
  dim_t fSize;
};

class FunctionPointerExecutor : public Executor {
public:
  FunctionPointerExecutor(const std::string& ret, const std::string& sig)
      : fRetType(ret), fSignature(sig) {}
  PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,
                    CallContext*) override;

protected:
  std::string fRetType;
  std::string fSignature;
};

} // unnamed namespace

} // namespace cppjit::cpyrt

#endif // !CPYRT_DECLAREEXECUTORS_H
