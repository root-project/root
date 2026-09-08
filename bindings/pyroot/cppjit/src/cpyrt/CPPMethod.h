#ifndef CPYRT_CPPMETHOD_H
#define CPYRT_CPPMETHOD_H

// Bindings
#include "PyCallable.h"

// Standard
#include <string>
#include <unordered_map>
#include <vector>

namespace cppjit::cpyrt {

class Executor;
class Converter;

class PyCallArgs {
public:
  PyCallArgs(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
             PyObject* kwds)
      : fSelf(self), fArgs(args), fNArgsf(nargsf), fKwds(kwds), fFlags(kNone) {}
  ~PyCallArgs();

  enum ECleanupFlags {
    kNone = 0x0000,
    kIsOffset = 0x0001,    // args were offset by 1 to drop self
    kSelfSwap = 0x0002,    // args[-1] and self need swapping
    kArgsSwap = 0x0004,    // args[0] and args[1] need swapping
    kDoFree = 0x0008,      // args need to be free'd (vector call only)
    kDoItemDecref = 0x0010 // items in args need a decref (vector call only)
  };

public:
  CPPInstance*& fSelf;
  cpyrt_PyArgs_t fArgs;
  size_t fNArgsf;
  PyObject* fKwds;
  int fFlags;
};

class CPPMethod : public PyCallable {
public:
  CPPMethod(interop::TCppScope_t scope, interop::TCppMethod_t method);
  CPPMethod(const CPPMethod&);
  CPPMethod& operator=(const CPPMethod&);
  virtual ~CPPMethod();

public:
  PyObject* GetSignature(bool show_formalargs = true) override;
  PyObject* GetSignatureNames() override;
  PyObject* GetSignatureTypes() override;
  PyObject* GetPrototype(bool show_formalargs = true) override;
  PyObject* GetTypeName() override;
  PyObject*
  Reflex(interop::Reflex::RequestId_t request,
         interop::Reflex::FormatId_t = interop::Reflex::OPTIMAL) override;

  int GetPriority() override;
  bool IsGreedy() override;

  int GetMaxArgs() override;
  PyObject* GetCoVarNames() override;
  PyObject* GetArgDefault(int iarg, bool silent = true) override;
  bool IsConst() override;

  PyObject* GetScopeProxy() override;
  interop::TCppFuncAddr_t GetFunctionAddress() override;

  PyCallable* Clone() override { return new CPPMethod(*this); }

  int GetArgMatchScore(PyObject* args_tuple) override;

public:
  PyObject* Call(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
                 PyObject* kwds, CallContext* ctxt = nullptr) override;

protected:
  virtual bool ProcessArgs(PyCallArgs& args);

  bool Initialize(CallContext* ctxt = nullptr);
  bool ProcessKwds(PyObject* self_in, PyCallArgs& args);
  bool ConvertAndSetArgs(cpyrt_PyArgs_t, size_t nargsf,
                         CallContext* ctxt = nullptr);
  PyObject* Execute(void* self, ptrdiff_t offset, CallContext* ctxt = nullptr);

  interop::TCppMethod_t GetMethod() { return fMethod; }
  interop::TCppScope_t GetScope() { return fScope; }
  Executor* GetExecutor() { return fExecutor; }
  std::string GetSignatureString(bool show_formalargs = true);
  std::string GetReturnTypeName();

  virtual bool InitExecutor_(Executor*&, CallContext* ctxt = nullptr);

  // whether fExecutor's result is a real PyObject* (ConstructorExecutor
  // returns the new object's address instead)
  virtual bool ResultIsPyObject() const { return true; }

private:
  void Copy_(const CPPMethod&);
  void Destroy_();
  bool VerifyArgCount_(Py_ssize_t);

  PyObject* ExecuteFast(void*, ptrdiff_t, CallContext*);
  PyObject* ExecuteProtected(void*, ptrdiff_t, CallContext*);

  bool InitConverters_();

  void SetPyError_(PyObject* msg);

private:
  // representation
  interop::TCppMethod_t fMethod;
  interop::TCppScope_t fScope;
  Executor* fExecutor;

  // call dispatch buffers
  std::vector<Converter*> fConverters;
  std::unordered_map<std::string, int>* fArgIndices;

protected:
  // cached value that doubles as initialized flag (uninitialized if -1)
  int fArgsRequired;
};

} // namespace cppjit::cpyrt

#endif // !CPYRT_CPPMETHOD_H
