#ifndef CPYCPPYY_UTILITY_H
#define CPYCPPYY_UTILITY_H

// Standard
#include <map>
#include <memory>
#include <string>
#include <vector>


namespace CPyCppyy {

class PyCallable;

#if PY_VERSION_HEX < 0x030b0000
extern dict_lookup_func gDictLookupOrg;
extern bool gDictLookupActive;
#endif

// additional converter functions
unsigned long PyLongOrInt_AsULong(PyObject* pyobject);
PY_ULONG_LONG PyLongOrInt_AsULong64(PyObject* pyobject);

namespace Utility {

// convenience functions for adding methods to classes
bool AddToClass(PyObject* pyclass, const char* label, PyCFunction cfunc,
    int flags = METH_VARARGS);
bool AddToClass(PyObject* pyclass, const char* label, const char* func);
bool AddToClass(PyObject* pyclass, const char* label, PyCallable* pyfunc);

// helpers for dynamically constructing operators
PyCallable* FindUnaryOperator(PyObject* pyclass, const char* op);
PyCallable* FindBinaryOperator(PyObject* left, PyObject* right,
    const char* op, Cppyy::TCppScope_t scope = 0);
PyCallable* FindBinaryOperator(const std::string& lcname, const std::string& rcname,
    const char* op, Cppyy::TCppScope_t scope = 0, bool reverse = false);

// helper for template classes and methods
enum ArgPreference { kNone, kPointer, kReference, kValue };
std::string ConstructTemplateArgs(
    PyObject* pyname, PyObject* tpArgs, PyObject* args = nullptr, ArgPreference = kNone, int argoff = 0, int* pcnt = nullptr);
std::string CT2CppNameS(PyObject* pytc, bool allow_voidp);
inline PyObject* CT2CppName(PyObject* pytc, const char* cpd, bool allow_voidp)
{
    const std::string& name = CT2CppNameS(pytc, allow_voidp);
    if (!name.empty()) {
        if (name == "const char*") cpd = "";
        return CPyCppyy_PyText_FromString((std::string{name}+cpd).c_str());
    }
    return nullptr;
}

// helper for generating callbacks
void ConstructCallbackPreamble(const std::string& retType,
    const std::vector<std::string>& argtypes, std::ostringstream& code);
void ConstructCallbackReturn(const std::string& retType, int nArgs, std::ostringstream& code);

// helper for function pointer conversions
PyObject* FuncPtr2StdFunction(const std::string& retType, const std::string& signature, void* address);

// initialize proxy type objects
bool InitProxy(PyObject* module, PyTypeObject* pytype, const char* name);

// retrieve the memory buffer from pyobject, return buflength, tc (optional) is python
// array.array type code, size is type size, buf will point to buffer, and if check is
// true, some heuristics will be applied to check buffer compatibility with the type
Py_ssize_t GetBuffer(PyObject* pyobject, char tc, int size, void*& buf, bool check = true);

// data/operator mappings
std::string MapOperatorName(const std::string& name, bool bTakesParames, bool* stubbed = nullptr);

struct PyOperators {
    PyOperators() : fEq(nullptr), fNe(nullptr), fLt(nullptr), fLe(nullptr), fGt(nullptr), fGe(nullptr),
        fLAdd(nullptr), fRAdd(nullptr), fSub(nullptr), fLMul(nullptr), fRMul(nullptr), fDiv(nullptr),
        fHash(nullptr) {}
    ~PyOperators();

    PyObject* fEq;
    PyObject* fNe;
    PyObject *fLt, *fLe;
    PyObject *fGt, *fGe;
    PyObject *fLAdd, *fRAdd;
    PyObject* fSub;
    PyObject *fLMul, *fRMul;
    PyObject* fDiv;
    PyObject* fHash;
};

// meta information
std::string ClassName(PyObject* pyobj);
bool IsSTLIterator(const std::string& classname);

// for threading: save call to PyErr_Occurred()
PyObject* PyErr_Occurred_WithGIL();

// helpers for collecting/maintaining python exception data
struct PyError_t {
   struct PyObjectDeleter {
      void operator()(PyObject *obj) { Py_XDECREF(obj); }
   };
#if PY_VERSION_HEX < 0x030c0000
   std::unique_ptr<PyObject, PyObjectDeleter> fType;
   std::unique_ptr<PyObject, PyObjectDeleter> fTrace;
#endif
   std::unique_ptr<PyObject, PyObjectDeleter> fValue;
   bool fIsCpp = false;
};

PyError_t FetchPyError();
void RestorePyError(PyError_t &error);

size_t FetchError(std::vector<PyError_t>&, bool is_cpp = false);
void SetDetailedException(
    std::vector<PyError_t>&& errors /* clears */, PyObject* topmsg /* steals ref */, PyObject* defexc);

// setup Python API for callbacks
bool IncludePython();

} // namespace Utility

} // namespace CPyCppyy

#endif // !CPYCPPYY_UTILITY_H
