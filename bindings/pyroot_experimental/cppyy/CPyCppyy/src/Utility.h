#ifndef CPYCPPYY_UTILITY_H
#define CPYCPPYY_UTILITY_H

// Standard
#include <map>
#include <string>
#include <vector>


namespace CPyCppyy {

class PyCallable;

extern dict_lookup_func gDictLookupOrg;
extern bool gDictLookupActive;

// additional converter functions
unsigned long PyLongOrInt_AsULong(PyObject* pyobject);
ULong64_t     PyLongOrInt_AsULong64(PyObject* pyobject);

namespace Utility {

// convenience functions for adding methods to classes
bool AddToClass(PyObject* pyclass, const char* label, PyCFunction cfunc,
    int flags = METH_VARARGS);
bool AddToClass(PyObject* pyclass, const char* label, const char* func);
bool AddToClass(PyObject* pyclass, const char* label, PyCallable* pyfunc);

bool AddUsingToClass(PyObject* pyclass, const char* method);

// helpers for dynamically constructing binary operators
bool AddBinaryOperator(PyObject* left, PyObject* right,
    const char* op, const char* label, const char* alt_label = nullptr);
bool AddBinaryOperator(PyObject* pyclass,
    const char* op, const char* label, const char* alt_label = nullptr);
bool AddBinaryOperator(PyObject* pyclass, const std::string& lcname, const std::string& rcname,
    const char* op, const char* label, const char* alt_label = nullptr);

// helper for template classes and methods
std::string ConstructTemplateArgs(PyObject* pyname, PyObject* args, int argoff);

// initialize proxy type objects
bool InitProxy(PyObject* module, PyTypeObject* pytype, const char* name);

// retrieve the memory buffer from pyobject, return buflength, tc (optional) is python
// array.array type code, size is type size, buf will point to buffer, and if check is
// true, some heuristics will be applied to check buffer compatibility with the type
int GetBuffer(PyObject* pyobject, char tc, int size, void*& buf, bool check = true);

// data/operator mappings
std::string MapOperatorName(const std::string& name, bool bTakesParames);

// meta information
const std::string Compound(const std::string& name);
Py_ssize_t ArraySize(const std::string& name);
std::string ClassName(PyObject* pyobj);

// for threading: save call to PyErr_Occurred()
PyObject* PyErr_Occurred_WithGIL();

// helpers for collecting/maintaining python exception data
struct PyError_t {
    PyError_t() { fType = fValue = fTrace = 0; }

    static void Clear(PyError_t& e)
    {
    // Remove exception information.
        Py_XDECREF(e.fType); Py_XDECREF(e.fValue); Py_XDECREF(e.fTrace);
        e.fType = e.fValue = e.fTrace = 0;
    }

    PyObject *fType, *fValue, *fTrace;
};

size_t FetchError(std::vector<PyError_t>&);
void SetDetailedException(
    std::vector<PyError_t>& errors /* clears */, PyObject* topmsg /* steals ref */, PyObject* defexc);

} // namespace Utility

} // namespace CPyCppyy

#endif // !CPYCPPYY_UTILITY_H
