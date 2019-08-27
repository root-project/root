#ifndef CPYCPPYY_MEMORYREGULATOR_H
#define CPYCPPYY_MEMORYREGULATOR_H

#include <functional>
#include <utility>

namespace CPyCppyy {

class CPPInstance;

typedef std::function<std::pair<bool, bool>(Cppyy::TCppObject_t, Cppyy::TCppType_t)> MemHook_t;

class MemoryRegulator {
private:
    static MemHook_t registerHook, unregisterHook;
public:
    MemoryRegulator();

// callback from C++-side frameworks
    static bool RecursiveRemove(Cppyy::TCppObject_t cppobj, Cppyy::TCppType_t klass);

// add a python object to the table of managed objects
    static bool RegisterPyObject(CPPInstance* pyobj, void* cppobj);

// remove a python object from the table of managed objects, w/o notification
    static bool UnregisterPyObject(Cppyy::TCppObject_t cppobj, Cppyy::TCppType_t klass);

// new reference to python object matching cppobj, or 0 on failure
    static PyObject* RetrievePyObject(Cppyy::TCppObject_t cppobj, Cppyy::TCppType_t klass);

// callback when weak refs to managed objects are destroyed
    static PyObject* EraseCallback(PyObject*, PyObject* pyref);

// set hooks for custom memory regulation
    static void SetRegisterHook(MemHook_t h) { registerHook = h; }
    static void SetUnregisterHook(MemHook_t h) { unregisterHook = h; }
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_MEMORYREGULATOR_H
