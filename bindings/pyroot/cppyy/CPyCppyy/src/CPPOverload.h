#ifndef CPYCPPYY_CPPOVERLOAD_H
#define CPYCPPYY_CPPOVERLOAD_H

// Bindings
#include "PyCallable.h"

// Standard
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>


namespace CPyCppyy {

// signature hashes are also used by TemplateProxy
// Forward declare for use in HashSignature before class definition
CPYCPPYY_IMPORT PyTypeObject CPPOverload_Type;
CPYCPPYY_IMPORT PyTypeObject TemplateProxy_Type;

inline uint64_t HashSignature(CPyCppyy_PyArgs_t args, size_t nargsf)
{
// Build a hash from the types of the given python function arguments.
    uint64_t hash = 0;

    Py_ssize_t nargs = CPyCppyy_PyArgs_GET_SIZE(args, nargsf);
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        PyObject* pyobj = CPyCppyy_PyArgs_GET_ITEM(args, i);
        
        // For CPPOverload and TemplateProxy, mix in object identity (pointer address)
        // to distinguish different C++ callables with same Python type
        if (Py_TYPE(pyobj) == &CPPOverload_Type || Py_TYPE(pyobj) == &TemplateProxy_Type) {
            // Use golden ratio mixing: shift by 3 (pointers are 8-byte aligned),
            // then apply proper bit mixing with golden ratio constant
            hash ^= ((uint64_t)(uintptr_t)pyobj >> 3) + 0x9e3779b9ULL + (hash << 6) + (hash >> 2);
        } else {
            // Standard type-based hashing for other objects
            hash += (uint64_t)Py_TYPE(pyobj);
#if PY_VERSION_HEX >= 0x030e0000
            hash += (uint64_t)(PyUnstable_Object_IsUniqueReferencedTemporary(pyobj) ? 1 : 0);
#else
            hash += (uint64_t)(Py_REFCNT(pyobj) == 1 ? 1 : 0);
#endif
        }
        hash += (hash << 10); hash ^= (hash >> 6);
    }

    hash += (hash << 3); hash ^= (hash >> 11); hash += (hash << 15);

    return hash;
}

class CPPOverload {
public:
    typedef std::vector<std::pair<uint64_t, PyCallable*>> DispatchMap_t;
    typedef std::vector<PyCallable*> Methods_t;

    struct MethodInfo_t {
        MethodInfo_t() : fDoc(nullptr), fFlags(CallContext::kNone)
            { fRefCount = new int(1); }
        ~MethodInfo_t();

        std::string                 fName;
        CPPOverload::DispatchMap_t  fDispatchMap;
        CPPOverload::Methods_t      fMethods;
        PyObject*                   fDoc;
        uint32_t                    fFlags;

        int* fRefCount;

    private:
        MethodInfo_t(const MethodInfo_t&) = delete;
        MethodInfo_t& operator=(const MethodInfo_t&) = delete;
    };

public:
    void Set(const std::string& name, std::vector<PyCallable*>& methods);
    void AdoptMethod(PyCallable* pc);
    void MergeOverload(CPPOverload* meth);

    const std::string& GetName() const { return fMethodInfo->fName; }
    bool HasMethods() const { return !fMethodInfo->fMethods.empty(); }

// find a method based on the provided signature
    PyObject* FindOverload(const std::string& signature, int want_const = -1);
    PyObject* FindOverload(PyObject *args_tuple, int want_const = -1);

public:                 // public, as the python C-API works with C structs
    PyObject_HEAD
    CPPInstance*   fSelf;         // must be first (same layout as TemplateProxy)
    MethodInfo_t*  fMethodInfo;
    uint32_t       fFlags;
#if PY_VERSION_HEX >= 0x03080000
    vectorcallfunc fVectorCall;
#endif

private:
    CPPOverload() = delete;
};


//- method proxy type and type verification ----------------------------------
CPYCPPYY_IMPORT PyTypeObject CPPOverload_Type;

template<typename T>
inline bool CPPOverload_Check(T* object)
{
    return object && PyObject_TypeCheck(object, &CPPOverload_Type);
}

template<typename T>
inline bool CPPOverload_CheckExact(T* object)
{
    return object && Py_TYPE(object) == &CPPOverload_Type;
}

//- creation -----------------------------------------------------------------
inline CPPOverload* CPPOverload_New(
    const std::string& name, std::vector<PyCallable*>& methods)
{
// Create and initialize a new method proxy from the overloads.
    CPPOverload* pymeth = (CPPOverload*)CPPOverload_Type.tp_new(&CPPOverload_Type, nullptr, nullptr);
    pymeth->Set(name, methods);
    return pymeth;
}

inline CPPOverload* CPPOverload_New(const std::string& name, PyCallable* method)
{
// Create and initialize a new method proxy from the method.
    std::vector<PyCallable*> p;
    p.push_back(method);
    return CPPOverload_New(name, p);
}

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPOVERLOAD_H
