#ifndef CPYCPPYY_CPPOVERLOAD_H
#define CPYCPPYY_CPPOVERLOAD_H

// Bindings
#include "PyCallable.h"

// Standard
#include <map>
#include <string>
#include <vector>


namespace CPyCppyy {

class CPPOverload {
public:
    typedef std::map<uint64_t, int>  DispatchMap_t;
    typedef std::vector<PyCallable*> Methods_t;

    struct MethodInfo_t {
        MethodInfo_t() : fFlags(CallContext::kNone) { fRefCount = new int(1); }
        ~MethodInfo_t();

        std::string                 fName;
        CPPOverload::DispatchMap_t  fDispatchMap;
        CPPOverload::Methods_t      fMethods;
        uint64_t                    fFlags;

        int* fRefCount;

    private:
        MethodInfo_t(const MethodInfo_t&) = delete;
        MethodInfo_t& operator=(const MethodInfo_t&) = delete;
    };

public:
    void Set(const std::string& name, std::vector<PyCallable*>& methods);

    const std::string& GetName() const { return fMethodInfo->fName; }
    void AddMethod(PyCallable* pc);
    void AddMethod(CPPOverload* meth);

public:                 // public, as the python C-API works with C structs
    PyObject_HEAD
    CPPInstance*   fSelf;         // must be first (same layout as TemplateProxy)
    MethodInfo_t*  fMethodInfo;

private:
    CPPOverload() = delete;
};


//- method proxy type and type verification ----------------------------------
extern PyTypeObject CPPOverload_Type;

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
