#ifndef CPYCPPYY_CPPINSTANCE_H
#define CPYCPPYY_CPPINSTANCE_H

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// CpyCppyy::CPPInstance                                                    //
//                                                                          //
// Python-side proxy, encapsulaties a C++ object.                           //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


// Bindings
#include "CPPScope.h"
#include "Cppyy.h"
#include "CallContext.h"     // for Parameter

// Standard
#include <utility>
#include <vector>


// TODO: have an CPPInstance derived or alternative type for smart pointers

namespace CPyCppyy {

typedef std::vector<std::pair<ptrdiff_t, PyObject*>> CI_DatamemberCache_t;

class CPPInstance {
public:
    enum EFlags {
        kNone        = 0x0,
        kIsOwner     = 0x0001,
        kIsReference = 0x0002,
        kIsRValue    = 0x0004,
        kIsValue     = 0x0008,
        kIsSmartPtr  = 0x0010,
        kIsPtrPtr    = 0x0020 };

public:
    void Set(void* address, EFlags flags = kNone)
    {
    // Initialize the proxy with the pointer value 'address.'
        fObject = address;
        fFlags  = flags;
        fSmartPtrType = (Cppyy::TCppType_t)0;
        fDereferencer = (Cppyy::TCppMethod_t)0;
    }

    void SetSmartPtr(Cppyy::TCppType_t ptrtype, Cppyy::TCppMethod_t deref)
    {
        fFlags |= kIsSmartPtr;
        fSmartPtrType = ptrtype;
        fDereferencer = deref;
    }

    void* GetObject() const
    {
    // Retrieve a pointer to the held C++ object.

    // We get the raw pointer from the smart pointer each time, in case
    // it has changed or has been freed.
        if (fFlags & kIsSmartPtr) {
            return Cppyy::CallR(fDereferencer, fObject, 0, nullptr);
        }

        if (fObject && (fFlags & kIsReference))
            return *(reinterpret_cast<void**>(const_cast<void*>(fObject)));
        else
            return const_cast<void*>(fObject);             // may be null
    }

    Cppyy::TCppType_t ObjectIsA() const
    {
    // Retrieve a pointer to the C++ type; may return nullptr.
        return ((CPPClass*)Py_TYPE(this))->fCppType;
    }

    void PythonOwns() { fFlags |= kIsOwner; }
    void CppOwns()    { fFlags &= ~kIsOwner; }

public:                 // public, as the python C-API works with C structs
    PyObject_HEAD
    void*     fObject;
    int       fFlags;

// cache for expensive to create data member objects
    CI_DatamemberCache_t fDatamemberCache;

// TODO: should be its own version of CPPInstance so as not to clutter the
// normal instances
    Cppyy::TCppType_t   fSmartPtrType;
    Cppyy::TCppMethod_t fDereferencer;

private:
    CPPInstance() = delete;
};


//- object proxy type and type verification ----------------------------------
CPYCPPYY_IMPORT PyTypeObject CPPInstance_Type;

template<typename T>
inline bool CPPInstance_Check(T* object)
{
    return object && (PyObject*)object != Py_None && PyObject_TypeCheck(object, &CPPInstance_Type);
}

template<typename T>
inline bool CPPInstance_CheckExact(T* object)
{
    return object && Py_TYPE(object) == &CPPInstance_Type;
}


//- helper for memory regulation (no PyTypeObject equiv. member in p2.2) -----
void op_dealloc_nofree(CPPInstance*);

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPINSTANCE_H
