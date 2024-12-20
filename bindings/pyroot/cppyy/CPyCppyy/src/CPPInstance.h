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


namespace CPyCppyy {

typedef std::vector<std::pair<ptrdiff_t, PyObject*>> CI_DatamemberCache_t;

class CPPInstance {
public:
    enum EFlags {
        kDefault     = 0x0000,
        kNoWrapConv  = 0x0001,    // use type as-is (eg. no smart ptr wrap)
        kIsOwner     = 0x0002,    // Python instance owns C++ object/memory
        kIsExtended  = 0x0004,    // has extended data
        kIsValue     = 0x0008,    // was created from a by-value return
        kIsReference = 0x0010,    // represents one indirection
        kIsArray     = 0x0020,    // represents an array of objects
        kIsSmartPtr  = 0x0040,    // is or embeds a smart pointer
        kIsPtrPtr    = 0x0080,    // represents two indirections
        kIsRValue    = 0x0100,    // can be used as an r-value
        kIsLValue    = 0x0200,    // can be used as an l-value
        kNoMemReg    = 0x0400,    // do not register with memory regulator
        kIsRegulated = 0x0800,    // is registered with memory regulator
        kIsActual    = 0x1000,    // has been downcasted to actual type
        kHasLifeLine = 0x2000,    // has a life line set
    };

public:                 // public, as the python C-API works with C structs
    PyObject_HEAD
    void*     fObject;
    uint32_t  fFlags;

public:
// construction (never done directly)
    CPPInstance() = delete;

    void Set(void* address, EFlags flags = kDefault);
    CPPInstance* Copy(void* cppinst, PyTypeObject* target = nullptr);

// state checking
    bool  IsExtended() const { return fFlags & kIsExtended; }
    bool  IsSmart() const { return fFlags & kIsSmartPtr; }

// access to C++ pointer and type
    void*  GetObject();
    void*& GetObjectRaw() { return IsExtended() ? *(void**) fObject : fObject; }
    Cppyy::TCppType_t ObjectIsA(bool check_smart = true) const;

// memory management: ownership of the underlying C++ object
    void PythonOwns();
    void CppOwns();

// data member cache
    CI_DatamemberCache_t& GetDatamemberCache();

// smart pointer management
    void SetSmart(PyObject* smart_type);
    void* GetSmartObject() { return GetObjectRaw(); }
    Cppyy::TCppType_t GetSmartIsA() const;

// cross-inheritance dispatch
    void SetDispatchPtr(void*);

// redefine pointer to object as fixed-size array
    void CastToArray(Py_ssize_t sz);
    Py_ssize_t ArrayLength();

private:
    void  CreateExtension();
    void* GetExtendedObject();
};


//- public methods -----------------------------------------------------------
inline void CPPInstance::Set(void* address, EFlags flags)
{
// Initialize the proxy with the pointer value 'address.'
    if (flags != kDefault) fFlags = flags;
    GetObjectRaw() = address;
}

//----------------------------------------------------------------------------
inline void* CPPInstance::GetObject()
{
// Retrieve a pointer to the held C++ object.
    if (!IsExtended()) {
        if (fObject && (fFlags & kIsReference))
            return *(reinterpret_cast<void**>(fObject));
        else
            return fObject;   // may be null
    } else
        return GetExtendedObject();
}

//----------------------------------------------------------------------------
inline Cppyy::TCppType_t CPPInstance::ObjectIsA(bool check_smart) const
{
// Retrieve the C++ type identifier (or raw type if smart).
    if (check_smart || !IsSmart()) return ((CPPClass*)Py_TYPE(this))->fCppType;
    return GetSmartIsA();
}


//- object proxy type and type verification ----------------------------------
CPYCPPYY_IMPORT PyTypeObject CPPInstance_Type;

template<typename T>
inline bool CPPInstance_Check(T* object)
{
// Short-circuit the type check by checking tp_new which all generated subclasses
// of CPPInstance inherit.
    return object && \
        (Py_TYPE(object)->tp_new == CPPInstance_Type.tp_new || \
         PyObject_TypeCheck(object, &CPPInstance_Type));
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
