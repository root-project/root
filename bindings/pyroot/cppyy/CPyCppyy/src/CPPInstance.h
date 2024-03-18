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
        kNoWrapConv  = 0x0001,
        kIsOwner     = 0x0002,
        kIsExtended  = 0x0004,
        kIsReference = 0x0008,
        kIsRValue    = 0x0010,
        kIsLValue    = 0x0020,
        kIsValue     = 0x0040,
        kIsPtrPtr    = 0x0080,
        kIsArray     = 0x0100,
        kIsSmartPtr  = 0x0200,
        kNoMemReg    = 0x0400,
        kHasLifeLine = 0x0800,
        kIsRegulated = 0x1000,
        kIsActual    = 0x2000 };

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
