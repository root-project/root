// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005

#ifndef PYROOT_OBJECTPROXY_H
#define PYROOT_OBJECTPROXY_H

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// PyROOT::ObjectProxy                                                      //
//                                                                          //
// Python-side proxy, encapsulaties a C++ object.                           //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


// Bindings
#include "PyRootType.h"

// ROOT
#include "DllImport.h"
#include "TInterpreterValue.h"


namespace PyROOT {

   class ObjectProxy {
   public:
      enum EFlags { kNone = 0x0, kIsOwner = 0x0001, kIsReference = 0x0002, kIsValue = 0x0004 };

   public:
      void Set( void* address, EFlags flags = kNone )
      {
      // Initialize the proxy with the pointer value 'address.'
         fObject = address;
         fFlags  = flags;
      }

      void* GetObject() const
      {
      // Retrieve a pointer to the held C++ object.
         if ( fObject && ( fFlags & kIsReference ) )
            return *(reinterpret_cast< void** >( const_cast< void* >( fObject ) ));
         else if ( fObject && ( fFlags & kIsValue ) )
            return ((TInterpreterValue*)fObject)->GetAsPointer();
         else
            return const_cast< void* >( fObject );          // may be null
      }

      Cppyy::TCppType_t ObjectIsA() const
      {
      // Retrieve a pointer to the C++ type; may return NULL.
         return ((PyRootClass*)Py_TYPE(this))->fCppType;
      }

      void HoldOn() { fFlags |= kIsOwner; }
      void Release() { fFlags &= ~kIsOwner; }

   public:               // public, as the python C-API works with C structs
      PyObject_HEAD
      void*     fObject;
      int       fFlags;

   private:              // private, as the python C-API will handle creation
      ObjectProxy() {}
   };


//- object proxy type and type verification ----------------------------------
   R__EXTERN PyTypeObject ObjectProxy_Type;

   template< typename T >
   inline Bool_t ObjectProxy_Check( T* object )
   {
      return object && PyObject_TypeCheck( object, &ObjectProxy_Type );
   }

   template< typename T >
   inline Bool_t ObjectProxy_CheckExact( T* object )
   {
      return object && Py_TYPE(object) == &ObjectProxy_Type;
   }


//- helper for memory regulation (no PyTypeObject equiv. member in p2.2) -----
   void op_dealloc_nofree( ObjectProxy* );

} // namespace PyROOT

#endif // !PYROOT_OBJECTPROXY_H
