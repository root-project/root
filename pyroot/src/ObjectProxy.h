// @(#)root/pyroot:$Name:  $:$Id: ObjectProxy.h,v 1.3 2005/08/25 06:44:15 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

#ifndef PYROOT_OBJECTPROXY_H
#define PYROOT_OBJECTPROXY_H

// ROOT
#include "DllImport.h"
#include "TClassRef.h"
class TClass;


namespace PyROOT {

/** Object proxy object to hold ROOT instance
      @author  WLAV
      @date    01/04/2005
      @version 1.0
 */

   class ObjectProxy {
   public:
      enum EFlags { kNone = 0x0, kIsOwner = 0x0001, kIsReference = 0x0002 };

   public:
      void Set( void** address, TClass* klass, EFlags flags = kNone )
      {
         fObject = (void*) address;
         fClass  = klass;
         fFlags  = flags | kIsReference;
      }
 
      void Set( void* object, TClass* klass, EFlags flags = kNone )
      {
         fObject = object;
         fClass  = klass;
         fFlags  = flags & ~kIsReference;
      }

      void* GetObject() const
      {
         if ( fObject && ( fFlags & kIsReference ) )
            return *(reinterpret_cast< void** >( const_cast< void* >( fObject ) ));
         else
            return const_cast< void* >( fObject );        // may be null
      }

      TClass* ObjectIsA() const
      {
         return fClass.GetClass();                        // may return null
      }

      void HoldOn() { fFlags |= kIsOwner; }
      void Release() { fFlags &= ~kIsOwner; }

   public:               // public, as the python C-API works with C structs
      PyObject_HEAD
      void*     fObject;
      TClassRef fClass;
      int       fFlags;

   private:              // private, as the python C-API will handle creation
      ObjectProxy() {}
   };


//- object proxy type and type verification ----------------------------------
   R__EXTERN PyTypeObject ObjectProxy_Type;

   template< typename T >
   inline bool ObjectProxy_Check( T* object )
   {
      return object && PyObject_TypeCheck( object, &ObjectProxy_Type );
   }

   template< typename T >
   inline bool ObjectProxy_CheckExact( T* object )
   {
      return object && object->ob_type == &ObjectProxy_Type;
   }


//- helper for memory regulation (no PyTypeObject equiv. member in p2.2) -----
   void op_dealloc_nofree( ObjectProxy* );   

} // namespace PyROOT

#endif // !PYROOT_OBJECTPROXY_H
