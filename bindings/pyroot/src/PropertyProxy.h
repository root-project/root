// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005

#ifndef PYROOT_PROPERTYPROXY_H
#define PYROOT_PROPERTYPROXY_H

// Bindings
#include "Converters.h"

// ROOT
#include "DllImport.h"
#include "TClassRef.h"
class TDataMember;
class TGlobal;

// Reflex
#ifdef PYROOT_USE_REFLEX
#include "Reflex/Member.h"
#endif

// CINT
#include "DataMbr.h"

// Standard
#include <string>


namespace PyROOT {

/** Proxy to ROOT data presented as python property
      @author  WLAV
      @date    02/12/2005
      @version 2.0
 */

   class ObjectProxy;

   class PropertyProxy {
   public:
      void Set( TDataMember* );
      void Set( TGlobal* );
#ifdef PYROOT_USE_REFLEX
      void Set( const ROOT::Reflex::Member& );
#endif

      std::string GetName() { return fName; }
      Long_t GetAddress( ObjectProxy* pyobj /* owner */ );

   public:               // public, as the python C-API works with C structs
      PyObject_HEAD
      Long_t       fOffset;
      Long_t       fProperty;
      TConverter*  fConverter;
      Int_t        fOwnerTagnum;   // TODO: wrap up ...
      std::string  fName;
      Int_t        fOwnerIsNamespace;

   private:              // private, as the python C-API will handle creation
      PropertyProxy() {}
   };


//- property proxy type and type verification --------------------------------
   R__EXTERN PyTypeObject PropertyProxy_Type;

   template< typename T >
   inline Bool_t PropertyProxy_Check( T* object )
   {
      return object && PyObject_TypeCheck( object, &PropertyProxy_Type );
   }

   template< typename T >
   inline Bool_t PropertyProxy_CheckExact( T* object )
   {
      return object && Py_TYPE(object) == &PropertyProxy_Type;
   }

//- creation -----------------------------------------------------------------
   template< class T >
   inline PropertyProxy* PropertyProxy_New( const T& dmi )
   {
   // Create an initialize a new property descriptor, given the C++ datum.
      PropertyProxy* pyprop =
         (PropertyProxy*)PropertyProxy_Type.tp_new( &PropertyProxy_Type, 0, 0 );
      pyprop->Set( dmi );
      return pyprop;
   }

} // namespace PyROOT

#endif // !PYROOT_PROPERTYPROXY_H
