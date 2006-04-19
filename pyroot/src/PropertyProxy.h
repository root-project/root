// @(#)root/pyroot:$Name:  $:$Id: PropertyProxy.h,v 1.4 2005/10/25 05:13:15 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

#ifndef PYROOT_PROPERTYPROXY_H
#define PYROOT_PROPERTYPROXY_H

// Bindings
#include "Converters.h"

// ROOT
#include "DllImport.h"
class TDataMember;
class TGlobal;

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
 
      std::string GetName() { return fDMInfo.Name(); }
      Long_t GetAddress( ObjectProxy* pyobj /* owner */ );

   public:               // public, as the python C-API works with C structs
      PyObject_HEAD
      G__DataMemberInfo fDMInfo;
      Long_t            fProperty;
      TConverter*       fConverter;

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
      return object && object->ob_type == &PropertyProxy_Type;
   }

//- creation -----------------------------------------------------------------
   template< class T >
   inline PropertyProxy* PropertyProxy_New( T* dmi )
   {
      PropertyProxy* pyprop =
         (PropertyProxy*)PropertyProxy_Type.tp_new( &PropertyProxy_Type, 0, 0 );
      pyprop->Set( dmi );
      return pyprop;
   }

} // namespace PyROOT

#endif // !PYROOT_PROPERTYPROXY_H
