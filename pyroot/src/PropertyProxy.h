// @(#)root/pyroot:$Name:  $:$Id: PropertyProxy.h,v 1.1 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

#ifndef PYROOT_PROPERTYPROXY_H
#define PYROOT_PROPERTYPROXY_H

// Bindings
#include "Converters.h"

// ROOT
#include "DllImport.h"
class TDataMember;

// Standard
#include <string>


namespace PyROOT {

/** Proxy to ROOT data presented as python property
      @author  WLAV
      @date    02/12/2005
      @version 1.1
 */

   class ObjectProxy;

   class PropertyProxy {
   public:
      void Set( TDataMember* );
 
      const std::string& GetName() const { return fName; }
      long GetAddress( ObjectProxy* pyobj /* owner */ );

   public:               // public, as the python C-API works with C structs
      PyObject_HEAD
      std::string    fName;
      TDataMember*   fDataMember;
      Converter*     fConverter;

   private:              // private, as the python C-API will handle creation
      PropertyProxy() {}
   };


//- property proxy type and type verification --------------------------------
   R__EXTERN PyTypeObject PropertyProxy_Type;

   template< typename T >
   inline bool PropertyProxy_Check( T* object )
   {
      return object && PyObject_TypeCheck( object, &PropertyProxy_Type );
   }

   template< typename T >
   inline bool PropertyProxy_CheckExact( T* object )
   {
      return object && object->ob_type == &PropertyProxy_Type;
   }

//- creation -----------------------------------------------------------------
   inline PropertyProxy* PropertyProxy_New( TDataMember* dataMember )
   {
      PropertyProxy* pyprop =
         (PropertyProxy*)PropertyProxy_Type.tp_new( &PropertyProxy_Type, 0, 0 );
      pyprop->Set( dataMember );
      return pyprop;
   }

} // namespace PyROOT

#endif // !PYROOT_PROPERTYPROXY_H
