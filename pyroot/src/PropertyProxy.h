// @(#)root/pyroot:$Name:  $:$Id: PropertyProxy.h,v 1.68 2005/01/28 05:45:41 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

#ifndef PYROOT_PROPERTYPROXY_H
#define PYROOT_PROPERTYPROXY_H

// Bindings
#include "Utility.h"

// ROOT
#include "DllImport.h"
class TDataMember;

// Standard
#include <string>


namespace PyROOT {

/** Proxy to ROOT data presented as python property
      @author  WLAV
      @date    02/12/2005
      @version 1.0
 */

   class PropertyProxy {
   public:
      void Set( TDataMember* );
 
      const std::string& GetName() const { return fName; }

   public:               // public, as the python C-API works with C structs
      PyObject_HEAD
      std::string        fName;
      TDataMember*       fDataMember;
      Utility::EDataType fDataType;

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
