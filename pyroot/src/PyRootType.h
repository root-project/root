// @(#)root/pyroot:$Name:  $:$Id: ObjectProxy.h,v 1.2 2005/03/30 05:16:19 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

#ifndef PYROOT_PYROOTTYPE_H
#define PYROOT_PYROOTTYPE_H

// ROOT
#include "DllImport.h"


namespace PyROOT {

//- metatype type and type verification --------------------------------------
   R__EXTERN PyTypeObject PyRootType_Type;

   template< typename T >
   inline bool PyRootType_Check( T* object )
   {
      return object && PyObject_TypeCheck( object, &PyRootType_Type );
   }

   template< typename T >
   inline bool PyRootType_CheckExact( T* object )
   {
      return object && object->ob_type == &PyRootType_Type;
   }

} // namespace PyROOT

#endif // !PYROOT_OBJECTPROXY_H
