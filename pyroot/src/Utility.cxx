// @(#)root/pyroot:$Name:  $:$Id:  $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "Utility.h"
#include "ObjectHolder.h"


//- data ------------------------------------------------------------------------
char* PyROOT::Utility::theObject_ = const_cast< char* >( "_theObject" );

PyObject* PyROOT::Utility::theObjectString_ =
   PyString_FromString( PyROOT::Utility::theObject_ );


//- public functions ------------------------------------------------------------
PyROOT::ObjectHolder* PyROOT::Utility::getObjectHolder( PyObject* self ) {
   if ( self !=  0  ) {
      PyObject* cobj = PyObject_GetAttr( self, theObjectString_ );
      if ( cobj != 0 ) {
         ObjectHolder* holder =
            reinterpret_cast< PyROOT::ObjectHolder* >( PyCObject_AsVoidPtr( cobj ) );
         Py_DECREF( cobj );
         return holder;
      }
      else
         PyErr_Clear();
   }

   return 0;
}
