// @(#)root/pyroot:$Name:  $:$Id: Utility.cxx,v 1.3 2004/06/12 05:35:10 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "Utility.h"
#include "ObjectHolder.h"

// ROOT
#include "TClassEdit.h"

// CINT
#include "Api.h"


//- data ------------------------------------------------------------------------
char* PyROOT::Utility::theObject_ = const_cast< char* >( "_theObject" );

PyObject* PyROOT::Utility::theObjectString_ =
   PyString_FromString( PyROOT::Utility::theObject_ );


//- public functions ------------------------------------------------------------
void PyROOT::Utility::addToClass(
      const char* label, PyCFunction cfunc, PyObject* cls, int flags ) {
   PyMethodDef* pdef = new PyMethodDef;
   pdef->ml_name  = const_cast< char* >( label );
   pdef->ml_meth  = cfunc;
   pdef->ml_flags = flags;
   pdef->ml_doc   = NULL;

   PyObject* func = PyCFunction_New( pdef, NULL );
   PyObject* method = PyMethod_New( func, NULL, cls );
   PyObject_SetAttrString( cls, pdef->ml_name, method );
   Py_DECREF( func );
   Py_DECREF( method );
}


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


void* PyROOT::Utility::getObjectFromHolderFromArgs( PyObject* argsTuple ) {
   PyObject* self = PyTuple_GetItem( argsTuple, 0 );
   Py_INCREF( self );

   PyROOT::ObjectHolder* holder = getObjectHolder( self );
   Py_DECREF( self );

   if ( holder != 0 )
      return holder->getObject();
   return 0;
}


PyROOT::Utility::EDataType PyROOT::Utility::effectiveType( const std::string& typeName ) {
   EDataType effType = kOther;

   std::string shortName = TClassEdit::ShortType( G__TypeInfo( typeName.c_str() ).TrueName(), 1 );

   if ( isPointer( typeName ) ) {
      if ( shortName == "char" )
         effType = kString;
      else if ( shortName == "double" )
         effType = kDoublePtr;
      else if ( shortName == "float" )
         effType = kFloatPtr;
      else if ( shortName == "long" )
         effType = kLongPtr;
      else if ( shortName == "int" )
         effType = kIntPtr;
      else if ( shortName == "void" )
         effType = kVoidPtr;
      else
         effType = kOther;
   }
   else if ( shortName == "bool" )
      effType = kBool;
   else if ( shortName == "char" )
      effType = kChar;
   else if ( shortName == "short" )
      effType = kShort;
   else if ( shortName == "int" )
      effType = kInt;
   else if ( shortName == "long" )
      effType = kLong;
   else if ( shortName == "float" )
      effType = kFloat;
   else if ( shortName == "double" )
      effType = kDouble;
   else if ( shortName == "void" )
      effType = kVoid;
   else
      effType = kOther;

   return effType;
}


bool PyROOT::Utility::isPointer( const std::string& tn ) {
   bool isp = false;
   for ( std::string::const_reverse_iterator it = tn.rbegin(); it != tn.rend(); ++it ) {
      if ( *it == '*' || *it == '&' ) {
         isp = true;
         break;
      }
      else if ( isalnum( *it ) )
         break;
   }

   return isp;
}
