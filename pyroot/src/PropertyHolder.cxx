// Author: Wim Lavrijsen, Jun 2004

// Bindings
#include "PyROOT.h"
#include "PropertyHolder.h"
#include "PyBufferFactory.h"

// ROOT
#include "TDataMember.h"
#include "TDataType.h"

// Standard
#include <string.h>
#include <string>
#include <iostream>


//- protected class members --------------------------------------------------
void PyROOT::PropertyHolder::destroy( void* pp ) {
   delete reinterpret_cast< PyROOT::PropertyHolder* >( pp );
}


PyObject* PyROOT::PropertyHolder::invoke_get( PyObject* self, PyObject* args, PyObject* kws ) {
   return reinterpret_cast< PyROOT::PropertyHolder* >( PyCObject_AsVoidPtr( self ) )->get( args, kws );
}

PyObject* PyROOT::PropertyHolder::invoke_set( PyObject* self, PyObject* args, PyObject* kws ) {
   reinterpret_cast< PyROOT::PropertyHolder* >( PyCObject_AsVoidPtr( self ) )->set( args, kws );
   Py_INCREF( Py_None );
   return Py_None;
}


//- public class members -----------------------------------------------------
bool PyROOT::PropertyHolder::addToClass( PropertyHolder* pp, PyObject* pyclass ) {
   const std::string& name = pp->getName();

// setup a property for the member
   PyMethodDef* pdef = new PyMethodDef;
   pdef->ml_name  = new char[ 3 + name.length() + 1 ];
   strcpy( pdef->ml_name, ("get"+name).c_str() );
   pdef->ml_meth  = (PyCFunction) &invoke_get;
   pdef->ml_flags = METH_VARARGS | METH_KEYWORDS;
   pdef->ml_doc   = NULL;

   PyObject* propget = PyCFunction_New( pdef, PyCObject_FromVoidPtr( pp, NULL ) );

   pdef = new PyMethodDef;
   pdef->ml_name  = new char[ 3 + name.length() + 1 ];
   strcpy( pdef->ml_name, ("set"+name).c_str() );
   pdef->ml_meth = (PyCFunction) &invoke_set;
   pdef->ml_flags = METH_VARARGS | METH_KEYWORDS;
   pdef->ml_doc   = NULL;

   PyObject* propset = PyCFunction_New( pdef, PyCObject_FromVoidPtr( pp, NULL ) );

   PyObject* property = PyObject_CallFunctionObjArgs(
      (PyObject*)&PyProperty_Type, propget, propset, NULL );
   Py_DECREF( propget );
   Py_DECREF( propset );

   PyObject_SetAttrString( pyclass, const_cast< char* >( name.c_str() ), property );
   Py_DECREF( property );

   return true;
}


//- constructor --------------------------------------------------------------
PyROOT::PropertyHolder::PropertyHolder( TDataMember* dm ) : m_name( dm->GetName() ) {
   m_dataMember = dm;
   m_dataType = Utility::effectiveType( dm->GetFullTypeName() );
}


//- public members -----------------------------------------------------------
PyObject* PyROOT::PropertyHolder::get( PyObject* args, PyObject* ) {
   int offset = m_dataMember->GetOffset();
   void* obj = Utility::getObjectFromHolderFromArgs( args );

   switch ( m_dataType ) {
   case Utility::kInt:
   case Utility::kLong: {
      return PyLong_FromLong( *((int*)((int)obj+offset)) );
   }
   case Utility::kFloat: {
      return PyFloat_FromDouble( *((float*)((int)obj+offset)) );
   }
   case Utility::kDouble: {
      return PyFloat_FromDouble( *((double*)((int)obj+offset)) );
   }
   case Utility::kIntPtr: {
      return PyBufferFactory::getInstance()->PyBuffer_FromMemory( *((int**)((int)obj+offset)) );
   }
   case Utility::kLongPtr: {
      return PyBufferFactory::getInstance()->PyBuffer_FromMemory( *((long**)((int)obj+offset)) );
   }
   case Utility::kFloatPtr: {
      return PyBufferFactory::getInstance()->PyBuffer_FromMemory( *((float**)((int)obj+offset)) );
   }
   case Utility::kDoublePtr: {
      return PyBufferFactory::getInstance()->PyBuffer_FromMemory( *((double**)((int)obj+offset)) );
   }
   default:
      PyErr_SetString( PyExc_RuntimeError, "sorry this is experimental ... stay tuned" );
   }

   return 0;
}

void PyROOT::PropertyHolder::set( PyObject* args, PyObject* ) {
   int offset = m_dataMember->GetOffset();
   void* obj = Utility::getObjectFromHolderFromArgs( args );

   switch( m_dataType ) {
   case Utility::kInt:
   case Utility::kLong: {
      *((int*)((int)obj+offset)) = PyLong_AsLong( PyTuple_GetItem( args, 1 ) );
      break;
   }
   case Utility::kFloat: {
      *((float*)((int)obj+offset)) = PyFloat_AsDouble( PyTuple_GetItem( args, 1 ) );
      break;
   }
   case Utility::kDouble: {
      *((double*)((int)obj+offset)) = PyFloat_AsDouble( PyTuple_GetItem( args, 1 ) );
      break;
   }
   default:
      PyErr_SetString( PyExc_RuntimeError, "this property doesn't allow assignment" );
   }

   if ( PyErr_Occurred() )
      PyErr_Print();
}
