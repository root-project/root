// Author: Wim Lavrijsen, Jun 2004

// Bindings
#include "PyROOT.h"
#include "PropertyHolder.h"
#include "PyBufferFactory.h"
#include "RootWrapper.h"
#include "ObjectHolder.h"

// ROOT
#include "TROOT.h"
#include "TClass.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TClassEdit.h"

// CINT
#include "Api.h"

// Standard
#include <string.h>
#include <string>


//- destructor callback --------------------------------------------------------
extern "C" void destroyPropertyHolder( void* pph ) {
   delete reinterpret_cast< PyROOT::PropertyHolder* >( pph );
}


//- protected class members --------------------------------------------------
PyObject* PyROOT::PropertyHolder::invoke_get( PyObject* self, PyObject* args, PyObject* kws ) {
   return reinterpret_cast< PropertyHolder* >( PyCObject_AsVoidPtr( self ) )->get( args, kws );
}

PyObject* PyROOT::PropertyHolder::invoke_set( PyObject* self, PyObject* args, PyObject* kws ) {
   return reinterpret_cast< PropertyHolder* >( PyCObject_AsVoidPtr( self ) )->set( args, kws );
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
   int offset = m_dataMember->GetOffsetCint();
   void* obj = Utility::getObjectFromHolderFromArgs( args );
   if ( ! obj ) {
      PyErr_SetString( PyExc_ReferenceError, "attempt to access a null-pointer" );
      return 0;
   }

   switch ( m_dataType ) {
   case Utility::kShort:
      return PyLong_FromLong( (long) *((Short_t*)((long)obj+offset)) );
   case Utility::kInt:
   case Utility::kLong:
   case Utility::kEnum: {
      return PyLong_FromLong( *((Long_t*)((long)obj+offset)) );
   }
   case Utility::kUInt:
   case Utility::kULong: {
      return PyLong_FromLong( *((ULong_t*)((long)obj+offset)) );
   }
   case Utility::kFloat: {
      return PyFloat_FromDouble( *((Float_t*)((long)obj+offset)) );
   }
   case Utility::kDouble: {
      return PyFloat_FromDouble( *((Double_t*)((long)obj+offset)) );
   }
   case Utility::kIntPtr: {
      return PyBufferFactory::getInstance()->PyBuffer_FromMemory( *((Int_t**)((long)obj+offset)) );
   }
   case Utility::kLongPtr: {
      return PyBufferFactory::getInstance()->PyBuffer_FromMemory( *((Long_t**)((long)obj+offset)) );
   }
   case Utility::kFloatPtr: {
      return PyBufferFactory::getInstance()->PyBuffer_FromMemory( *((Float_t**)((long)obj+offset)) );
   }
   case Utility::kDoublePtr: {
      return PyBufferFactory::getInstance()->PyBuffer_FromMemory( *((Double_t**)((long)obj+offset)) );
   }
   case Utility::kOther: {
   // TODO: refactor this code with TMethodHolder returns
      std::string sname = TClassEdit::ShortType(
         G__TypeInfo( m_dataMember->GetFullTypeName() ).TrueName(), 1 );

      TClass* cls = gROOT->GetClass( sname.c_str(), 1 );
      long* address = *((long**)((long)obj+offset));

      if ( cls && address ) {
      // special case: cross-cast to real class for TGlobal returns
         if ( sname == "TGlobal" )
            return bindRootGlobal( (TGlobal*)address );

      // upgrade to real class for TObject returns
         TClass* clActual = cls->GetActualClass( (void*)address );
         if ( clActual ) {
            int offset = (cls != clActual) ? clActual->GetBaseClassOffset( cls ) : 0;
            address -= offset;
         }

         return bindRootObject( new ObjectHolder( (void*)address, clActual, false ) );
      }

      // fall through ...
   }
   default:
      PyErr_SetString( PyExc_RuntimeError, "there is no converter available for this property" );
   }

   return 0;
}

PyObject* PyROOT::PropertyHolder::set( PyObject* args, PyObject* ) {
   int offset = m_dataMember->GetOffsetCint();
   void* obj = Utility::getObjectFromHolderFromArgs( args );
   if ( ! obj ) {
      PyErr_SetString( PyExc_ReferenceError, "attempt to access a null-pointer" );
      return 0;
   }

   PyObject* dm = PyTuple_GET_ITEM( args, 1 );

   switch( m_dataType ) {
   case Utility::kShort: {
      *((Short_t*)((long)obj+offset))  = (Short_t) PyLong_AsLong( dm );
      break;
   }
   case Utility::kInt:
   case Utility::kLong:
   case Utility::kEnum: {
      *((Long_t*)((long)obj+offset))   = PyLong_AsLong( dm );
      break;
   }
   case Utility::kFloat: {
      *((Float_t*)((long)obj+offset))  = PyFloat_AsDouble( dm );
      break;
   }
   case Utility::kDouble: {
      *((Double_t*)((long)obj+offset)) = PyFloat_AsDouble( dm );
      break;
   }
   default:
      PyErr_SetString( PyExc_RuntimeError, "this property doesn't allow assignment" );
   }

   if ( PyErr_Occurred() )
      return 0;

   Py_INCREF( Py_None );
   return Py_None;
}
