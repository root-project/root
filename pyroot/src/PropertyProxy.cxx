// @(#)root/pyroot:$Name:  $:$Id: PropertyProxy.cxx,v 1.68 2005/01/28 05:45:41 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "PropertyProxy.h"
#include "ObjectProxy.h"
#include "PyBufferFactory.h"
#include "RootWrapper.h"

// ROOT
#include "TROOT.h"
#include "TClass.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TClassEdit.h"

// CINT
#include "Api.h"


namespace PyROOT {

namespace {

//= PyROOT property proxy property behaviour =================================
   PyObject* pp_get( PropertyProxy* pyprop, ObjectProxy* pyobj, PyObject* )
   {
      if ( pyprop->fDataMember->Property() & G__BIT_ISSTATIC ) {
         long offset = 0;
         G__DataMemberInfo dmi = pyprop->fDataMember->GetClass()->GetClassInfo()->GetDataMember(
               pyprop->GetName().c_str(), &offset );

         switch ( pyprop->fDataType ) {
         case Utility::kShort:
            return PyInt_FromLong( (long) *((int*)((G__var_array*)dmi.Handle())->p[dmi.Index()]) );
         case Utility::kEnum:
         case Utility::kInt:
            return PyInt_FromLong( *((int*)((G__var_array*)dmi.Handle())->p[dmi.Index()]) );
         case Utility::kLong:
            return PyLong_FromLong( *((long*)((G__var_array*)dmi.Handle())->p[dmi.Index()]) );
         case Utility::kFloat:
            return PyFloat_FromDouble( *((float*)((G__var_array*)dmi.Handle())->p[dmi.Index()]) );
         case Utility::kDouble:
            return PyFloat_FromDouble( *((double*)((G__var_array*)dmi.Handle())->p[dmi.Index()]) );
         default:
            PyErr_SetString( PyExc_RuntimeError, "no converter available for this property" );
         }
         
         return 0;
      }

      int offset = pyprop->fDataMember->GetOffsetCint();
      void* obj = pyobj->GetObject();
      if ( ! obj ) {
         PyErr_SetString( PyExc_ReferenceError, "attempt to access a null-pointer" );
         return 0;
      }

      switch ( pyprop->fDataType ) {
      case Utility::kShort:
         return PyInt_FromLong( (long) *((Short_t*)((long)obj+offset)) );
      case Utility::kEnum:
      case Utility::kInt:
         return PyInt_FromLong( *((Long_t*)((long)obj+offset)) );
      case Utility::kLong:
         return PyLong_FromLong( *((Long_t*)((long)obj+offset)) );
      case Utility::kUInt:
      case Utility::kULong:
         return PyLong_FromLong( *((ULong_t*)((long)obj+offset)) );
      case Utility::kFloat:
         return PyFloat_FromDouble( *((Float_t*)((long)obj+offset)) );
      case Utility::kDouble:
         return PyFloat_FromDouble( *((Double_t*)((long)obj+offset)) );
      case Utility::kIntPtr:
         return PyBufferFactory::Instance()->PyBuffer_FromMemory( *((Int_t**)((long)obj+offset)) );
      case Utility::kLongPtr:
         return PyBufferFactory::Instance()->PyBuffer_FromMemory( *((Long_t**)((long)obj+offset)) );
      case Utility::kFloatPtr:
         return PyBufferFactory::Instance()->PyBuffer_FromMemory( *((Float_t**)((long)obj+offset)) );
      case Utility::kDoublePtr:
         return PyBufferFactory::Instance()->PyBuffer_FromMemory( *((Double_t**)((long)obj+offset)) );
      case Utility::kOther: {
      // TODO: refactor this code with TMethodHolder returns
         std::string sname = TClassEdit::ShortType(
            G__TypeInfo( pyprop->fDataMember->GetFullTypeName() ).TrueName(), 1 );

         TClass* klass = gROOT->GetClass( sname.c_str(), 1 );
         long* address = *((long**)((int)obj+offset));

         if ( klass && address ) {
         // special case: cross-cast to real class for TGlobal returns
            if ( sname == "TGlobal" )
               return BindRootGlobal( (TGlobal*)address );

            return BindRootObject( (void*)address, klass );
         }

      // fall through ...
      }
      default:
         PyErr_SetString( PyExc_RuntimeError, "no converter available for this property" );
      }

      return 0;
   }

//____________________________________________________________________________
   PyObject* pp_set( PropertyProxy* pyprop, ObjectProxy* pyobj, PyObject* value )
   {
      int offset = pyprop->fDataMember->GetOffsetCint();
      void* obj = pyobj->GetObject();
      if ( ! obj ) {
         PyErr_SetString( PyExc_ReferenceError, "attempt to access a null-pointer" );
         return 0;
      }

      switch( pyprop->fDataType ) {
      case Utility::kShort: {
         *((Short_t*)((int)obj+offset))  = (Short_t) PyLong_AsLong( value );
         break;
      }
      case Utility::kInt:
      case Utility::kLong:
      case Utility::kEnum: {
         *((Long_t*)((int)obj+offset))   = PyLong_AsLong( value );
         break;
      }
      case Utility::kFloat: {
         *((Float_t*)((int)obj+offset))  = PyFloat_AsDouble( value );
         break;
      }
      case Utility::kDouble: {
         *((Double_t*)((int)obj+offset)) = PyFloat_AsDouble( value );
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


//= PyROOT property proxy construciton =======================================
   PropertyProxy* pp_new( PyTypeObject* pytype, PyObject*, PyObject* )
   {
      PropertyProxy* pyprop = (PropertyProxy*)pytype->tp_alloc( pytype, 0 );
      new ( &pyprop->fName ) std::string("");

      return pyprop;
   }

} // unnamed namespace


//= PyROOT property proxy type ===============================================
PyTypeObject PropertyProxy_Type = {
   PyObject_HEAD_INIT( &PyType_Type )
   0,                         // ob_size
   "ROOT.PropertyProxy",      // tp_name
   sizeof(PropertyProxy),     // tp_basicsize
   0,                         // tp_itemsize
   0,                         // tp_dealloc
   0,                         // tp_print
   0,                         // tp_getattr
   0,                         // tp_setattr
   0,                         // tp_compare
   0,                         // tp_repr
   0,                         // tp_as_number
   0,                         // tp_as_sequence
   0,                         // tp_as_mapping
   0,                         // tp_hash
   0,                         // tp_call
   0,                         // tp_str
   0,                         // tp_getattro
   0,                         // tp_setattro
   0,                         // tp_as_buffer
   Py_TPFLAGS_DEFAULT,        // tp_flags
   "PyROOT property proxy (internal)",            // tp_doc
   0,                         // tp_traverse
   0,                         // tp_clear
   0,                         // tp_richcompare
   0,                         // tp_weaklistoffset
   0,                         // tp_iter
   0,                         // tp_iternext
   0,                         // tp_methods
   0,                         // tp_members
   0,                         // tp_getset
   0,                         // tp_base
   0,                         // tp_dict
   (descrgetfunc)pp_get,      // tp_descr_get
   (descrsetfunc)pp_set,      // tp_descr_set
   0,                         // tp_dictoffset
   0,                         // tp_init
   0,                         // tp_alloc
   (newfunc)pp_new,           // tp_new
   0,                         // tp_free
   0,                         // tp_is_gc
   0,                         // tp_bases
   0,                         // tp_mro
   0,                         // tp_cache
   0,                         // tp_subclasses
   0,                         // tp_weaklist

#if PY_MAJOR_VERSION >= 2 && PY_MINOR_VERSION >= 3
   0                          // tp_del
#endif
};

} // namespace PyROOT


//- public members -----------------------------------------------------------
void PyROOT::PropertyProxy::Set( TDataMember* dataMember )
{
   fName       = dataMember->GetName();
   fDataMember = dataMember;
   fDataType   = Utility::effectiveType( dataMember->GetFullTypeName() );
}
