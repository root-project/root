// @(#)root/pyroot:$Name:  $:$Id: PropertyProxy.cxx,v 1.3 2005/05/25 06:23:36 brun Exp $
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
      long address = pyprop->GetAddress( pyobj );
      if ( address < 0 )
         return 0;

   // for fixed size arrays
      void* ptr = (void*)address;
      if ( pyprop->fDataMember->GetArrayDim() != 0 )
         ptr = &address;

      if ( pyprop->fConverter != 0 )
         return pyprop->fConverter->FromMemory( ptr );

      PyErr_Format( PyExc_NotImplementedError,
         "could not convert %s", pyprop->fDataMember->GetName() );
      return 0;

   /*
      case Utility::kOther: {
      // TODO: refactor this code with TMethodHolder returns
         std::string sname = TClassEdit::ShortType(
            G__TypeInfo( pyprop->fDataMember->GetFullTypeName() ).TrueName(), 1 );

         TClass* klass = gROOT->GetClass( sname.c_str(), 1 );
         long* ref = *((long**)address);

         if ( klass && ref ) {
         // special case: cross-cast to real class for TGlobal returns
            if ( sname == "TGlobal" )
               return BindRootGlobal( (TGlobal*)ref );

            return BindRootObject( (void*)ref, klass );
         }
      }
   */
   }

//____________________________________________________________________________
   int pp_set( PropertyProxy* pyprop, ObjectProxy* pyobj, PyObject* value )
   {
      const int errret = -1;

      long address = pyprop->GetAddress( pyobj );
      if ( address < 0 )
         return errret;

   // fixed size arrays and object instances are not passed as address, but as &address
      void* ptr = (void*)address;
      if ( pyprop->fDataMember->GetArrayDim() != 0 )
         ptr = &address;

   // actual conversion; return on success
      if ( pyprop->fConverter && pyprop->fConverter->ToMemory( value, ptr ) )
         return 0;

   // set a python error, if not already done
      if ( ! PyErr_Occurred() )
         PyErr_SetString( PyExc_RuntimeError, "property type mismatch or assignment not allowed" );

   // failure ...
      return errret;
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
   (char*)"ROOT.PropertyProxy",                  // tp_name
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
   (char*)"PyROOT property proxy (internal)",    // tp_doc
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

   std::string fullType = fDataMember->GetFullTypeName();
   if ( (int)fDataMember->GetArrayDim() != 0 ||
        ( ! fDataMember->IsBasic() && fDataMember->IsaPointer() ) ) {
      fullType.append( "*" );
   }

   fConverter  = CreateConverter( fullType, fDataMember->GetMaxIndex( 0 ) );
}

//____________________________________________________________________________
long PyROOT::PropertyProxy::GetAddress( ObjectProxy* pyobj ) {
   const int errret = -1;

// get offsets from CINT
   G__ClassInfo* clInfo = fDataMember->GetClass()->GetClassInfo();

// class attributes
   if ( fDataMember->Property() & G__BIT_ISSTATIC ) {
      long offset = 0;
      G__DataMemberInfo dmi = clInfo->GetDataMember( fName.c_str(), &offset );
      return (long)((G__var_array*)dmi.Handle())->p[dmi.Index()];
   }

// instance attributes; requires object for full address
   if ( ! ObjectProxy_Check( pyobj ) ) {
      PyErr_Format( PyExc_TypeError,
         "object instance required for access to property \"%s\"", fName.c_str() );
      return errret;
   }

   void* obj = pyobj->GetObject();
   if ( ! obj ) {
      PyErr_SetString( PyExc_ReferenceError, "attempt to access a null-pointer" );
      return errret;
   }

   long offset = G__isanybase(
      clInfo->Tagnum(), pyobj->ObjectIsA()->GetClassInfo()->Tagnum(), (long)obj );
   return (long)obj + offset + fDataMember->GetOffsetCint();
}
