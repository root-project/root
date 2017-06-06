// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "PyStrings.h"
#include "PropertyProxy.h"
#include "ObjectProxy.h"
#include "Utility.h"


namespace PyROOT {

   enum ETypeDetails {
      kNone           =    0,
      kIsStaticData   =    1,
      kIsEnumData     =    2,
      kIsConstData    =    4,
      kIsArrayType    =    8
   };

namespace {

//= PyROOT property proxy property behaviour =================================
   PyObject* pp_get( PropertyProxy* pyprop, ObjectProxy* pyobj, PyObject* )
   {
   // normal getter access
      void* address = pyprop->GetAddress( pyobj );
      if (!address || (ptrdiff_t)address == -1 /* Cling error */) return nullptr;

      // for fixed size arrays
      void* ptr = address;
      if ( pyprop->fProperty & kIsArrayType )
         ptr = &address;

   // not-initialized or public data accesses through class (e.g. by help())
      if ( ! ptr || (ptrdiff_t)ptr == -1 /* Cling error */ ) {
         Py_INCREF( pyprop );
         return (PyObject*)pyprop;
      }

      if (pyprop->fConverter != nullptr) {
         PyObject* result = pyprop->fConverter->FromMemory( ptr );
         if ( ! result )
            return result;

         // ensure that the encapsulating class does not go away for the duration
         // of the data member's lifetime, if it is a bound type (it doesn't matter
         // for builtin types, b/c those are copied over into python types and thus
         // end up being "stand-alone")
         if ( pyobj && ObjectProxy_Check( result ) ) {
            if ( PyObject_SetAttr( result, PyStrings::gLifeLine, (PyObject*)pyobj ) == -1 )
               PyErr_Clear();     // ignored
         }
         return result;
      }

      PyErr_Format( PyExc_NotImplementedError,
         "no converter available for \"%s\"", pyprop->GetName().c_str() );
      return nullptr;
   }

////////////////////////////////////////////////////////////////////////////////
/// Set the value of the C++ datum held.

   int pp_set( PropertyProxy* pyprop, ObjectProxy* pyobj, PyObject* value )
   {
      const int errret = -1;

   // filter const objects to prevent changing their values
      if ( ( pyprop->fProperty & kIsConstData ) ) {
         PyErr_SetString( PyExc_TypeError, "assignment to const data not allowed" );
         return errret;
      }

      ptrdiff_t address = (ptrdiff_t)pyprop->GetAddress( pyobj );
      if ( ! address || address == -1 /* Cling error */ )
         return errret;

   // for fixed size arrays
      void* ptr = (void*)address;
      if ( pyprop->fProperty & kIsArrayType )
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

//= PyROOT property proxy construction/destruction ===========================
   PropertyProxy* pp_new( PyTypeObject* pytype, PyObject*, PyObject* )
   {
   // Create and initialize a new property descriptor.
      PropertyProxy* pyprop = (PropertyProxy*)pytype->tp_alloc( pytype, 0 );

      pyprop->fOffset         = 0;
      pyprop->fProperty       = 0;
      pyprop->fConverter = nullptr;
      pyprop->fEnclosingScope = 0;
      new ( &pyprop->fName ) std::string();

      return pyprop;
   }

////////////////////////////////////////////////////////////////////////////////
/// Deallocate memory held by this descriptor.

   void pp_dealloc( PropertyProxy* pyprop )
   {
      using namespace std;
      delete pyprop->fConverter;
      pyprop->fName.~string();

      Py_TYPE(pyprop)->tp_free( (PyObject*)pyprop );
   }


} // unnamed namespace


//= PyROOT property proxy type ===============================================
PyTypeObject PropertyProxy_Type = {
   PyVarObject_HEAD_INIT(&PyType_Type, 0)(char *) "ROOT.PropertyProxy", // tp_name
   sizeof(PropertyProxy),                                               // tp_basicsize
   0,                                                                   // tp_itemsize
   (destructor)pp_dealloc,                                              // tp_dealloc
   nullptr,                                                             // tp_print
   nullptr,                                                             // tp_getattr
   nullptr,                                                             // tp_setattr
   nullptr,                                                             // tp_compare
   nullptr,                                                             // tp_repr
   nullptr,                                                             // tp_as_number
   nullptr,                                                             // tp_as_sequence
   nullptr,                                                             // tp_as_mapping
   nullptr,                                                             // tp_hash
   nullptr,                                                             // tp_call
   nullptr,                                                             // tp_str
   nullptr,                                                             // tp_getattro
   nullptr,                                                             // tp_setattro
   nullptr,                                                             // tp_as_buffer
   Py_TPFLAGS_DEFAULT,                                                  // tp_flags
   (char *)"PyROOT property proxy (internal)",                          // tp_doc
   nullptr,                                                             // tp_traverse
   nullptr,                                                             // tp_clear
   nullptr,                                                             // tp_richcompare
   0,                                                                   // tp_weaklistoffset
   nullptr,                                                             // tp_iter
   nullptr,                                                             // tp_iternext
   nullptr,                                                             // tp_methods
   nullptr,                                                             // tp_members
   nullptr,                                                             // tp_getset
   nullptr,                                                             // tp_base
   nullptr,                                                             // tp_dict
   (descrgetfunc)pp_get,                                                // tp_descr_get
   (descrsetfunc)pp_set,                                                // tp_descr_set
   0,                                                                   // tp_dictoffset
   nullptr,                                                             // tp_init
   nullptr,                                                             // tp_alloc
   (newfunc)pp_new,                                                     // tp_new
   nullptr,                                                             // tp_free
   nullptr,                                                             // tp_is_gc
   nullptr,                                                             // tp_bases
   nullptr,                                                             // tp_mro
   nullptr,                                                             // tp_cache
   nullptr,                                                             // tp_subclasses
   nullptr                                                              // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
   ,
   nullptr // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
   ,
   0 // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
   ,
   0 // tp_finalize
#endif
};

} // namespace PyROOT


//- public members -----------------------------------------------------------
void PyROOT::PropertyProxy::Set( Cppyy::TCppScope_t scope, Cppyy::TCppIndex_t idata )
{
   fEnclosingScope = scope;
   fName           = Cppyy::GetDatamemberName( scope, idata );
   fOffset         = Cppyy::GetDatamemberOffset( scope, idata );
   fProperty       = Cppyy::IsStaticData( scope, idata ) ? kIsStaticData : 0;

   Int_t size = Cppyy::GetDimensionSize( scope, idata, 0 );
   if ( 0 < size )
      fProperty |= kIsArrayType;

   std::string fullType = Cppyy::GetDatamemberType( scope, idata );
   if ( Cppyy::IsEnumData( scope, idata ) ) {
      fullType = "UInt_t";
      fProperty |= kIsEnumData;
   }

   if ( Cppyy::IsConstData( scope, idata ) )
      fProperty |= kIsConstData;

   fConverter = CreateConverter( fullType, size );
}

////////////////////////////////////////////////////////////////////////////////

void PyROOT::PropertyProxy::Set( Cppyy::TCppScope_t scope, const std::string& name, void* address )
{
   fEnclosingScope = scope;
   fName           = name;
   fOffset         = (ptrdiff_t)address;
   fProperty       = (kIsStaticData | kIsConstData | kIsEnumData /* true, but may chance */ );
   fConverter      = CreateConverter( "UInt_t", -1 );
}

////////////////////////////////////////////////////////////////////////////////
/// class attributes, global properties

void* PyROOT::PropertyProxy::GetAddress( ObjectProxy* pyobj ) {
   if ( fProperty & kIsStaticData )
      return (void*)fOffset;

// special case: non-static lookup through class
   if (!pyobj) return nullptr;

   // instance attributes; requires valid object for full address
   if ( ! ObjectProxy_Check( pyobj ) ) {
      PyErr_Format( PyExc_TypeError,
         "object instance required for access to property \"%s\"", GetName().c_str() );
      return nullptr;
   }

   void* obj = pyobj->GetObject();
   if ( ! obj ) {
      PyErr_SetString( PyExc_ReferenceError, "attempt to access a null-pointer" );
      return nullptr;
   }

// the proxy's internal offset is calculated from the enclosing class
   ptrdiff_t offset = 0;
   if ( pyobj->ObjectIsA() != fEnclosingScope)
      offset = Cppyy::GetBaseOffset( pyobj->ObjectIsA(), fEnclosingScope, obj, 1 /* up-cast */ );

   return (void*)((ptrdiff_t)obj + offset + fOffset);
}
