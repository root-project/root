// @(#)root/pyroot:$Name:  $:$Id: Converters.cxx,v 1.34 2006/12/08 07:42:31 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "Converters.h"
#include "ObjectProxy.h"
#include "PyBufferFactory.h"
#include "Utility.h"
#include "RootWrapper.h"

// ROOT
#include "TClass.h"
#include "TClassEdit.h"

// CINT
#include "Api.h"

// Standard
#include <limits.h>
#include <string.h>
#include <utility>


//- data ______________________________________________________________________
PyROOT::ConvFactories_t PyROOT::gConvFactories;


//- base converter implementation ---------------------------------------------
PyObject* PyROOT::TConverter::FromMemory( void* )
{
// could happen if no derived class override
   PyErr_SetString( PyExc_TypeError, "unknown type can not be converted from memory" );
   return 0;
}

//_____________________________________________________________________________
Bool_t PyROOT::TConverter::ToMemory( PyObject*, void* )
{
// could happen if no derived class override
   PyErr_SetString( PyExc_TypeError, "unknown type can not be converted to memory" );
   return kFALSE;
}


//- helper macro's ------------------------------------------------------------
#define PYROOT_IMPLEMENT_BASIC_CONVERTER( name, type, stype, F1, F2 )         \
PyObject* PyROOT::T##name##Converter::FromMemory( void* address )             \
{                                                                             \
   return F1( (stype)*((type*)address) );                                     \
}                                                                             \
                                                                              \
Bool_t PyROOT::T##name##Converter::ToMemory( PyObject* value, void* address ) \
{                                                                             \
   type s = (type)F2( value );                                                \
   if ( PyErr_Occurred() )                                                    \
      return kFALSE;                                                          \
   *((type*)address) = (type)s;                                               \
   return kTRUE;                                                              \
}

#define PYROOT_IMPLEMENT_BASIC_REF_CONVERTER( name )                          \
PyObject* PyROOT::T##name##Converter::FromMemory( void* )                     \
{                                                                             \
   return 0;                                                                  \
}                                                                             \
                                                                              \
Bool_t PyROOT::T##name##Converter::ToMemory( PyObject*, void* )               \
{                                                                             \
   return kFALSE;                                                             \
}


//_____________________________________________________________________________
#define PYROOT_IMPLEMENT_BASIC_CHAR_CONVERTER( name, type, low, high )        \
Bool_t PyROOT::T##name##Converter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )\
{                                                                             \
/* convert <pyobject> to C++ <<type>>, set arg for call, allow int -> char */ \
   if ( PyString_Check( pyobject ) ) {                                        \
      if ( PyString_GET_SIZE( pyobject ) == 1 ) {                             \
         para.fl = (Long_t)PyString_AS_STRING( pyobject )[0];                 \
         if ( func )                                                          \
            func->SetArg( para.fl );                                          \
      } else {                                                                \
         PyErr_Format( PyExc_TypeError,                                       \
            #type" expected, got string of size "PY_SSIZE_T_FORMAT, PyString_GET_SIZE( pyobject ) );\
         return kFALSE;                                                       \
      }                                                                       \
   } else {                                                                   \
      para.fl = PyLong_AsLong( pyobject );                                    \
      if ( PyErr_Occurred() ) {                                               \
         return kFALSE;                                                       \
      } else if ( ! ( low <= para.fl && para.fl <= high ) ) {                 \
         PyErr_SetString( PyExc_ValueError, "integer to character: value out of range" );\
         return kFALSE;                                                       \
      } else if ( func )                                                      \
         func->SetArg( para.fl );                                             \
   }                                                                          \
   return kTRUE;                                                              \
}                                                                             \
                                                                              \
PyObject* PyROOT::T##name##Converter::FromMemory( void* address )             \
{                                                                             \
   return PyString_FromFormat( "%c", *((type*)address) );                     \
}                                                                             \
                                                                              \
Bool_t PyROOT::T##name##Converter::ToMemory( PyObject* value, void* address ) \
{                                                                             \
   if ( PyString_Check( value ) ) {                                           \
      const char* buf = PyString_AsString( value );                           \
      if ( PyErr_Occurred() )                                                 \
         return kFALSE;                                                       \
      int len = strlen( buf );                                                \
      if ( len != 1 ) {                                                       \
         PyErr_Format( PyExc_TypeError, #type" expected, got string of size %d", len );\
         return kFALSE;                                                       \
      }                                                                       \
      *((type*)address) = (type)buf[0];                                       \
   } else {                                                                   \
      Long_t l = PyLong_AsLong( value );                                      \
      if ( PyErr_Occurred() )                                                 \
         return kFALSE;                                                       \
      if ( ! ( low <= l && l <= high ) ) {                                    \
         PyErr_SetString( PyExc_ValueError, "integer to character: value out of range" );\
         return kFALSE;                                                       \
      }                                                                       \
      *((type*)address) = (type)l;                                            \
   }                                                                          \
   return kTRUE;                                                              \
}


//- converters for built-ins --------------------------------------------------
Bool_t PyROOT::TLongConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ long, set arg for call
   para.fl = PyLong_AsLong( pyobject );
   if ( PyErr_Occurred() )
      return kFALSE;
   else if ( func )
      func->SetArg( para.fl );
   return kTRUE;
}

PYROOT_IMPLEMENT_BASIC_CONVERTER( Long, Long_t, Long_t, PyLong_FromLong, PyLong_AsLong )

//____________________________________________________________________________
Bool_t PyROOT::TLongRefConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ long&, set arg for call
   if ( ! PyInt_CheckExact( pyobject ) )
      return kFALSE;

   para.fl = (Long_t)&((PyIntObject*)pyobject)->ob_ival;
   if ( func )
      func->SetArgRef( ((PyIntObject*)pyobject)->ob_ival );
   return kTRUE;
}

PYROOT_IMPLEMENT_BASIC_REF_CONVERTER( LongRef )

//____________________________________________________________________________
Bool_t PyROOT::TBoolConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ bool, allow int/long -> bool, set arg for call
   para.fl = PyLong_AsLong( pyobject );
   if ( PyErr_Occurred() )
      return kFALSE;

   if ( ! ( para.fl == 0 || para.fl == 1 ) ) {
      PyErr_SetString( PyExc_TypeError, "boolean value should be bool, or integer 1 or 0" );
      return kFALSE;
   }

   if ( func )
      func->SetArg( para.fl );
   return kTRUE;
}

PYROOT_IMPLEMENT_BASIC_CONVERTER( Bool, Bool_t, Long_t, PyInt_FromLong, PyInt_AsLong )

//____________________________________________________________________________
PYROOT_IMPLEMENT_BASIC_CHAR_CONVERTER( Char,  Char_t,  CHAR_MIN, CHAR_MAX  )
PYROOT_IMPLEMENT_BASIC_CHAR_CONVERTER( UChar, UChar_t,        0, UCHAR_MAX )

//____________________________________________________________________________
PYROOT_IMPLEMENT_BASIC_CONVERTER( Short,  Short_t,  Long_t, PyInt_FromLong,  PyInt_AsLong )
PYROOT_IMPLEMENT_BASIC_CONVERTER( UShort, UShort_t, Long_t, PyInt_FromLong,  PyInt_AsLong )
PYROOT_IMPLEMENT_BASIC_CONVERTER( Int,    Int_t,    Long_t, PyInt_FromLong,  PyInt_AsLong )

//____________________________________________________________________________
namespace {

   ULong_t ConvertULong( PyObject* pyobject )
   {
   // convert <pybject> to C++ unsigned long, with bounds checking, allow int -> ulong
      ULong_t ul = PyLong_AsUnsignedLong( pyobject );
      if ( PyErr_Occurred() && PyInt_Check( pyobject ) ) {
         PyErr_Clear();
         Int_t i = PyInt_AS_LONG( pyobject );
         if ( 0 <= i ) {
            ul = (ULong_t)i;
         } else {
            PyErr_SetString( PyExc_OverflowError,
               "can\'t convert negative value to unsigned long" );
         }
      }

      return ul;
   }

} // unnamed namespace

Bool_t PyROOT::TULongConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ unsigned long, set arg for call
   para.ful = ConvertULong( pyobject );
   if ( PyErr_Occurred() )
      return kFALSE;
   else if ( func )
      func->SetArg( para.ful );
   return kTRUE;
}

PyObject* PyROOT::TULongConverter::FromMemory( void* address )
{
// construct python object from C++ unsigned long read at <address>
   return PyLong_FromUnsignedLong( *((ULong_t*)address) );
}

Bool_t PyROOT::TULongConverter::ToMemory( PyObject* value, void* address )
{
// convert <value> to C++ unsigned long, write it at <address>
   ULong_t u = ConvertULong( value );
   if ( PyErr_Occurred() )
      return kFALSE;
   *((ULong_t*)address) = u;
   return kTRUE;
}

//____________________________________________________________________________
PyObject* PyROOT::TUIntConverter::FromMemory( void* address )
{
// construct python object from C++ unsigned int read at <address>
   return PyLong_FromUnsignedLong( *((UInt_t*)address) );
}

Bool_t PyROOT::TUIntConverter::ToMemory( PyObject* value, void* address )
{
// convert <value> to C++ unsigned int, write it at <address>
   ULong_t u = ConvertULong( value );
   if ( PyErr_Occurred() )
      return kFALSE;

   if ( u > (ULong_t)UINT_MAX ) {
      PyErr_SetString( PyExc_OverflowError, "value to large for unsigned int" );
      return kFALSE;
   }

   *((UInt_t*)address) = (UInt_t)u;
   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::TDoubleConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ double, set arg for call
   para.fd = PyFloat_AsDouble( pyobject );
   if ( PyErr_Occurred() )
      return kFALSE;
   else if ( func )
      func->SetArg( para.fd );
   return kTRUE;
}

PYROOT_IMPLEMENT_BASIC_CONVERTER( Double, Double_t, Double_t, PyFloat_FromDouble, PyFloat_AsDouble )
PYROOT_IMPLEMENT_BASIC_CONVERTER( Float,  Float_t,  Double_t, PyFloat_FromDouble, PyFloat_AsDouble )

//____________________________________________________________________________
Bool_t PyROOT::TDoubleRefConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ double&, set arg for call
   if ( ! PyFloat_CheckExact( pyobject ) )
      return kFALSE;

   para.fl = (Long_t)&((PyFloatObject*)pyobject)->ob_fval;
   if ( func )
      func->SetArgRef( ((PyFloatObject*)pyobject)->ob_fval );
   return kTRUE;
}

PYROOT_IMPLEMENT_BASIC_REF_CONVERTER( DoubleRef )

//____________________________________________________________________________
Bool_t PyROOT::TConstDoubleRefConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ const double&, set arg for call using buffer
   para.fd = fBuffer = PyFloat_AsDouble( pyobject );
   if ( PyErr_Occurred() )
      return kFALSE;
   else if ( func )
      func->SetArgRef( fBuffer );
   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::TVoidConverter::SetArg( PyObject*, TParameter&, G__CallFunc* )
{
// can't happen (unless a type is mapped wrongly), but implemented for completeness
   PyErr_SetString( PyExc_SystemError, "void/unknown arguments can\'t be set" );
   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::TMacroConverter::SetArg( PyObject*, TParameter&, G__CallFunc* )
{
// C++ macro's are not acceptable function args (but their values could be)
   PyErr_SetString( PyExc_SystemError, "macro arguments can\'t be set" );
   return kFALSE;
}

PyObject* PyROOT::TMacroConverter::FromMemory( void* address )
{
// no info available from ROOT/meta; go directly to CINT for the type info
   G__DataMemberInfo dmi;
   while ( dmi.Next() ) {    // using G__ClassInfo().GetDataMember() would cause overwrite

      if ( (Long_t)address == dmi.Offset() ) {
      // for now, only handle int, double, and C-string
         switch ( dmi.Type()->Type() ) {
         case 'p':
            return PyInt_FromLong( (Long_t) *(Int_t*)address );
         case 'P':
            return PyFloat_FromDouble( (double) *(Double_t*)address );
         case 'T':
            return PyString_FromString( *(char**)address );
         default:
         // type unknown/not implemented
            PyErr_SetString( PyExc_NotImplementedError, "macro value could not be converted" );
            return 0;
         }
      }
   }

// type unknown/not implemented
   PyErr_SetString( PyExc_AttributeError, "requested macro not found" );
   return 0;
}

//____________________________________________________________________________
Bool_t PyROOT::TLongLongConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ long long, set arg for call
   para.fll = PyLong_AsLongLong( pyobject );
   if ( PyErr_Occurred() )
      return kFALSE;
   else if ( func )
      func->SetArg( para.fll );
   return kTRUE;
}

PyObject* PyROOT::TLongLongConverter::FromMemory( void* address )
{
// construct python object from C++ long long read at <address>
   return PyLong_FromLongLong( *(Long64_t*)address );
}

Bool_t PyROOT::TLongLongConverter::ToMemory( PyObject* value, void* address )
{
// convert <value> to C++ long long, write it at <address>
   Long64_t ll = PyLong_AsLongLong( value );
   if ( PyErr_Occurred() )
      return kFALSE;
   *((Long64_t*)address) = ll;
   return kTRUE;
}

//____________________________________________________________________________
namespace {

   ULong64_t ConvertULongLong( PyObject* pyobject )
   {
   // convert <pyobject> to C++ unsigned long long, with bounds checking
      ULong64_t ull = PyLong_AsUnsignedLongLong( pyobject );
      if ( PyErr_Occurred() && PyInt_Check( pyobject ) ) {
         PyErr_Clear();
         Int_t i = PyInt_AS_LONG( pyobject );
         if ( 0 <= i ) {
            ull = (ULong64_t)i;
         } else {
            PyErr_SetString( PyExc_OverflowError,
               "can\'t convert negative value to unsigned long long" );
         }
      }

      return ull;
   }

} // unnamed namespace

Bool_t PyROOT::TULongLongConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ unsigned long long, set arg for call
   para.full = ConvertULongLong( pyobject );
   if ( PyErr_Occurred() )
      return kFALSE;
   else if ( func )
      func->SetArg( para.full );
   return kTRUE;
}

PyObject* PyROOT::TULongLongConverter::FromMemory( void* address )
{
// construct python object from C++ unsigned long long read at <address>
   return PyLong_FromUnsignedLongLong( *(ULong64_t*)address );
}

Bool_t PyROOT::TULongLongConverter::ToMemory( PyObject* value, void* address )
{
// convert <value> to C++ unsigned long long, write it at <address>
   Long64_t ull = ConvertULongLong( value );
   if ( PyErr_Occurred() )
      return kFALSE;
   *((ULong64_t*)address) = ull;
   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::TCStringConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// construct a new string and copy it in new memory
   const char* s = PyString_AsString( pyobject );
   if ( PyErr_Occurred() )
      return kFALSE;

   fBuffer = s;
   para.fv = (void*)fBuffer.c_str();

// verify (too long string will cause truncation, no crash)
   if ( fMaxSize < (UInt_t)fBuffer.size() )
      PyErr_Warn( PyExc_RuntimeWarning, "string too long for char array (truncated)" );
   else if ( fMaxSize != UINT_MAX )
      fBuffer.resize( fMaxSize, '\0' );      // padd remainder of buffer as needed

// set the value and declare success
   if ( func )
      func->SetArg( reinterpret_cast< Long_t >( fBuffer.c_str() ) );
   return kTRUE;
}

PyObject* PyROOT::TCStringConverter::FromMemory( void* address )
{
// construct python object from C++ const char* read at <address>
   if ( address && *(char**)address ) {
      if ( fMaxSize != UINT_MAX ) {          // need to prevent reading beyond boundary
         std::string buf( *(char**)address, fMaxSize );
         return PyString_FromString( buf.c_str() );
      }

      return PyString_FromString( *(char**)address );
   }

// empty string in case there's no address
   return PyString_FromString( const_cast< char* >( "" ) );
}

Bool_t PyROOT::TCStringConverter::ToMemory( PyObject* value, void* address )
{
// convert <value> to C++ const char*, write it at <address>
   const char* s = PyString_AsString( value );
   if ( PyErr_Occurred() )
      return kFALSE;

// verify (too long string will cause truncation, no crash)
   if ( fMaxSize < (UInt_t)PyString_GET_SIZE( value ) )
      PyErr_Warn( PyExc_RuntimeWarning, "string too long for char array (truncated)" );

   if ( fMaxSize != UINT_MAX )
      strncpy( *(char**)address, s, fMaxSize );   // padds remainder
   else
      strcpy( *(char**)address, s );

   return kTRUE;
}


//- pointer/array conversions -------------------------------------------------
namespace {

   inline Bool_t CArraySetArg(
      PyObject* pyobject, PyROOT::TParameter& para, G__CallFunc* func, char tc, int size )
   {
      int buflen = PyROOT::Utility::GetBuffer( pyobject, tc, size, para.fv );
      if ( ! para.fv || buflen == 0 )
         return kFALSE;
      else if ( func )
         func->SetArg( para.fl );
      return kTRUE;
   }

} // unnamed namespace

//____________________________________________________________________________
Bool_t PyROOT::TVoidArrayConverter::GetAddressSpecialCase( PyObject* pyobject, void*& address )
{
// (1): "null pointer"
   if ( pyobject == Py_None ) {
      address = (void*)0;
      return kTRUE;
   }

// (2): allow integer zero to act as a null pointer, no deriveds
   if ( PyInt_CheckExact( pyobject ) || PyLong_CheckExact( pyobject ) ) {
      Long_t val = (Long_t)PyLong_AsLong( pyobject );
      if ( val == 0l ) {
         address = (void*)val;
         return kTRUE;
      }

      return kFALSE;
   }

// (3): opaque CObject from somewhere
   if ( PyCObject_Check( pyobject ) ) {
      address = (void*)PyCObject_AsVoidPtr( pyobject );
      return kTRUE;
   }

   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::TVoidArrayConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// just convert pointer if it is a ROOT object
   if ( ObjectProxy_Check( pyobject ) ) {
   // depending on memory policy, some objects are no longer owned when passed to C++
      if ( ! fKeepControl && Utility::gMemoryPolicy != Utility::kStrict )
         ((ObjectProxy*)pyobject)->Release();

   // set pointer (may be null) and declare success
      para.fv = ((ObjectProxy*)pyobject)->GetObject();
      if ( func )
         func->SetArg( para.fl );
      return kTRUE;
   }

// handle special cases
   if ( GetAddressSpecialCase( pyobject, para.fv ) ) {
      if ( func )
         func->SetArg( para.fl );
      return kTRUE;
   }

// final try: attempt to get buffer
   int buflen = Utility::GetBuffer( pyobject, '*', 1, para.fv, kFALSE );

// ok if buffer exists (can't perform any useful size checks)
   if ( para.fv && buflen != 0 ) {
      if ( func )
         func->SetArg( para.fl );
      return kTRUE;
   }

// give up
   return kFALSE;
}

//____________________________________________________________________________
PyObject* PyROOT::TVoidArrayConverter::FromMemory( void* address )
{
// nothing sensible can be done, just return <address> as pylong
   return PyLong_FromLong( (Long_t)address );
}

//____________________________________________________________________________
Bool_t PyROOT::TVoidArrayConverter::ToMemory( PyObject* value, void* address )
{
// just convert pointer if it is a ROOT object
   if ( ObjectProxy_Check( value ) ) {
   // depending on memory policy, some objects are no longer owned when passed to C++
      if ( ! fKeepControl && Utility::gMemoryPolicy != Utility::kStrict )
         ((ObjectProxy*)value)->Release();

   // set pointer (may be null) and declare success
      *(void**)address = ((ObjectProxy*)value)->GetObject();
      return kTRUE;
   }

// handle special cases
   void* ptr = 0;
   if ( GetAddressSpecialCase( value, ptr ) ) {
      *(void**)address = ptr;
      return kTRUE;
   }

// final try: attempt to get buffer
   void* buf = 0;
   int buflen = Utility::GetBuffer( value, '*', 1, buf, kFALSE );
   if ( ! buf || buflen == 0 )
      return kFALSE;

   *(void**)address = buf;
   return kTRUE;
}

//____________________________________________________________________________
#define PYROOT_IMPLEMENT_ARRAY_CONVERTER( name, type, code )                 \
Bool_t PyROOT::T##name##ArrayConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )\
{                                                                            \
   return CArraySetArg( pyobject, para, func, code, sizeof(type) );          \
}                                                                            \
                                                                             \
PyObject* PyROOT::T##name##ArrayConverter::FromMemory( void* address )       \
{                                                                            \
   return BufFac_t::Instance()->PyBuffer_FromMemory( *(type**)address, fSize );\
}                                                                            \
                                                                             \
Bool_t PyROOT::T##name##ArrayConverter::ToMemory( PyObject* value, void* address )\
{                                                                            \
   void* buf = 0;                                                            \
   int buflen = Utility::GetBuffer( value, code, sizeof(type), buf );        \
   if ( ! buf || buflen == 0 )                                               \
      return kFALSE;                                                         \
   if ( 0 <= fSize ) {                                                       \
      if ( fSize < buflen/(int)sizeof(type) ) {                              \
         PyErr_SetString( PyExc_ValueError, "buffer too large for value" );  \
         return kFALSE;                                                      \
      }                                                                      \
      memcpy( *(type**)address, buf, 0 < buflen ? buflen : sizeof(type) );   \
   } else                                                                    \
      *(type**)address = (type*)buf;                                         \
   return kTRUE;                                                             \
}

//____________________________________________________________________________
PYROOT_IMPLEMENT_ARRAY_CONVERTER( Short,  Short_t,  'h' )
PYROOT_IMPLEMENT_ARRAY_CONVERTER( UShort, UShort_t, 'H' )
PYROOT_IMPLEMENT_ARRAY_CONVERTER( Int,    Int_t,    'i' )
PYROOT_IMPLEMENT_ARRAY_CONVERTER( UInt,   UInt_t,   'I' )
PYROOT_IMPLEMENT_ARRAY_CONVERTER( Long,   Long_t,   'l' )
PYROOT_IMPLEMENT_ARRAY_CONVERTER( ULong,  ULong_t,  'L' )
PYROOT_IMPLEMENT_ARRAY_CONVERTER( Float,  Float_t,  'f' )
PYROOT_IMPLEMENT_ARRAY_CONVERTER( Double, Double_t, 'd' )

//____________________________________________________________________________
Bool_t PyROOT::TLongLongArrayConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ long long*, set arg for call
   PyObject* pytc = PyObject_GetAttrString( pyobject, const_cast< char* >( "typecode" ) );
   if ( pytc != 0 ) {              // iow, this array has a known type, but there's no
      Py_DECREF( pytc );           // such thing for long long in module array
      return kFALSE;
   }

   return TVoidArrayConverter::SetArg( pyobject, para, func );
}


//- converters for special cases ----------------------------------------------
#define PYROOT_IMPLEMENT_STRING_AS_PRIMITIVE_CONVERTER( name, strtype, DF1 )  \
PyROOT::T##name##Converter::T##name##Converter() :                            \
      TRootObjectConverter( gROOT->GetClass( #strtype ) ) {}                  \
                                                                              \
Bool_t PyROOT::T##name##Converter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )\
{                                                                             \
   if ( PyString_Check( pyobject ) ) {                                        \
      fBuffer = PyString_AS_STRING( pyobject );                               \
      para.fv = &fBuffer;                                                     \
      if ( func )                                                             \
         func->SetArg( para.fl );                                             \
      return kTRUE;                                                           \
   }                                                                          \
                                                                              \
   return TRootObjectConverter::SetArg( pyobject, para, func );               \
}                                                                             \
                                                                              \
PyObject* PyROOT::T##name##Converter::FromMemory( void* address )             \
{                                                                             \
   if ( address )                                                             \
      return PyString_FromString( ((strtype*)address)->DF1() );               \
   return PyString_FromString( const_cast< char* >( "" ) );                   \
}                                                                             \
                                                                              \
Bool_t PyROOT::T##name##Converter::ToMemory( PyObject* value, void* address ) \
{                                                                             \
   if ( PyString_Check( value ) ) {                                           \
      *((strtype*)address) = PyString_AS_STRING( value );                     \
      return kTRUE;                                                           \
   }                                                                          \
                                                                              \
   return TRootObjectConverter::ToMemory( value, address );                   \
}

PYROOT_IMPLEMENT_STRING_AS_PRIMITIVE_CONVERTER( TString,   TString,     Data )
PYROOT_IMPLEMENT_STRING_AS_PRIMITIVE_CONVERTER( STLString, std::string, c_str )

//____________________________________________________________________________
Bool_t PyROOT::TRootObjectConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ instance*, set arg for call
   if ( ! ObjectProxy_Check( pyobject ) ) {
      if ( GetAddressSpecialCase( pyobject, para.fv ) ) {
         if ( func )
            func->SetArg( para.fl );         // allow special cases such as NULL
         return kTRUE;
      }

   // not a PyROOT object (TODO: handle SWIG etc.)
      return kFALSE;
   }

   ObjectProxy* pyobj = (ObjectProxy*)pyobject;
   if ( pyobj->ObjectIsA() && pyobj->ObjectIsA()->GetBaseClass( fClass.GetClass() ) ) {
   // depending on memory policy, some objects need releasing when passed into functions
      if ( ! KeepControl() && Utility::gMemoryPolicy != Utility::kStrict )
         ((ObjectProxy*)pyobject)->Release();

   // calculate offset between formal and actual arguments
      para.fv = pyobj->GetObject();
      G__ClassInfo* clFormalInfo = fClass->GetClassInfo();
      G__ClassInfo* clActualInfo = pyobj->ObjectIsA()->GetClassInfo();
      Long_t offset = 0;
      if ( clFormalInfo && clActualInfo && clFormalInfo != clActualInfo )
         offset = G__isanybase( clFormalInfo->Tagnum(), clActualInfo->Tagnum(), para.fl );

   // set pointer (may be null) and declare success
      para.fl += offset;
      if ( func )
         func->SetArg( para.fl );
      return kTRUE;

   } else if ( ! fClass.GetClass()->GetClassInfo() ) {
   // assume "user knows best" to allow anonymous pointer passing
      para.fv = pyobj->GetObject();
      if ( func )
         func->SetArg( para.fl );
      return kTRUE;
   }

   return kFALSE;
}

//____________________________________________________________________________
PyObject* PyROOT::TRootObjectConverter::FromMemory( void* address )
{
// construct python object from C++ instance read at <address>
   return BindRootObject( address, fClass, kFALSE );
}

//____________________________________________________________________________
Bool_t PyROOT::TRootObjectConverter::ToMemory( PyObject* value, void* address )
{
// convert <value> to C++ instance, write it at <address>
   if ( ! ObjectProxy_Check( value ) ) {
      void* ptr = 0;
      if ( GetAddressSpecialCase( value, ptr ) ) {
         *(void**)address = ptr;             // allow special cases such as NULL
         return kTRUE;
      }

   // not a PyROOT object (TODO: handle SWIG etc.)
      return kFALSE;
   }

   if ( ((ObjectProxy*)value)->ObjectIsA()->GetBaseClass( fClass.GetClass() ) ) {
   // depending on memory policy, some objects need releasing when passed into functions
      if ( ! KeepControl() && Utility::gMemoryPolicy != Utility::kStrict )
         ((ObjectProxy*)value)->Release();

   // TODO: fix this, as this is sooo wrong ...
      memcpy( (void*)address, ((ObjectProxy*)value)->GetObject(), fClass->Size() );
      return kTRUE;
   }

   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::TRootObjectPtrConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ instance**, set arg for call
   if ( ! ObjectProxy_Check( pyobject ) )
      return kFALSE;              // not a PyROOT object (TODO: handle SWIG etc.)

   if ( ((ObjectProxy*)pyobject)->ObjectIsA()->GetBaseClass( fClass.GetClass() ) ) {
   // depending on memory policy, some objects need releasing when passed into functions
      if ( ! KeepControl() && Utility::gMemoryPolicy != Utility::kStrict )
         ((ObjectProxy*)pyobject)->Release();

   // set pointer (may be null) and declare success
      para.fv = &((ObjectProxy*)pyobject)->fObject;
      if ( func )
         func->SetArg( para.fl );
      return kTRUE;
   }

   return kFALSE;
}

//____________________________________________________________________________
PyObject* PyROOT::TRootObjectPtrConverter::FromMemory( void* address )
{
// construct python object from C++ instance* read at <address>
   return BindRootObject( address, fClass, kTRUE );
}

//____________________________________________________________________________
Bool_t PyROOT::TRootObjectPtrConverter::ToMemory( PyObject* value, void* address )
{
// convert <value> to C++ instance*, write it at <address>
   if ( ! ObjectProxy_Check( value ) )
      return kFALSE;              // not a PyROOT object (TODO: handle SWIG etc.)

   if ( ((ObjectProxy*)value)->ObjectIsA()->GetBaseClass( fClass.GetClass() ) ) {
   // depending on memory policy, some objects need releasing when passed into functions
      if ( ! KeepControl() && Utility::gMemoryPolicy != Utility::kStrict )
         ((ObjectProxy*)value)->Release();

   // set pointer (may be null) and declare success
      *(void**)address = ((ObjectProxy*)value)->GetObject();
      return kTRUE;
   }

   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::TVoidPtrRefConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ void*&, set arg for call
   if ( ObjectProxy_Check( pyobject ) ) {
      para.fv = ((ObjectProxy*)pyobject)->fObject;
      if ( func )
         func->SetArgRef( reinterpret_cast< Long_t& >( para.fl ) );
      return kTRUE;
   }

   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::TVoidPtrPtrConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// convert <pyobject> to C++ void**, set arg for call
   if ( ObjectProxy_Check( pyobject ) ) {
   // this is a ROOT object, take and set its address
      para.fv = &((ObjectProxy*)pyobject)->fObject;
      if ( func )
         func->SetArg( para.fl );
      return kTRUE;
   }

// buffer objects are allowed under "user knows best"
   int buflen = Utility::GetBuffer( pyobject, '*', 1, para.fv, kFALSE );

// ok if buffer exists (can't perform any useful size checks)
   if ( para.fv && buflen != 0 ) {
      if ( func )
         func->SetArg( para.fl );
      return kTRUE;
   }

   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::TPyObjectConverter::SetArg( PyObject* pyobject, TParameter& para, G__CallFunc* func )
{
// by definition: set and declare success
   para.fv = pyobject;
   if ( func )
      func->SetArg( para.fl );
   return kTRUE;
}

PyObject* PyROOT::TPyObjectConverter::FromMemory( void* address )
{
// construct python object from C++ PyObject* read at <address>
   PyObject* pyobject = *((PyObject**)address);

   if ( ! pyobject ) {
      Py_INCREF( Py_None );
      return Py_None;
   }

   Py_INCREF( pyobject );
   return pyobject;
}

Bool_t PyROOT::TPyObjectConverter::ToMemory( PyObject* value, void* address )
{
// no conversion needed, write <value> at <address>
   Py_INCREF( value );
   *((PyObject**)address) = value;
   return kTRUE;
}


//- factories -----------------------------------------------------------------
PyROOT::TConverter* PyROOT::CreateConverter( const std::string& fullType, Long_t user )
{
// The matching of the fulltype to a converter factory goes through up to five levels:
//   1) full, exact match
//   2) match of decorated, unqualified type
//   3) accept const ref as by value
//   4) accept ref as pointer
//   5) generalized cases (covers basically all ROOT classes)
//
// If all fails, void is used, which will generate a run-time warning when used.

// resolve typedefs etc.
   G__TypeInfo ti( fullType.c_str() );
   std::string resolvedType = ti.TrueName();
   if ( ! ti.IsValid() )
      resolvedType = fullType;     // otherwise, resolvedType will be "(unknown)"

// an exactly matching converter is preferred
   ConvFactories_t::iterator h = gConvFactories.find( resolvedType );
   if ( h != gConvFactories.end() )
      return (h->second)( user );

//-- nothing? ok, collect information about the type and possible qualifiers/decorators
   const std::string& cpd = Utility::Compound( resolvedType );
   std::string realType   = TClassEdit::ShortType( resolvedType.c_str(), 1 );

// accept unqualified type (as python does not know about qualifiers)
   h = gConvFactories.find( realType + cpd );
   if ( h != gConvFactories.end() )
      return (h->second)( user );

//-- nothing? collect qualifier information
   Bool_t isConst = resolvedType.find( "const" ) != std::string::npos;

// accept const <type>& as converter by value (as python copies most types)
   if ( isConst && cpd == "&" ) {
      h = gConvFactories.find( realType );
      if ( h != gConvFactories.end() )
         return (h->second)( user );
   }

//-- still nothing? try pointer instead of ref, if ref
   if ( cpd == "&" ) {
      h = gConvFactories.find( realType + "*" );
      if ( h != gConvFactories.end() )
         return (h->second)( user );
   }

//-- still nothing? use a generalized converter
   Bool_t control = kTRUE;
   if ( Utility::gMemoryPolicy == Utility::kHeuristics )
      control = cpd == "&" || isConst;

// converters for known/ROOT classes and default (void*)
   TConverter* result = 0;
   if ( TClass* klass = gROOT->GetClass( realType.c_str() ) ) {
      if ( cpd == "**" || cpd == "*&" || cpd == "&*" )
         result = new TRootObjectPtrConverter( klass, control );
      else if ( cpd == "*" )
         result = new TRootObjectConverter( klass, control );
      else if ( cpd == "&" )
         result = new TStrictRootObjectConverter( klass, control );
      else if ( cpd == "" )               // by value
         result = new TStrictRootObjectConverter( klass, kTRUE );

   } else if ( ti.Property() & G__BIT_ISENUM ) {
   // special case (CINT): represent enums as unsigned integers
      if ( cpd == "&" )
         h = gConvFactories.find( "long&" );
      else
         h = gConvFactories.find( "UInt_t" );
   }

   if ( ! result && h != gConvFactories.end() )
   // converter factory available, use it to create converter
      result = (h->second)( user );
   else if ( ! result ) {
      if ( cpd != "" )
         result = new TVoidArrayConverter();       // "user knows best"
      else
         result = new TVoidConverter();            // fails on use
   }

   return result;
}

//____________________________________________________________________________
#define PYROOT_BASIC_CONVERTER_FACTORY( name )                               \
TConverter* Create##name##Converter( Long_t )                                \
{                                                                            \
   return new T##name##Converter();                                          \
}

#define PYROOT_ARRAY_CONVERTER_FACTORY( name )                               \
TConverter* Create##name##Converter( Long_t user )                           \
{                                                                            \
   return new T##name##Converter( (Int_t)user );                             \
}

//____________________________________________________________________________
namespace {

   using namespace PyROOT;

// use macro rather than template for portability ...
   PYROOT_BASIC_CONVERTER_FACTORY( Bool )
   PYROOT_BASIC_CONVERTER_FACTORY( Char )
   PYROOT_BASIC_CONVERTER_FACTORY( UChar )
   PYROOT_BASIC_CONVERTER_FACTORY( Short )
   PYROOT_BASIC_CONVERTER_FACTORY( UShort )
   PYROOT_BASIC_CONVERTER_FACTORY( Int )
   PYROOT_BASIC_CONVERTER_FACTORY( UInt )
   PYROOT_BASIC_CONVERTER_FACTORY( Long )
   PYROOT_BASIC_CONVERTER_FACTORY( LongRef )
   PYROOT_BASIC_CONVERTER_FACTORY( ULong )
   PYROOT_BASIC_CONVERTER_FACTORY( Float )
   PYROOT_BASIC_CONVERTER_FACTORY( Double )
   PYROOT_BASIC_CONVERTER_FACTORY( DoubleRef )
   PYROOT_BASIC_CONVERTER_FACTORY( ConstDoubleRef )
   PYROOT_BASIC_CONVERTER_FACTORY( Void )
   PYROOT_BASIC_CONVERTER_FACTORY( Macro )
   PYROOT_BASIC_CONVERTER_FACTORY( LongLong )
   PYROOT_BASIC_CONVERTER_FACTORY( ULongLong )
   PYROOT_ARRAY_CONVERTER_FACTORY( CString )
   PYROOT_ARRAY_CONVERTER_FACTORY( ShortArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( UShortArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( IntArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( UIntArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( LongArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( ULongArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( FloatArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( DoubleArray )
   PYROOT_BASIC_CONVERTER_FACTORY( VoidArray )
   PYROOT_BASIC_CONVERTER_FACTORY( LongLongArray )
   PYROOT_BASIC_CONVERTER_FACTORY( TString )
   PYROOT_BASIC_CONVERTER_FACTORY( STLString )
   PYROOT_BASIC_CONVERTER_FACTORY( VoidPtrRef )
   PYROOT_BASIC_CONVERTER_FACTORY( VoidPtrPtr )
   PYROOT_BASIC_CONVERTER_FACTORY( PyObject )

// converter factories for ROOT types
   typedef std::pair< const char*, ConverterFactory_t > NFp_t;

   NFp_t factories_[] = {
   // factories for built-ins
      NFp_t( "bool",               &CreateBoolConverter               ),
      NFp_t( "char",               &CreateCharConverter               ),
      NFp_t( "unsigned char",      &CreateUCharConverter              ),
      NFp_t( "short",              &CreateShortConverter              ),
      NFp_t( "unsigned short",     &CreateUShortConverter             ),
      NFp_t( "int",                &CreateIntConverter                ),
      NFp_t( "int&",               &CreateLongRefConverter            ),
      NFp_t( "unsigned int",       &CreateUIntConverter               ),
      NFp_t( "UInt_t", /* enum */  &CreateUIntConverter               ),
      NFp_t( "long",               &CreateLongConverter               ),
      NFp_t( "long&",              &CreateLongRefConverter            ),
      NFp_t( "unsigned long",      &CreateULongConverter              ),
      NFp_t( "long long",          &CreateLongLongConverter           ),
      NFp_t( "unsigned long long", &CreateULongLongConverter          ),
      NFp_t( "float",              &CreateFloatConverter              ),
      NFp_t( "double",             &CreateDoubleConverter             ),
      NFp_t( "double&",            &CreateDoubleRefConverter          ),
      NFp_t( "const double&",      &CreateConstDoubleRefConverter     ),
      NFp_t( "void",               &CreateVoidConverter               ),
      NFp_t( "#define",            &CreateMacroConverter              ),

   // pointer/array factories
      NFp_t( "short*",             &CreateShortArrayConverter         ),
      NFp_t( "unsigned short*",    &CreateUShortArrayConverter        ),
      NFp_t( "int*",               &CreateIntArrayConverter           ),
      NFp_t( "unsigned int*",      &CreateUIntArrayConverter          ),
      NFp_t( "long*",              &CreateLongArrayConverter          ),
      NFp_t( "unsigned long*",     &CreateULongArrayConverter         ),
      NFp_t( "float*",             &CreateFloatArrayConverter         ),
      NFp_t( "double*",            &CreateDoubleArrayConverter        ),
      NFp_t( "long long*",         &CreateLongLongArrayConverter      ),
      NFp_t( "void*",              &CreateVoidArrayConverter          ),

   // factories for special cases
      NFp_t( "const char*",        &CreateCStringConverter            ),
      NFp_t( "char*",              &CreateCStringConverter            ),
      NFp_t( "TString",            &CreateTStringConverter            ),
      NFp_t( "TString&",           &CreateTStringConverter            ),
      NFp_t( "std::string",        &CreateSTLStringConverter          ),
      NFp_t( "string",             &CreateSTLStringConverter          ),
      NFp_t( "const std::string&", &CreateSTLStringConverter          ),
      NFp_t( "const string&",      &CreateSTLStringConverter          ),
      NFp_t( "void*&",             &CreateVoidPtrRefConverter         ),
      NFp_t( "void**",             &CreateVoidPtrPtrConverter         ),
      NFp_t( "PyObject*",          &CreatePyObjectConverter           ),
      NFp_t( "_object*",           &CreatePyObjectConverter           )
   };

   struct InitConvFactories_t {
   public:
      InitConvFactories_t()
      {
         int nf = sizeof( factories_ ) / sizeof( factories_[ 0 ] );
         for ( int i = 0; i < nf; ++i ) {
            gConvFactories[ factories_[ i ].first ] = factories_[ i ].second;
         }
      }
   } initConvFactories_;

} // unnamed namespace
