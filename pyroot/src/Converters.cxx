// @(#)root/pyroot:$Name:  $:$Id: Converters.cxx,v 1.14 2005/08/10 05:25:41 brun Exp $
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
#include <string.h>
#include <utility>


//- data ______________________________________________________________________
PyROOT::ConvFactories_t PyROOT::gConvFactories;

//- base converter implementation ---------------------------------------------
PyObject* PyROOT::Converter::FromMemory( void* )
{
   PyErr_SetString( PyExc_TypeError, "unknown type can not be converted from memory" );
   return 0;
}

//_____________________________________________________________________________
bool PyROOT::Converter::ToMemory( PyObject*, void* )
{
   PyErr_SetString( PyExc_TypeError, "unknown type can not be converted to memory" );
   return false;
}


//- helper macro's ------------------------------------------------------------
#define PYROOT_IMPLEMENT_BASIC_CONVERTER( name, type, stype, F1, F2 )         \
PyObject* PyROOT::name##Converter::FromMemory( void* address )                \
{                                                                             \
   return F1( (stype)*((type*)address) );                                     \
}                                                                             \
                                                                              \
bool PyROOT::name##Converter::ToMemory( PyObject* value, void* address )      \
{                                                                             \
   type s = (type)F2( value );                                                \
   if ( PyErr_Occurred() )                                                    \
      return false;                                                           \
   *((type*)address) = (type)s;                                               \
   return true;                                                               \
}

#define PYROOT_IMPLEMENT_BASIC_REF_CONVERTER( name )                          \
PyObject* PyROOT::name##Converter::FromMemory( void* )                        \
{                                                                             \
   return 0;                                                                  \
}                                                                             \
                                                                              \
bool PyROOT::name##Converter::ToMemory( PyObject*, void* )                    \
{                                                                             \
   return false;                                                              \
}


//_____________________________________________________________________________
#define PYROOT_IMPLEMENT_BASIC_CHAR_CONVERTER( name, type, low, high )        \
bool PyROOT::name##Converter::SetArg( PyObject* pyobject, G__CallFunc* func ) \
{                                                                             \
   if ( PyString_Check( pyobject ) ) {                                        \
      if ( PyString_GET_SIZE( pyobject ) == 1 )                               \
         func->SetArg( (long)PyString_AS_STRING( pyobject )[0] );             \
      else                                                                    \
         PyErr_Format( PyExc_TypeError,                                       \
            #type" expected, got string of size %d", PyString_GET_SIZE( pyobject ) );\
   } else {                                                                  \
      long l = PyLong_AsLong( pyobject );                                    \
      if ( PyErr_Occurred() )                                                \
         return false;                                                       \
      if ( ! ( low <= l && l <= high ) ) {                                   \
         PyErr_SetString( PyExc_ValueError, "integer to character: value out of range" );\
         return false;                                                       \
      }                                                                      \
      func->SetArg( l );                                                     \
   }                                                                         \
   return true;                                                              \
}                                                                            \
                                                                             \
PyObject* PyROOT::name##Converter::FromMemory( void* address )               \
{                                                                            \
   type buf[2]; buf[1] = (type)'\0';                                         \
   buf[0] = *((type*)address);                                               \
   return PyString_FromString( (char*)buf );                                 \
}                                                                            \
                                                                             \
bool PyROOT::name##Converter::ToMemory( PyObject* value, void* address )     \
{                                                                            \
   const char* buf = PyString_AsString( value );                             \
   if ( PyErr_Occurred() )                                                   \
      return false;                                                          \
                                                                             \
   int len = strlen( buf );                                                  \
   if ( len != 1 ) {                                                         \
      PyErr_Format( PyExc_TypeError, #type" expected, got string of size %d", len );\
      return false;                                                          \
   }                                                                         \
                                                                             \
   *((type*)address) = (type)buf[0];                                         \
   return true;                                                              \
}


//- converters for built-ins --------------------------------------------------
bool PyROOT::LongConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   func->SetArg( PyLong_AsLong( pyobject ) );
   if ( PyErr_Occurred() )
      return false;
   return true;
}

PYROOT_IMPLEMENT_BASIC_CONVERTER( Long, Long_t, Long_t, PyLong_FromLong, PyLong_AsLong )

//____________________________________________________________________________
bool PyROOT::LongRefConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   if ( ! PyInt_CheckExact( pyobject ) )
      return false;

   func->SetArgRef( ((PyIntObject*)pyobject)->ob_ival );
   return true;
}

PYROOT_IMPLEMENT_BASIC_REF_CONVERTER( LongRef )

//____________________________________________________________________________
bool PyROOT::BoolConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   long l = PyLong_AsLong( pyobject );
   if ( PyErr_Occurred() )
      return false;

   if ( ! ( l == 0 || l == 1 ) ) {
      PyErr_SetString( PyExc_TypeError, "boolean value should be bool, or integer 1 or 0" );
      return false;
   }

   func->SetArg( l );
   return true;
}

PYROOT_IMPLEMENT_BASIC_CONVERTER( Bool, Bool_t, Long_t, PyInt_FromLong, PyInt_AsLong )

//____________________________________________________________________________
PYROOT_IMPLEMENT_BASIC_CHAR_CONVERTER( Char,  Char_t, -128, 127 )
PYROOT_IMPLEMENT_BASIC_CHAR_CONVERTER( UChar, UChar_t,   0, 255 )

//____________________________________________________________________________
PYROOT_IMPLEMENT_BASIC_CONVERTER( Short,  Short_t,  Long_t, PyInt_FromLong,  PyInt_AsLong )
PYROOT_IMPLEMENT_BASIC_CONVERTER( UShort, UShort_t, Long_t, PyInt_FromLong,  PyInt_AsLong )
PYROOT_IMPLEMENT_BASIC_CONVERTER( Int,    Int_t,    Long_t, PyInt_FromLong,  PyInt_AsLong )
PYROOT_IMPLEMENT_BASIC_CONVERTER( UInt,   UInt_t,   Long_t, PyInt_FromLong,  PyInt_AsLong )

PyObject* PyROOT::ULongConverter::FromMemory( void* address )
{
   return PyLong_FromUnsignedLong( *((ULong_t*)address) );
}

bool PyROOT::ULongConverter::ToMemory( PyObject* value, void* address )
{
   ULong_t s = (ULong_t)PyLong_AsUnsignedLong( value );
   if ( PyErr_Occurred() ) {
      if ( PyInt_Check( value ) ) {    // shouldn't be ... bug in python?
         PyErr_Clear();
         s = (ULong_t)PyInt_AS_LONG( value ); 
      } else
         return false;
   }

   *((ULong_t*)address) = s;
   return true;
}

//____________________________________________________________________________
bool PyROOT::DoubleConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   func->SetArg( PyFloat_AsDouble( pyobject ) );
   if ( PyErr_Occurred() )
      return false;
   return true;
}

PYROOT_IMPLEMENT_BASIC_CONVERTER( Double, Double_t, Double_t, PyFloat_FromDouble, PyFloat_AsDouble )
PYROOT_IMPLEMENT_BASIC_CONVERTER( Float,  Float_t,  Double_t, PyFloat_FromDouble, PyFloat_AsDouble )

//____________________________________________________________________________
bool PyROOT::DoubleRefConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   if ( ! PyFloat_CheckExact( pyobject ) )
      return false;

   func->SetArgRef( ((PyFloatObject*)pyobject)->ob_fval );
   return true;
}

PYROOT_IMPLEMENT_BASIC_REF_CONVERTER( DoubleRef )

//____________________________________________________________________________
bool PyROOT::VoidConverter::SetArg( PyObject*, G__CallFunc* )
{
   PyErr_SetString( PyExc_SystemError, "void/unknown arguments can\'t be set" );
   return false;
}

//____________________________________________________________________________
bool PyROOT::LongLongConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   func->SetArg( PyLong_AsLongLong( pyobject ) );
   if ( PyErr_Occurred() )
      return false;
   return true;
}

PyObject* PyROOT::LongLongConverter::FromMemory( void* address )
{
   return PyLong_FromLongLong( *(Long64_t*)address );
}

bool PyROOT::LongLongConverter::ToMemory( PyObject* value, void* address )
{
   Long64_t ll = PyLong_AsLongLong( value );
   if ( PyErr_Occurred() )
      return false;
   *((Long64_t*)address) = ll;
   return true;
}

//____________________________________________________________________________
bool PyROOT::CStringConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
// construct a new string and copy it in new memory
   const char* s = PyString_AsString( pyobject );
   if ( PyErr_Occurred() )
      return false;
   fBuffer = s;

// set the value and declare success
   func->SetArg( reinterpret_cast< long >( fBuffer.c_str() ) );
   return true;
}

PyObject* PyROOT::CStringConverter::FromMemory( void* address ) {
   if ( address )
      return PyString_FromString( *(char**)address );
   return PyString_FromString( const_cast< char* >( "" ) );
}

bool PyROOT::CStringConverter::ToMemory( PyObject* value, void* address ) {
   const char* s = PyString_AsString( value );
   if ( PyErr_Occurred() )
      return false;

   strcpy( *(char**)address, s );
   return true;
}


//- pointer/array conversions -------------------------------------------------
namespace {

   inline bool CArraySetArg( PyObject* pyobject, G__CallFunc* func, char tc, int size )
   {
      void* buf = 0;
      int buflen = PyROOT::Utility::GetBuffer( pyobject, tc, size, buf );
      if ( ! buf || buflen == 0 )
         return false;

      func->SetArg( (long) buf );
      return true;
   }

} // unnamed namespace

bool PyROOT::VoidArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
// just convert pointer if it is a ROOT object
   if ( ObjectProxy_Check( pyobject ) ) {
   // depending on memory policy, some objects are no longer owned when passed to C++
      if ( ! fKeepControl )
         ((ObjectProxy*)pyobject)->Release();

   // set pointer (may be null) and declare success
      func->SetArg( reinterpret_cast< long >( ((ObjectProxy*)pyobject)->GetObject() ) );
      return true;
   }

// special case: NULL pointer
   if ( pyobject == gNullObject ) {
      func->SetArg( 0l );
      return true;
   }

// get the buffer
   void* buf = 0;
   int buflen = Utility::GetBuffer( pyobject, '*', 1, buf, false );

// ok if buffer exists (can't perform any useful size checks)
   if ( buf && buflen != 0 ) {
      func->SetArg( (long)buf );
      return true;
   }

// give up
   return false;
}

//____________________________________________________________________________
PyObject* PyROOT::VoidArrayConverter::FromMemory( void* address )
{
   return PyLong_FromLong( (long)address );
}

//____________________________________________________________________________
bool PyROOT::VoidArrayConverter::ToMemory( PyObject* value, void* address )
{
// just convert pointer if it is a ROOT object
   if ( ObjectProxy_Check( value ) ) {
   // depending on memory policy, some objects are no longer owned when passed to C++
      if ( ! fKeepControl )
         ((ObjectProxy*)value)->Release();

   // set pointer (may be null) and declare success
      *(void**)address = ((ObjectProxy*)value)->GetObject();
      return true;
   }

// special case: NULL pointer
   if ( value == gNullObject ) {
      *(void**)address = 0;
      return true;
   }

   void* buf = 0;
   int buflen = Utility::GetBuffer( value, '*', 1, buf, false );
   if ( ! buf || buflen == 0 )
      return false;

   *(void**)address = buf;
   return true;
}

//____________________________________________________________________________
#define PYROOT_IMPLEMENT_ARRAY_CONVERTER( name, type, code )                 \
bool PyROOT::name##ArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )\
{                                                                            \
   return CArraySetArg( pyobject, func, code, sizeof(type) );                \
}                                                                            \
                                                                             \
PyObject* PyROOT::name##ArrayConverter::FromMemory( void* address )          \
{                                                                            \
   return BufFac_t::Instance()->PyBuffer_FromMemory( *(type**)address, fSize );\
}                                                                            \
                                                                             \
bool PyROOT::name##ArrayConverter::ToMemory( PyObject* value, void* address )\
{                                                                            \
   void* buf = 0;                                                            \
   int buflen = Utility::GetBuffer( value, code, sizeof(type), buf );        \
   if ( ! buf || buflen == 0 )                                               \
      return false;                                                          \
   if ( 0 <= fSize ) {                                                       \
      if ( fSize < buflen/(int)sizeof(type) ) {                              \
         PyErr_SetString( PyExc_ValueError, "buffer too large for value" );  \
         return false;                                                       \
      }                                                                      \
      memcpy( *(type**)address, buf, 0 < buflen ? buflen : sizeof(type) );   \
   } else                                                                    \
      *(type**)address = (type*)buf;                                         \
   return true;                                                              \
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
bool PyROOT::LongLongArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   PyObject* pytc = PyObject_GetAttrString( pyobject, const_cast< char* >( "typecode" ) );
   if ( pytc != 0 ) {              // iow, this array has a known type, but there's no
      Py_DECREF( pytc );           // such thing for long long in module array
      return false;
   }
   
   return VoidArrayConverter::SetArg( pyobject, func );
}


//- converters for special cases ----------------------------------------------
#define PYROOT_IMPLEMENT_STRING_AS_PRIMITIVE_CONVERTER( name, strtype, DF1 )  \
bool PyROOT::name##Converter::SetArg( PyObject* pyobject, G__CallFunc* func ) \
{                                                                             \
   const char* s = PyString_AsString( pyobject );                             \
   if ( PyErr_Occurred() )                                                    \
      return false;                                                           \
   fBuffer = s;                                                               \
   func->SetArg( reinterpret_cast< long >( &fBuffer ) );                      \
   return true;                                                               \
}                                                                             \
                                                                              \
PyObject* PyROOT::name##Converter::FromMemory( void* address ) {              \
   if ( address )                                                             \
      return PyString_FromString( ((strtype*)address)->DF1() );               \
   return PyString_FromString( const_cast< char* >( "" ) );                   \
}                                                                             \
                                                                              \
bool PyROOT::name##Converter::ToMemory( PyObject* value, void* address )      \
{                                                                             \
   const char* buf = PyString_AsString( value );                              \
   if ( PyErr_Occurred() )                                                    \
      return false;                                                           \
                                                                              \
   *((strtype*)address) = buf;                                                \
   return true;                                                               \
}


PYROOT_IMPLEMENT_STRING_AS_PRIMITIVE_CONVERTER( TString,   TString,     Data )
PYROOT_IMPLEMENT_STRING_AS_PRIMITIVE_CONVERTER( STLString, std::string, c_str )

//____________________________________________________________________________
bool PyROOT::RootObjectConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   if ( ! ObjectProxy_Check( pyobject ) ) {
      if ( pyobject == gNullObject ) {   // allow NULL pointer as a special case
          func->SetArg( 0l );
          return true;
      }

   // not a PyROOT object (TODO: handle SWIG etc.)
      return false;
   }

   ObjectProxy* pyobj = (ObjectProxy*)pyobject;
   if ( pyobj->ObjectIsA()->GetBaseClass( fClass.GetClass() ) ) {
   // depending on memory policy, some objects need releasing when passed into functions
      if ( ! KeepControl() )
         ((ObjectProxy*)pyobject)->Release();

   // calculate offset between formal and actual arguments
      void* obj = pyobj->GetObject();
      G__ClassInfo* clFormalInfo = fClass->GetClassInfo();
      G__ClassInfo* clActualInfo = pyobj->ObjectIsA()->GetClassInfo();
      long offset = 0;
      if ( clFormalInfo && clActualInfo )
         offset = G__isanybase( clFormalInfo->Tagnum(), clActualInfo->Tagnum(), (long)obj );

   // set pointer (may be null) and declare success
      func->SetArg( reinterpret_cast< long >( obj ) + offset );
      return true;

   } else if ( ! fClass.GetClass()->GetClassInfo() ) {
   // assume "user knows best" to allow anonymous pointer passing
      func->SetArg( reinterpret_cast< long >( pyobj->GetObject() ) );
      return true;
   }

   return false;
}

//____________________________________________________________________________
PyObject* PyROOT::RootObjectConverter::FromMemory( void* address )
{
   return BindRootObject( address, fClass, false );
}

//____________________________________________________________________________
bool PyROOT::RootObjectConverter::ToMemory( PyObject* value, void* address )
{
   if ( ! ObjectProxy_Check( value ) ) {
      if ( value == gNullObject ) {   // allow NULL pointer as a special case
          *(long**)address = 0l;
          return true;
      }

   // not a PyROOT object (TODO: handle SWIG etc.)
      return false;
   }

   if ( ((ObjectProxy*)value)->ObjectIsA()->GetBaseClass( fClass.GetClass() ) ) {
   // depending on memory policy, some objects need releasing when passed into functions
      if ( ! KeepControl() )
         ((ObjectProxy*)value)->Release();

   // TODO: fix this, as this is sooo wrong ...
      memcpy( (void*)address, ((ObjectProxy*)value)->GetObject(), fClass->Size() );
      return true;
   }

   return false;
}

//____________________________________________________________________________
bool PyROOT::RootObjectPtrConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   if ( ! ObjectProxy_Check( pyobject ) )
      return false;               // not a PyROOT object (TODO: handle SWIG etc.)

   if ( ((ObjectProxy*)pyobject)->ObjectIsA()->GetBaseClass( fClass.GetClass() ) ) {
   // depending on memory policy, some objects need releasing when passed into functions
      if ( ! KeepControl() )
         ((ObjectProxy*)pyobject)->Release();

   // set pointer (may be null) and declare success
      func->SetArg( reinterpret_cast< long >( &((ObjectProxy*)pyobject)->fObject ) );
      return true;
   }

   return false;
}

//____________________________________________________________________________
PyObject* PyROOT::RootObjectPtrConverter::FromMemory( void* address )
{
   return BindRootObject( address, fClass, true );
}

//____________________________________________________________________________
bool PyROOT::RootObjectPtrConverter::ToMemory( PyObject* value, void* address )
{
   if ( ! ObjectProxy_Check( value ) )
      return false;               // not a PyROOT object (TODO: handle SWIG etc.)

   if ( ((ObjectProxy*)value)->ObjectIsA()->GetBaseClass( fClass.GetClass() ) ) {
   // depending on memory policy, some objects need releasing when passed into functions
      if ( ! KeepControl() )
         ((ObjectProxy*)value)->Release();

   // set pointer (may be null) and declare success
      *(void**)address = ((ObjectProxy*)value)->GetObject();
      return true;
   }

   return false;
}

//____________________________________________________________________________
bool PyROOT::VoidPtrRefConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   if ( ObjectProxy_Check( pyobject ) ) {
      func->SetArgRef( reinterpret_cast< long& >( ((ObjectProxy*)pyobject)->fObject ) );
      return true;
   }

   return false;
}

//____________________________________________________________________________
bool PyROOT::VoidPtrPtrConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   if ( ObjectProxy_Check( pyobject ) ) {
   // this is a ROOT object, take and set its address
      func->SetArg( reinterpret_cast< long >( &((ObjectProxy*)pyobject)->fObject ) );
      return true;
   }

// buffer objects are allowed under "user knows best"
   void* buf = 0;
   int buflen = Utility::GetBuffer( pyobject, '*', 1, buf, false );

// ok if buffer exists (can't perform any useful size checks)
   if ( buf && buflen != 0 ) {
      func->SetArg( (long)buf );
      return true;
   }

   return false;
}

//____________________________________________________________________________
bool PyROOT::PyObjectConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
// by definition: set and declare success
   func->SetArg( reinterpret_cast< long >( pyobject ) );
   return true;
}

PyObject* PyROOT::PyObjectConverter::FromMemory( void* address )
{
   PyObject* pyobject = *((PyObject**)address);

   if ( ! pyobject ) {
      Py_INCREF( Py_None );
      return Py_None;
   }

   Py_INCREF( pyobject );
   return pyobject;
}

bool PyROOT::PyObjectConverter::ToMemory( PyObject* value, void* address )
{
   Py_INCREF( value );
   *((PyObject**)address) = value;
   return true;
}


//- factories -----------------------------------------------------------------
PyROOT::Converter* PyROOT::CreateConverter( const std::string& fullType, long user )
{
// The matching of the fulltype to a converter factory goes in five steps:
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
   bool isConst = resolvedType.find( "const" ) != std::string::npos;

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
   bool control = true;
   if ( Utility::gMemoryPolicy == Utility::kHeuristics )
      control = cpd == "&" || isConst;

// converters for known/ROOT classes and default (void*)
   Converter* result = 0;
   if ( TClass* klass = gROOT->GetClass( realType.c_str() ) ) {
      if ( cpd == "**" || cpd == "*&" || cpd == "&*" )
         result = new RootObjectPtrConverter( klass, control );
      else if ( cpd == "*" || cpd == "&" )
         result = new RootObjectConverter( klass, control );
      else if ( cpd == "" )               // by value
         result = new RootObjectConverter( klass, true );

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
         result = new VoidArrayConverter();       // "user knows best"
      else
         result = new VoidConverter();            // fails on use
   }

   return result;
}

//____________________________________________________________________________
#define PYROOT_BASIC_CONVERTER_FACTORY( name )                               \
Converter* Create##name##Converter( long )                                   \
{                                                                            \
   return new name##Converter();                                             \
}

#define PYROOT_ARRAY_CONVERTER_FACTORY( name )                               \
Converter* Create##name##Converter( long user )                              \
{                                                                            \
   return new name##Converter( (int)user );                                  \
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
   PYROOT_BASIC_CONVERTER_FACTORY( Void )
   PYROOT_BASIC_CONVERTER_FACTORY( LongLong )
   PYROOT_BASIC_CONVERTER_FACTORY( CString )
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
   typedef std::pair< const char*, ConverterFactory_t > ncp_t;

   ncp_t factories_[] = {
   // factories for built-ins
      ncp_t( "bool",               &CreateBoolConverter               ),
      ncp_t( "char",               &CreateCharConverter               ),
      ncp_t( "unsigned char",      &CreateUCharConverter              ),
      ncp_t( "short",              &CreateShortConverter              ),
      ncp_t( "unsigned short",     &CreateUShortConverter             ),
      ncp_t( "int",                &CreateIntConverter                ),
      ncp_t( "unsigned int",       &CreateUIntConverter               ),
      ncp_t( "UInt_t", /* enum */  &CreateUIntConverter               ),
      ncp_t( "long",               &CreateLongConverter               ),
      ncp_t( "long&",              &CreateLongRefConverter            ),
      ncp_t( "unsigned long",      &CreateULongConverter              ),
      ncp_t( "long long",          &CreateLongLongConverter           ),
      ncp_t( "float",              &CreateFloatConverter              ),
      ncp_t( "double",             &CreateDoubleConverter             ),
      ncp_t( "double&",            &CreateDoubleRefConverter          ),
      ncp_t( "void",               &CreateVoidConverter               ),

   // pointer/array factories
      ncp_t( "short*",             &CreateShortArrayConverter         ),
      ncp_t( "unsigned short*",    &CreateUShortArrayConverter        ),
      ncp_t( "int*",               &CreateIntArrayConverter           ),
      ncp_t( "unsigned int*",      &CreateUIntArrayConverter          ),
      ncp_t( "long*",              &CreateLongArrayConverter          ),
      ncp_t( "unsigned long*",     &CreateULongArrayConverter         ),
      ncp_t( "float*",             &CreateFloatArrayConverter         ),
      ncp_t( "double*",            &CreateDoubleArrayConverter        ),
      ncp_t( "long long*",         &CreateLongLongArrayConverter      ),
      ncp_t( "void*",              &CreateVoidArrayConverter          ),

   // factories for special cases
      ncp_t( "const char*",        &CreateCStringConverter            ),
      ncp_t( "char*",              &CreateCStringConverter            ),
      ncp_t( "TString",            &CreateTStringConverter            ),
      ncp_t( "TString&",           &CreateTStringConverter            ),
      ncp_t( "std::string",        &CreateSTLStringConverter          ),
      ncp_t( "string",             &CreateSTLStringConverter          ),
      ncp_t( "std::string&",       &CreateSTLStringConverter          ),
      ncp_t( "string&",            &CreateSTLStringConverter          ),
      ncp_t( "void*&",             &CreateVoidPtrRefConverter         ),
      ncp_t( "void**",             &CreateVoidPtrPtrConverter         ),
      ncp_t( "PyObject*",          &CreatePyObjectConverter           ),
      ncp_t( "_object*",           &CreatePyObjectConverter           )
   };

   class InitConvFactories_ {
   public:
      InitConvFactories_()
      {
         int nf = sizeof( factories_ ) / sizeof( factories_[ 0 ] );
         for ( int i = 0; i < nf; ++i ) {
            gConvFactories[ factories_[ i ].first ] = factories_[ i ].second;
         }
      }
   } initConvFactories_;

} // unnamed namespace
