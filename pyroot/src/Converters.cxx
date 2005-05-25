// @(#)root/pyroot:$Name:  $:$Id: Converters.cxx,v 1.8 2005/05/25 06:23:36 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "Converters.h"
#include "ObjectProxy.h"
#include "PyBufferFactory.h"
#include "Utility.h"

// ROOT
#include "TClassEdit.h"

// CINT
#include "Api.h"

// Standard
#include <Riostream.h>
#include <string.h>
#include <utility>


//- temporary hacks -----------------------------------------------------------
PyObject* PyROOT::Converter::FromMemory( void* )
{
   PyErr_SetString( PyExc_NotImplementedError, "converter not implemented" );
   return 0;
}

bool PyROOT::Converter::ToMemory( PyObject*, void* )
{
   PyErr_SetString( PyExc_NotImplementedError, "converter not implemented" );
   return false;
}

//- data ______________________________________________________________________
PyROOT::ConvFactories_t PyROOT::gConvFactories;


//- helper macro's -----------------------------------------------------------
#define PYROOT_IMPLEMENT_BASIC_CONVERTER( name, type, stype, F1, F2 )        \
PyObject* PyROOT::name##Converter::FromMemory( void* address )               \
{                                                                            \
   return F1( (stype)*((type*)address) );                                    \
}                                                                            \
                                                                             \
bool PyROOT::name##Converter::ToMemory( PyObject* value, void* address )     \
{                                                                            \
   type s = (type)F2( value );                                               \
   if ( PyErr_Occurred() )                                                   \
      return false;                                                          \
   *((type*)address) = s;                                                    \
   return true;                                                              \
}

//____________________________________________________________________________
#define PYROOT_IMPLEMENT_BASIC_CHAR_CONVERTER( name, type )                  \
bool PyROOT::name##Converter::SetArg( PyObject* pyobject, G__CallFunc* func )\
{                                                                            \
   if ( PyString_Check( pyobject ) ) {                                       \
      if ( PyString_GET_SIZE( pyobject ) == 1 )                              \
         func->SetArg( (long)PyString_AS_STRING( pyobject )[0] );            \
      else                                                                   \
         PyErr_Format( PyExc_ValueError,                                     \
            #type" expected, got string of size %d", PyString_GET_SIZE( pyobject ) );\
   } else                                                                    \
      func->SetArg( PyLong_AsLong( pyobject ) );                             \
   if ( PyErr_Occurred() )                                                   \
      return false;                                                          \
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
      PyErr_Format( PyExc_ValueError, "char expected, but got string of length %d", len ); \
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

PYROOT_IMPLEMENT_BASIC_CONVERTER( Long,  Long_t, Long_t, PyLong_FromLong, PyLong_AsLong )

//____________________________________________________________________________
PYROOT_IMPLEMENT_BASIC_CHAR_CONVERTER( Char, Char_t )
PYROOT_IMPLEMENT_BASIC_CHAR_CONVERTER( UChar, UChar_t )

//____________________________________________________________________________
PYROOT_IMPLEMENT_BASIC_CONVERTER( Bool,   Bool_t,   Long_t, PyInt_FromLong,  PyInt_AsLong )
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
bool PyROOT::VoidConverter::SetArg( PyObject*, G__CallFunc* func )
{
// TODO: verify to see if this is the proper approach
   std::cerr << "VoidConverter::SetArg called ... may not be proper\n";
   func->SetArg( 0l );
   return true;
}

PyObject* PyROOT::VoidConverter::FromMemory( void* )
{
// TODO: verify to see if this is the proper approach
   Py_INCREF( Py_None );
   return Py_None;
}

bool PyROOT::VoidConverter::ToMemory( PyObject*, void* address )
{
// TODO: verify to see if this is the proper approach
   std::cerr << "VoidConverter::ToMemory called ... may not be proper\n";
   *((int*)address) = 0;
   return true;
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
bool PyROOT::VoidArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
// just convert pointer if it is a ROOT object
   if ( ObjectProxy_Check( pyobject ) ) {
   // if non-const, object can no longer be held, as pointer to it may get copied
      if ( ! fIsConst )
         ((ObjectProxy*)pyobject)->Release();

   // set pointer (may be null) and declare success
      func->SetArg( reinterpret_cast< long >( ((ObjectProxy*)pyobject)->GetObject() ) );
      return true;
   }

// special case: allow integer to act as a null pointer
   long val = PyLong_AsLong( pyobject );
   if ( PyErr_Occurred() )
      PyErr_Clear();
   else if ( val == 0l ) {
      func->SetArg( val );
      return true;
   }

// special case: don't handle strings here (yes, they're buffers, but not quite)
   if ( PyString_Check( pyobject ) )
      return false;
      
// ok, then attempt to retrieve pointer to buffer interface
   PyBufferProcs* bufprocs = pyobject->ob_type->tp_as_buffer;
   PySequenceMethods* seqmeths = pyobject->ob_type->tp_as_sequence;
   if ( seqmeths != 0 && bufprocs != 0 && bufprocs->bf_getwritebuffer != 0 &&
        (*(bufprocs->bf_getsegcount))( pyobject, 0 ) == 1 ) {

   // get the buffer
      void* buf = 0;
      int buflen = (*(bufprocs->bf_getwritebuffer))( pyobject, 0, &buf );

   // ok if buffer exists (can't perform any useful size checks)
      if ( buflen ) {
      // this is a gamble ... may or may not be ok, but that's for the user
         func->SetArg( (long) buf );
         return true;
      }
   }

// give up
   return false;
}


namespace {

   inline int getbuffer( PyObject* pyobject, char tc, int size, void*& buf )
   {
   // special case: don't handle strings here (yes, they're buffers, but not quite)
      if ( PyString_Check( pyobject ) )
         return 0;

   // attempt to retrieve pointer to buffer interface
      PyBufferProcs* bufprocs = pyobject->ob_type->tp_as_buffer;
      PySequenceMethods* seqmeths = pyobject->ob_type->tp_as_sequence;
      if ( seqmeths != 0 && bufprocs != 0 && bufprocs->bf_getwritebuffer != 0 &&
           (*(bufprocs->bf_getsegcount))( pyobject, 0 ) == 1 ) {

      // get the buffer
         int buflen = (*(bufprocs->bf_getwritebuffer))( pyobject, 0, &buf );

      // determine buffer compatibility (use "buf" as a status flag)
         PyObject* pytc = PyObject_GetAttrString( pyobject, const_cast< char* >( "typecode" ) );
         if ( pytc != 0 ) {        // for array objects
            if ( PyString_AS_STRING( pytc )[0] != tc )
               buf = 0;            // no match
            Py_DECREF( pytc );
         } else if ( buflen / (*(seqmeths->sq_length))( pyobject ) == size ) {
         // this is a gamble ... may or may not be ok, but that's for the user
            PyErr_Clear();
         } else
            buf = 0;                         // not compatible

         return buflen;
      }

      return 0;
   }

   inline bool CArraySetArg( PyObject* pyobject, G__CallFunc* func, char tc, int size )
   {
      void* buf = 0;
      int buflen = getbuffer( pyobject, tc, size, buf );
      if ( ! buf || buflen == 0 )
         return false;

      func->SetArg( (long) buf );
      return true;
   }

} // unnamed namespace

//____________________________________________________________________________
#define PYROOT_IMPLEMENT_ARRAY_CONVERTER( name, type, code )                 \
bool PyROOT::name##ArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )\
{                                                                            \
   return CArraySetArg( pyobject, func, code, sizeof(type) );                \
}                                                                            \
                                                                             \
PyObject* PyROOT::name##ArrayConverter::FromMemory( void* address ) {         \
   return BufFac_t::Instance()->PyBuffer_FromMemory( *(type**)address, fSize );\
}                                                                            \
                                                                             \
bool PyROOT::name##ArrayConverter::ToMemory( PyObject* value, void* address ) {\
   void* buf = 0;                                                            \
   int buflen = getbuffer( value, code, sizeof(type), buf );                 \
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

//- converters for special cases ----------------------------------------------
bool PyROOT::TStringConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
// similar to cstring, see above
   const char* s = PyString_AsString( pyobject );
   if ( PyErr_Occurred() )
      return false;
   fBuffer = s;

// set the value and declare success
   func->SetArg( reinterpret_cast< long >( &fBuffer ) );
   return true;
}

//____________________________________________________________________________
bool PyROOT::KnownClassConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   if ( ! ObjectProxy_Check( pyobject ) )
      return false;

   if ( ((ObjectProxy*)pyobject)->ObjectIsA()->GetBaseClass( fClass.GetClass() ) ) {
   // if non-const, object can no longer be held, as pointer to it may get copied
      if ( ! IsConst() )
         ((ObjectProxy*)pyobject)->Release();

   // set pointer (may be null) and declare success
      func->SetArg( reinterpret_cast< long >( ((ObjectProxy*)pyobject)->GetObject() ) );
      return true;
   }

   return false;
}

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
   Converter* result = 0;

   G__TypeInfo ti( fullType.c_str() );
   std::string realType = TClassEdit::ShortType( ti.TrueName(), 1 );
   int isp = Utility::isPointer( fullType );

   if ( 0 < isp ) {
      ConvFactories_t::iterator h = gConvFactories.find( realType + "*" );
      if ( h == gConvFactories.end() ) {
         bool isConst = isp == 2 || fullType.find( "const" ) != std::string::npos;
         if ( TClass* klass = gROOT->GetClass( realType.c_str() ) )
            result = new KnownClassConverter( klass, isConst );
         else {
            h = isConst ? gConvFactories.find( "const void*" ) : gConvFactories.find( "void*" );
            result = (h->second)( user );
         }
      }
      else
         result = (h->second)( user );
   } else if ( ti.Property() & G__BIT_ISENUM ) {
      result = (gConvFactories.find( "UInt_t" )->second)( user );
   } else {
      ConvFactories_t::iterator h = gConvFactories.find( realType );
      if ( h != gConvFactories.end() )
         result = (h->second)( user );
   }

   return result;                  // may be null
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
   PYROOT_BASIC_CONVERTER_FACTORY( ULong )
   PYROOT_BASIC_CONVERTER_FACTORY( Float )
   PYROOT_BASIC_CONVERTER_FACTORY( Double )
   PYROOT_BASIC_CONVERTER_FACTORY( Void )
   PYROOT_BASIC_CONVERTER_FACTORY( LongLong )
   PYROOT_BASIC_CONVERTER_FACTORY( CString )
   PYROOT_BASIC_CONVERTER_FACTORY( VoidArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( ShortArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( UShortArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( IntArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( UIntArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( LongArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( ULongArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( FloatArray )
   PYROOT_ARRAY_CONVERTER_FACTORY( DoubleArray )
   PYROOT_BASIC_CONVERTER_FACTORY( TString )
   PYROOT_BASIC_CONVERTER_FACTORY( LongLongArray )
   PYROOT_BASIC_CONVERTER_FACTORY( PyObject )

   Converter* CreateConstVoidArrayConverter( long )
   {
      return new VoidArrayConverter( true );
   }

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
      ncp_t( "UInt_t", /* enum */  &CreateLongConverter               ),
      ncp_t( "long",               &CreateLongConverter               ),
      ncp_t( "unsigned long",      &CreateULongConverter              ),
      ncp_t( "long long",          &CreateLongLongConverter           ),
      ncp_t( "float",              &CreateFloatConverter              ),
      ncp_t( "double",             &CreateDoubleConverter             ),
      ncp_t( "void",               &CreateVoidConverter               ),
      ncp_t( "const char*",        &CreateCStringConverter            ),
      ncp_t( "char*",              &CreateCStringConverter            ),

   // pointer/array factories
      ncp_t( "void*",              &CreateVoidArrayConverter          ),
      ncp_t( "const void*",        &CreateConstVoidArrayConverter     ),
      ncp_t( "short*",             &CreateShortArrayConverter         ),
      ncp_t( "unsigned short*",    &CreateUShortArrayConverter        ),
      ncp_t( "int*",               &CreateIntArrayConverter           ),
      ncp_t( "unsigned int*",      &CreateUIntArrayConverter          ),
      ncp_t( "long*",              &CreateLongArrayConverter          ),
      ncp_t( "unsigned long*",     &CreateULongArrayConverter         ),
      ncp_t( "float*",             &CreateFloatArrayConverter         ),
      ncp_t( "double*",            &CreateDoubleArrayConverter        ),

   // factories for special cases
      ncp_t( "TString",            &CreateTStringConverter            ),
      ncp_t( "long long*",         &CreateLongLongArrayConverter      ),
      ncp_t( "PyObject*",          &CreatePyObjectConverter           ),
      ncp_t( "_object*",           &CreatePyObjectConverter           )
   };

   const int nFactories_ = sizeof( factories_ ) / sizeof( factories_[ 0 ] );

   class InitConvFactories_ {
   public:
      InitConvFactories_()
      {
         for ( int i = 0; i < nFactories_; ++i ) {
            gConvFactories[ factories_[ i ].first ] = factories_[ i ].second;
         }
      }
   } initConvFactories_;

} // unnamed namespace
