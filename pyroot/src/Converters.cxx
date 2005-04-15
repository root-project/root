// @(#)root/pyroot:$Name:  $:$Id: Converters.cxx,v 1.4 2005/04/14 21:53:47 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "Converters.h"
#include "ObjectProxy.h"

// CINT
#include "Api.h"

// Standard
#include <Riostream.h>
#include <utility>


//- data ______________________________________________________________________
PyROOT::ConvFactories_t PyROOT::gConvFactories;


//- converters for built-ins --------------------------------------------------
bool PyROOT::LongConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   func->SetArg( PyLong_AsLong( pyobject ) );
   if ( PyErr_Occurred() )
      return false;
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

//____________________________________________________________________________
bool PyROOT::VoidConverter::SetArg( PyObject*, G__CallFunc* func )
{
// TODO: verify to see if this is the proper approach
   func->SetArg( 0l );
   std::cerr << "convert< void > called ... may not be proper\n";
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

   bool CArraySetArg( PyObject* pyobject, G__CallFunc* func, char tc, int size )
   {
   // special case: don't handle strings here (yes, they're buffers, but not quite)
      if ( PyString_Check( pyobject ) )
         return false;

   // attempt to retrieve pointer to buffer interface
      PyBufferProcs* bufprocs = pyobject->ob_type->tp_as_buffer;
      PySequenceMethods* seqmeths = pyobject->ob_type->tp_as_sequence;
      if ( seqmeths != 0 && bufprocs != 0 && bufprocs->bf_getwritebuffer != 0 &&
           (*(bufprocs->bf_getsegcount))( pyobject, 0 ) == 1 ) {

      // get the buffer
         void* buf = 0;
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

         if ( buf != 0 ) {
            func->SetArg( (long) buf );
            return true;
         }
      }

   // give up
      return false;
   }

} // unnamed namespace

//____________________________________________________________________________
bool PyROOT::ShortArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   return CArraySetArg( pyobject, func, 'h', sizeof(int) );
}

//____________________________________________________________________________
bool PyROOT::UShortArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   return CArraySetArg( pyobject, func, 'H', sizeof(int) );
}

//____________________________________________________________________________
bool PyROOT::IntArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   return CArraySetArg( pyobject, func, 'i', sizeof(int) );
}

//____________________________________________________________________________
bool PyROOT::UIntArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   return CArraySetArg( pyobject, func, 'I', sizeof(int) );
}

//____________________________________________________________________________
bool PyROOT::LongArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   return CArraySetArg( pyobject, func, 'l', sizeof(long) );
}

//____________________________________________________________________________
bool PyROOT::ULongArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   return CArraySetArg( pyobject, func, 'L', sizeof(int) );
}

//____________________________________________________________________________
bool PyROOT::FloatArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   return CArraySetArg( pyobject, func, 'f', sizeof(float) );
}

//____________________________________________________________________________
bool PyROOT::DoubleArrayConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
   return CArraySetArg( pyobject, func, 'd', sizeof(double) );
}


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
bool PyROOT::PyObjectConverter::SetArg( PyObject* pyobject, G__CallFunc* func )
{
// by definition: set and declare success
   func->SetArg( reinterpret_cast< long >( pyobject ) );
   return true;
}


//- factories -----------------------------------------------------------------
#define PYROOT_CONVERTER_FACTORY( name )               \
Converter* Create##name()                              \
{                                                      \
   return new name ();                                 \
}

namespace {

   using namespace PyROOT;

// use macro rather than template for portability ...
   PYROOT_CONVERTER_FACTORY( LongConverter )
   PYROOT_CONVERTER_FACTORY( DoubleConverter )
   PYROOT_CONVERTER_FACTORY( VoidConverter )
   PYROOT_CONVERTER_FACTORY( LongLongConverter )
   PYROOT_CONVERTER_FACTORY( CStringConverter )
   PYROOT_CONVERTER_FACTORY( VoidArrayConverter )
   PYROOT_CONVERTER_FACTORY( ShortArrayConverter )
   PYROOT_CONVERTER_FACTORY( UShortArrayConverter )
   PYROOT_CONVERTER_FACTORY( IntArrayConverter )
   PYROOT_CONVERTER_FACTORY( UIntArrayConverter )
   PYROOT_CONVERTER_FACTORY( LongArrayConverter )
   PYROOT_CONVERTER_FACTORY( ULongArrayConverter )
   PYROOT_CONVERTER_FACTORY( FloatArrayConverter )
   PYROOT_CONVERTER_FACTORY( DoubleArrayConverter )
   PYROOT_CONVERTER_FACTORY( TStringConverter )
   PYROOT_CONVERTER_FACTORY( PyObjectConverter )

   Converter* CreateConstVoidArrayConverter()
   {
      return new VoidArrayConverter( true );
   }

// converter factories for ROOT types
   typedef std::pair< const char*, ConverterFactory_t > ncp_t;

   ncp_t factories_[] = {
   // factories for built-ins
      ncp_t( "char",               &CreateLongConverter               ),
      ncp_t( "unsigned char",      &CreateLongConverter               ),
      ncp_t( "short",              &CreateLongConverter               ),
      ncp_t( "unsigned short",     &CreateLongConverter               ),
      ncp_t( "int",                &CreateLongConverter               ),
      ncp_t( "unsigned int",       &CreateLongConverter               ),
      ncp_t( "UInt_t",             &CreateLongConverter               ),
      ncp_t( "long",               &CreateLongConverter               ),
      ncp_t( "unsigned long",      &CreateLongConverter               ),
      ncp_t( "long long",          &CreateLongLongConverter           ),
      ncp_t( "float",              &CreateDoubleConverter             ),
      ncp_t( "double",             &CreateDoubleConverter             ),
      ncp_t( "void",               &CreateVoidConverter               ),
      ncp_t( "bool",               &CreateLongConverter               ),
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
