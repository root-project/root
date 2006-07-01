// @(#)root/pyroot:$Name:  $:$Id: PyBufferFactory.cxx,v 1.10 2005/09/09 05:19:10 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "PyBufferFactory.h"

// Standard
#include <map>


//- data ---------------------------------------------------------------------
namespace {

// size callback label
   char* sizeCallback = const_cast< char* >( "_size" );
   PyObject* sizeCallbackString_ = PyString_FromString( sizeCallback );

// callable cache
   std::map< PyObject*, PyObject* > gSizeCallbacks;

// make copies of buffer types
#define PYROOT_PREPARE_PYBUFFER_TYPE( name )                                 \
   PyTypeObject      Py##name##Buffer_Type      = PyBuffer_Type;             \
   PySequenceMethods Py##name##Buffer_SeqMethods = *(PyBuffer_Type.tp_as_sequence);

   PYROOT_PREPARE_PYBUFFER_TYPE( Short )
   PYROOT_PREPARE_PYBUFFER_TYPE( UShort )
   PYROOT_PREPARE_PYBUFFER_TYPE( Int )
   PYROOT_PREPARE_PYBUFFER_TYPE( UInt )
   PYROOT_PREPARE_PYBUFFER_TYPE( Long )
   PYROOT_PREPARE_PYBUFFER_TYPE( ULong )
   PYROOT_PREPARE_PYBUFFER_TYPE( Float )
   PYROOT_PREPARE_PYBUFFER_TYPE( Double )

// implement get, str, and length functions (use explicit funcs: vc++ can't handle templates)
   int buffer_length( PyObject* self, const int tsize )
   {
      std::map< PyObject*, PyObject* >::iterator iscbp = gSizeCallbacks.find( self );
      if ( iscbp != gSizeCallbacks.end() ) {
         PyObject* pylen = PyObject_CallObject( iscbp->second, NULL );
         int nlen = PyInt_AsLong( pylen );
         Py_DECREF( pylen );

         return nlen;
      }

      PyErr_Clear();
      return (*(PyBuffer_Type.tp_as_sequence->sq_length))(self) / tsize;
   }

//____________________________________________________________________________
   const char* buffer_get( PyObject* self, int idx, const int tsize )
   {
      if ( idx < 0 || idx >= buffer_length( self, tsize ) ) {
         PyErr_SetString( PyExc_IndexError, "buffer index out of range" );
         return 0;
      }

      const char* buf = 0;
      (*(PyBuffer_Type.tp_as_buffer->bf_getcharbuffer))( self, 0, &buf );

      return buf;
   }

//____________________________________________________________________________
#define PYROOT_IMPLEMENT_PYBUFFER_METHODS( name, type, stype, F1 )           \
   PyObject* name##_buffer_str( PyObject* self )                             \
   {                                                                         \
      int l = buffer_length( self, sizeof( type ) );                         \
      return PyString_FromFormat( "<"#type" buffer of length %d>", l );      \
   }                                                                         \
                                                                             \
   type name##_buffer_length( PyObject* self )                               \
   {                                                                         \
      return buffer_length( self, sizeof( type ) );                          \
   }                                                                         \
                                                                             \
   PyObject* name##_buffer_item( PyObject* self, int idx ) {                 \
      const char* buf = buffer_get( self, idx, sizeof( type ) );             \
      if ( ! buf ) {                                                         \
         PyErr_SetString( PyExc_IndexError, "attempt to index a null-buffer" );\
         return 0;                                                           \
      }                                                                      \
                                                                             \
      return F1( (stype)*((type*)buf + idx) );                               \
   }

   PYROOT_IMPLEMENT_PYBUFFER_METHODS( Short,  Short_t,  Long_t,   PyInt_FromLong )
   PYROOT_IMPLEMENT_PYBUFFER_METHODS( UShort, UShort_t, Long_t,   PyInt_FromLong )
   PYROOT_IMPLEMENT_PYBUFFER_METHODS( Int,    Int_t,    Long_t,   PyInt_FromLong )
   PYROOT_IMPLEMENT_PYBUFFER_METHODS( UInt,   UInt_t,   Long_t,   PyInt_FromLong )
   PYROOT_IMPLEMENT_PYBUFFER_METHODS( Long,   Long_t,   Long_t,   PyLong_FromLong )
   PYROOT_IMPLEMENT_PYBUFFER_METHODS( ULong,  ULong_t,  ULong_t,  PyLong_FromUnsignedLong )
   PYROOT_IMPLEMENT_PYBUFFER_METHODS( Float,  Float_t,  Double_t, PyFloat_FromDouble )
   PYROOT_IMPLEMENT_PYBUFFER_METHODS( Double, Double_t, Double_t, PyFloat_FromDouble )

} // unnamed namespace


//- instance handler ------------------------------------------------------------
PyROOT::TPyBufferFactory* PyROOT::TPyBufferFactory::Instance()
{
// singleton factory
   static TPyBufferFactory* fac = new TPyBufferFactory;
   return fac;
}


//- constructor/destructor ------------------------------------------------------
#define PYROOT_INSTALL_PYBUFFER_METHODS( name, type )                           \
   Py##name##Buffer_SeqMethods.sq_item      = (intargfunc) name##_buffer_item;  \
   Py##name##Buffer_SeqMethods.sq_length    = (inquiry) &name##_buffer_length;  \
   Py##name##Buffer_Type.tp_as_sequence     = &Py##name##Buffer_SeqMethods;     \
   Py##name##Buffer_Type.tp_str             = (reprfunc) name##_buffer_str;

PyROOT::TPyBufferFactory::TPyBufferFactory()
{
// construct python buffer types
   PYROOT_INSTALL_PYBUFFER_METHODS( Short,  Short_t )
   PYROOT_INSTALL_PYBUFFER_METHODS( UShort, UShort_t )
   PYROOT_INSTALL_PYBUFFER_METHODS( Int,    Int_t )
   PYROOT_INSTALL_PYBUFFER_METHODS( UInt,   UInt_t )
   PYROOT_INSTALL_PYBUFFER_METHODS( Long,   Long_t )
   PYROOT_INSTALL_PYBUFFER_METHODS( ULong,  ULong_t )
   PYROOT_INSTALL_PYBUFFER_METHODS( Float,  Float_t )
   PYROOT_INSTALL_PYBUFFER_METHODS( Double, Double_t )
}

//____________________________________________________________________________
PyROOT::TPyBufferFactory::~TPyBufferFactory()
{
}


//- public members --------------------------------------------------------------
#define PYROOT_IMPLEMENT_PYBUFFER_FROM_MEMORY( name, type )                     \
PyObject* PyROOT::TPyBufferFactory::PyBuffer_FromMemory( type* address, int size )\
{                                                                               \
   size = size < 0 ? int(INT_MAX/double(sizeof(type)))*sizeof(type) : size*sizeof(type);\
   PyObject* buf = PyBuffer_FromReadWriteMemory( (void*)address, size );        \
   Py_INCREF( &Py##name##Buffer_Type );                                         \
   buf->ob_type = &Py##name##Buffer_Type;                                       \
   return buf;                                                                  \
}                                                                               \
                                                                                \
PyObject* PyROOT::TPyBufferFactory::PyBuffer_FromMemory( type* address, PyObject* scb )\
{                                                                               \
   PyObject* buf = PyBuffer_FromMemory( address, 0 );                           \
   if ( buf != 0 && PyCallable_Check( scb ) ) {                                 \
      Py_INCREF( scb );                                                         \
      gSizeCallbacks[ buf ] = scb;                                              \
   }                                                                            \
   return buf;                                                                  \
}

PYROOT_IMPLEMENT_PYBUFFER_FROM_MEMORY( Short,  Short_t )
PYROOT_IMPLEMENT_PYBUFFER_FROM_MEMORY( UShort, UShort_t )
PYROOT_IMPLEMENT_PYBUFFER_FROM_MEMORY( Int,    Int_t )
PYROOT_IMPLEMENT_PYBUFFER_FROM_MEMORY( UInt,   UInt_t )
PYROOT_IMPLEMENT_PYBUFFER_FROM_MEMORY( Long,   Long_t )
PYROOT_IMPLEMENT_PYBUFFER_FROM_MEMORY( ULong,  ULong_t )
PYROOT_IMPLEMENT_PYBUFFER_FROM_MEMORY( Float,  Float_t )
PYROOT_IMPLEMENT_PYBUFFER_FROM_MEMORY( Double, Double_t )
