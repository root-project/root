// @(#)root/pyroot:$Name:  $:$Id: PyBufferFactory.cxx,v 1.2 2004/05/07 20:47:20 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "PyBufferFactory.h"


//- data ------------------------------------------------------------------------
namespace {

// make copies of buffer types
   PyTypeObject      PyLongBuffer_Type           = PyBuffer_Type;
   PySequenceMethods PyLongBuffer_SeqMethods     = *(PyBuffer_Type.tp_as_sequence);
   PyTypeObject      PyIntBuffer_Type            = PyBuffer_Type;
   PySequenceMethods PyIntBuffer_SeqMethods      = *(PyBuffer_Type.tp_as_sequence);
   PyTypeObject      PyDoubleBuffer_Type         = PyBuffer_Type;
   PySequenceMethods PyDoubleBuffer_SeqMethods   = *(PyBuffer_Type.tp_as_sequence);
   PyTypeObject      PyFloatBuffer_Type          = PyBuffer_Type;
   PySequenceMethods PyFloatBuffer_SeqMethods    = *(PyBuffer_Type.tp_as_sequence);


// implement 'length' and 'get' functions
   template< typename T >
   int buffer_length( PyObject* self ) {
      return (*(PyBuffer_Type.tp_as_sequence->sq_length))(self) / sizeof( T );
   }

   PyObject* long_buffer_item( PyObject* self, int idx ) {
      const char* buf = 0;
      (*(PyBuffer_Type.tp_as_buffer->bf_getcharbuffer))( self, 0, &buf );
      return PyLong_FromLong( *((long*)buf + idx) );
   }

   PyObject* int_buffer_item( PyObject* self, int idx ) {
      const char* buf = 0;
      (*(PyBuffer_Type.tp_as_buffer->bf_getcharbuffer))( self, 0, &buf );
      return PyInt_FromLong( *((int*)buf + idx) );
   }

   PyObject* double_buffer_item( PyObject* self, int idx ) {
      const char* buf = 0;
      (*(PyBuffer_Type.tp_as_buffer->bf_getcharbuffer))( self, 0, &buf );
      return PyFloat_FromDouble( *((double*)buf + idx) );
   }

   PyObject* float_buffer_item( PyObject* self, int idx ) {
      const char* buf = 0;
      (*(PyBuffer_Type.tp_as_buffer->bf_getcharbuffer))( self, 0, &buf );
      return PyFloat_FromDouble( *((float*)buf + idx) );
   }

}


//- instance handler ------------------------------------------------------------
PyROOT::PyBufferFactory* PyROOT::PyBufferFactory::getInstance() {
   static PyBufferFactory* fac = 0;

   if ( ! fac ) {
      fac = new PyBufferFactory();
   }

   return fac;
}


//- constructor/destructor ------------------------------------------------------
PyROOT::PyBufferFactory::PyBufferFactory() {
   PyLongBuffer_SeqMethods.sq_item      = (intargfunc) long_buffer_item;
   PyLongBuffer_SeqMethods.sq_length    = (inquiry) &buffer_length< long >;
   PyLongBuffer_Type.tp_as_sequence     = &PyLongBuffer_SeqMethods;

   PyIntBuffer_SeqMethods.sq_item       = (intargfunc) int_buffer_item;
   PyIntBuffer_SeqMethods.sq_length     = (inquiry) &buffer_length< int >;
   PyIntBuffer_Type.tp_as_sequence      = &PyIntBuffer_SeqMethods;

   PyDoubleBuffer_SeqMethods.sq_item    = (intargfunc) double_buffer_item;
   PyDoubleBuffer_SeqMethods.sq_length  = (inquiry) &buffer_length< double >;
   PyDoubleBuffer_Type.tp_as_sequence   = &PyDoubleBuffer_SeqMethods;

   PyFloatBuffer_SeqMethods.sq_item     = (intargfunc) float_buffer_item;
   PyFloatBuffer_SeqMethods.sq_length   = (inquiry) &buffer_length< float >;
   PyFloatBuffer_Type.tp_as_sequence    = &PyFloatBuffer_SeqMethods;
}


PyROOT::PyBufferFactory::~PyBufferFactory() {

}


//- public members --------------------------------------------------------------
PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( long* address, int size ) {
   PyObject* buf = PyBuffer_FromReadWriteMemory( (void*)address, size*sizeof(long) );
   Py_INCREF( &PyLongBuffer_Type );
   buf->ob_type = &PyLongBuffer_Type;
   return buf;
}

PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( int* address, int size ) {
   PyObject* buf = PyBuffer_FromReadWriteMemory( (void*)address, size*sizeof(int) );
   Py_INCREF( &PyIntBuffer_Type );
   buf->ob_type = &PyIntBuffer_Type;
   return buf;
}

PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( double* address, int size ) {
   PyObject* buf = PyBuffer_FromReadWriteMemory( (void*)address, size*sizeof(double) );
   Py_INCREF( &PyDoubleBuffer_Type );
   buf->ob_type = &PyDoubleBuffer_Type;
   return buf;
}

PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( float* address, int size ) {
   PyObject* buf = PyBuffer_FromReadWriteMemory( (void*)address, size*sizeof(float) );
   Py_INCREF( &PyFloatBuffer_Type );
   buf->ob_type = &PyFloatBuffer_Type;
   return buf;
}
