// @(#)root/pyroot:$Name:  $:$Id: PyBufferFactory.cxx,v 1.5 2004/08/13 06:02:40 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "PyBufferFactory.h"

// Standard
#include <map>


//- data ------------------------------------------------------------------------
namespace {

// size callback label
   char* sizeCallback = const_cast< char* >( "_size" );
   PyObject* sizeCallbackString_ = PyString_FromString( sizeCallback );

// callable cache
   std::map< PyObject*, PyObject* > s_sizeCallbacks;

// make copies of buffer types
   PyTypeObject      PyLongBuffer_Type           = PyBuffer_Type;
   PySequenceMethods PyLongBuffer_SeqMethods     = *(PyBuffer_Type.tp_as_sequence);
   PyTypeObject      PyIntBuffer_Type            = PyBuffer_Type;
   PySequenceMethods PyIntBuffer_SeqMethods      = *(PyBuffer_Type.tp_as_sequence);
   PyTypeObject      PyDoubleBuffer_Type         = PyBuffer_Type;
   PySequenceMethods PyDoubleBuffer_SeqMethods   = *(PyBuffer_Type.tp_as_sequence);
   PyTypeObject      PyFloatBuffer_Type          = PyBuffer_Type;
   PySequenceMethods PyFloatBuffer_SeqMethods    = *(PyBuffer_Type.tp_as_sequence);


// implement 'length' and 'get' functions (use explicit funcs: vc++ can't handle templates)
   int buffer_length( PyObject* self, const int tsize ) {
      std::map< PyObject*, PyObject* >::iterator iscbp = s_sizeCallbacks.find( self );
      if ( iscbp != s_sizeCallbacks.end() ) {
         PyObject* pylen = PyObject_CallObject( iscbp->second, NULL );
         int nlen = PyInt_AsLong( pylen );
         Py_DECREF( pylen );

         return nlen;
      }

      PyErr_Clear();
      return (*(PyBuffer_Type.tp_as_sequence->sq_length))(self) / tsize;
   }

   const char* get_buffer( PyObject* self, int idx, const int tsize ) {
      if ( idx < 0 || idx >= buffer_length( self, tsize ) ) {
         PyErr_SetString( PyExc_IndexError, "buffer index out of range" );
         return 0;
      }

      const char* buf = 0;
      (*(PyBuffer_Type.tp_as_buffer->bf_getcharbuffer))( self, 0, &buf );

      return buf;
   }


   int long_buffer_length( PyObject* self ) {
      return buffer_length( self, sizeof( long ) );
   }

   PyObject* long_buffer_item( PyObject* self, int idx ) {
      const char* buf = get_buffer( self, idx, sizeof( long ) );
      if ( ! buf )
         return 0;

      return PyLong_FromLong( *((long*)buf + idx) );
   }

   int int_buffer_length( PyObject* self ) {
      return buffer_length( self, sizeof( int ) );
   }

   PyObject* int_buffer_item( PyObject* self, int idx ) {
      const char* buf = get_buffer( self, idx, sizeof( int ) );
      if ( ! buf )
         return 0;

      return PyInt_FromLong( *((int*)buf + idx) );
   }

   int double_buffer_length( PyObject* self ) {
      return buffer_length( self, sizeof( double ) );
   }

   PyObject* double_buffer_item( PyObject* self, int idx ) {
      const char* buf = get_buffer( self, idx, sizeof( double ) );
      if ( ! buf )
         return 0;

      return PyFloat_FromDouble( *((double*)buf + idx) );
   }

   int float_buffer_length( PyObject* self ) {
      return buffer_length( self, sizeof( float ) );
   }

   PyObject* float_buffer_item( PyObject* self, int idx ) {
      const char* buf = get_buffer( self, idx, sizeof( float ) );
      if ( ! buf )
         return 0;

      return PyFloat_FromDouble( *((float*)buf + idx) );
   }

}


//- instance handler ------------------------------------------------------------
PyROOT::PyBufferFactory* PyROOT::PyBufferFactory::getInstance() {
   static PyBufferFactory* fac = new PyBufferFactory();
   return fac;
}


//- constructor/destructor ------------------------------------------------------
PyROOT::PyBufferFactory::PyBufferFactory() {
   PyLongBuffer_SeqMethods.sq_item      = (intargfunc) long_buffer_item;
   PyLongBuffer_SeqMethods.sq_length    = (inquiry) &long_buffer_length;
   PyLongBuffer_Type.tp_as_sequence     = &PyLongBuffer_SeqMethods;

   PyIntBuffer_SeqMethods.sq_item       = (intargfunc) int_buffer_item;
   PyIntBuffer_SeqMethods.sq_length     = (inquiry) &int_buffer_length;
   PyIntBuffer_Type.tp_as_sequence      = &PyIntBuffer_SeqMethods;

   PyDoubleBuffer_SeqMethods.sq_item    = (intargfunc) double_buffer_item;
   PyDoubleBuffer_SeqMethods.sq_length  = (inquiry) &double_buffer_length;
   PyDoubleBuffer_Type.tp_as_sequence   = &PyDoubleBuffer_SeqMethods;

   PyFloatBuffer_SeqMethods.sq_item     = (intargfunc) float_buffer_item;
   PyFloatBuffer_SeqMethods.sq_length   = (inquiry) &float_buffer_length;
   PyFloatBuffer_Type.tp_as_sequence    = &PyFloatBuffer_SeqMethods;
}


PyROOT::PyBufferFactory::~PyBufferFactory() {

}


//- public members --------------------------------------------------------------
PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( long* address, int size ) {
   size = size < 0 ? int(INT_MAX/double(sizeof(long)))*sizeof(long) : size*sizeof(long);
   PyObject* buf = PyBuffer_FromReadWriteMemory( (void*)address, size );
   Py_INCREF( &PyLongBuffer_Type );
   buf->ob_type = &PyLongBuffer_Type;
   return buf;
}

PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( long* address, PyObject* scb ) {
   PyObject* buf = PyBuffer_FromMemory( address, 0 );
   if ( buf != 0 && PyCallable_Check( scb ) ) {
      Py_INCREF( scb );
      s_sizeCallbacks[ buf ] = scb;
   }
   return buf;
}

PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( int* address, int size ) {
   size = size < 0 ? int(INT_MAX/double(sizeof(int)))*sizeof(int) : size*sizeof(int);
   PyObject* buf = PyBuffer_FromReadWriteMemory( (void*)address, size );
   Py_INCREF( &PyIntBuffer_Type );
   buf->ob_type = &PyIntBuffer_Type;
   return buf;
}

PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( int* address, PyObject* scb ) {
   PyObject* buf = PyBuffer_FromMemory( address, 0 );
   if ( buf != 0 && PyCallable_Check( scb ) ) {
      Py_INCREF( scb );
      s_sizeCallbacks[ buf ] = scb;
   }
   return buf;
}

PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( double* address, int size ) {
   size = size < 0 ? int(INT_MAX/double(sizeof(double)))*sizeof(double) : size*sizeof(double);
   PyObject* buf = PyBuffer_FromReadWriteMemory( (void*)address, size );
   Py_INCREF( &PyDoubleBuffer_Type );
   buf->ob_type = &PyDoubleBuffer_Type;
   return buf;
}

PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( double* address, PyObject* scb ) {
   PyObject* buf = PyBuffer_FromMemory( address, 0 );
   if ( buf != 0 && PyCallable_Check( scb ) ) {
      Py_INCREF( scb );
      s_sizeCallbacks[ buf ] = scb;
   }  
   return buf;
}

PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( float* address, int size ) {
   size = size < 0 ? int(INT_MAX/double(sizeof(float)))*sizeof(float) : size*sizeof(float);
   PyObject* buf = PyBuffer_FromReadWriteMemory( (void*)address, size );
   Py_INCREF( &PyFloatBuffer_Type );
   buf->ob_type = &PyFloatBuffer_Type;
   return buf;
}

PyObject* PyROOT::PyBufferFactory::PyBuffer_FromMemory( float* address, PyObject* scb ) {
   PyObject* buf = PyBuffer_FromMemory( address, 0 );
   if ( buf != 0 && PyCallable_Check( scb ) ) {
      Py_INCREF( scb );
      s_sizeCallbacks[ buf ] = scb;
   }
   return buf;
}
