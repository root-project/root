// @(#)root/pyroot:$Name:  $:$Id: MethodDispatcher.cxx,v 1.6 2004/08/04 20:46:10 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "MethodDispatcher.h"

// Standard
#include <stdio.h>


//- protected class members -----------------------------------------------------
void PyROOT::MethodDispatcher::destroy( void* pmd ) {
   delete reinterpret_cast< PyROOT::MethodDispatcher* >( pmd );
}


PyObject* PyROOT::MethodDispatcher::invoke( PyObject* self, PyObject* args, PyObject* kws ) {
   return (*(reinterpret_cast< PyROOT::MethodDispatcher* >( PyCObject_AsVoidPtr( self ) )) )( args, kws );
}


//- protected members -----------------------------------------------------------
unsigned long PyROOT::MethodDispatcher::hashSignature( PyObject* aTuple ){
   unsigned long hash = 0;

   int nargs = PyTuple_GET_SIZE( aTuple );
   for ( int i = 0; i < nargs; ++i ) {
      hash += (unsigned long) PyTuple_GET_ITEM( aTuple, i )->ob_type;
      hash += (hash << 10); hash ^= (hash >> 6); 
   }

   hash += (hash << 3); hash ^= (hash >> 11); hash += (hash << 15);

   return hash;
}


//- public class members --------------------------------------------------------
bool PyROOT::MethodDispatcher::addToClass( MethodDispatcher* pmd, PyObject* cls ) {
   PyMethodDef* pdef = new PyMethodDef;
   pdef->ml_name  = const_cast< char* >( pmd->getName().c_str() );
   pdef->ml_meth  = (PyCFunction) PyROOT::MethodDispatcher::invoke;
   pdef->ml_flags = METH_VARARGS | METH_KEYWORDS;
   pdef->ml_doc   = NULL;

   PyObject* func = PyCFunction_New( pdef, PyCObject_FromVoidPtr( pmd, MethodDispatcher::destroy ) );
   PyObject* method = 0;
   if ( pmd->m_isStatic == true )
      method = PyStaticMethod_New( func );
   else
      method = PyMethod_New( func, NULL, cls );
   PyObject_SetAttrString( cls, pdef->ml_name, method );
   Py_DECREF( func );
   Py_DECREF( method );

   return true;
}


//- public members --------------------------------------------------------------
PyObject* PyROOT::MethodDispatcher::operator()( PyObject* aTuple, PyObject* aDict ) {
   int nMethods = m_methods.size();

// simple case
   if ( nMethods == 1 )
      return (*m_methods[0])( aTuple, aDict );

// handle overloading
   unsigned long sighash = hashSignature( aTuple );

// look for known signatures ...
   DispatchMap_t::iterator m = m_dispatchMap.find( sighash );
   if ( m != m_dispatchMap.end() ) {
      int index = m->second;
      return (*m_methods[ index ])( aTuple, aDict );
   }

// ... otherwise loop over all methods and find the one that does not fail
   for ( int i = 0; i < nMethods; ++i ) {
      PyObject* result = (*m_methods[i])( aTuple, aDict );

      if ( result != 0 ) {
      // success: update the dispatch map
         m_dispatchMap[ sighash ] = i;
         return result;
      }

   // failure: reset and try again
      PyErr_Clear();
   }

   char txt[ 256 ];
   sprintf( txt, "none of the %d overloaded methods succeeded", m_methods.size() );
   PyErr_SetString( PyExc_TypeError, txt );
   return 0;
}


void PyROOT::MethodDispatcher::addMethod( PyCallable* pmh ) {
   m_methods.push_back( pmh );
}
