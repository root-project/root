// @(#)root/pyroot:$Name:  $:$Id: MethodProxy.cxx,v 1.11 2006/03/24 06:04:09 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "structmember.h"
#include "MethodProxy.h"
#include "ObjectProxy.h"
#include "TPyException.h"

// Standard
#include <algorithm>
#include <functional>
#include <vector>
#include <algorithm>


namespace PyROOT {

namespace {

// helper for collecting/maintaining exception data in overload dispatch
   struct PyError_t {
      PyError_t() { fType = fValue = fTrace = 0; }

      static void Clear( PyError_t& e )
      {
         Py_XDECREF( e.fType ); Py_XDECREF( e.fValue ); Py_XDECREF( e.fTrace );
         e.fType = e.fValue = e.fTrace = 0;
      }

      PyObject *fType, *fValue, *fTrace;
   };

// helper to hash tuple (using tuple hash would cause self-tailing loops)
   inline Long_t HashSignature( PyObject* args )
   {
      ULong_t hash = 0;

      Int_t nargs = PyTuple_GET_SIZE( args );
      for ( Int_t i = 0; i < nargs; ++i ) {
         hash += (ULong_t) PyTuple_GET_ITEM( args, i )->ob_type;
         hash += (hash << 10); hash ^= (hash >> 6);
      }

      hash += (hash << 3); hash ^= (hash >> 11); hash += (hash << 15);

      return hash;
   }

// helper to sort on method priority
   int PriorityCmp( PyCallable* left, PyCallable* right )
   {
      return left->GetPriority() > right->GetPriority();
   }

//= PyROOT method proxy object behaviour =====================================
   PyObject* mp_name( MethodProxy* meth, void* )
   {
      return PyString_FromString( meth->GetName().c_str() );
   }

//____________________________________________________________________________
   PyObject* mp_doc( MethodProxy* meth, void* )
   {
      MethodProxy::Methods_t& methods = meth->fMethodInfo->fMethods;

   // collect doc strings
      Int_t nMethods = methods.size();
      PyObject* doc = methods[0]->GetDocString();

   // simple case
      if ( nMethods == 1 )
         return doc;

   // overloaded method
      PyObject* separator = PyString_FromString( "\n" );
      for ( Int_t i = 1; i < nMethods; ++i ) {
         PyString_Concat( &doc, separator );
         PyString_ConcatAndDel( &doc, methods[i]->GetDocString() );
      }
      Py_DECREF( separator );

      return doc;
   }

//____________________________________________________________________________
   PyGetSetDef mp_getset[] = {
      { (char*)"__name__", (getter)mp_name, NULL, NULL, NULL },
      { (char*)"__doc__", (getter)mp_doc, NULL, NULL, NULL },
      { (char*)NULL, NULL, NULL, NULL, NULL }
   };

//= PyROOT method proxy function behaviour ===================================
   PyObject* mp_call( MethodProxy* meth, PyObject* args, PyObject* kwds )
   {
   // get local handles to proxy internals
      MethodProxy::Methods_t&     methods     = meth->fMethodInfo->fMethods;
      MethodProxy::DispatchMap_t& dispatchMap = meth->fMethodInfo->fDispatchMap;

      Int_t nMethods = methods.size();

   // simple case
      if ( nMethods == 1 )
         return (*methods[0])( meth->fSelf, args, kwds );

   // handle overloading
      Long_t sighash = HashSignature( args );

   // look for known signatures ...
      MethodProxy::DispatchMap_t::iterator m = dispatchMap.find( sighash );
      if ( m != dispatchMap.end() ) {
         Int_t index = m->second;
         PyObject* result = (*methods[ index ])( meth->fSelf, args, kwds );

         if ( result == (PyObject*)TPyExceptionMagic )
            return 0;              // exception info was already set

         if ( result != 0 )
            return result;

      // fall through: python is dynamic, and so, the hashing isn't infallible
         PyErr_Clear();
      }

   // ... otherwise loop over all methods and find the one that does not fail
      if ( ! meth->fMethodInfo->fIsSorted ) {
         std::stable_sort( methods.begin(), methods.end(), PriorityCmp );
         meth->fMethodInfo->fIsSorted = kTRUE;
      }

      std::vector< PyError_t > errors;
      for ( Int_t i = 0; i < nMethods; ++i ) {
         PyObject* result = (*methods[i])( meth->fSelf, args, kwds );

         if ( result == (PyObject*)TPyExceptionMagic ) {
            std::for_each( errors.begin(), errors.end(), PyError_t::Clear );
            return 0;              // exception info was already set
         }

         if ( result != 0 ) {
         // success: update the dispatch map for subsequent calls
            dispatchMap[ sighash ] = i;
            std::for_each( errors.begin(), errors.end(), PyError_t::Clear );
            return result;
         }

      // failure: collect error message/trace (automatically clears exception, too)
         if ( ! PyErr_Occurred() ) {
         // this should not happen; set an error to prevent core dump and report
            PyObject* sig = methods[i]->GetPrototype();
            PyErr_Format( PyExc_SystemError, "%s =>\n    %s",
               PyString_AS_STRING( sig ), (char*)"NULL result without error in mp_call" );
            Py_DECREF( sig );
         }
         PyError_t e;
         PyErr_Fetch( &e.fType, &e.fValue, &e.fTrace );
         errors.push_back( e );
      }

   // first summarize, then add details
      PyObject* value = PyString_FromFormat(
         "none of the %d overloaded methods succeeded. Full details:", nMethods );
      PyObject* separator = PyString_FromString( "\n  " );

   // if this point is reached, none of the overloads succeeded: notify user
      for ( std::vector< PyError_t >::iterator e = errors.begin(); e != errors.end(); ++e ) {
         PyString_Concat( &value, separator );
         PyString_Concat( &value, e->fValue );
      }

      Py_DECREF( separator );
      std::for_each( errors.begin(), errors.end(), PyError_t::Clear );

   // report failure
      PyErr_SetObject( PyExc_TypeError, value );
      Py_DECREF( value );
      return 0;
   }

//____________________________________________________________________________
   MethodProxy* mp_descrget( MethodProxy* pymeth, ObjectProxy* pyobj, PyObject* )
   {
   // create and use a new one method proxy (language requirement)
      MethodProxy* newPyMeth = (MethodProxy*)MethodProxy_Type.tp_alloc( &MethodProxy_Type, 0 );

   // method info is shared, as it contains the collected overload knowledge
      *pymeth->fMethodInfo->fRefCount += 1;
      newPyMeth->fMethodInfo = pymeth->fMethodInfo;

   // new method is to be bound to current object (may be NULL)
      Py_XINCREF( (PyObject*)pyobj );
      newPyMeth->fSelf = pyobj;

      return newPyMeth;
   }


//= PyROOT method proxy construction/destruction =================================
   MethodProxy* mp_new( PyTypeObject*, PyObject*, PyObject* )
   {
      MethodProxy* pymeth = PyObject_GC_New( MethodProxy, &MethodProxy_Type );
      pymeth->fSelf = NULL;
      pymeth->fMethodInfo = new MethodProxy::MethodInfo_t;

      PyObject_GC_Track( pymeth );
      return pymeth;
   }

//____________________________________________________________________________
   void mp_dealloc( MethodProxy* pymeth )
   {
      PyObject_GC_UnTrack( pymeth );

      Py_XDECREF( (PyObject*)pymeth->fSelf );
      pymeth->fSelf = NULL;

      if ( --(*pymeth->fMethodInfo->fRefCount) <= 0 ) {
         delete pymeth->fMethodInfo;
      }
 
      PyObject_GC_Del( pymeth );
   }

//____________________________________________________________________________
   int mp_traverse( MethodProxy* pymeth, visitproc visit, void* args )
   {
      if ( pymeth->fSelf )
         return visit( (PyObject*)pymeth->fSelf, args );
         
      return 0;
   }

//____________________________________________________________________________
   int mp_clear( MethodProxy* pymeth )
   {
      Py_XDECREF( (PyObject*)pymeth->fSelf );
      pymeth->fSelf = NULL;

      return 0;
   }

//= PyROOT method proxy access to internals =================================
   PyObject* mp_disp( MethodProxy* meth, PyObject* args, PyObject* )
   {
      PyObject* sigarg = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "S:disp" ), &sigarg ) )
         return 0;

      PyObject* sig1 = PyString_FromFormat( "(%s)", PyString_AS_STRING( sigarg ) );

      MethodProxy::Methods_t& methods = meth->fMethodInfo->fMethods;
      for ( Int_t i = 0; i < (Int_t)methods.size(); ++i ) {
         PyObject* sig2 = methods[ i ]->GetSignature();
         if ( PyObject_Compare( sig1, sig2 ) == 0 ) {
            Py_DECREF( sig2 );

            MethodProxy* newmeth = mp_new( NULL, NULL, NULL );
            MethodProxy::Methods_t vec; vec.push_back( methods[ i ] );
            newmeth->Set( meth->fMethodInfo->fName, vec );

            Py_DECREF( sig1 );
            return (PyObject*)newmeth;
         }

         Py_DECREF( sig2 );
      }

      Py_DECREF( sig1 );
      PyErr_Format( PyExc_LookupError, "signature \"%s\" not found", PyString_AS_STRING( sigarg ) );
      return 0;
   }

//____________________________________________________________________________
   PyMethodDef mp_methods[] = {
      { (char*)"disp", (PyCFunction)mp_disp, METH_VARARGS, (char*)"select overload for dispatch" },
      { (char*)NULL, NULL, 0, NULL }
   };

} // unnamed namespace


//= PyROOT method proxy type =================================================
PyTypeObject MethodProxy_Type = {
   PyObject_HEAD_INIT( &PyType_Type )
   0,                         // ob_size
   (char*)"ROOT.MethodProxy", // tp_name
   sizeof(MethodProxy),       // tp_basicsize
   0,                         // tp_itemsize
   (destructor)mp_dealloc,    // tp_dealloc
   0,                         // tp_print
   0,                         // tp_getattr
   0,                         // tp_setattr
   0,                         // tp_compare
   0,                         // tp_repr
   0,                         // tp_as_number
   0,                         // tp_as_sequence
   0,                         // tp_as_mapping
   0,                         // tp_hash
   (ternaryfunc)mp_call,      // tp_call
   0,                         // tp_str
   0,                         // tp_getattro
   0,                         // tp_setattro
   0,                         // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,      // tp_flags
   (char*)"PyROOT method proxy (internal)",      // tp_doc
   (traverseproc)mp_traverse, // tp_traverse
   (inquiry)mp_clear,         // tp_clear
   0,                         // tp_richcompare
   0,                         // tp_weaklistoffset
   0,                         // tp_iter
   0,                         // tp_iternext
   mp_methods,                // tp_methods
   0,                         // tp_members
   mp_getset,                 // tp_getset
   0,                         // tp_base
   0,                         // tp_dict
   (descrgetfunc)mp_descrget, // tp_descr_get
   0,                         // tp_descr_set
   0,                         // tp_dictoffset
   0,                         // tp_init
   0,                         // tp_alloc
   (newfunc)mp_new,           // tp_new
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
void PyROOT::MethodProxy::Set( const std::string& name, std::vector< PyCallable* >& methods )
{
// set method data
   fMethodInfo->fName = name;
   fMethodInfo->fMethods.swap( methods );
   fMethodInfo->fIsSorted = kFALSE;
}
