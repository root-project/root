// @(#)root/pyroot:$Name:  $:$Id: MethodProxy.cxx,v 1.68 2005/01/28 05:45:41 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "structmember.h"
#include "MethodProxy.h"
#include "ObjectProxy.h"
#include "TPyException.h"


namespace PyROOT {

namespace {

// helper to hash tuple (using tuple hash would cause self-tailing loops)
   inline long HashSignature( PyObject* args )
   {
      unsigned long hash = 0;

      int nargs = PyTuple_GET_SIZE( args );
      for ( int i = 0; i < nargs; ++i ) {
         hash += (unsigned long) PyTuple_GET_ITEM( args, i )->ob_type;
         hash += (hash << 10); hash ^= (hash >> 6);
      }

      hash += (hash << 3); hash ^= (hash >> 11); hash += (hash << 15);

      return hash;
   }


//= PyROOT method proxy object behaviour =====================================
   PyObject* mp_name( MethodProxy* meth, void* )
   {
      return PyString_FromString( meth->GetName().c_str() );
   }

   PyObject* mp_doc( MethodProxy* meth, void* )
   {
      MethodProxy::Methods_t& methods = meth->fMethodInfo->fMethods;

   // collect doc strings
      int nMethods = methods.size();
      PyObject* doc = methods[0]->GetDocString();

   // simple case
      if ( nMethods == 1 )
         return doc;

   // overloaded method
      for ( int i = 1; i < nMethods; ++i ) {
         PyString_ConcatAndDel( &doc, PyString_FromString( "\n" ) );
         PyString_ConcatAndDel( &doc, methods[i]->GetDocString() );
      }

      return doc;
   }

   PyGetSetDef mp_getset[] = {
      { "__name_", (getter)mp_name, NULL, NULL, NULL },
      { "__doc__", (getter)mp_doc, NULL, NULL, NULL },
      { (char*)NULL, NULL, NULL, NULL, NULL }
   };


//= PyROOT method proxy function behaviour ===================================
   PyObject* mp_call( MethodProxy* meth, PyObject* args, PyObject* kwds )
   {
   // get local handles to proxy internals
      MethodProxy::Methods_t&     methods     = meth->fMethodInfo->fMethods;
      MethodProxy::DispatchMap_t& dispatchMap = meth->fMethodInfo->fDispatchMap;

      int nMethods = methods.size();

   // simple case
      if ( nMethods == 1 )
         return (*methods[0])( meth->fSelf, args, kwds );

   // handle overloading
      long sighash = HashSignature( args );

   // look for known signatures ...
      MethodProxy::DispatchMap_t::iterator m = dispatchMap.find( sighash );
      if ( m != dispatchMap.end() ) {
         int index = m->second;
         PyObject* result = (*methods[ index ])( meth->fSelf, args, kwds );

         if ( result == TPyExceptionMagic )
            return 0;              // exception info was already set

         return result;
      }

   // ... otherwise loop over all methods and find the one that does not fail
      for ( int i = 0; i < nMethods; ++i ) {
         PyObject* result = (*methods[i])( meth->fSelf, args, kwds );

         if ( result == TPyExceptionMagic )
            return 0;              // exception info was already set

         if ( result != 0 ) {
         // success: update the dispatch map
            dispatchMap[ sighash ] = i;
            return result;
         }

      // failure: reset and try again
         PyErr_Clear();
      }

      PyErr_Format( PyExc_TypeError, "none of the %d overloaded methods succeeded", nMethods );
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
      Py_XINCREF( pyobj );
      newPyMeth->fSelf = pyobj;

      return newPyMeth;
   }


//= PyROOT method proxy construction/destruction =================================
   MethodProxy* mp_new( PyTypeObject*, PyObject*, PyObject* )
   {
      MethodProxy* pymeth = PyObject_GC_New( MethodProxy, &MethodProxy_Type );
      pymeth->fSelf = NULL;
      pymeth->fMethodInfo = new MethodProxy::MethodInfo();

      PyObject_GC_Track( pymeth );
      return pymeth;
   }

//____________________________________________________________________________
   void mp_dealloc( MethodProxy* pymeth )
   {
      PyObject_GC_UnTrack( pymeth );
      Py_XDECREF( pymeth->fSelf );

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
      PyObject* pyobject = (PyObject*)pymeth->fSelf;
      pymeth->fSelf = NULL;
      Py_XDECREF( pyobject );

      return 0;
   }

} // unnamed namespace


//= PyROOT method proxy type =================================================
PyTypeObject MethodProxy_Type = {
   PyObject_HEAD_INIT( &PyType_Type )
   0,                         // ob_size
   "ROOT.MethodProxy",        // tp_name
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
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,       // tp_flags
   "PyROOT method proxy (internal)",              // tp_doc
   (traverseproc)mp_traverse, // tp_traverse
   (inquiry)mp_clear,         // tp_clear
   0,                         // tp_richcompare
   0,                         // tp_weaklistoffset
   0,                         // tp_iter
   0,                         // tp_iternext
   0,                         // tp_methods
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
}
