// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "structmember.h"    // from Python
#if PY_VERSION_HEX >= 0x02050000
#include "code.h"            // from Python
#else
#include "compile.h"         // from Python
#endif
#ifndef CO_NOFREE
// python2.2 does not have CO_NOFREE defined
#define CO_NOFREE       0x0040
#endif
#include "MethodProxy.h"
#include "ObjectProxy.h"
#include "TPyException.h"
#include "Utility.h"
#include "PyStrings.h"

// Standard
#include <algorithm>
#include <functional>
#include <vector>
#include <algorithm>


namespace PyROOT {

namespace {

// helper to test whether a method is used in a pseudo-function modus
   Bool_t inline IsPseudoFunc( MethodProxy* pymeth )
   {
      return (void*)pymeth == (void*)pymeth->fSelf;
   }

// helper for collecting/maintaining exception data in overload dispatch
   struct PyError_t {
      PyError_t() { fType = fValue = fTrace = 0; }

      static void Clear( PyError_t& e )
      {
      // Remove exception information.
         Py_XDECREF( e.fType ); Py_XDECREF( e.fValue ); Py_XDECREF( e.fTrace );
         e.fType = e.fValue = e.fTrace = 0;
      }

      PyObject *fType, *fValue, *fTrace;
   };

// helper to hash tuple (using tuple hash would cause self-tailing loops)
   inline Long_t HashSignature( PyObject* args )
   {
   // Build a hash from the types of the given python function arguments.
      ULong_t hash = 0;

      Int_t nargs = PyTuple_GET_SIZE( args );
      for ( Int_t i = 0; i < nargs; ++i ) {
         hash += (ULong_t) Py_TYPE( PyTuple_GET_ITEM( args, i ) );
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

// helper to factor out return logic of mp_call
   inline PyObject* HandleReturn( MethodProxy* pymeth, PyObject* result ) {

   // special case for python exceptions, propagated through C++ layer
      if ( result == (PyObject*)TPyExceptionMagic )
         return 0;              // exception info was already set

   // if this method creates new objects, always take ownership
      if ( pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kIsCreator ) {

      // either be a constructor with a fresh object proxy self ... 
         if ( pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kIsConstructor ) {
            if ( pymeth->fSelf )
               pymeth->fSelf->HoldOn();
         }

      // ... or be a method with an object proxy return value
         else if ( ObjectProxy_Check( result ) )
            ((ObjectProxy*)result)->HoldOn();
      }

   // if this new object falls inside self, make sure its lifetime is proper
      if ( ObjectProxy_Check( pymeth->fSelf ) && ObjectProxy_Check( result ) ) {
         Long_t ptrdiff = (Long_t)((ObjectProxy*)result)->GetObject() - (Long_t)pymeth->fSelf->GetObject();
         if ( 0 <= ptrdiff && ptrdiff < (Long_t)pymeth->fSelf->ObjectIsA()->Size() ) {
            if ( PyObject_SetAttr( result, PyStrings::gLifeLine, (PyObject*)pymeth->fSelf ) == -1 )
               PyErr_Clear();     // ignored
         }
      }

      return result;
   }


//= PyROOT method proxy object behaviour =====================================
   PyObject* mp_name( MethodProxy* pymeth, void* )
   {
      return PyROOT_PyUnicode_FromString( pymeth->GetName().c_str() );
   }

//____________________________________________________________________________
   PyObject* mp_module( MethodProxy* /* pymeth */, void* )
   {
      Py_INCREF( PyStrings::gROOTns );
      return PyStrings::gROOTns;
   }

//____________________________________________________________________________
   PyObject* mp_doc( MethodProxy* pymeth, void* )
   {
   // Build python document string ('__doc__') from all C++-side overloads.
      MethodProxy::Methods_t& methods = pymeth->fMethodInfo->fMethods;

   // collect doc strings
      Int_t nMethods = methods.size();
      PyObject* doc = methods[0]->GetDocString();

   // simple case
      if ( nMethods == 1 )
         return doc;

   // overloaded method
      PyObject* separator = PyROOT_PyUnicode_FromString( "\n" );
      for ( Int_t i = 1; i < nMethods; ++i ) {
         PyROOT_PyUnicode_Append( &doc, separator );
         PyROOT_PyUnicode_AppendAndDel( &doc, methods[i]->GetDocString() );
      }
      Py_DECREF( separator );

      return doc;
   }

//____________________________________________________________________________
   PyObject* mp_meth_func( MethodProxy* pymeth, void* )
   {
   // Create a new method proxy to be returned.
      MethodProxy* newPyMeth = (MethodProxy*)MethodProxy_Type.tp_alloc( &MethodProxy_Type, 0 );

   // method info is shared, as it contains the collected overload knowledge
      *pymeth->fMethodInfo->fRefCount += 1;
      newPyMeth->fMethodInfo = pymeth->fMethodInfo;

   // new method is unbound, use of 'meth' is for keeping track whether this
   // proxy is used in the capacity of a method or a function
      newPyMeth->fSelf = (ObjectProxy*)newPyMeth;

      return (PyObject*)newPyMeth;
   }

//____________________________________________________________________________
   PyObject* mp_meth_self( MethodProxy* pymeth, void* )
   {
   // Return the bound self, if any; in case of pseudo-function role, pretend
   // that the data member im_self does not exist.
      if ( IsPseudoFunc( pymeth ) ) {
         PyErr_Format( PyExc_AttributeError,
            "function %s has no attribute \'im_self\'", pymeth->fMethodInfo->fName.c_str() );
         return 0;
      } else if ( pymeth->fSelf != 0 ) {
         Py_INCREF( (PyObject*)pymeth->fSelf );
         return (PyObject*)pymeth->fSelf;
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* mp_meth_class( MethodProxy* pymeth, void* )
   {
   // Return scoping class; in case of pseudo-function role, pretend that there
   // is no encompassing class (i.e. global scope).
      if ( ! IsPseudoFunc( pymeth ) ) {
         PyObject* pyclass = pymeth->fMethodInfo->fMethods[0]->GetScope();
         if ( ! pyclass )
            PyErr_Format( PyExc_AttributeError,
               "function %s has no attribute \'im_class\'", pymeth->fMethodInfo->fName.c_str() );
         return pyclass;
      }

      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* mp_func_closure( MethodProxy* /* pymeth */, void* )
   {
   // Stub only, to fill out the python function interface.
      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* mp_func_code( MethodProxy* pymeth, void* )
   {
   // Stub only, to fill out the python function interface.
#if PY_VERSION_HEX < 0x03000000
      MethodProxy::Methods_t& methods = pymeth->fMethodInfo->fMethods;

   // collect maximum number of arguments in set of overloads; this is also used
   // for the number of locals ("stack variables")
      int co_argcount = 0;
      MethodProxy::Methods_t::iterator maxargmeth;
      for ( MethodProxy::Methods_t::iterator imeth = methods.begin();
            imeth != methods.end(); ++imeth ) {
         if ( co_argcount < (*imeth)->GetMaxArgs() ) {
            co_argcount = (*imeth)->GetMaxArgs();
            maxargmeth = imeth;
         }
      }
      co_argcount += 1;       // for 'self'

   // for now, code object representing the statement 'pass'
      PyObject* co_code = PyString_FromStringAndSize( "d\x00\x00S", 4 );

   // tuple with all the const literals used in the function
      PyObject* co_consts = PyTuple_New( 2 );
      Py_INCREF( Py_None );
      PyTuple_SET_ITEM( co_consts, 0, Py_None );
      PyObject* val1 = PyFloat_FromDouble( -1.0 );
      PyTuple_SET_ITEM( co_consts, 1, val1 );

      PyObject* co_names = PyTuple_New( 2 );
      PyTuple_SET_ITEM( co_names, 0, PyString_FromString( "dafunc" ) );
      PyTuple_SET_ITEM( co_names, 1, PyString_FromString( "acos" ) );

   // names, freevars, and cellvars go unused
      PyObject* co_unused = PyTuple_New( 0 );

   // variable names are both the argument and local names
      PyObject* co_varnames = PyTuple_New( co_argcount + 1 );
      PyTuple_SET_ITEM( co_varnames, 0, PyString_FromString( "self" ) );
      for ( int iarg = 1; iarg < co_argcount; ++iarg ) {
         PyTuple_SET_ITEM( co_varnames, iarg, (*maxargmeth)->GetArgSpec( iarg - 1 ) );
      }
      PyTuple_SET_ITEM( co_varnames, co_argcount, PyString_FromString( "d" ) );

   // filename is made-up
      PyObject* co_filename = PyString_FromString( "ROOT.py" );

   // name is the function name, also through __name__ on the function itself
      PyObject* co_name = PyString_FromString( pymeth->GetName().c_str() );

   // firstlineno is the line number of first function code in the containing scope

   // lnotab is a packed table that maps instruction count and line number
      PyObject* co_lnotab = PyString_FromString( "\x00\x01\x0c\x01" );

      PyObject* code = (PyObject*)PyCode_New(
         co_argcount,                             // argcount
         co_argcount + 1,                         // nlocals
         2,                                       // stacksize  
         CO_OPTIMIZED | CO_NEWLOCALS | CO_NOFREE, // flags  
         co_code,                                 // code  
         co_consts,                               // consts  
         co_names,                                // names  
         co_varnames,                             // varnames  
         co_unused,                               // freevars  
         co_unused,                               // cellvars  
         co_filename,                             // filename  
         co_name,                                 // name  
         1,                                       // firstlineno  
         co_lnotab );                             // lnotab  
   
      Py_DECREF( co_lnotab );  
      Py_DECREF( co_name );  
      Py_DECREF( co_unused );  
      Py_DECREF( co_filename );  
      Py_DECREF( co_varnames );  
      Py_DECREF( co_names );  
      Py_DECREF( co_consts );  
      Py_DECREF( co_code );  
   
      return code;  
#else  
// not important for functioning of most code, so not implemented for p3 for now (TODO)  
      pymeth = 0;
      if ( pymeth || !pymeth) Py_INCREF( Py_None );
      return Py_None;
#endif
   }

//____________________________________________________________________________
   PyObject* mp_func_defaults( MethodProxy* pymeth, void* )
   {
   // Create a tuple of default values for the overload with the most arguments.
      MethodProxy::Methods_t& methods = pymeth->fMethodInfo->fMethods;

      int maxarg = 0;
      MethodProxy::Methods_t::iterator maxargmeth;
      for ( MethodProxy::Methods_t::iterator imeth = methods.begin(); imeth != methods.end(); ++imeth ) {
         if ( maxarg < (*imeth)->GetMaxArgs() ) {
            maxarg = (*imeth)->GetMaxArgs();
            maxargmeth = imeth;
         }
      }

      PyObject* defaults = PyTuple_New( maxarg );

      int itup = 0;
      for ( int iarg = 0; iarg < maxarg; ++iarg ) {
         PyObject* defvalue = (*maxargmeth)->GetArgDefault( iarg );
         if ( defvalue )
            PyTuple_SET_ITEM( defaults, itup++, defvalue );
      }
      _PyTuple_Resize( &defaults, itup );

      return defaults;
   }

//____________________________________________________________________________
   PyObject* mp_func_globals( MethodProxy* /* pymeth */, void* )
   {
   // Return this function's global dict (hard-wired to be the ROOT module); used
   // for lookup of names from co_code indexing into co_names.
      PyObject* pyglobal = PyModule_GetDict( PyImport_AddModule( (char*)"ROOT" ) );
      Py_XINCREF( pyglobal );
      return pyglobal;
   }

//____________________________________________________________________________
   PyObject* mp_getcreates( MethodProxy* pymeth, void* )
   {
   // Get '_creates' boolean, which determines ownership of return values.
      return PyInt_FromLong(
         (Bool_t)(pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kIsCreator) );
   }

//____________________________________________________________________________
   int mp_setcreates( MethodProxy* pymeth, PyObject* value, void* )
   {
   // Set '_creates' boolean, which determines ownership of return values.
      if ( ! value ) {        // means that _creates is being deleted
         pymeth->fMethodInfo->fFlags &= ~MethodProxy::MethodInfo_t::kIsCreator;
         return 0;
      }

      Long_t iscreator = PyLong_AsLong( value );
      if ( iscreator == -1 && PyErr_Occurred() ) {
         PyErr_SetString( PyExc_ValueError, "a boolean 1 or 0 is required for _creates" );
         return -1;
      }

      if ( iscreator )
         pymeth->fMethodInfo->fFlags |= MethodProxy::MethodInfo_t::kIsCreator;
      else
         pymeth->fMethodInfo->fFlags &= ~MethodProxy::MethodInfo_t::kIsCreator;

      return 0;
   }

//____________________________________________________________________________
   PyObject* mp_getmempolicy( MethodProxy* pymeth, void* )
   {
   // Get '_mempolicy' enum, which determines ownership of call arguments.
      if ( (Bool_t)(pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kIsHeuristics ) )
         return PyInt_FromLong( Utility::kHeuristics );

      if ( (Bool_t)(pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kIsStrict ) )
         return PyInt_FromLong( Utility::kStrict );

      return PyInt_FromLong( -1 );
   }

//____________________________________________________________________________
   int mp_setmempolicy( MethodProxy* pymeth, PyObject* value, void* )
   {
   // Set '_mempolicy' enum, which determines ownership of call arguments.
      Long_t mempolicy = PyLong_AsLong( value );
      if ( mempolicy == Utility::kHeuristics ) {
         pymeth->fMethodInfo->fFlags |= MethodProxy::MethodInfo_t::kIsHeuristics;
         pymeth->fMethodInfo->fFlags &= ~MethodProxy::MethodInfo_t::kIsStrict;
      } else if ( mempolicy == Utility::kStrict ) {
         pymeth->fMethodInfo->fFlags |= MethodProxy::MethodInfo_t::kIsStrict;
         pymeth->fMethodInfo->fFlags &= ~MethodProxy::MethodInfo_t::kIsHeuristics;
      } else {
         PyErr_SetString( PyExc_ValueError,
            "expected kMemoryStrict or kMemoryHeuristics as value for _mempolicy" );
         return -1;
      }

      return 0;
   }

//____________________________________________________________________________
   PyObject* mp_getthreaded( MethodProxy* pymeth, void* )
   {
   // Get '_threaded' boolean, which determines whether the GIL will be released.
      return PyInt_FromLong(
         (Bool_t)(pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kReleaseGIL) );
   }

//____________________________________________________________________________
   int mp_setthreaded( MethodProxy* pymeth, PyObject* value, void* )
   {
   // Set '_threaded' boolean, which determines whether the GIL will be released.
      Long_t isthreaded = PyLong_AsLong( value );
      if ( isthreaded == -1 && PyErr_Occurred() ) {
         PyErr_SetString( PyExc_ValueError, "a boolean 1 or 0 is required for _creates" );
         return -1;
      }

      if ( isthreaded )
         pymeth->fMethodInfo->fFlags |= MethodProxy::MethodInfo_t::kReleaseGIL;
      else
         pymeth->fMethodInfo->fFlags &= ~MethodProxy::MethodInfo_t::kReleaseGIL;

      return 0;
   }

//____________________________________________________________________________
   PyGetSetDef mp_getset[] = {
      { (char*)"__name__",   (getter)mp_name,   NULL, NULL, NULL },
      { (char*)"__module__", (getter)mp_module, NULL, NULL, NULL },
      { (char*)"__doc__",    (getter)mp_doc,    NULL, NULL, NULL },

   // to be more python-like, where these are duplicated as well; to actually
   // derive from the python method or function type is too memory-expensive,
   // given that most of the members of those types would not be used
      { (char*)"im_func",  (getter)mp_meth_func,  NULL, NULL, NULL },
      { (char*)"im_self",  (getter)mp_meth_self,  NULL, NULL, NULL },
      { (char*)"im_class", (getter)mp_meth_class, NULL, NULL, NULL },

      { (char*)"func_closure",  (getter)mp_func_closure,  NULL, NULL, NULL },
      { (char*)"func_code",     (getter)mp_func_code,     NULL, NULL, NULL },
      { (char*)"func_defaults", (getter)mp_func_defaults, NULL, NULL, NULL },
      { (char*)"func_globals",  (getter)mp_func_globals,  NULL, NULL, NULL },
      { (char*)"func_doc",      (getter)mp_doc,           NULL, NULL, NULL },
      { (char*)"func_name",     (getter)mp_name,          NULL, NULL, NULL },

      { (char*)"_creates", (getter)mp_getcreates, (setter)mp_setcreates,
            (char*)"For ownership rules of result: if true, objects are python-owned", NULL },
      { (char*)"_mempolicy", (getter)mp_getmempolicy, (setter)mp_setmempolicy,
            (char*)"For argument ownership rules: like global, either heuristic or strict", NULL },
      { (char*)"_threaded", (getter)mp_getthreaded, (setter)mp_setthreaded,
            (char*)"If true, releases GIL on call into C++", NULL },
      { (char*)NULL, NULL, NULL, NULL, NULL }
   };

//= PyROOT method proxy function behavior ====================================
   PyObject* mp_call( MethodProxy* pymeth, PyObject* args, PyObject* kwds )
   {
   // Call the appropriate overload of this method.

   // if called through im_func pseudo-representation (this can be gamed if the
   // user really wants to ... )
      if ( IsPseudoFunc( pymeth ) )
         pymeth->fSelf = NULL;

   // get local handles to proxy internals
      MethodProxy::Methods_t&     methods     = pymeth->fMethodInfo->fMethods;
      MethodProxy::DispatchMap_t& dispatchMap = pymeth->fMethodInfo->fDispatchMap;

      Int_t nMethods = methods.size();

      Long_t user = 0;
      if ( (Bool_t)(pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kIsHeuristics ) )
         user = Utility::kHeuristics;
      else if ( (Bool_t)(pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kIsStrict ) )
         user = Utility::kStrict;
      else
         user = Utility::gMemoryPolicy;

   // simple case
      if ( nMethods == 1 ) {
         PyObject* result = (*methods[0])( pymeth->fSelf, args, kwds, user,
            (pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kReleaseGIL) );
         return HandleReturn( pymeth, result );
      }

   // otherwise, handle overloading
      Long_t sighash = HashSignature( args );

   // look for known signatures ...
      MethodProxy::DispatchMap_t::iterator m = dispatchMap.find( sighash );
      if ( m != dispatchMap.end() ) {
         Int_t index = m->second;
         PyObject* result = 0;

         if ( pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kReleaseGIL ){
            Py_BEGIN_ALLOW_THREADS
            result = (*methods[ index ])( pymeth->fSelf, args, kwds, user );
            Py_END_ALLOW_THREADS
         } else
            result = (*methods[ index ])( pymeth->fSelf, args, kwds, user );

         result = HandleReturn( pymeth, result );

         if ( result != 0 )
            return result;

      // fall through: python is dynamic, and so, the hashing isn't infallible
         PyErr_Clear();
      }

   // ... otherwise loop over all methods and find the one that does not fail
      if ( ! ( pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kIsSorted ) ) {
         std::stable_sort( methods.begin(), methods.end(), PriorityCmp );
         pymeth->fMethodInfo->fFlags |= MethodProxy::MethodInfo_t::kIsSorted;
      }

      std::vector< PyError_t > errors;
      for ( Int_t i = 0; i < nMethods; ++i ) {
         PyObject* result = 0;

         if ( pymeth->fMethodInfo->fFlags & MethodProxy::MethodInfo_t::kReleaseGIL ){
            Py_BEGIN_ALLOW_THREADS
            result = (*methods[i])( pymeth->fSelf, args, kwds, user );
            Py_END_ALLOW_THREADS
         } else
            result = (*methods[i])( pymeth->fSelf, args, kwds, user );

         if ( result == (PyObject*)TPyExceptionMagic ) {
            std::for_each( errors.begin(), errors.end(), PyError_t::Clear );
            return 0;              // exception info was already set
         }

         if ( result != 0 ) {
         // success: update the dispatch map for subsequent calls
            dispatchMap[ sighash ] = i;
            std::for_each( errors.begin(), errors.end(), PyError_t::Clear );
            return HandleReturn( pymeth, result );
         }

      // failure: collect error message/trace (automatically clears exception, too)
         if ( ! PyErr_Occurred() ) {
         // this should not happen; set an error to prevent core dump and report
            PyObject* sig = methods[i]->GetPrototype();
            PyErr_Format( PyExc_SystemError, "%s =>\n    %s",
               PyROOT_PyUnicode_AsString( sig ), (char*)"NULL result without error in mp_call" );
            Py_DECREF( sig );
         }
         PyError_t e;
         PyErr_Fetch( &e.fType, &e.fValue, &e.fTrace );
         errors.push_back( e );
      }

   // first summarize, then add details
      PyObject* value = PyROOT_PyUnicode_FromFormat(
         "none of the %d overloaded methods succeeded. Full details:", nMethods );
      PyObject* separator = PyROOT_PyUnicode_FromString( "\n  " );

   // if this point is reached, none of the overloads succeeded: notify user
      for ( std::vector< PyError_t >::iterator e = errors.begin(); e != errors.end(); ++e ) {
         PyROOT_PyUnicode_Append( &value, separator );
         PyROOT_PyUnicode_Append( &value, e->fValue );
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
   // Descriptor; create and return a new bound method proxy (language requirement).
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
   // Create a new method proxy object.
      MethodProxy* pymeth = PyObject_GC_New( MethodProxy, &MethodProxy_Type );
      pymeth->fSelf = NULL;
      pymeth->fMethodInfo = new MethodProxy::MethodInfo_t;

      PyObject_GC_Track( pymeth );
      return pymeth;
   }

//____________________________________________________________________________
   void mp_dealloc( MethodProxy* pymeth )
   {
   // Deallocate memory held by method proxy object.
      PyObject_GC_UnTrack( pymeth );

      if ( ! IsPseudoFunc( pymeth ) ) {
         Py_XDECREF( (PyObject*)pymeth->fSelf );
      }

      pymeth->fSelf = NULL;

      if ( --(*pymeth->fMethodInfo->fRefCount) <= 0 ) {
         delete pymeth->fMethodInfo;
      }
 
      PyObject_GC_Del( pymeth );
   }


//____________________________________________________________________________
   Long_t mp_hash( MethodProxy* pymeth )
   {
   // Hash of method proxy object for insertion into dictionaries; with actual
   // method (fMethodInfo) shared, its address is best suited.
      return _Py_HashPointer( pymeth->fMethodInfo );
   }

//____________________________________________________________________________
   int mp_traverse( MethodProxy* pymeth, visitproc visit, void* args )
   {
   // Garbage collector traverse of held python member objects.
      if ( pymeth->fSelf && ! IsPseudoFunc( pymeth ) )
         return visit( (PyObject*)pymeth->fSelf, args );

      return 0;
   }

//____________________________________________________________________________
   int mp_clear( MethodProxy* pymeth )
   {
   // Garbage collector clear of held python member objects.
      if ( ! IsPseudoFunc( pymeth ) ) {
         Py_XDECREF( (PyObject*)pymeth->fSelf );
      }

      pymeth->fSelf = NULL;

      return 0;
   }

//____________________________________________________________________________
   PyObject* mp_richcompare( MethodProxy* self, MethodProxy* other, int op )
   {
   // Rich set of comparison objects; only equals is defined.
      if ( op != Py_EQ )
         return PyType_Type.tp_richcompare( (PyObject*)self, (PyObject*)other, op );

      // defined by type + (shared) MethodInfo + bound self, with special case for fSelf (i.e. pseudo-function)
      if ( ( Py_TYPE(self) == Py_TYPE(other) && self->fMethodInfo == other->fMethodInfo ) && \
           ( ( IsPseudoFunc( self ) && IsPseudoFunc( other ) ) || self->fSelf == other->fSelf ) ) {
         Py_INCREF( Py_True );
         return Py_True;
      }
      Py_INCREF( Py_False );
      return Py_False;
   }


//= PyROOT method proxy access to internals =================================
   PyObject* mp_disp( MethodProxy* pymeth, PyObject* sigarg )
   {
   // Select and call a specific C++ overload, based on its signature.
      if ( ! PyROOT_PyUnicode_Check( sigarg ) ) {
         PyErr_Format( PyExc_TypeError, "disp() argument 1 must be string, not %.50s",
                       sigarg == Py_None ? "None" : Py_TYPE(sigarg)->tp_name );
         return 0;
      }

      PyObject* sig1 = PyROOT_PyUnicode_FromFormat( "(%s)", PyROOT_PyUnicode_AsString( sigarg ) );

      MethodProxy::Methods_t& methods = pymeth->fMethodInfo->fMethods;
      for ( Int_t i = 0; i < (Int_t)methods.size(); ++i ) {

         PyObject* sig2 = methods[ i ]->GetSignature();
         if ( PyObject_RichCompareBool( sig1, sig2, Py_EQ ) ) {
            Py_DECREF( sig2 );

            MethodProxy* newmeth = mp_new( NULL, NULL, NULL );
            MethodProxy::Methods_t vec; vec.push_back( methods[ i ]->Clone() );
            newmeth->Set( pymeth->fMethodInfo->fName, vec );

            if ( pymeth->fSelf && ! IsPseudoFunc( pymeth ) ) {
               Py_INCREF( pymeth->fSelf );
               newmeth->fSelf = pymeth->fSelf;
            }

            Py_DECREF( sig1 );
            return (PyObject*)newmeth;
         }

         Py_DECREF( sig2 );
      }

      Py_DECREF( sig1 );
      PyErr_Format( PyExc_LookupError, "signature \"%s\" not found", PyROOT_PyUnicode_AsString( sigarg ) );
      return 0;
   }

//____________________________________________________________________________
   PyMethodDef mp_methods[] = {
      { (char*)"disp", (PyCFunction)mp_disp, METH_O, (char*)"select overload for dispatch" },
      { (char*)NULL, NULL, 0, NULL }
   };

} // unnamed namespace


//= PyROOT method proxy type =================================================
PyTypeObject MethodProxy_Type = {
   PyVarObject_HEAD_INIT( &PyType_Type, 0 )
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
   (hashfunc)mp_hash,         // tp_hash
   (ternaryfunc)mp_call,      // tp_call
   0,                         // tp_str
   0,                         // tp_getattro
   0,                         // tp_setattro
   0,                         // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,      // tp_flags
   (char*)"PyROOT method proxy (internal)",      // tp_doc
   (traverseproc)mp_traverse, // tp_traverse
   (inquiry)mp_clear,         // tp_clear
   (richcmpfunc)mp_richcompare,                  // tp_richcompare
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
   0                          // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
   , 0                        // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
   , 0                        // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
   , 0                        // tp_finalize
#endif
};

} // namespace PyROOT


//- public members -----------------------------------------------------------
void PyROOT::MethodProxy::Set( const std::string& name, std::vector< PyCallable* >& methods )
{
// Fill in the data of a freshly created method proxy.
   fMethodInfo->fName = name;
   fMethodInfo->fMethods.swap( methods );
   fMethodInfo->fFlags &= ~MethodInfo_t::kIsSorted;

// special case: all constructors are considered creators by default
   if ( name == "__init__" )
      fMethodInfo->fFlags |= (MethodInfo_t::kIsCreator | MethodInfo_t::kIsConstructor);

// special case, in heuristics mode also tag *Clone* methods as creators
   if ( Utility::gMemoryPolicy == Utility::kHeuristics && name.find( "Clone" ) != std::string::npos )
      fMethodInfo->fFlags |= MethodInfo_t::kIsCreator;
}

//____________________________________________________________________________
void PyROOT::MethodProxy::AddMethod( PyCallable* pc )
{
// Fill in the data of a freshly created method proxy.
   fMethodInfo->fFlags &= ~MethodInfo_t::kIsSorted;
   fMethodInfo->fMethods.push_back( pc );
}
      
//____________________________________________________________________________
void PyROOT::MethodProxy::AddMethod( MethodProxy* meth )
{
   fMethodInfo->fMethods.insert( fMethodInfo->fMethods.end(),
      meth->fMethodInfo->fMethods.begin(), meth->fMethodInfo->fMethods.end() );
}

//____________________________________________________________________________
PyROOT::MethodProxy::MethodInfo_t::MethodInfo_t( const MethodInfo_t& s ) :
   fName( s.fName ), fDispatchMap( s.fDispatchMap ), fMethods( s.fMethods ), fFlags( s.fFlags ) 
{
   *s.fRefCount += 1;
   fRefCount = s.fRefCount;
}

//____________________________________________________________________________
PyROOT::MethodProxy::MethodInfo_t& PyROOT::MethodProxy::MethodInfo_t::operator=( const MethodInfo_t& s )
{
   if ( this != &s ) {
      *s.fRefCount += 1;
      fRefCount = s.fRefCount;

      fName        = s.fName;
      fDispatchMap = s.fDispatchMap;
      fMethods     = s.fMethods;
      fFlags       = s.fFlags;
   }

   return *this;
}

//____________________________________________________________________________
PyROOT::MethodProxy::MethodInfo_t::~MethodInfo_t()
{
// Destructor (this object is reference counted).
   for ( Methods_t::iterator it = fMethods.begin(); it != fMethods.end(); ++it ) {
      delete *it;
   }
   fMethods.clear();
   delete fRefCount;
}
