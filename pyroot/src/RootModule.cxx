// @(#)root/pyroot:$Name:  $:$Id: RootModule.cxx,v 1.4 2004/10/30 06:26:43 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "RootWrapper.h"

// Standard
#include <string>


//- data -----------------------------------------------------------------------
PyObject* g_modroot = 0;


//- private helpers ------------------------------------------------------------
namespace {

   typedef PyDictEntry* (*dictlookup) ( PyDictObject*, PyObject*, long );
   dictlookup dictLookupOrg = 0;

   PyDictEntry* rootLookDictString( PyDictObject* mp, PyObject* key, long hash ) {
   // first search dictionary itself
      PyDictEntry* ep = (*dictLookupOrg)( mp, key, hash );
      if ( ! ep || ep->me_value != 0 )
         return ep;

   // then search builtins
      if ( PyDict_GetItem( PyEval_GetBuiltins(), key ) != 0 ) {
         return ep;
      }

   // filter python private variables (C++ discourages __ as variable start)
      if ( ! PyString_CheckExact( key ) )
         return ep;

      std::string strkey = PyString_AS_STRING( key );
      if ( 2 < strkey.size() && strkey.substr( 0, 2 ) == "__" )
         return ep;

   // all failed, attempt to get ROOT enum/global/class
      PyObject* val = PyObject_GetAttr( g_modroot, key );

      if ( ! val ) {
         PyErr_Clear();
         val = PyROOT::makeRootClassFromString( strkey );
      }

      if ( ! val ) {
         PyErr_Clear();
         val = PyROOT::getRootGlobalFromString( strkey );
      }

      if ( val != Py_None ) {
      // success ... store reference to ROOT entity in the dictionary
         Py_INCREF( key );

         if ( ! ep->me_key )
            mp->ma_fill++;
         else
            Py_DECREF( ep->me_key );
         ep->me_key   = key;
         ep->me_hash  = hash;
         ep->me_value = val;
         mp->ma_used++;
      }
      else
      // failure ...
         Py_DECREF( val );

      return ep;
   }

   PyObject* setRootLazyLookup( PyObject*, PyObject* args ) {
      PyObject* dict = 0;
      if ( ! PyArg_ParseTuple( args, "O!", &PyDict_Type, &dict ) )
         return 0;

      ((PyDictObject*)dict)->ma_lookup = rootLookDictString;

      Py_INCREF( Py_None );
      return Py_None;
   }

} // unnamed namespace


//- data -----------------------------------------------------------------------
static PyMethodDef PyROOTMethods[] = {
   { (char*) "makeRootClass", (PyCFunction) PyROOT::makeRootClass,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "getRootGlobal", (PyCFunction) PyROOT::getRootGlobal,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "setRootLazyLookup", (PyCFunction) setRootLazyLookup,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { NULL, NULL, 0, NULL }
};


extern "C" void initlibPyROOT() {
// prepare for lazyness
   dictLookupOrg = ((PyDictObject*)PyDict_New())->ma_lookup;

// setup PyROOT
   g_modroot = Py_InitModule( const_cast< char* >( "libPyROOT" ), PyROOTMethods );

// setup ROOT
   PyROOT::initRoot();
}
