// @(#)root/pyroot:$Name:  $:$Id: RootModule.cxx,v 1.5 2004/11/23 21:45:06 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "PropertyProxy.h"
#include "RootWrapper.h"
#include "Utility.h"

// Standard
#include <string>


//- data -----------------------------------------------------------------------
PyObject* gRootModule = 0;


//- private helpers ------------------------------------------------------------
namespace {

   typedef PyDictEntry* (*dictlookup) ( PyDictObject*, PyObject*, long );
   dictlookup dictLookupOrg = 0;

   PyDictEntry* RootLookDictString( PyDictObject* mp, PyObject* key, long hash )
   {
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
      PyObject* val = PyObject_GetAttr( gRootModule, key );

      if ( ! val ) {
         PyErr_Clear();
         val = PyROOT::MakeRootClassFromString( strkey );
      }

      if ( ! val ) {
         PyErr_Clear();
         val = PyROOT::GetRootGlobalFromString( strkey );
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
      } else
      // failure ...
         Py_DECREF( val );

      return ep;
   }

//____________________________________________________________________________
   PyObject* SetRootLazyLookup( PyObject*, PyObject* args )
   {
      PyObject* dict = 0;
      if ( ! PyArg_ParseTuple( args, "O!", &PyDict_Type, &dict ) )
         return 0;

      ((PyDictObject*)dict)->ma_lookup = RootLookDictString;

      Py_INCREF( Py_None );
      return Py_None;
   }

} // unnamed namespace


//- data -----------------------------------------------------------------------
static PyMethodDef PyROOTMethods[] = {
   { (char*) "makeRootClass", (PyCFunction) PyROOT::MakeRootClass,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "getRootGlobal", (PyCFunction) PyROOT::GetRootGlobal,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "setRootLazyLookup", (PyCFunction) SetRootLazyLookup,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { NULL, NULL, 0, NULL }
};


//____________________________________________________________________________
extern "C" void initlibPyROOT()
{
   using namespace PyROOT;

// prepare for lazyness
   PyObject* dict = PyDict_New();
   dictLookupOrg = ((PyDictObject*)dict)->ma_lookup;
   Py_DECREF( dict );

// setup PyROOT
   gRootModule = Py_InitModule( const_cast< char* >( "libPyROOT" ), PyROOTMethods );
   if ( ! gRootModule )
      return;
   Py_INCREF( gRootModule );

// inject object proxy type
   if ( ! Utility::InitProxy( gRootModule, &ObjectProxy_Type, "ObjectProxy" ) )
      return;

// inject method proxy type
   if ( ! Utility::InitProxy( gRootModule, &MethodProxy_Type, "MethodProxy" ) )
      return;

// inject property proxy type
   if ( ! Utility::InitProxy( gRootModule, &PropertyProxy_Type, "PropertyProxy" ) )
      return;

// setup ROOT
   PyROOT::InitRoot();
}
