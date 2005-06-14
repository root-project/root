// @(#)root/pyroot:$Name:  $:$Id: RootModule.cxx,v 1.13 2005/06/12 17:21:53 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "PyRootType.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "PropertyProxy.h"
#include "PyBufferFactory.h"
#include "RootWrapper.h"
#include "Utility.h"
#include "DllImport.h"

// Standard
#include <string>


//- data -----------------------------------------------------------------------
PyObject* gRootModule = 0;


//- private helpers ------------------------------------------------------------
namespace {

   using namespace PyROOT;

   PyDictEntry* RootLookDictString( PyDictObject* mp, PyObject* key, long hash )
   {
   // first search dictionary itself
      PyDictEntry* ep = (*gDictLookupOrg)( mp, key, hash );
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

   // all failed, start calling into ROOT
      gDictLookupActive = true;

   // attempt to get ROOT enum/global/class
      PyObject* val = PyObject_GetAttr( gRootModule, key );

      if ( ! val ) {
         PyErr_Clear();
         val = MakeRootClassFromString( strkey );
      }

      if ( ! val ) {
         PyErr_Clear();
         val = GetRootGlobalFromString( strkey );
      }

      if ( ! val ) {
         PyErr_Clear();
         TObject* object = gROOT->FindObject( strkey.c_str() );
         if ( object != 0 )
            val = BindRootObject( object, object->IsA() );
      }

      if ( val != 0 ) {
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

   // stopped calling into ROOT
      gDictLookupActive = false;

      return ep;
   }

//____________________________________________________________________________
   PyObject* SetRootLazyLookup( PyObject*, PyObject* args )
   {
      PyObject* dict = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!" ), &PyDict_Type, &dict ) )
         return 0;

      ((dictlookup&)((PyDictObject*)dict)->ma_lookup) = RootLookDictString;

      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* MakeRootTemplateClass( PyObject*, PyObject* args )
   {
   // args is class name + template arguments, build full instantiation
      int nArgs = (int)PyTuple_GET_SIZE( args );
      if ( nArgs < 2 ) {
         PyErr_Format( PyExc_TypeError, "too few arguments for template instantiation" );
         return 0;
      }

   // copy initial argument (no check, comes from internal class)
      PyObject* pyname = PyString_FromString(
         PyString_AS_STRING( PyTuple_GET_ITEM( args, 0 ) ) );

   // build "< type, type, ... >" part of class name
      PyString_ConcatAndDel( &pyname, PyString_FromString( "<" ) );
      for ( int i = 1; i < nArgs; ++i ) {
      // add type as string to name
         PyObject* tn = PyTuple_GET_ITEM( args, i );
         if ( PyString_Check( tn ) )
            PyString_Concat( &pyname, tn );
         else {
            PyObject* tpName = PyObject_GetAttrString( tn, const_cast< char* >( "__name__" ) );
            if ( PyErr_Occurred() ) {
               Py_DECREF( pyname );
               return 0;
            }

         // special case for strings
            if ( strcmp( PyString_AS_STRING( tpName ), "str" ) == 0 ) {
               Py_DECREF( tpName );
               tpName = PyString_FromString( "std::string" );
            }

            PyString_ConcatAndDel( &pyname, tpName );
         }

      // add a comma, as needed
         if ( i != nArgs - 1 )
            PyString_ConcatAndDel( &pyname, PyString_FromString( "," ) );
      }
      PyString_ConcatAndDel( &pyname, PyString_FromString( ">" ) );

      std::string name = PyString_AS_STRING( pyname );
      Py_DECREF( pyname );

      return MakeRootClassFromString( name );
   }

//____________________________________________________________________________
   PyObject* AddressOf( PyObject*, PyObject* args )
   {
      ObjectProxy* pyobj = 0;
      PyObject* pyname = 0;
      if ( PyArg_ParseTuple( args, const_cast< char* >( "O|S" ), &pyobj, &pyname ) &&
           ObjectProxy_Check( pyobj ) && pyobj->fObject ) {

         if ( pyname != 0 ) {
         // locate property proxy for offset info
            PropertyProxy* pyprop = 0;

            PyObject* pyclass = PyObject_GetAttrString(
               (PyObject*)pyobj, const_cast< char* >( "__class__" ) );

            if ( pyclass ) {
               PyObject* dict = PyObject_GetAttrString( pyclass, const_cast< char* >( "__dict__" ) );
               pyprop = (PropertyProxy*)PyObject_GetItem( dict, pyname );
               Py_DECREF( dict );
            }
            Py_XDECREF( pyclass );

            if ( PropertyProxy_Check( pyprop ) ) {
            // this is an address of a value (i.e. &myobj->prop)
               PyObject* pybuf = BufFac_t::Instance()->PyBuffer_FromMemory(
                  (Long_t*)pyprop->GetAddress( pyobj ), 1 );
               Py_DECREF( pyprop );
               return pybuf;
            }

            Py_XDECREF( pyprop );

            PyErr_Format( PyExc_TypeError,
               "%s is not a valid data member", PyString_AS_STRING( pyname ) );
            return 0;
         }

      // this is an address of an address (i.e. &myobj, with myobj of type MyObj*)
         return BufFac_t::Instance()->PyBuffer_FromMemory( (Long_t*)&pyobj->fObject, 1 );
      }

      PyErr_SetString( PyExc_ValueError, "invalid argument for AddressOf()" );
      return 0;
   }

//____________________________________________________________________________
   PyObject* SetMemoryPolicy( PyObject*, PyObject* args )
   {
      PyObject* policy = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!" ), &PyInt_Type, &policy ) )
         return 0;

      long l = PyInt_AS_LONG( policy );
      if ( Utility::SetMemoryPolicy( (Utility::EMemoryPolicy)l ) ) {
         Py_INCREF( Py_None );
         return Py_None;
      }

      PyErr_Format( PyExc_ValueError, "Unknown policy %ld", l );
      return 0;
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
   { (char*) "MakeRootTemplateClass", (PyCFunction)MakeRootTemplateClass,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "AddressOf", (PyCFunction)AddressOf,
     METH_VARARGS, (char*) "Retrieve address of held object" },
   { (char*) "SetMemoryPolicy", (PyCFunction)SetMemoryPolicy,
     METH_VARARGS, (char*) "Determines object ownership model" },
   { NULL, NULL, 0, NULL }
};


//____________________________________________________________________________
extern "C" void initlibPyROOT()
{
   using namespace PyROOT;

// prepare for lazyness
   PyObject* dict = PyDict_New();
   gDictLookupOrg = (dictlookup)((PyDictObject*)dict)->ma_lookup;
   Py_DECREF( dict );

// setup PyROOT
   gRootModule = Py_InitModule( const_cast< char* >( "libPyROOT" ), PyROOTMethods );
   if ( ! gRootModule )
      return;
   Py_INCREF( gRootModule );

// inject meta type
   if ( ! Utility::InitProxy( gRootModule, &PyRootType_Type, "PyRootType" ) )
      return;

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
