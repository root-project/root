// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_ROOTWRAPPER_H
#define PYROOT_ROOTWRAPPER_H

// ROOT
class TClass;
class TGlobal;

// Standard
#include <string>


namespace PyROOT {

   class TScopeAdapter;

// initialize ROOT
   void InitRoot();

// construct the dictionary of the given ROOT class in pyclass
   int BuildRootClassDict( const TScopeAdapter& klass, PyObject* pyclass );

// construct a tuple of base classes for the given ROOT class
   PyObject* BuildRootClassBases( const TScopeAdapter& klass );

// construct a Python shadow class for the named ROOT class
   PyObject* MakeRootClassFromString(
      const std::string& name, PyObject* scope = 0 );

   PyObject* MakeRootClass( PyObject*, PyObject* args );
   PyObject* MakeRootClassFromType( TClass* );

// convenience function to retrieve global variables and enums
   PyObject* GetRootGlobal( PyObject*, PyObject* args );
   PyObject* GetRootGlobalFromString( const std::string& name );

// bind a ROOT object into a Python object
   PyObject* BindRootObjectNoCast( void* object, TClass* klass, Bool_t isRef = kFALSE );
   PyObject* BindRootObject( void* object, TClass* klass, Bool_t isRef = kFALSE );
   PyObject* BindRootGlobal( TGlobal* );

} // namespace PyROOT

#endif // !PYROOT_ROOTWRAPPER_H
