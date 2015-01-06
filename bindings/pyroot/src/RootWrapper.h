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

// initialize ROOT
   void InitRoot();

// construct a Python shadow class for the named C++ class
   PyObject* CreateScopeProxy(
      const std::string& scope_name, PyObject* parent = 0 );
   PyObject* CreateScopeProxy( PyObject*, PyObject* args );

// convenience function to retrieve global variables and enums
   PyObject* GetCppGlobal( const std::string& name );
   PyObject* GetCppGlobal( PyObject*, PyObject* args );

// bind a ROOT object into a Python object
   PyObject* BindRootObjectNoCast(
      void* object, TClass* klass, Bool_t isRef = kFALSE, Bool_t isValue = kFALSE );
   PyObject* BindRootObject( void* object, TClass* klass, Bool_t isRef = kFALSE );
   PyObject* BindRootObjectArray( void* address, TClass* klass, Int_t size );
   PyObject* BindRootGlobal( TGlobal* );

} // namespace PyROOT

#endif // !PYROOT_ROOTWRAPPER_H
