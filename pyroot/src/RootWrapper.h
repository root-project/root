// @(#)root/pyroot:$Name:  $:$Id: RootWrapper.h,v 1.5 2004/10/30 06:26:43 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_ROOTWRAPPER_H
#define PYROOT_ROOTWRAPPER_H

// ROOT
class TClass;
class TGlobal;

// Standard
#include <string>


namespace PyROOT {

// bindings
   class ObjectHolder;


// initialize ROOT
   void initRoot();

// construct the dictionary of the given ROOT class in pyclass
   int buildRootClassDict( TClass* cls, PyObject* pyclass );

// construct a tuple of base classes for the given ROOT class
   PyObject* buildRootClassBases( TClass* cls );

// construct a Python shadow class for the named ROOT class
   PyObject* makeRootClass( PyObject* self, PyObject* args );
   PyObject* makeRootClassFromString( const std::string& className );

// convenience function to retrieve global variables and enums
   PyObject* getRootGlobal( PyObject* self, PyObject* args );
   PyObject* getRootGlobalFromString( const std::string& globalName );

// bind a ROOT object into a Python object (if force is false, allow recycle)
   PyObject* bindRootObject( ObjectHolder* obh, bool force = false );
   PyObject* bindRootGlobal( TGlobal* );

} // namespace PyROOT

#endif // !PYROOT_ROOTWRAPPER_H
