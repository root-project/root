// @(#)root/pyroot:$Name:  $:$Id: RootWrapper.h,v 1.2 2004/05/07 20:47:20 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_ROOTWRAPPER_H
#define PYROOT_ROOTWRAPPER_H

// ROOT
class TClass;


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
   PyObject* makeRootClassFromString( const char* className );

// bind a held ROOT object into a Python object
   PyObject* bindRootObject( ObjectHolder* obj );

} // namespace PyROOT

#endif // !PYROOT_ROOTWRAPPER_H
