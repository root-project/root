// @(#)root/pyroot:$Name:  $:$Id: RootWrapper.h,v 1.1 2004/04/27 06:28:48 brun Exp $
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

// construct a dictionary of method dispatchers in the given ROOT class
   int buildRootClassDict( TClass* cls, PyObject* pyclass, PyObject* dct );

// construct a tuple of base classes for the given ROOT class
   PyObject* buildRootClassBases( TClass* cls );

// construct a Python shadow class for the named ROOT class
   PyObject* makeRootClass( PyObject* self, PyObject* args );
   PyObject* makeRootClassFromString( const char* className );

// bind a held ROOT object into a Python object
   PyObject* bindRootObject( ObjectHolder* obj );

} // namespace PyROOT

#endif // !PYROOT_ROOTWRAPPER_H
