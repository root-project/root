// @(#)root/pyroot:$Name:  $:$Id: RootWrapper.h,v 1.3 2004/06/12 05:35:10 brun Exp $
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

// bind a held ROOT object into a Python object (if force is false, allow recycle)
   PyObject* bindRootObject( ObjectHolder* obj, bool force = false );

} // namespace PyROOT

#endif // !PYROOT_ROOTWRAPPER_H
