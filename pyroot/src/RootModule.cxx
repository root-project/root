// @(#)root/pyroot:$Name:  $:$Id: RootModule.cxx,v 1.3 2004/05/07 20:47:20 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "RootWrapper.h"


//------------------------------------------------------------------------------
static PyMethodDef PyROOTMethods[] = {
   { (char*) "makeRootClass", (PyCFunction) PyROOT::makeRootClass,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "getRootGlobalEnum", (PyCFunction) PyROOT::getRootGlobalEnum,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { NULL, NULL, 0, NULL }
};

extern "C" void initlibPyROOT() {
// setup PyROOT
   Py_InitModule( const_cast< char* >( "libPyROOT" ), PyROOTMethods );

// setup ROOT
   PyROOT::initRoot();
}
