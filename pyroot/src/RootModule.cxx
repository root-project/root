// @(#)root/pyroot:$Name:  $:$Id:  $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "ObjectHolder.h"
#include "RootWrapper.h"

// ROOT
#include "TObject.h"
#include "TGlobal.h"


//------------------------------------------------------------------------------
static PyMethodDef PyROOTMethods[] = {
   { (char*) "makeRootClass", (PyCFunction) PyROOT::makeRootClass,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { NULL, NULL, 0, NULL }
};

extern "C" void initPyROOT() {
// setup PyROOT
   Py_InitModule( const_cast< char* >( "PyROOT" ), PyROOTMethods );

// setup ROOT
   PyROOT::initRoot();
}
