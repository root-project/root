// @(#)root/pyroot:$Name:  $:$Id: RootModule.cxx,v 1.2 2004/04/28 22:04:06 rdm Exp $
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

extern "C" void initlibPyROOT() {
// setup PyROOT
   Py_InitModule( const_cast< char* >( "libPyROOT" ), PyROOTMethods );

// setup ROOT
   PyROOT::initRoot();
}
