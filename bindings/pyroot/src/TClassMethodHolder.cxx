// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Aug 2004

// Bindings
#include "PyROOT.h"
#include "TClassMethodHolder.h"


//- public members --------------------------------------------------------------
PyObject* PyROOT::TClassMethodHolder::Call(
      ObjectProxy*&, PyObject* args, PyObject* kwds, TCallContext* ctxt )
{
// preliminary check in case keywords are accidently used (they are ignored otherwise)
if (kwds != nullptr && PyDict_Size(kwds)) {
   PyErr_SetString(PyExc_TypeError, "keyword arguments are not yet supported");
   return nullptr;
   }

// setup as necessary
   if (!this->Initialize(ctxt)) return nullptr; // important: 0, not Py_None

   // translate the arguments
   if (!this->ConvertAndSetArgs(args, ctxt)) return nullptr; // important: 0, not Py_None

   // execute function
   return this->Execute(nullptr, 0, ctxt);
}
