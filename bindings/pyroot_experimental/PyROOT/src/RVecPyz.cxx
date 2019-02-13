// Author: Stefan Wunsch CERN  02/2019
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "CPyCppyy.h"
#include "CPPInstance.h"
#include "ProxyWrappers.h"
#include "PyROOTPythonize.h"
#include "RConfig.h"
#include "TInterpreter.h"

////////////////////////////////////////////////////////////////////////////
/// \brief Adopt memory of a Python object with array interface using an RVec
/// \param[in] obj PyObject with array interface
///
/// This function returns an RVec which adopts the memory of the given
/// PyObject. The RVec takes the data pointer and the size from the array
/// interface dictionary.
PyObject *PyROOT::AsRVec(PyObject * /*self*/, PyObject * obj)
{
   if (!obj) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: Invalid Python object.");
      return NULL;
   }

   // Get array interface of object
   auto pyinterface = PyObject_GetAttrString(obj, "__array_interface__");
   if (!pyinterface) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: __array_interface__ does not exist.");
      return NULL;
   }
   if (!PyDict_Check(pyinterface)) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: __array_interface__ is not a dictionary.");
      return NULL;
   }

   // Get the data-pointer
   auto pydata = PyDict_GetItemString(pyinterface, "data");
   if (!pydata) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: __array_interface__['data'] does not exist.");
      return NULL;
   }
   long data = PyLong_AsLong(PyTuple_GetItem(pydata, 0));

   // Get the size of the contiguous memory
   auto pyshape = PyDict_GetItemString(pyinterface, "shape");
   if (!pyshape) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: __array_interface__['shape'] does not exist.");
      return NULL;
   }
   long size = 0;
   for (unsigned int i = 0; i < PyTuple_Size(pyshape); i++) {
      if (size == 0) size = 1;
      size *= PyLong_AsLong(PyTuple_GetItem(pyshape, i));
   }

   // Get the typestring
   auto pytypestr = PyDict_GetItemString(pyinterface, "typestr");
   if (!pytypestr) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: __array_interface__['typestr'] does not exist.");
      return NULL;
   }
   std::string typestr = CPyCppyy_PyUnicode_AsString(pytypestr);
   const auto length = typestr.length();
   if(length != 3) {
      PyErr_SetString(PyExc_RuntimeError,
              ("Object not convertible: __array_interface__['typestr'] returned '" + typestr + "' with invalid length unequal 3.").c_str());
      return NULL;
   }

   // Verify correct endianess
   const auto endianess = typestr.substr(1, 2);
#ifdef R__BYTESWAP
   const auto byteswap = "<";
#else
   const auto byteswap = ">";
#endif
   if (!endianess.compare(byteswap)) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: Endianess of __array_interface__['typestr'] does not match endianess of ROOT.");
      return NULL;
   }

   const auto dtype = typestr.substr(1, length);
   std::string cppdtype;
   if (dtype == "i4") {
      cppdtype = "int";
   } else if (dtype == "u4") {
      cppdtype = "unsigned int";
   } else if (dtype == "i8") {
      cppdtype = "long";
   } else if (dtype == "u8") {
      cppdtype = "unsigned long";
   } else if (dtype == "f4") {
      cppdtype = "float";
   } else if (dtype == "f8") {
      cppdtype = "double";
   } else {
      PyErr_SetString(PyExc_RuntimeError, ("Object not convertible: Python object has unknown data-type '" + dtype + "'.").c_str());
      return NULL;
   }

   // Construct an RVec of the correct data-type
   const std::string klassname = "ROOT::VecOps::RVec<" + cppdtype + ">";
   auto address = (void*) gInterpreter->Calc("new " + klassname + "(reinterpret_cast<" + cppdtype + "*>(" + data + ")," + size + ")");

   // Bind the object to a Python-side proxy
   auto klass = (Cppyy::TCppType_t)Cppyy::GetScope(klassname);
   auto pyobj = CPyCppyy::BindCppObject(address, klass);

   // Give Python the ownership of the underlying C++ object
   ((CPyCppyy::CPPInstance*)pyobj)->PythonOwns();

   // Bind pyobject holding adopted memory to the RVec
   if (PyObject_SetAttrString(pyobj, "__adopted__", obj)) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: Failed to set Python object as attribute __adopted__.");
      return NULL;
   }

   // Clean-up and return
   Py_DECREF(pyinterface);
   return pyobj;
}
