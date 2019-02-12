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

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/PyROOTHelpers.hxx" // RResultPtrDummy

#include <utility> // std::pair
#include <sstream> // std::stringstream

////////////////////////////////////////////////////////////////////////////
/// \brief Make an RDataFrame from a dictionary of numpy arrays
/// \param[in] pydata Dictionary with numpy arrays
///
/// This function takes a dictionary of numpy arrays and creates an RDataFrame
/// using the keys as column names and the numpy arrays as data.
PyObject *PyROOT::MakeRDataFrame(PyObject * /*self*/, PyObject * pydata)
{
   if (!pydata) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: Invalid Python object.");
      return NULL;
   }

   if (!PyDict_Check(pydata)) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: Python object is not a dictionary.");
      return NULL;
   }

   // Iterate over dictionary, convert numpy arrays to RVecs and put together interpreter code
   PyObject *key, *value;
   Py_ssize_t pos = 0;
   std::string code = "ROOT::Internal::RDF::NewLazyDataFrame(";
   auto pyvecs = PyDict_New();
   while (PyDict_Next(pydata, &pos, &key, &value)) {
      // Get name of key
      if (!CPyCppyy_PyUnicode_Check(key)) {
         PyErr_SetString(PyExc_RuntimeError, "Object not convertible: Dictionary key is not convertible to a string.");
         return NULL;
      }
      std::string keystr = CPyCppyy_PyUnicode_AsString(key);

      // Convert value to RVec and attach to dictionary
      auto pyvec = PyROOT::AsRVec(NULL, value);
      if (pyvec == NULL) {
         PyErr_SetString(PyExc_RuntimeError,
                         ("Object not convertible: Dictionary entry " + keystr + " is not convertible with AsRVec.").c_str());
         return NULL;
      }
      PyDict_SetItem(pyvecs, key, pyvec);

      // Prepare interpreter code
      std::string vectype = Cppyy::GetScopedFinalName(((CPyCppyy::CPPInstance*)pyvec)->ObjectIsA());
      std::string ptrtype = "ROOT::Internal::RDF::RResultPtrDummy<" + vectype + ">";
      std::stringstream ss;
      ss << ((CPyCppyy::CPPInstance*)pyvec)->GetObject();
      auto vecaddress = ss.str();
      code += "std::pair<std::string," + ptrtype +  ">(\"" + keystr
           + "\"," + ptrtype + "(*reinterpret_cast<" + vectype+ "*>(" + vecaddress + ")))";
      code += ",";
   }
   code.pop_back();
   code += ");";

   // Create RDataFrame and build Python proxy
   auto address = (void*) gInterpreter->Calc(code.c_str());
   auto klass = (Cppyy::TCppType_t)Cppyy::GetScope("ROOT::RDataFrame");
   auto pyobj = CPyCppyy::BindCppObject(address, klass);

   // Give Python the ownership of the underlying C++ object
   ((CPyCppyy::CPPInstance*)pyobj)->PythonOwns();

   // Bind pyobject holding adopted memory to the RVec
   // TODO: Do we need to keep a reference to the data dictionary or is this
   // already fixed by the reference from the rvec to the numpy arrays?
   if (PyObject_SetAttrString(pyobj, "__data__", pyvecs)) {
      PyErr_SetString(PyExc_RuntimeError, "Object not convertible: Failed to set dictionary as attribute __data__.");
      return NULL;
   }

   return pyobj;
}
