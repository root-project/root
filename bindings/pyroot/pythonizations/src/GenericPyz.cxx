// Author: Stefan Wunsch, Enric Tejedor CERN  06/2018
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Python.h"

#include "CPyCppyy/API.h"

#include "../../cppyy/CPyCppyy/src/CPyCppyy.h"
#include "../../cppyy/CPyCppyy/src/CPPInstance.h"
#include "../../cppyy/CPyCppyy/src/Utility.h"

#include "PyROOTPythonize.h"

#include "TClass.h"
#include "TInterpreter.h"
#include "TInterpreterValue.h"

#include <map>

namespace {

std::string GetScopedFinalNameFromPyObject(const PyObject *pyobj)
{
   return Cppyy::GetScopedFinalName(((CPyCppyy::CPPInstance *)pyobj)->ObjectIsA());
}

} // namespace

using namespace CPyCppyy;

// We take as unique identifier the declId of the class to
// treat the case where a class is loaded, an instance printed,
// the class unloaded and reloaded with changes.
static ULong64_t GetClassID(const char *clName)
{
   if (auto cl = TClass::GetClass(clName)) {
      if (auto clInfo = cl->GetClassInfo()) {
         return reinterpret_cast<ULong64_t>(gInterpreter->GetDeclId(clInfo));
      }
   }
   return 0;
}

PyObject *ClingPrintValue(PyObject *self, PyObject * /* args */)
{
   // Map holding the classID of the classes and the pointer
   // to the printer function.
   static std::map<ULong64_t, void *> declIDPrinterMap;

   auto cppObj = CPyCppyy::Instance_AsVoidPtr(self);
   if (!cppObj)
      // Proxied cpp object is null, use cppyy's generic __repr__
      return PyObject_Repr(self);

   // We jit the helper only once, at the first invocation of any
   // printer. The integer parameter is there to make sure we have
   // different instances of the printing function in presence of
   // unload-reload events.
   if (0 == declIDPrinterMap.size()) {
      std::string printerCode = "namespace ROOT::Internal::Pythonizations::ValuePrinters"
                                "{"
                                "   template<class T, ULong64_t> std::string ValuePrinter(void *obj)"
                                "   {"
                                "      return cling::printValue((T *)obj);"
                                "   }"
                                "}";
      gInterpreter->Declare(printerCode.c_str());
   }

   const std::string className = GetScopedFinalNameFromPyObject(self);

   std::string printResult;

   if (const auto classID = GetClassID(className.c_str())) {
      // If we never encountered this class, we jit the function which
      // is necessary to print it and store it in the map instantiated
      // above. Otherwise, we just use the pointer to the previously
      // jitted function. This allows to jit the printer only once per
      // type, at the modest price of a typename and pointer stored in
      // memory.
      auto &printerFuncrPtr = declIDPrinterMap[classID];

      if (!printerFuncrPtr) {
         std::string printFuncName = "ROOT::Internal::Pythonizations::ValuePrinters::ValuePrinter<" + className + ", " +
                                     std::to_string(classID) + ">";
         printerFuncrPtr = (void *)gInterpreter->Calc(printFuncName.c_str());
      }
      printResult = ((std::string(*)(void *))printerFuncrPtr)(cppObj);
   } else {
      // If something went wrong, we use the slow method
      printResult = gInterpreter->ToString(className.c_str(), cppObj);
   }

   if (printResult.find("@0x") == 0) {
      // Fall back to __repr__ if we just get an address from cling
      return PyObject_Repr(self);
   } else {
      return PyUnicode_FromString(printResult.c_str());
   }
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pretty printing pythonization
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// received from Python.
///
/// This function adds the following pythonizations to print the object more
/// user-friendly than cppyy by using the output of cling::printValue as the
/// return value of the special method __str__.
PyObject *PyROOT::AddPrettyPrintingPyz(PyObject * /* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);
   Utility::AddToClass(pyclass, "__str__", (PyCFunction)ClingPrintValue);
   Py_RETURN_NONE;
}
