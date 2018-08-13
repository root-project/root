#include "CPyCppyy.h"
#include "PyROOTPythonize.h"
#include "CPPInstance.h"
#include "Utility.h"
#include "TInterpreter.h"
#include "TInterpreterValue.h"

#include <sstream>
#include <algorithm>

using namespace CPyCppyy;

std::string GetCppName(CPPInstance *self)
{
   return Cppyy::GetScopedFinalName(self->ObjectIsA());
}

PyObject *ClingPrintValue(CPPInstance *self)
{
   const std::string className = GetCppName(self);
   std::stringstream code;
   code << "*((" << className << "*)" << self->GetObject() << ")";

   auto value = gInterpreter->CreateTemporary();
   std::string pprint = "";
   if (gInterpreter->Evaluate(code.str().c_str(), *value) == 1 /*success*/)
      pprint = value->ToTypeAndValueString().second;
   delete value;
   pprint.erase(std::remove(pprint.begin(), pprint.end(), '\n'), pprint.end());
   return CPyCppyy_PyUnicode_FromString(pprint.c_str());
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
