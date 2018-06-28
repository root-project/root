#include "CPyCppyy.h"
#include "PyROOTPythonize.h"
#include "CPPInstance.h"
#include "Utility.h"
#include "TInterpreter.h"

#include <sstream>

using namespace CPyCppyy;

std::string GetCppName(CPPInstance *self)
{
   return Cppyy::GetScopedFinalName(self->ObjectIsA());
}

PyObject *ClingPrintValue(CPPInstance *self)
{
   const std::string className = GetCppName(self);
   std::string pprint;
   std::stringstream calcPrintValue;
   calcPrintValue << "*((std::string*)" << &pprint << ") = cling::printValue((" << className << "*)"
                  << self->GetObject() << ");";
   gInterpreter->Calc(calcPrintValue.str().c_str());
   return CPyCppyy_PyUnicode_FromString(pprint.c_str());
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add generic features to any class
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// received from Python.
///
/// This function adds the following pythonizations:
/// - Prints the object more user-friendly than cppyy by using the output of
///   cling::printValue as the return value of the special method __str__.
PyObject *PyROOT::PythonizeGeneric(PyObject * /* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);
   Utility::AddToClass(pyclass, "__str__", (PyCFunction)ClingPrintValue);
   Py_RETURN_NONE;
}
