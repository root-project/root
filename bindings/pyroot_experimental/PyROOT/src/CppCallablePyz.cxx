// Author: Stefan Wunsch, Enric Tejedor CERN  04/2019
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Python.h"

#include "CPyCppyy.h"
#include "PyROOTPythonize.h"
#include "CPPInstance.h"
#include "Utility.h"
#include "TInterpreter.h"
#include "TInterpreterValue.h"

#include <sstream>


// Parse positional arguments of the decorator
bool ParsePositionalArgs(PyObject* args)
{
   if (!PyTuple_Check(args)) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to parse positional arguments: Invalid tuple.");
      return false;
   }

   if (PyTuple_Size(args) != 3) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to parse positional arguments: Expect exactly two positional arguments (list of input types, return type).");
      return false;
   }

   auto instance = PyTuple_GetItem(args, 0);
   auto inputTypes = PyTuple_GetItem(args, 1);
   auto returnType = PyTuple_GetItem(args, 2);

   // Attach arguments to instance
   PyObject_SetAttrString(instance, "input_types", inputTypes);
   PyObject_SetAttrString(instance, "return_type", returnType);

   return true;
}


// Attach keyword to instance
bool AttachKeyword(PyObject* kwargs, PyObject* instance, const char* name)
{
   PyObject* p;
   if ((p = PyDict_GetItemString(kwargs, name))) {
      const auto status = PyObject_IsTrue(p);
      if (status == 1) {
         PyObject_SetAttrString(instance, name, Py_True);
      } else if (status == 0) {
         PyObject_SetAttrString(instance, name, Py_False);
      } else {
         return false;
      }
   }
   return true;
}


// Parse keyword arguments of the decorator
bool ParseKeywordArguments(PyObject* args, PyObject* kwargs)
{
   if (!PyDict_Check(kwargs)) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to parse keyword arguments: Invalid dictionary.");
      return false;
   }

   // Attach optional name to instance
   auto instance = PyTuple_GetItem(args, 0);
   PyObject* p;
   if ((p = PyDict_GetItemString(kwargs, "name"))) {
      if (!CPyCppyy_PyUnicode_Check(p)) {
         PyErr_SetString(PyExc_RuntimeError,
                 "Failed to parse arguments: Given name is not a valid string.");
         return false;
      }
      PyObject_SetAttrString(instance, "name", p);
   }

   // Attach optional numpy_only flag to instance
   if (!AttachKeyword(kwargs, instance, "numba_only")) {
      PyErr_SetString(PyExc_RuntimeError,
              "Failed to parse arguments: Given object for numba_only cannot be evaluated as a boolean.");
      return false;
   }

   // Attach optional generic_only flag to instance
   if (!AttachKeyword(kwargs, instance, "generic_only")) {
      PyErr_SetString(PyExc_RuntimeError,
              "Failed to parse arguments: Given object for generic_only cannot be evaluated as a boolean.");
      return false;
   }

   // Attach optional verbose flag
   if (!AttachKeyword(kwargs, instance, "verbose")) {
      PyErr_SetString(PyExc_RuntimeError,
              "Failed to parse arguments: Given object for verbose flag cannot be evaluated as a boolean.");
      return false;
   }

   return true;
}


// Init of class used as decorator to create generic C++ wrapper
// The init parses the arguments passed to the decorator.
PyObject* GenericCallableImpl_init(PyObject * /*self*/, PyObject *args, PyObject *kwargs)
{
   if(!ParsePositionalArgs(args)) return NULL;

   if(kwargs != 0) {
      if(!ParseKeywordArguments(args, kwargs)) return NULL;
   }

   Py_RETURN_NONE;
}


// Check arguments given to call operator of the decorator
bool CheckCallArgs(PyObject* args)
{
   if (!PyTuple_Check(args)) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to parse arguments: Invalid tuple.");
      return false;
   }

   if (!(PyTuple_Size(args) == 2)) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to parse arguments: Expect exactly one argument (Python callable).");
      return false;
   }

   return true;
}


// Check instance passed from init to call operator of the decorator
bool CheckInstance(PyObject* instance)
{
   if (!PyObject_HasAttrString(instance, "input_types")) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: No input_types attribute found.");
      return false;
   }

   if (!PyObject_HasAttrString(instance, "return_type")) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: No return_type attribute found.");
      return false;
   }

   return true;
}


// Check callable passed to decorator
bool CheckCallable(PyObject* callable)
{
   if (!PyCallable_Check(callable)) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: Given PyObject is not callable.");
      return false;
   }
   return true;
}


// Extract name of callable passed to call operator of the decorator.
// Either extract the name from the optional keyword argument or from the
// __name__ property of the callable itself.
std::string ExtractName(PyObject* instance, PyObject* pyfunc)
{
   PyObject* pyname;
   if (PyObject_HasAttrString(instance, "name")) {
      pyname = PyObject_GetAttrString(instance, "name");
   } else {
      if (!PyObject_HasAttrString(pyfunc, "__name__")) {
         PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: Python callable does not have attribute __name__.");
         return "";
      }
      pyname = PyObject_GetAttrString(pyfunc, "__name__");
   }
   std::string name = CPyCppyy_PyUnicode_AsString(pyname);
   Py_DECREF(pyname);
   return name;
}


// Call method of class used as decorator to create generic C++ wrapper
// The call method creates the C++ wrapper class for the Python callable and
// passes through the actual callable.
PyObject* GenericCallableImpl_call(PyObject * /*self*/, PyObject *args)
{
   // Parse arguments
   if(!CheckCallArgs(args)) return NULL;
   auto instance = PyTuple_GetItem(args, 0);
   auto pyfunc = PyTuple_GetItem(args, 1);
   if(!CheckCallable(pyfunc)) return NULL;
   Py_INCREF(pyfunc);
   if(!CheckInstance(instance)) return NULL;

   auto inputTypes = PyObject_GetAttrString(instance, "input_types");
   auto returnType = PyObject_GetAttrString(instance, "return_type");

   // Extract name of Python callable
   auto name = ExtractName(instance, pyfunc);
   if (name.compare("") == 0) return NULL;

   // Get C++ return type
   if (!CPyCppyy_PyUnicode_Check(returnType)) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: Return type argument cannot be interpreted as string.");
      return NULL;
   }
   std::string returnTypeStr = CPyCppyy_PyUnicode_AsString(returnType);
   Py_DECREF(returnType);
   if (returnTypeStr.compare("") == 0) {
      returnTypeStr = "void";
   }

   // Put function in namespace
   std::stringstream code;
   code << "namespace CppCallable {\n";

   // Set return type
   code << returnTypeStr << " ";

   // Set name of Python callable as function name
   code << name;

   // Build function signature, type string and list of variables
   code << "(";

   auto iter = PyObject_GetIter(inputTypes);
   auto inputTypesSize = PyObject_Size(inputTypes);
   Py_DECREF(inputTypes);
   if (!iter) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: Failed to iterate over input types.");
      return NULL;
   }

   // Map C++ types to type characters of Python/C API (PyObject_CallFunction)
   std::map<std::string, std::string> typemap = {
       {"float", "f"},
       {"double", "f"},
       {"int", "i"},
       {"unsigned int", "I"},
       {"long", "l"},
       {"unsigned long", "k"},
   };

   PyObject *item;
   auto idx = 0u;
   std::stringstream typestr;
   std::vector<std::string> pytypes(inputTypesSize);
   std::vector<std::string> inputTypesStr(inputTypesSize);
   std::stringstream vars;
   while ((item = PyIter_Next(iter))) {
      // Convert argument to string
      if (!CPyCppyy_PyUnicode_Check(item)) {
         Py_DECREF(iter);
         Py_DECREF(item);
         PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: Failed to interpret input type as string.");
         return NULL;
      }

      inputTypesStr[idx] = CPyCppyy_PyUnicode_AsString(item);
      Py_DECREF(item);

      auto pytype = typemap.find(inputTypesStr[idx]);
      if (pytype != typemap.end()) { // Types in typemap
         pytypes[idx] = pytype->second;
         vars << pytypes[idx] << "_" << idx;
         code << inputTypesStr[idx] << " " << pytypes[idx] << "_" << idx;
      } else if (inputTypesStr[idx].compare("") == 0) { // No input type
         pytypes[idx] = "";
      } else if (inputTypesStr[idx].compare("bool") == 0) { // Bool
         pytypes[idx] = "O";
         vars << "pyb_" << idx;
         code << inputTypesStr[idx] << " b_" << idx;
      } else { // C++ object
         pytypes[idx] = "O";
         vars << "pyo_" << idx;
         code << inputTypesStr[idx] << "& o_" << idx;
      }
      typestr << pytypes[idx];

      if (idx != inputTypesSize - 1 && pytypes[idx].compare("") != 0) {
         code << ", ";
         vars << ", ";
      }

      idx++;
   }
   Py_DECREF(iter);

   // Acquire lock to protect multi-threaded scenarios
   code << ") {\n"
        << "   // Acquire lock to protect multi-threaded scenarios\n"
        << "   R__WRITE_LOCKGUARD(ROOT::gCoreMutex);\n\n";

   // Get pointer to Python callable
   code << "   // Get Python callable from pointer\n"
        << "   auto pyfunc = reinterpret_cast<PyObject*>(" << pyfunc << ");\n"
        << "   if (!PyCallable_Check(pyfunc)) {\n"
        << "      throw std::runtime_error(\"Python object " << name << " is not callable.\");\n"
        << "   }\n\n";

   // Build Python proxies of the C++ objects for Python
   code << "   // Build Python proxies for C++ objects\n";
   std::stringstream cleanup; // Register clean-up code
   bool hasPyBoolIncref = false;
   for (std::size_t i = 0; i < pytypes.size(); i++) {
      if (inputTypesStr[i].compare("bool") == 0) { // Bool
         if (!hasPyBoolIncref) {
            code << "   Py_INCREF(Py_True);\n"
                 << "   Py_INCREF(Py_False);\n";
            cleanup << "   Py_DECREF(Py_True);\n"
                    << "   Py_DECREF(Py_False);\n";
            hasPyBoolIncref = true;
         }
         code << "   auto pyb_" << i << " = b_" << i << " ? Py_True : Py_False;\n";
      } else if (pytypes[i] == "O") { // C++ objects
         code << "   auto pyo_" << i << " = TPython::CPPInstance_FromVoidPtr("
              << "&o_" << i << ", \"" << inputTypesStr[i] << "\");\n";
         cleanup << "   Py_DECREF(pyo_" << i << ");\n";
      }
   }
   code << "\n";

   // Call Python callable
   auto typestr_str = typestr.str();
   auto vars_str = vars.str();
   if (vars_str.compare("") != 0) {
      vars_str = ", " + vars_str;
   }
   code << "   // Call Python callable\n"
        << "   auto pyresult = PyObject_CallFunction(pyfunc, (char*)\"" << typestr_str << "\"" << vars_str << ");\n"
        << "   if (pyresult == 0) {\n"
        << "      PyErr_Print();\n"
        << "      throw std::runtime_error(\"Failed to call Python callable " << name << ".\");\n"
        << "   }\n\n";

   // Clean-up Python proxies
   code << "   // Clean-up Python proxies\n"
        << cleanup.str()
        << "\n";

   // Convert result to C++ type
   code << "   // Convert result to C++ type\n";
   if (returnTypeStr.compare("void") == 0) {
      code << "   Py_DECREF(pyresult);\n\n"
           << "   return;\n";
   } else if (returnTypeStr.compare("bool") == 0) {
      code << "   if (pyresult == Py_True) {\n"
           << "      Py_DECREF(pyresult);\n"
           << "      return true;\n"
           << "   } else if (pyresult == Py_False) {\n"
           << "      Py_DECREF(pyresult);\n"
           << "      return false;\n"
           << "   } else {\n"
           << "      PyErr_Print();\n"
           << "      throw std::runtime_error(\"Failed to convert return value of Python callable to C++ object: Python object is not of type PyBool.\");\n"
           << "   }\n";
   } else {
      auto pytype = typemap.find(returnTypeStr);
      if (pytype != typemap.end()) {
         if (pytype->second == "f") {
            code << "   auto result = PyFloat_AsDouble(pyresult);\n";
         } else if (pytype->second == "i" || pytype->second == "l") {
            code << "   auto result = PyLong_AsLong(pyresult);\n";
         } else if (pytype->second == "I" || pytype->second == "k") {
            code << "   auto result = PyLong_AsUnsignedLong(pyresult);\n";
         }
      } else {
         code << "   if (!TPython::CPPInstance_Check(pyresult)) {\n"
              << "      throw std::runtime_error(\"Failed to convert return value of Python callable to C++ object: Python object is not created by cppyy (CPPInstance).\");\n"
              << "      \n"
              << "   }\n";
         code << "   auto result = *reinterpret_cast<" << returnTypeStr << "*>(TPython::CPPInstance_AsVoidPtr(pyresult));\n";
      }

      code << "   Py_DECREF(pyresult);\n\n"
           << "   return result;\n";
   }
   code << "}\n}";

   // Attach C++ wrapper code to callable
   auto code_str = code.str();
   auto code_cstr = code_str.c_str();
   auto pycode = CPyCppyy_PyUnicode_FromString(code_cstr);
   PyObject_SetAttrString(pyfunc, "__cpp_wrapper__", pycode);
   Py_DECREF(pycode);

   // Jit C++ wrapper
   auto err = gInterpreter->Declare("#include \"Python.h\"");
   if (!err) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to compile C++ wrapper: Failed to include Python.h.");
      return NULL;
   }

   err = gInterpreter->Declare("#include \"CPyCppyy/TPython.h\"");
   if (!err) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to compile C++ wrapper: Failed to include TPython.h.");
      return NULL;
   }

   err = gInterpreter->Declare(code_cstr);
   if (!err) {
      PyErr_SetString(PyExc_RuntimeError,
              ("Failed to compile C++ wrapper: Compilation error from following wrapper code.\n" + code.str()).c_str());
      return NULL;
   }

   // Pass through Python callable
   return pyfunc;
}


// Call method of class used as decorator to create C++ wrapper using numba
// The call method creates the C++ wrapper class for the Python callable and
// passes through the actual callable.
PyObject* NumbaCallableImpl_call(PyObject * /*self*/, PyObject *args)
{
   // Parse arguments
   if(!CheckCallArgs(args)) return NULL;
   auto instance = PyTuple_GetItem(args, 0);
   auto pyfunc = PyTuple_GetItem(args, 1);
   if(!CheckCallable(pyfunc)) return NULL;
   Py_INCREF(pyfunc);
   if(!CheckInstance(instance)) return NULL;

   auto inputTypes = PyObject_GetAttrString(instance, "input_types");
   auto returnType = PyObject_GetAttrString(instance, "return_type");

   // Extract name of Python callable
   auto name = ExtractName(instance, pyfunc);
   if (name.compare("") == 0) return NULL;

   // Get C++ return type
   if (!CPyCppyy_PyUnicode_Check(returnType)) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: Return type argument cannot be interpreted as string.");
      return NULL;
   }
   std::string returnTypeStr = CPyCppyy_PyUnicode_AsString(returnType);
   Py_DECREF(returnType);
   if (returnTypeStr.compare("") == 0) {
      returnTypeStr = "void";
   }

   // Find numba types for C++ types
   std::map<std::string, std::string> typemap = {
       {"float", "float32"},
       {"double", "float64"},
       {"int", "int32"},
       {"unsigned int", "uint32"},
       {"long", "int64"},
       {"unsigned long", "uint64"},
       {"bool", "boolean"},
   };

   auto iter = PyObject_GetIter(inputTypes);
   Py_DECREF(inputTypes);
   if (!iter) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: Failed to iterate over input types.");
      return NULL;
   }

   PyObject *item;
   std::vector<std::string> numbaTypes;
   std::vector<std::string> cppTypes;
   while ((item = PyIter_Next(iter))) {
      // Convert argument to string
      if (!CPyCppyy_PyUnicode_Check(item)) {
         Py_DECREF(iter);
         Py_DECREF(item);
         PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: Failed to interpret input type as string.");
         return NULL;
      }

      const std::string cpptype = CPyCppyy_PyUnicode_AsString(item);
      Py_DECREF(item);

      auto t = typemap.find(cpptype);
      if (t != typemap.end()) { // Types in typemap
         numbaTypes.emplace_back(t->second);
         cppTypes.emplace_back(cpptype);
      } else if (cpptype.compare("") == 0) { // No input, skip
      } else {
         Py_DECREF(iter);
         Py_DECREF(item);
         PyErr_SetString(PyExc_RuntimeError,
                 ("Failed to create C++ callable: Input type " + cpptype + " is not valid for jitting with numba.").c_str());
         return NULL;
      }
   }
   Py_DECREF(iter);

   // Import numba
   auto numba = PyImport_ImportModule("numba");
   if (!numba) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to import numba.");
      return NULL;
   }

   // Get cfunc method
   auto cfunc = PyObject_GetAttrString(numba, "cfunc");
   Py_DECREF(numba);
   if (!cfunc) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to import cfunc from numba.");
      return NULL;
   }

   // Jit Python callable
   std::stringstream numbaSignature;
   auto t = typemap.find(returnTypeStr);
   if (t != typemap.end()) {
      numbaSignature << typemap[returnTypeStr] << "(";
   } else if (returnTypeStr.compare("") == 0 || returnTypeStr.compare("void") == 0) {
      numbaSignature << "void(";
   } else {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: Return type is not valid for jitting with numba.");
      return NULL;
   }
   for(std::size_t i = 0; i < numbaTypes.size(); i++) {
      numbaSignature << numbaTypes[i];
      if (i != numbaTypes.size() - 1) {
         numbaSignature << ", ";
      }
   }
   numbaSignature << ")";
   auto numbaSignatureStr = numbaSignature.str();
   auto args_ = Py_BuildValue("(s)", (char*)numbaSignatureStr.c_str());
   auto kwargs_ = Py_BuildValue("{s:O}", (char*)"nopython", Py_True);
   auto decorator = PyObject_Call(cfunc, args_, kwargs_);
   Py_DECREF(cfunc);
   if (!decorator) {
      PyErr_SetString(PyExc_RuntimeError,
              ("Failed to create C++ callable: Unable to create instance of numba.cfunc with signature "
              + numbaSignatureStr + ".").c_str());
      return NULL;
   }
   Py_DECREF(args_);
   Py_DECREF(kwargs_);

   auto jitted = PyObject_CallFunction(decorator, (char*)"O", pyfunc);
   Py_DECREF(decorator);
   if (!jitted) {
      PyObject *type, *value, *traceback;
      PyErr_Fetch(&type, &value, &traceback);
      auto pyerr = PyObject_Str(value);
      std::string pyerrstr = CPyCppyy_PyUnicode_AsString(pyerr);
      PyErr_SetString(PyExc_RuntimeError,
              ("Failed to create C++ callable: Unable to jit function using numba.cfunc with signature "
               + numbaSignatureStr + ":\n" + pyerrstr).c_str());
      Py_DECREF(pyerr);
      Py_DECREF(type);
      Py_DECREF(value);
      Py_DECREF(traceback);
      return NULL;
   }

   // Attach jitted function to callable
   PyObject_SetAttrString(pyfunc, "__numba_cfunc__", jitted);
   Py_DECREF(jitted);

   // Extract function pointer
   auto pyaddress = PyObject_GetAttrString(jitted, "address");
   if (!pyaddress) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create C++ callable: Unable to extract function pointer from numba.cfunc.");
      return NULL;
   }
   auto address = PyLong_AsUnsignedLongLong(pyaddress);

   // Put wrapper function in ROOT namespace
   std::stringstream code;
   code << "namespace CppCallable {\n";

   // Set return type of wrapper functoin
   code << returnTypeStr << " ";

   // Set name of Python callable as function name
   code << name;

   // Build function signature, function pointer cast and variable list
   code << "(";
   std::stringstream vars;
   std::stringstream fPtr;
   fPtr << returnTypeStr << "(*)(";
   for(std::size_t i = 0; i < cppTypes.size(); i++) {
      code << cppTypes[i] << " x_" << i;
      vars << "x_" << i;
      fPtr << cppTypes[i];
      if (i != cppTypes.size() - 1) {
         code << ", ";
         vars << ", ";
         fPtr << ", ";
      }
   }
   code << ") {\n";
   fPtr << ")";

   // Cast int to C function pointer
   code << "   auto funcptr = reinterpret_cast<" << fPtr.str() << ">(" << address << ");\n";

   // Return result
   code << "   return funcptr(" << vars.str() << ");\n";

   // Close function and namespace
   code << "}\n}";

   // Jit C++ wrapper
   auto code_str = code.str();
   auto code_cstr = code_str.c_str();

   auto err = gInterpreter->Declare(code_cstr);
   if (!err) {
      PyErr_SetString(PyExc_RuntimeError,
              ("Failed to compile C++ wrapper: Compilation error from following wrapper code.\n" + code.str()).c_str());
      return NULL;
   }

   // Attach code function to callable
   auto pycode = CPyCppyy_PyUnicode_FromString(code_cstr);
   PyObject_SetAttrString(pyfunc, "__cpp_wrapper__", pycode);
   Py_DECREF(pycode);

   // Pass through Python callable
   return pyfunc;
}


bool GetKeyword(PyObject* obj, const char* name, bool defaultVal)
{
   bool prop = defaultVal;
   if (PyObject_HasAttrString(obj, name)) {
      auto attr = PyObject_GetAttrString(obj, name);
      prop = PyObject_IsTrue(attr);
      Py_DECREF(attr);
   }
   return prop;
}


// Call method of class used as decorator to create either generic or numba C++ wrapper.
// The call method creates the C++ wrapper class for the Python callable and
// passes through the actual callable.
PyObject* ProxyCallableImpl_call(PyObject * /*self*/, PyObject *args)
{
   // Get numba_only and generic_only optional arguments
   // The arguments are interpreted as follows (in this order):
   // 1) numba_only = true , generic_only = true or false: Just try numba implementation
   // 2) numba_only = false, generic_only = true: Just try generic implementation
   // 3) numba_only = false, generic_only = false: Try numba first, fail silently, go to generic (default setting)
   auto instance = PyTuple_GetItem(args, 0);

   auto numbaOnly = GetKeyword(instance, "numba_only", false);
   auto genericOnly = GetKeyword(instance, "generic_only", false);
   auto verbose = GetKeyword(instance, "verbose", true);

   // Case 1) Use only numba
   if (numbaOnly) {
      return NumbaCallableImpl_call(NULL, args);
   }

   // Case 2) Use only generic
   else if (genericOnly) {
      return GenericCallableImpl_call(NULL, args);
   }

   // Case 3) Try first numba and then fall back to generic
   else {
      auto pyfunc = NumbaCallableImpl_call(NULL, args);
      if (pyfunc) {
         return pyfunc;
      } else {
         PyErr_Clear();
         if (verbose) {
            PyErr_WarnEx(PyExc_RuntimeWarning,
                    "Failed to compile Python callable using numba, fall back to generic implementation. Note that the generic implementation is potentially slow and does not allow multi-threading.", 1);
         }
         return GenericCallableImpl_call(NULL, args);
      }
   }
}


// Method definition for class used as decorator to create C++ wrapper
static PyMethodDef CallableImplMethods[] =
{
    {"__init__", (PyCFunction)GenericCallableImpl_init, METH_VARARGS|METH_KEYWORDS, "Parse decorator arguments"},
    {"__call__", ProxyCallableImpl_call, METH_VARARGS, "Create C++ wrapper function"},
    {NULL, NULL, 0, NULL}
};


// Proxy to return the C++ wrapper class which can be used as decorator
PyObject *PyROOT::GetCppCallableClass(PyObject * /*self*/, PyObject * args) {
   // Parse argument to get type of callable class
   if (!PyTuple_Check(args)) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create callable class: Invalid tuple.");
      return NULL;
   }

   // Create wrapper class for decorator
   auto classDict = PyDict_New();
   auto className = CPyCppyy_PyUnicode_FromString("CppCallableImpl");
   auto classBases = PyTuple_New(0);

   // Add methods
   for (auto def = CallableImplMethods; def->ml_name != NULL; def++) {
      auto func = PyCFunction_New(def, NULL);
#if PY_VERSION_HEX < 0x03000000
      auto method = PyMethod_New(func, NULL, NULL);
#else
      auto method = PyInstanceMethod_New(func);
#endif
	  PyDict_SetItemString(classDict, def->ml_name, method);
	  Py_DECREF(func);
	  Py_DECREF(method);
   }

   auto callableClass = PyObject_CallFunctionObjArgs(
           (PyObject*)&PyType_Type, className, classBases, classDict, NULL);
   Py_DECREF(className);
   Py_DECREF(classBases);
   Py_DECREF(classDict);

   // Return implementation class
   return callableClass;
}
