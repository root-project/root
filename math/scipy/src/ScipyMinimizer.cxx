#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h> // Needs to be included first to avoid redefinition of _POSIX_C_SOURCE
#include <Math/ScipyMinimizer.h>
#include <Fit/ParameterSettings.h>
#include <Math/IFunction.h>
#include <Math/FitMethodFunction.h>
#include <Math/GenAlgoOptions.h>
#include <Math/Error.h>
#include <TString.h>
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

using namespace ROOT;
using namespace ROOT::Math;
using namespace ROOT::Math::Experimental;
using namespace ROOT::Fit;

/// function wrapper for the function to be minimized
const ROOT::Math::IMultiGenFunction *gFunction = nullptr;
/// function wrapper for the gradient of the function to be minimized
const ROOT::Math::IMultiGradFunction *gGradFunction = nullptr;
/// function wrapper for the hessian of the function to be minimized
std::function<bool(const std::vector<double> &, double *)> gfHessianFunction;

/// simple function for debugging
#define PyPrint(pyo) PyObject_Print(pyo, stdout, Py_PRINT_RAW)

/// function to wrap into Python the C/C++ target function to be minimized
PyObject *target_function(PyObject * /*self*/, PyObject *args)
{
   PyArrayObject *arr = (PyArrayObject *)PyTuple_GetItem(args, 0);

   auto params = (double *)PyArray_DATA(arr);
   auto r = (*gFunction)(params);

   return PyFloat_FromDouble(r);
};

/// function to wrap into Python the C/C++ jacobian function
PyObject *jac_function(PyObject * /*self*/, PyObject *args)
{
   PyArrayObject *arr = (PyArrayObject *)PyTuple_GetItem(args, 0);

   uint size = PyArray_SIZE(arr);
   auto params = (double *)PyArray_DATA(arr);
   double *values = new double[size];
   gGradFunction->Gradient(params, values);
   npy_intp dims[1] = {size};
   PyObject *py_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, values);
   return py_array;
};

/// function to wrap into Python the C/C++ hessian function
PyObject *hessian_function(PyObject * /*self*/, PyObject *args)
{
   PyArrayObject *arr = (PyArrayObject *)PyTuple_GetItem(args, 0);

   uint size = PyArray_SIZE(arr);
   auto params = (double *)PyArray_DATA(arr);
   double *values = new double[size * size];
   std::vector<double> x(params, params + size);
   gfHessianFunction(x, values);
   npy_intp dims[2] = {size, size};
   PyObject *py_array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, values);
   return py_array;
};

//_______________________________________________________________________
ScipyMinimizer::ScipyMinimizer() : BasicMinimizer()
{
   fCalls = 0;
   fOptions.SetMinimizerType("Scipy");
   fOptions.SetMinimizerAlgorithm("L-BFGS-B");
   PyInitialize();
   fHessianFunc = nullptr;
   fConstraintsList = PyList_New(0);
   fConstN = 0;
   fExtraOpts = nullptr;
}

//_______________________________________________________________________
ScipyMinimizer::ScipyMinimizer(const char *type)
{
   fCalls = 0;
   fOptions.SetMinimizerType("Scipy");
   fOptions.SetMinimizerAlgorithm(type);
   PyInitialize();
   fHessianFunc = nullptr;
   fConstraintsList = PyList_New(0);
   fConstN = 0;
   fExtraOpts = nullptr;
}

//_______________________________________________________________________
void ScipyMinimizer::PyInitialize()
{
   static PyObject *ParamsError;
   static PyMethodDef ParamsMethods[] = {
      {"target_function", target_function, METH_VARARGS, "Target function to minimize."},
      {"jac_function", jac_function, METH_VARARGS, "Jacobian function."},
      {"hessian_function", hessian_function, METH_VARARGS, "Hessianfunction."},
      {NULL, NULL, 0, NULL} /* Sentinel */
   };

   static struct PyModuleDef paramsmodule = {
      PyModuleDef_HEAD_INIT,
      "params",                /* name of module */
      "ROOT Scipy parameters", /* module documentation, may be NULL */
      -1,                      /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
      ParamsMethods,
      NULL, /* m_slots */
      NULL, /* m_traverse */
      0,    /* m_clear */
      NULL  /* m_free */
   };

   auto PyInit_params = [](void) -> PyObject * {
      PyObject *m;

      m = PyModule_Create(&paramsmodule);
      if (m == NULL)
         return NULL;

      ParamsError = PyErr_NewException("params.error", NULL, NULL);
      Py_XINCREF(ParamsError);
      if (PyModule_AddObject(m, "error", ParamsError) < 0) {
         Py_XDECREF(ParamsError);
         Py_CLEAR(ParamsError);
         Py_DECREF(m);
         return NULL;
      }

      return m;
   };
   if (PyImport_AppendInittab("params", PyInit_params) == -1) {
      MATH_ERROR_MSG("ScipyMinimizer::Minimize", "could not extend in-built modules table");
      exit(1);
   }

   bool pyIsInitialized = PyIsInitialized();
   if (!pyIsInitialized) {
      Py_Initialize(); // Python initialization
   }
   fLocalNS = PyDict_New();
   fGlobalNS = PyDict_New();

   if (!pyIsInitialized) {
      _import_array(); // Numpy initialization
   }
   // Scipy initialization

   PyRunString("from scipy.optimize import minimize, Bounds");
   fMinimize = PyDict_GetItemString(fLocalNS, "minimize");
   fBoundsMod = PyDict_GetItemString(fLocalNS, "Bounds");
   PyRunString("from params import target_function, jac_function, hessian_function");
   fTarget = PyDict_GetItemString(fLocalNS, "target_function");
   fJacobian = PyDict_GetItemString(fLocalNS, "jac_function");
   fHessian = PyDict_GetItemString(fLocalNS, "hessian_function");
}

//_______________________________________________________________________
// Finalize Python interpreter
void ScipyMinimizer::PyFinalize()
{
   Py_Finalize();
}

//_______________________________________________________________________
int ScipyMinimizer::PyIsInitialized()
{
   return Py_IsInitialized();
}

//_______________________________________________________________________
ScipyMinimizer::~ScipyMinimizer()
{
   if (fMinimize)
      Py_DECREF(fMinimize);
   if (fBoundsMod)
      Py_DECREF(fBoundsMod);
}

//_______________________________________________________________________
void ScipyMinimizer::SetExtraOptions()
{
   auto constExtraOpts = dynamic_cast<const GenAlgoOptions *>(fOptions.ExtraOptions());
   fExtraOpts = const_cast<GenAlgoOptions *>(constExtraOpts);
}

//_______________________________________________________________________
bool ScipyMinimizer::Minimize()
{
   (gFunction) = ObjFunction();
   (gGradFunction) = GradObjFunction();
   gfHessianFunction = fHessianFunc;
   if (gGradFunction == nullptr) {
      fJacobian = Py_None;
   }
   if (gfHessianFunction == nullptr) {
      fHessian = Py_None;
   }
   auto method = fOptions.MinimizerAlgorithm();
   PyObject *pyoptions = PyDict_New();
   SetExtraOptions();
   if (fExtraOpts) {
      for (std::string key : fExtraOpts->GetAllRealKeys()) {
         double value = 0;
         fExtraOpts->GetRealValue(key.c_str(), value);
         PyDict_SetItemString(pyoptions, key.c_str(), PyFloat_FromDouble(value));
      }
      for (std::string key : fExtraOpts->GetAllIntKeys()) {
         int value = 0;
         fExtraOpts->GetIntValue(key.c_str(), value);
         PyDict_SetItemString(pyoptions, key.c_str(), PyLong_FromLong(value));
      }
      for (std::string key : fExtraOpts->GetAllNamedKeys()) {
         std::string value = "";
         fExtraOpts->GetNamedValue(key.c_str(), value);
         PyDict_SetItemString(pyoptions, key.c_str(), PyUnicode_FromString(value.c_str()));
      }
   }
   PyDict_SetItemString(pyoptions, "maxiter", PyLong_FromLong(MaxIterations()));
   if (PrintLevel() > 0) {
      PyDict_SetItemString(pyoptions, "disp", Py_True);
   }
   std::cout << "=== Scipy Minimization" << std::endl;
   std::cout << "=== Method: " << method << std::endl;
   std::cout << "=== Initial value: (";
   for (uint i = 0; i < NDim(); i++) {
      std::cout << X()[i];
      if (i < NDim() - 1)
         std::cout << ",";
   }
   std::cout << ")" << std::endl;

   double *values = const_cast<double *>(X());
   npy_intp dims[1] = {NDim()};
   PyObject *x0 = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, values);

   PyObject *pybounds_args = PyTuple_New(2);
   PyObject *pylimits_lower = PyList_New(NDim());
   PyObject *pylimits_upper = PyList_New(NDim());
   bool foundBounds = false;
   for (unsigned int i = 0; i < NDim(); i++) {
      ParameterSettings varsettings;

      if (GetVariableSettings(i, varsettings)) {
         if (varsettings.HasLowerLimit()) {
            foundBounds = true;
            PyList_SetItem(pylimits_lower, i, PyFloat_FromDouble(varsettings.LowerLimit()));
         } else {
            PyList_SetItem(pylimits_lower, i, PyFloat_FromDouble(-NPY_INFINITY));
         }
         if (varsettings.HasUpperLimit()) {
            foundBounds = true;
            PyList_SetItem(pylimits_upper, i, PyFloat_FromDouble(varsettings.UpperLimit()));
         } else {
            PyList_SetItem(pylimits_upper, i, PyFloat_FromDouble(NPY_INFINITY));
         }
      } else {
         MATH_ERROR_MSG("ScipyMinimizer::Minimize", Form("Variable index = %d not found", i));
      }
   }
   PyObject *pybounds = Py_None;
   if (foundBounds) {
      PyTuple_SetItem(pybounds_args, 0, pylimits_lower);
      PyTuple_SetItem(pybounds_args, 1, pylimits_upper);
      pybounds = PyObject_CallObject(fBoundsMod, pybounds_args);
   }

   // minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None,
   // callback=None, options=None)
   PyObject *args = Py_BuildValue("(OO)", fTarget, x0);
   PyObject *kw =
      Py_BuildValue("{s:s,s:O,,s:O,s:O,s:O,s:d,s:O}", "method", method.c_str(), "jac", fJacobian, "hess", fHessian,
                    "bounds", pybounds, "constraints", fConstraintsList, "tol", Tolerance(), "options", pyoptions);
   if (PrintLevel() > 0) {
      std::cout << "========Minimizer Parameters========\n";
      PyPrint(kw);
      std::cout << "====================================\n";
   }
   PyObject *result = PyObject_Call(fMinimize, args, kw);
   if (result == NULL) {
      PyErr_Print();
      return false;
   }
   if (PrintLevel() > 0) {
      std::cout << "========Minimizer Results========\n";
      PyPrint(result);
      std::cout << "=================================\n";
   }
   Py_DECREF(pybounds);
   Py_DECREF(args);
   Py_DECREF(kw);
   Py_DECREF(x0);
   Py_DECREF(fConstraintsList);
   fConstraintsList = PyList_New(0);

   // if the minimization works
   PyObject *pstatus = PyObject_GetAttrString(result, "status");
   int status = PyLong_AsLong(pstatus);
   Py_DECREF(pstatus);

   PyObject *psuccess = PyObject_GetAttrString(result, "success");
   bool success = PyLong_AsLong(psuccess);
   Py_DECREF(psuccess);

   // the x values for the minimum
   PyArrayObject *pyx = (PyArrayObject *)PyObject_GetAttrString(result, "x");
   const double *x = (const double *)PyArray_DATA(pyx);
   Py_DECREF(pyx);

   // number of function evaluations
   PyObject *pynfev = PyObject_GetAttrString(result, "nfev");
   long nfev = PyLong_AsLong(pynfev);
   Py_DECREF(pynfev);

   PyObject *pymessage = PyObject_GetAttrString(result, "message");
   const char *message = (const char *)PyUnicode_DATA(pymessage);
   Py_DECREF(pymessage);

   SetFinalValues(x);
   auto obj_value = (*gFunction)(x);
   SetMinValue(obj_value);
   fCalls = nfev; // number of function evaluations
   std::cout << "=== Success: " << success << std::endl;
   std::cout << "=== Status: " << status << std::endl;
   std::cout << "=== Message: " << message << std::endl;
   std::cout << "=== Function calls: " << nfev << std::endl;
   if (success)
      fStatus = 0;
   else
      fStatus = status; // suggested by Lorenzo.

   Py_DECREF(result);
   return success;
}

//_______________________________________________________________________
void ScipyMinimizer::PyRunString(TString code, TString errorMessage, int start)
{
   auto fPyReturn = PyRun_String(code, start, fGlobalNS, fLocalNS);
   if (!fPyReturn) {
      auto msg = errorMessage + Form(" \ncode = \"%s\"", code.Data());
      MATH_ERROR_MSG("ScipyMinimizer::PyRunString", msg.Data());
      PyErr_Print();
      exit(1);
   }
}

//_______________________________________________________________________
void ScipyMinimizer::SetHessianFunction(std::function<bool(std::span<const double>, double *)> func)
{
   fHessianFunc = func;
}

//_______________________________________________________________________
unsigned int ScipyMinimizer::NCalls() const
{
   return fCalls;
}

//_______________________________________________________________________
void ScipyMinimizer::AddConstraintFunction(std::function<double(const std::vector<double> &)> func, std::string type)
{
   if (type != "eq" && type != "ineq") {
      MATH_ERROR_MSG("ScipyMinimizer::AddConstraintFunction",
                     Form("Error in constraint type %s, it have to be \"eq\" or \"ineq\"", type.c_str()));
      exit(1);
   }
   static std::function<double(const std::vector<double> &)> cfunt = func;
   auto const_function = [](PyObject * /*self*/, PyObject *args) -> PyObject * {
      PyArrayObject *arr = (PyArrayObject *)PyTuple_GetItem(args, 0);

      uint size = PyArray_SIZE(arr);
      auto params = (double *)PyArray_DATA(arr);
      std::vector<double> x(params, params + size);
      auto r = cfunt(x);
      return PyFloat_FromDouble(r);
   };

   static const char *name = Form("const_function%d", fConstN);
   static const char *name_error = Form("const_function%d.error", fConstN);
   static PyObject *ConstError;
   static PyMethodDef ConstMethods[] = {
      {name, const_function, METH_VARARGS, "Constraint function to minimize."}, {NULL, NULL, 0, NULL} /* Sentinel */
   };
   static struct PyModuleDef constmodule = {
      PyModuleDef_HEAD_INIT,
      name,                    /* name of module */
      "ROOT Scipy parameters", /* module documentation, may be NULL */
      -1,                      /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
      ConstMethods,
      NULL, /* m_slots */
      NULL, /* m_traverse */
      0,    /* m_clear */
      NULL  /* m_free */
   };

   auto PyInit_const = [](void) -> PyObject * {
      PyObject *m;

      m = PyModule_Create(&constmodule);
      if (m == NULL)
         return NULL;
      ConstError = PyErr_NewException(name_error, NULL, NULL);
      Py_XINCREF(ConstError);
      if (PyModule_AddObject(m, "error", ConstError) < 0) {
         Py_XDECREF(ConstError);
         Py_CLEAR(ConstError);
         Py_DECREF(m);
         return NULL;
      }
      return m;
   };
   PyImport_AddModule(name);
   PyObject *module = PyInit_const();
   PyObject *sys_modules = PyImport_GetModuleDict();
   PyDict_SetItemString(sys_modules, name, module);

   PyRunString(Form("from %s import %s", name, name));
   PyObject *pyconstfun = PyDict_GetItemString(fLocalNS, name);

   PyObject *pyconst = PyDict_New();
   PyDict_SetItemString(pyconst, "type", PyUnicode_FromString(type.c_str()));
   PyDict_SetItemString(pyconst, "fun", pyconstfun);
   PyList_Append(fConstraintsList, pyconst);

   Py_DECREF(ConstError);
   Py_DECREF(module);
}
