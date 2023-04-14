#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h> // Needs to be included first to avoid redefinition of _POSIX_C_SOURCE
#include <Math/ScipyMinimizer.h>
#include <Fit/ParameterSettings.h>
#include <Math/IFunction.h>
#include <Math/FitMethodFunction.h>
#include <Math/GenAlgoOptions.h>
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
   fOptions.SetMinimizerType("Scipy");
   fOptions.SetMinimizerAlgorithm("L-BFGS-B");
   PyInitialize();
   fHessianFunc = [](const std::vector<double> &, double *) -> bool { return false; };
   // set extra options
   SetAlgoExtraOptions();
}

//_______________________________________________________________________
ScipyMinimizer::ScipyMinimizer(const char *type)
{
   fOptions.SetMinimizerType("Scipy");
   fOptions.SetMinimizerAlgorithm(type);
   PyInitialize();
   fHessianFunc = [](const std::vector<double> &, double *) -> bool { return false; };
   // set extra options
   SetAlgoExtraOptions();
}

//_______________________________________________________________________
void ScipyMinimizer::SetAlgoExtraOptions()
{
   std::string type = fOptions.MinimizerAlgorithm();
   SetExtraOptions(fExtraOpts);
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

   static struct PyModuleDef paramsmodule = {PyModuleDef_HEAD_INIT, "params", /* name of module */
                                             "ROOT Scipy parameters",         /* module documentation, may be NULL */
                                             -1, /* size of per-interpreter state of the module,
                                       or -1 if the module keeps state in global variables. */
                                             ParamsMethods,
                                             NULL,  /* m_slots */
                                             NULL,  /* m_traverse */
                                             0,     /* m_clear */
                                             NULL   /* m_free */
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
bool ScipyMinimizer::Minimize()
{
   (gFunction) = ObjFunction();
   (gGradFunction) = GradObjFunction();
   gfHessianFunction = fHessianFunc;
   if (gGradFunction == nullptr) {
      fJacobian = Py_None;
   }
   if (!gfHessianFunction) {
      fHessian = Py_None;
   }
   auto method = fOptions.MinimizerAlgorithm();
   PyObject *pyoptions = PyDict_New();
   if (method == "L-BFGS-B") {
      for (std::string key : fExtraOpts.GetAllRealKeys()) {
         double value = 0;
         fExtraOpts.GetRealValue(key.c_str(), value);
         PyDict_SetItemString(pyoptions, key.c_str(), PyFloat_FromDouble(value));
      }
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
   for (unsigned int i = 0; i < NDim(); i++) {
      ParameterSettings varsettings;

      if (GetVariableSettings(i, varsettings)) {
         if (varsettings.HasLowerLimit()) {
            PyList_SetItem(pylimits_lower, i, PyFloat_FromDouble(varsettings.LowerLimit()));
         } else {
            PyList_SetItem(pylimits_lower, i, PyFloat_FromDouble(-NPY_INFINITY));
         }
         if (varsettings.HasUpperLimit()) {
            PyList_SetItem(pylimits_upper, i, PyFloat_FromDouble(varsettings.UpperLimit()));
         } else {
            PyList_SetItem(pylimits_upper, i, PyFloat_FromDouble(NPY_INFINITY));
         }
      } else {
         MATH_ERROR_MSG("ScipyMinimizer::Minimize", Form("Variable index = %d not found", i));
      }
   }
   PyTuple_SetItem(pybounds_args, 0, pylimits_lower);
   PyTuple_SetItem(pybounds_args, 1, pylimits_upper);

   PyObject *pybounds = PyObject_CallObject(fBoundsMod, pybounds_args);

   // minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None,
   // callback=None, options=None)
   PyObject *args = Py_BuildValue("(OO)", fTarget, x0);
   PyObject *kw = Py_BuildValue("{s:s,s:O,,s:O,s:O,s:d,s:O}", "method", method.c_str(), "jac", fJacobian, "hess",
                                fHessian, "bounds", pybounds, "tol", Tolerance(), "options", pyoptions);

   PyObject *result = PyObject_Call(fMinimize, args, kw);
   if (result == NULL) {
      PyErr_Print();
      return false;
   }
   // PyPrint(result);
   Py_DECREF(pylimits_lower);
   Py_DECREF(pylimits_upper);
   Py_DECREF(pybounds);
   Py_DECREF(args);
   Py_DECREF(kw);
   Py_DECREF(x0);

   // if the minimization works
   PyObject *pstatus = PyObject_GetAttrString(result, "status");
   bool status = PyBool_Check(pstatus);
   Py_DECREF(pstatus);

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

   std::cout << "=== Status: " << status << std::endl;
   std::cout << "=== Message: " << message << std::endl;
   std::cout << "=== Function calls: " << nfev << std::endl;
   return status;
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
void ScipyMinimizer::SetHessianFunction(std::function<bool(const std::vector<double> &, double *)> func)
{
   fHessianFunc = func;
}
