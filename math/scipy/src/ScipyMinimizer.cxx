#include <Python.h> // Needs to be included first to avoid redefinition of _POSIX_C_SOURCE
#include <Math/ScipyMinimizer.h>
#include <Fit/ParameterSettings.h>
#include <Math/IFunction.h>
#include <Math/FitMethodFunction.h>
#include <TString.h>
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace ROOT;
using namespace ROOT::Math;
using namespace ROOT::Math::Experimental;
using namespace ROOT::Fit;

/// function wrapper for the function to be minimized
const ROOT::Math::IMultiGenFunction *gFunction;
/// function wrapper for the gradient of the function to be minimized
const ROOT::Math::IMultiGradFunction *gGradFunction;

PyObject *target_function(PyObject * /*self*/, PyObject *args)
{
   PyArrayObject *arr = (PyArrayObject *)PyTuple_GetItem(args, 0);

   auto params = (double *)PyArray_DATA(arr);
   auto r = (*gFunction)(params);

   return PyFloat_FromDouble(r);
};

PyObject *jac_function(PyObject * /*self*/, PyObject *args)
{
   PyArrayObject *arr = (PyArrayObject *)PyTuple_GetItem(args, 0);

   uint size = PyArray_SIZE(arr);
   auto params = (double *)PyArray_DATA(arr);
   double values[size];
   gGradFunction->Gradient(params, values);
   npy_intp dims[1] = {size};
   PyObject *py_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, values);
   return py_array;
};

//_______________________________________________________________________
ScipyMinimizer::ScipyMinimizer() : BasicMinimizer()
{
   fOptions.SetMinimizerType("Scipy");
   fOptions.SetMinimizerAlgorithm("lbfgsb");
   if (!PyIsInitialized()) {
      PyInitialize();
   }
}

//_______________________________________________________________________
ScipyMinimizer::ScipyMinimizer(const char *type)
{
   fOptions.SetMinimizerType("Scipy");
   fOptions.SetMinimizerAlgorithm(type);
   if (!PyIsInitialized()) {
      PyInitialize();
   }
}

//_______________________________________________________________________
void ScipyMinimizer::PyInitialize()
{
   static PyObject *ParamsError;
   static PyMethodDef ParamsMethods[] = {
      {"target_function", target_function, METH_VARARGS, "Target function to minimize."},
      {"jac_function", jac_function, METH_VARARGS, "Jacobian function."},
      {NULL, NULL, 0, NULL} /* Sentinel */
   };

   static struct PyModuleDef paramsmodule = {PyModuleDef_HEAD_INIT, "params", /* name of module */
                                             "ROOT Scipy parameters",         /* module documentation, may be NULL */
                                             -1, /* size of per-interpreter state of the module,
                                       or -1 if the module keeps state in global variables. */
                                             ParamsMethods};

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
   PyRunString("from scipy.optimize import minimize");
   fMinimize = PyDict_GetItemString(fLocalNS, "minimize");
   PyRunString("from params import target_function, jac_function");
   fTarget = PyDict_GetItemString(fLocalNS, "target_function");
   fJacobian = PyDict_GetItemString(fLocalNS, "jac_function");
}

//_______________________________________________________________________
// Finalize Python interpreter
void ScipyMinimizer::PyFinalize()
{
   if (fMinimize)
      Py_DECREF(fMinimize);
   Py_Finalize();
}

//_______________________________________________________________________
int ScipyMinimizer::PyIsInitialized()
{
   if (!Py_IsInitialized())
      return kFALSE;
   return kTRUE;
}

//_______________________________________________________________________
ScipyMinimizer::~ScipyMinimizer() {}

//_______________________________________________________________________
bool ScipyMinimizer::Minimize()
{
   (gFunction) = ObjFunction();
   (gGradFunction) = GradObjFunction();
   auto method = fOptions.MinimizerAlgorithm();
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
   PyObject *py_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, values);

   PyObject *pargs = PyTuple_New(0);

   auto pyvalues = Py_BuildValue("(OOOsO)", fTarget, py_array, pargs, method.c_str(), fJacobian);

   PyObject *result = PyObject_CallObject(fMinimize, pyvalues);
   Py_DECREF(pyvalues);
   Py_DECREF(py_array);

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
