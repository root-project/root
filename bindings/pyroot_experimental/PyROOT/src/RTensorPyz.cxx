#include "CPyCppyy.h"
#include "PyROOTPythonize.h"
#include "CPPInstance.h"
#include "Utility.h"

#include "TMVA/RTensor.hxx"

#include <string>
#include <vector>

using namespace CPyCppyy;

inline std::vector<size_t> GetIndicesFromTuple(PyObject *obj)
{
   std::vector<size_t> idx;
   for (unsigned int i = 0; i < PyTuple_Size(obj); i++)
      idx.push_back(PyInt_AsLong(PyTuple_GetItem(obj, i)));
   return idx;
}

template <typename dtype>
PyObject *RTensorGetItemFloat(CPPInstance *self, PyObject *obj)
{
   auto cobj = (TMVA::Experimental::RTensor<dtype> *)(self->GetObject());
   auto idx = GetIndicesFromTuple(obj);
   return PyFloat_FromDouble(cobj->At(idx));
}

template <typename dtype>
PyObject *RTensorGetItemInt(CPPInstance *self, PyObject *obj)
{
   auto cobj = (TMVA::Experimental::RTensor<dtype> *)(self->GetObject());
   auto idx = GetIndicesFromTuple(obj);
   return PyLong_FromLong(cobj->At(idx));
}

template <typename dtype>
PyObject *RTensorSetItemFloat(CPPInstance *self, PyObject *obj)
{
   auto cobj = (TMVA::Experimental::RTensor<dtype> *)(self->GetObject());
   std::vector<size_t> idx = GetIndicesFromTuple(PyTuple_GetItem(obj, 0));
   cobj->At(idx) = PyFloat_AsDouble(PyTuple_GetItem(obj, 1));
   Py_INCREF(Py_None);
   return Py_None;
}

template <typename dtype>
PyObject *RTensorSetItemInt(CPPInstance *self, PyObject *obj)
{
   auto cobj = (TMVA::Experimental::RTensor<dtype> *)(self->GetObject());
   std::vector<size_t> idx = GetIndicesFromTuple(PyTuple_GetItem(obj, 0));
   cobj->At(idx) = PyInt_AsLong(PyTuple_GetItem(obj, 1));
   Py_INCREF(Py_None);
   return Py_None;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pythonizations for __getitem__ and __setitem__
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// received from Python.
///
/// This functions adds the pythonizations for __getitem__ and __setitem__
/// so that elements of the object can be accessed in python with the []
/// operator.
PyObject *PyROOT::AddRTensorGetSetItem(PyObject * /* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);
   std::string dtype = CPyCppyy_PyUnicode_AsString(PyTuple_GetItem(args, 1));
   if (dtype == "float") {
      Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)RTensorGetItemFloat<float>, METH_O);
      Utility::AddToClass(pyclass, "__setitem__", (PyCFunction)RTensorSetItemFloat<float>, METH_VARARGS);
   } else if (dtype == "double") {
      Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)RTensorGetItemFloat<double>, METH_O);
      Utility::AddToClass(pyclass, "__setitem__", (PyCFunction)RTensorSetItemFloat<double>, METH_VARARGS);
   } else if (dtype == "int") {
      Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)RTensorGetItemInt<int>, METH_O);
      Utility::AddToClass(pyclass, "__setitem__", (PyCFunction)RTensorSetItemFloat<int>, METH_VARARGS);
   } else if (dtype == "long") {
      Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)RTensorGetItemInt<long>, METH_O);
      Utility::AddToClass(pyclass, "__setitem__", (PyCFunction)RTensorSetItemFloat<long>, METH_VARARGS);
   } else if (dtype == "unsigned int") {
      Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)RTensorGetItemInt<unsigned int>, METH_O);
      Utility::AddToClass(pyclass, "__setitem__", (PyCFunction)RTensorSetItemFloat<unsigned int>, METH_VARARGS);
   } else if (dtype == "unsigned long") {
      Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)RTensorGetItemInt<unsigned long>, METH_O);
      Utility::AddToClass(pyclass, "__setitem__", (PyCFunction)RTensorSetItemFloat<unsigned long>, METH_VARARGS);
   }
   Py_RETURN_NONE;
}
