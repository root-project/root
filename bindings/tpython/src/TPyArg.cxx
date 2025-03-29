// Author: Enric Tejedor CERN  08/2019
// Original PyROOT code by Wim Lavrijsen, LBL
//
// /*************************************************************************
//  * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
//  * All rights reserved.                                                  *
//  *                                                                       *
//  * For the licensing terms see $ROOTSYS/LICENSE.                         *
//  * For the list of contributors see $ROOTSYS/README/CREDITS.             *
//  *************************************************************************/

// Bindings
#include "Python.h"

#include "TPyArg.h"

//______________________________________________________________________________
//                        Generic wrapper for arguments
//                        =============================
//
// Transport class for bringing C++ values and objects from Cling to Python. It
// provides, from the selected constructor, the proper conversion to a PyObject.
// In principle, there should be no need to use this class directly: it relies
// on implicit conversions.

//- data ---------------------------------------------------------------------
ClassImp(TPyArg);

namespace {
   class PyGILRAII {
      PyGILState_STATE m_GILState;
   public:
      PyGILRAII() : m_GILState(PyGILState_Ensure()) { }
      ~PyGILRAII() { PyGILState_Release(m_GILState); }
   };
}

//- constructor dispatcher ---------------------------------------------------
void TPyArg::CallConstructor(PyObject *&pyself, PyObject *pyclass, const std::vector<TPyArg> &args)
{
   PyGILRAII gilRaii;

   int nArgs = args.size();
   PyObject *pyargs = PyTuple_New(nArgs);
   for (int i = 0; i < nArgs; ++i)
      PyTuple_SET_ITEM(pyargs, i, (PyObject *)args[i]);
   pyself = PyObject_Call(pyclass, pyargs, NULL);
   Py_DecRef(pyargs);
}

////////////////////////////////////////////////////////////////////////////////
void CallConstructor(PyObject *&pyself, PyObject *pyclass)
{
   PyGILRAII gilRaii;

   PyObject *pyargs = PyTuple_New(0);
   pyself = PyObject_Call(pyclass, pyargs, NULL);
   Py_DecRef(pyargs);
}

//- generic dispatcher -------------------------------------------------------
PyObject *TPyArg::CallMethod(PyObject *pymeth, const std::vector<TPyArg> &args)
{
   PyGILRAII gilRaii;

   int nArgs = args.size();
   PyObject *pyargs = PyTuple_New(nArgs);
   for (int i = 0; i < nArgs; ++i)
      PyTuple_SET_ITEM(pyargs, i, (PyObject *)args[i]);
   PyObject *result = PyObject_Call(pymeth, pyargs, NULL);
   Py_DecRef(pyargs);
   return result;
}

//- denstructor dispatcher ----------------------------------------------------
void TPyArg::CallDestructor(PyObject *&pyself, PyObject *, const std::vector<TPyArg> &)
{
   PyGILRAII gilRaii;

   Py_DecRef(pyself); // calls actual dtor if ref-count down to 0
}

////////////////////////////////////////////////////////////////////////////////
void TPyArg::CallDestructor(PyObject *&pyself)
{
   PyGILRAII gilRaii;

   Py_DecRef(pyself);
}

//- constructors/destructor --------------------------------------------------
TPyArg::TPyArg(PyObject *pyobject)
{
   PyGILRAII gilRaii;

   // Construct a TPyArg from a python object.
   Py_IncRef(pyobject);
   fPyObject = pyobject;
}

////////////////////////////////////////////////////////////////////////////////
/// Construct a TPyArg from an integer value.

TPyArg::TPyArg(Int_t value)
{
   PyGILRAII gilRaii;

   fPyObject = PyLong_FromLong(value);
}

////////////////////////////////////////////////////////////////////////////////
/// Construct a TPyArg from an integer value.

TPyArg::TPyArg(Long_t value)
{
   PyGILRAII gilRaii;

   fPyObject = PyLong_FromLong(value);
}

////////////////////////////////////////////////////////////////////////////////
/// Construct a TPyArg from a double value.

TPyArg::TPyArg(Double_t value)
{
   PyGILRAII gilRaii;

   fPyObject = PyFloat_FromDouble(value);
}

////////////////////////////////////////////////////////////////////////////////
/// Construct a TPyArg from a C-string.

TPyArg::TPyArg(const char *value)
{
   PyGILRAII gilRaii;

   fPyObject = PyUnicode_FromString(value);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TPyArg::TPyArg(const TPyArg &s)
{
   PyGILRAII gilRaii;

   Py_IncRef(s.fPyObject);
   fPyObject = s.fPyObject;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

TPyArg &TPyArg::operator=(const TPyArg &s)
{
   PyGILRAII gilRaii;

   if (&s != this) {
      Py_IncRef(s.fPyObject);
      fPyObject = s.fPyObject;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Done with held PyObject.

TPyArg::~TPyArg()
{
   PyGILRAII gilRaii;

   Py_DecRef(fPyObject);
   fPyObject = NULL;
}

//- public members -----------------------------------------------------------
TPyArg::operator PyObject *() const
{
   PyGILRAII gilRaii;

   // Extract the python object.
   Py_IncRef(fPyObject);
   return fPyObject;
}
