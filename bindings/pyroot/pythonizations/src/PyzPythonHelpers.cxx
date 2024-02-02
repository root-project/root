// Author: Stefan Wunsch CERN  08/2018
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**

Set of helper functions that are invoked from the pythonizors, on the
Python side. For that purpose, they are included in the interface of the
PyROOT extension module.

*/

#include "CPyCppyy/API.h"

#include "PyROOTPythonize.h"

#include "ROOT/RConfig.hxx"
#include "TInterpreter.h"

#include <sstream>

////////////////////////////////////////////////////////////////////////////
/// \brief Get size of C++ data-type
/// \param[in] self Always null, since this is a module function.
/// \param[in] args C++ data-type as Python string
///
/// This function returns the length of a C++ data-type in bytes
/// as a Python integer.
PyObject *PyROOT::GetSizeOfType(PyObject * /*self*/, PyObject *args)
{
   // Get name of data-type
   PyObject *pydtype = PyTuple_GetItem(args, 0);
   std::string dtype = PyUnicode_AsUTF8(pydtype);

   // Call interpreter to get size of data-type using `sizeof`
   size_t size = 0;
   std::stringstream code;
   code << "*((size_t*)" << std::showbase << (uintptr_t)&size << ") = (size_t)sizeof(" << dtype << ")";
   gInterpreter->Calc(code.str().c_str());

   // Return size of data-type as integer
   PyObject *pysize = PyLong_FromLong(size);
   return pysize;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get pointer to the data of an object
/// \param[in] self Always null, since this is a module function.
/// \param[in] args [0] Python representation of the C++ object.
///                 [1] Data-type of the C++ object as Python string.
///                 [2] Method to be called on the C++ object to get the data pointer as Python string
///
/// This function returns the pointer to the data of an object as an Python
/// integer retrieved by the given method.
PyObject *PyROOT::GetDataPointer(PyObject * /*self*/, PyObject *args)
{
   // Get pointer of C++ object
   PyObject *pyobj = PyTuple_GetItem(args, 0);
   void* cppobj = CPyCppyy::Instance_AsVoidPtr(pyobj);

   // Get name of C++ object as string
   PyObject *pycppname = PyTuple_GetItem(args, 1);
   std::string cppname = PyUnicode_AsUTF8(pycppname);

   // Get name of method to be called to get the data pointer
   PyObject *pymethodname = PyTuple_GetItem(args, 2);
   std::string methodname = PyUnicode_AsUTF8(pymethodname);

   // Call interpreter to get pointer to data
   uintptr_t pointer = 0;
   std::stringstream code;
   code << "*((intptr_t*)" << std::showbase << (uintptr_t)&pointer << ") = reinterpret_cast<uintptr_t>(reinterpret_cast<"
        << cppname << "*>(" << std::showbase << (uintptr_t)cppobj << ")->" << methodname << "())";
   gInterpreter->Calc(code.str().c_str());

   // Return pointer as integer
   PyObject *pypointer = PyLong_FromUnsignedLongLong(pointer);
   return pypointer;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get endianess of the system
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to an empty Python tuple.
/// \return Endianess as Python string
///
/// This function returns endianess of the system as a Python integer. The
/// return value is either '<' or '>' for little or big endian, respectively.
PyObject *PyROOT::GetEndianess(PyObject * /* self */, PyObject * /* args */)
{
#ifdef R__BYTESWAP
   return PyUnicode_FromString("<");
#else
   return PyUnicode_FromString(">");
#endif
}
