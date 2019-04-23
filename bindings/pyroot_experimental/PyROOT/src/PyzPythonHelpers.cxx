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

#include "CPyCppyy.h"
#include "CPPInstance.h"
#include "Utility.h"

#include "PyROOTPythonize.h"

#include "RConfig.h"
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
   std::string dtype = CPyCppyy_PyUnicode_AsString(pydtype);

   // Call interpreter to get size of data-type using `sizeof`
   long size = 0;
   std::stringstream code;
   code << "*((long*)" << &size << ") = (long)sizeof(" << dtype << ")";
   gInterpreter->Calc(code.str().c_str());

   // Return size of data-type as integer
   PyObject *pysize = PyInt_FromLong(size);
   return pysize;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get pointer to the data of a vector
/// \param[in] self Always null, since this is a module function.
/// \param[in] args[0] Data-type of the C++ object as Python string
/// \param[in] args[1] Python representation of the C++ object.
///
/// This function returns the pointer to the data of a vector as an Python
/// integer.
PyObject *PyROOT::GetVectorDataPointer(PyObject * /*self*/, PyObject *args)
{
   // Get pointer of C++ object
   PyObject *pyobj = PyTuple_GetItem(args, 0);
   auto instance = (CPyCppyy::CPPInstance *)(pyobj);
   auto cppobj = instance->GetObject();

   // Get name of C++ object as string
   PyObject *pycppname = PyTuple_GetItem(args, 1);
   std::string cppname = CPyCppyy_PyUnicode_AsString(pycppname);

   // Call interpreter to get pointer to data (using `data` method)
   unsigned long long pointer = 0;
   std::stringstream code;
   code << "*((long*)" << &pointer << ") = reinterpret_cast<long>(reinterpret_cast<" << cppname << "*>(" << cppobj
        << ")->data())";
   gInterpreter->Calc(code.str().c_str());

   // Return pointer as integer
   PyObject *pypointer = PyLong_FromUnsignedLongLong(pointer);
   return pypointer;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get endianess of the system
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to an empty Python tuple.
/// \param[out] Endianess as Python string
///
/// This function returns endianess of the system as a Python integer. The
/// return value is either '<' or '>' for little or big endian, respectively.
PyObject *PyROOT::GetEndianess(PyObject * /* self */, PyObject * /* args */)
{
#ifdef R__BYTESWAP
   return CPyCppyy_PyUnicode_FromString("<");
#else
   return CPyCppyy_PyUnicode_FromString(">");
#endif
}

