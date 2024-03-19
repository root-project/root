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
