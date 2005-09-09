// @(#)root/pyroot:$Name:  $:$Id: Pythonize.h,v 1.2 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen, Jul 2004

#ifndef PYROOT_PYTHONIZE_H
#define PYROOT_PYTHONIZE_H

// Standard
#include <string>


namespace PyROOT {

// make the named ROOT class more python-like
   Bool_t Pythonize( PyObject* pyclass, const std::string& name );

} // namespace PyROOT

#endif // !PYROOT_PYTHONIZE_H
