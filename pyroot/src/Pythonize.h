// @(#)root/pyroot:$Name:  $:$Id: Pythonize.h,v 1.68 2005/01/28 05:45:41 brun Exp $
// Author: Wim Lavrijsen, Jul 2004

#ifndef PYROOT_PYTHONIZE_H
#define PYROOT_PYTHONIZE_H

// Standard
#include <string>


namespace PyROOT {

// make the named ROOT class more python-like
   bool Pythonize( PyObject* pyclass, const std::string& name );

} // namespace PyROOT

#endif // !PYROOT_PYTHONIZE_H
