// Author: Wim Lavrijsen, Jul 2004

#ifndef PYROOT_PYTHONIZE_H
#define PYROOT_PYTHONIZE_H

// Standard
#include <string>


namespace PyROOT {

// make the named ROOT class more python-like
   bool pythonize( PyObject* pyclass, const std::string& name );

}

#endif // !PYROOT_PYTHONIZE_H
