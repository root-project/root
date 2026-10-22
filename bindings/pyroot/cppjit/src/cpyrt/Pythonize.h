#ifndef CPYRT_PYTHONIZE_H
#define CPYRT_PYTHONIZE_H

// Standard
#include <string>

namespace cppjit::cpyrt {

// make the named C++ class more python-like
bool Pythonize(PyObject* pyclass, interop::TCppScope_t scope);

} // namespace cppjit::cpyrt

#endif // !CPYRT_PYTHONIZE_H
