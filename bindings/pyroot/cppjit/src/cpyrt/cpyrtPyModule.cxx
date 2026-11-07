#include "cpyrtModule.h"

//----------------------------------------------------------------------------
extern "C" PyObject* PyInit_libcppjit() { return cppjit::cpyrt::Init(); }
