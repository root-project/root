#include "CPyCppyyModule.h"

//----------------------------------------------------------------------------
#if PY_VERSION_HEX >= 0x03000000
extern "C" PyObject* PyInit_libcppyy() {
#else
extern "C" void initlibcppyy() {
#endif
    PyObject *thisModule = CPyCppyy::Init();
#if PY_VERSION_HEX >= 0x03000000
    return thisModule;
#endif
}
