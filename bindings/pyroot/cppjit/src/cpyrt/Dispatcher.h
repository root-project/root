#ifndef CPYRT_DISPATCHER_H
#define CPYRT_DISPATCHER_H

// Standard
#include <sstream>

namespace cppjit::cpyrt {

class CPPScope;

// helper that inserts dispatchers for virtual methods
bool InsertDispatcher(CPPScope* klass, PyObject* bases, PyObject* dct,
                      std::ostringstream& err);

} // namespace cppjit::cpyrt

#endif // !CPYRT_DISPATCHER_H
