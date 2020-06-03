#ifndef CPYCPPYY_DISPATCHER_H
#define CPYCPPYY_DISPATCHER_H

namespace CPyCppyy {

class CPPScope;

// helper that inserts dispatchers for virtual methods
bool InsertDispatcher(CPPScope* klass, PyObject* dct);

} // namespace CPyCppyy

#endif // !CPYCPPYY_DISPATCHER_H
