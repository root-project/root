#ifndef CPYCPPYY_LOWLEVELVIEWS_H
#define CPYCPPYY_LOWLEVELVIEWS_H

#include <stddef.h>

namespace CPyCppyy {

PyObject* CreateLowLevelView(bool*,                   Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(unsigned char*,          Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(short*,                  Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(unsigned short*,         Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(int*,                    Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(unsigned int*,           Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(long*,                   Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(unsigned long*,          Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(long long*,              Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(unsigned long long*,     Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(float*,                  Py_ssize_t* shape = nullptr);
PyObject* CreateLowLevelView(double*,                 Py_ssize_t* shape = nullptr);

inline PyObject* CreatePointerView(void* ptr) {
    Py_ssize_t shape[] = {1, 1};
    return CreateLowLevelView((long*)ptr, shape);
}

//- low level view type and type verification --------------------------------
extern PyTypeObject LowLevelView_Type;

template<typename T>
inline bool LowLevelView_Check(T* object)
{
    return object && PyObject_TypeCheck(object, &LowLevelView_Type);
}

template<typename T>
inline bool LowLevelView_CheckExact(T* object)
{
    return object && Py_TYPE(object) == &LowLevelView_Type;
}

} // namespace CPyCppyy

#endif // !CPYCPPYY_LOWLEVELVIEWS_H
