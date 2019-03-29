#ifndef CPYCPPYY_CUSTOMPYTYPES_H
#define CPYCPPYY_CUSTOMPYTYPES_H

namespace CPyCppyy {

/** Custom "builtins," detectable by type, for pass by ref and improved
    performance.
 */

//- reference float object type and type verification ------------------------
extern PyTypeObject RefFloat_Type;

template<typename T>
inline bool RefFloat_Check(T* object)
{
    return object && PyObject_TypeCheck(object, &RefFloat_Type);
}

template<typename T>
inline bool RefFloat_CheckExact(T* object)
{
    return object && Py_TYPE(object) == &RefFloat_Type;
}

//- reference long object type and type verification -------------------------
extern PyTypeObject RefInt_Type;

template<typename T>
inline bool RefInt_Check(T* object)
{
    return object && PyObject_TypeCheck(object, &RefInt_Type);
}

template<typename T>
inline bool RefInt_CheckExact(T* object)
{
    return object && Py_TYPE(object) == &RefInt_Type;
}

//- custom instance method object type and type verification -----------------
extern PyTypeObject CustomInstanceMethod_Type;

template<typename T>
inline bool CustomInstanceMethod_Check(T* object)
{
    return object && PyObject_TypeCheck(object, &CustomInstanceMethod_Type);
}

template<typename T>
inline bool CustomInstanceMethod_CheckExact(T* object)
{
    return object && Py_TYPE(object) == &CustomInstanceMethod_Type;
}

PyObject* CustomInstanceMethod_New(PyObject* func, PyObject* self, PyObject* pyclass);

//- custom iterator for high performance std::vector iteration ---------------
struct indexiterobject {
    PyObject_HEAD
    PyObject*                ii_container;
    Py_ssize_t               ii_pos;
    Py_ssize_t               ii_len;
};

extern PyTypeObject IndexIter_Type;

class Converter;
struct vectoriterobject : public indexiterobject {
    CPyCppyy::Converter*     vi_converter;
    void*                    vi_data;
    Py_ssize_t               vi_stride;
};

extern PyTypeObject VectorIter_Type;

} // namespace CPyCppyy

#endif // !CPYCPPYY_CUSTOMPYTYPES_H
