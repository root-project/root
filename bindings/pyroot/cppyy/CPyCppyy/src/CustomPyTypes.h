#ifndef CPYCPPYY_CUSTOMPYTYPES_H
#define CPYCPPYY_CUSTOMPYTYPES_H

namespace CPyCppyy {

/** Custom "builtins," detectable by type, for pass by ref and improved
    performance.
 */

#if PY_VERSION_HEX < 0x03000000
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
#endif

//- custom type representing typedef to pointer of class ---------------------
struct typedefpointertoclassobject {
    PyObject_HEAD
    Cppyy::TCppType_t fCppType;
    PyObject*         fDict;

    ~typedefpointertoclassobject() {
        Py_DECREF(fDict);
    }
};

extern PyTypeObject TypedefPointerToClass_Type;

template<typename T>
inline bool TypedefPointerToClass_Check(T* object)
{
    return object && PyObject_TypeCheck(object, &TypedefPointerToClass_Type);
}

template<typename T>
inline bool TypedefPointerToClass_CheckExact(T* object)
{
    return object && Py_TYPE(object) == &TypedefPointerToClass_Type;
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
    void*                    vi_data;
    Py_ssize_t               vi_stride;
    CPyCppyy::Converter*     vi_converter;
    Cppyy::TCppType_t        vi_klass;
    int                      vi_flags;

    enum EFlags {
        kDefault        = 0x0000,
        kNeedLifeLine   = 0x0001,
        kIsPolymorphic  = 0x0002,
    };
};

extern PyTypeObject VectorIter_Type;

} // namespace CPyCppyy

#endif // !CPYCPPYY_CUSTOMPYTYPES_H
