// Bindings
#include "CPyCppyy.h"
#include "CustomPyTypes.h"
#include "CPPInstance.h"
#include "Converters.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"

// As of Python 3.12, we can't use the PyMethod_GET_FUNCTION and
// PyMethod_GET_SELF macros anymore, as the contain asserts that check if the
// Python type is actually PyMethod_Type. If the Python type is
// CustomInstanceMethod_Type, we need our own macros. Technically they do they
// same, because the actual C++ type of the PyObject is PyMethodObject anyway.
#define CustomInstanceMethod_GET_SELF(meth) reinterpret_cast<PyMethodObject *>(meth)->im_self
#define CustomInstanceMethod_GET_FUNCTION(meth) reinterpret_cast<PyMethodObject *>(meth)->im_func
#if PY_VERSION_HEX >= 0x03000000
// TODO: this will break functionality
#define CustomInstanceMethod_GET_CLASS(meth) Py_None
#else
#define CustomInstanceMethod_GET_CLASS(meth) PyMethod_GET_CLASS(meth)
#endif

namespace CPyCppyy {

#if PY_VERSION_HEX < 0x03000000
//= float type allowed for reference passing =================================
PyTypeObject RefFloat_Type = {     // python float is a C/C++ double
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.Double",         // tp_name
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES |
        Py_TPFLAGS_BASETYPE,       // tp_flags
    (char*)"CPyCppyy float object for pass by reference",   // tp_doc
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    &PyFloat_Type,                 // tp_base
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#if PY_VERSION_HEX >= 0x02030000
    , 0                            // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                            // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                            // tp_finalize
#endif
};

//= long type allowed for reference passing ==================================
PyTypeObject RefInt_Type = {       // python int is a C/C++ long
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.Long",           // tp_name
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES |
        Py_TPFLAGS_BASETYPE
#if PY_VERSION_HEX >= 0x03040000
        | Py_TPFLAGS_LONG_SUBCLASS
#endif
        ,                          // tp_flags
    (char*)"CPyCppyy long object for pass by reference",    // tp_doc
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    &PyInt_Type,                   // tp_base
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#if PY_VERSION_HEX >= 0x02030000
    , 0                            // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                            // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                            // tp_finalize
#endif
};
#endif

//- custom type representing typedef to pointer of class ---------------------
static PyObject* tptc_call(typedefpointertoclassobject* self, PyObject* args, PyObject* /* kwds */)
{
    long long addr = 0;
    if (!PyArg_ParseTuple(args, const_cast<char*>("|L"), &addr))
        return nullptr;
    return BindCppObjectNoCast((Cppyy::TCppObject_t)(intptr_t)addr, self->fCppType);
}

//-----------------------------------------------------------------------------
static PyObject* tptc_getcppname(typedefpointertoclassobject* self, void*)
{
    return CPyCppyy_PyText_FromString(
        (Cppyy::GetScopedFinalName(self->fCppType)+"*").c_str());
}

//-----------------------------------------------------------------------------
static PyObject* tptc_name(typedefpointertoclassobject* self, void*)
{
    PyObject* pyclass = CPyCppyy::GetScopeProxy(self->fCppType);
    if (pyclass) {
        PyObject* pyname = PyObject_GetAttr(pyclass, PyStrings::gName);
        Py_DECREF(pyclass);
        return pyname;
    }

    return CPyCppyy_PyText_FromString("<unknown>*");
}

//-----------------------------------------------------------------------------
static PyGetSetDef tptc_getset[] = {
    {(char*)"__name__",     (getter)tptc_name, nullptr, nullptr, nullptr},
    {(char*)"__cpp_name__", (getter)tptc_getcppname, nullptr, nullptr, nullptr},
    {(char*)nullptr, nullptr, nullptr, nullptr, nullptr}
};


PyTypeObject TypedefPointerToClass_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.TypedefPointerToClass",// tp_name
    sizeof(typedefpointertoclassobject), // tp_basicsize
    0,                              // tp_itemsize
    0,                              // tp_dealloc
    0,                              // tp_vectorcall_offset
    0,                              // tp_getattr
    0,                              // tp_setattr
    0,                              // tp_as_async
    0,                              // tp_repr
    0,                              // tp_as_number
    0,                              // tp_as_sequence
    0,                              // tp_as_mapping
    0,                              // tp_hash
    (ternaryfunc)tptc_call,         // tp_call
    0,                              // tp_str
    PyObject_GenericGetAttr,        // tp_getattro
    PyObject_GenericSetAttr,        // tp_setattro
    0,                              // tp_as_buffer
    Py_TPFLAGS_DEFAULT,             // tp_flags
    0,                              // tp_doc
    0,                              // tp_traverse
    0,                              // tp_clear
    0,                              // tp_richcompare
    0,                              // tp_weaklistoffset
    0,                              // tp_iter
    0,                              // tp_iternext
    0,                              // tp_methods
    0,                              // tp_members
    tptc_getset,                    // tp_getset
    0,                              // tp_base
    0,                              // tp_dict
    0,                              // tp_descr_get
    0,                              // tp_descr_set
    offsetof(typedefpointertoclassobject, fDict), // tp_dictoffset
    0,                              // tp_init
    0,                              // tp_alloc
    0,                              // tp_new
    0,                              // tp_free
    0,                              // tp_is_gc
    0,                              // tp_bases
    0,                              // tp_mro
    0,                              // tp_cache
    0,                              // tp_subclasses
    0                               // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
    , 0                           // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                           // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                           // tp_finalize
#endif
#if PY_VERSION_HEX >= 0x03080000
    , 0                           // tp_vectorcall
#endif
#if PY_VERSION_HEX >= 0x030c0000
    , 0                           // tp_watched
#endif
#if PY_VERSION_HEX >= 0x030d0000
    , 0                           // tp_versions_used
#endif
};

//= instancemethod object with a more efficient call function ================
static PyMethodObject* free_list;
static int numfree = 0;
#ifndef PyMethod_MAXFREELIST
#define PyMethod_MAXFREELIST 256
#endif

//-----------------------------------------------------------------------------
PyObject* CustomInstanceMethod_New(PyObject* func, PyObject* self, PyObject*
#if PY_VERSION_HEX < 0x03000000
        pyclass
#endif
    )
{
// from instancemethod, but with custom type (at issue is that instancemethod is not
// meant to be derived from)
    PyMethodObject* im;
    if (!PyCallable_Check(func)) {
        PyErr_Format(PyExc_SystemError,
            "%s:%d: bad argument to internal function", __FILE__, __LINE__);
        return nullptr;
    }

    im = free_list;
    if (im != nullptr) {
        free_list = (PyMethodObject*)(im->im_self);
        (void)PyObject_INIT(im, &CustomInstanceMethod_Type);
    }
    else {
        im = PyObject_GC_New(PyMethodObject, &CustomInstanceMethod_Type);
        if (im == nullptr)
            return nullptr;
    }

    im->im_weakreflist = nullptr;
    Py_INCREF(func);
    im->im_func = func;
    Py_XINCREF(self);
    im->im_self = self;
#if PY_VERSION_HEX < 0x03000000
    Py_XINCREF(pyclass);
    im->im_class = pyclass;
#endif
    PyObject_GC_Track(im);
    return (PyObject*)im;
}

//-----------------------------------------------------------------------------
static void im_dealloc(PyMethodObject* im)
{
// from instancemethod, but with custom type (at issue is that instancemethod is not
// meant to be derived from)
    PyObject_GC_UnTrack(im);

    if (im->im_weakreflist != nullptr)
        PyObject_ClearWeakRefs((PyObject*)im);

    Py_DECREF(im->im_func);
    Py_XDECREF(im->im_self);
#if PY_VERSION_HEX < 0x03000000
    Py_XDECREF(im->im_class);
#endif

    if (numfree < PyMethod_MAXFREELIST) {
        im->im_self = (PyObject*)free_list;
        free_list = im;
        numfree++;
    } else {
        PyObject_GC_Del(im);
    }
}

//-----------------------------------------------------------------------------
static PyObject* im_call(PyObject* meth, PyObject* args, PyObject* kw)
{
// The mapping from a method to a function involves reshuffling of self back
// into the list of arguments. However, the pythonized methods will then have
// to undo that shuffling, which is inefficient. This method is the same as
// the one for the instancemethod object, except for the shuffling.
    PyObject* self = CustomInstanceMethod_GET_SELF(meth);

    if (!self) {
    // unbound methods must be called with an instance of the class (or a
    // derived class) as first argument
        Py_ssize_t argc = PyTuple_GET_SIZE(args);
        PyObject* pyclass = CustomInstanceMethod_GET_CLASS(meth);
        if (1 <= argc && PyObject_IsInstance(PyTuple_GET_ITEM(args, 0), pyclass) == 1) {
            self = PyTuple_GET_ITEM(args, 0);

            PyObject* newArgs = PyTuple_New(argc-1);
            for (int i = 1; i < argc; ++i) {
                PyObject* v = PyTuple_GET_ITEM(args, i);
                Py_INCREF(v);
                PyTuple_SET_ITEM(newArgs, i-1, v);
            }

            args = newArgs;

        } else
            return PyMethod_Type.tp_call(meth, args, kw);   // will set proper error msg

    } else
        Py_INCREF(args);

    PyCFunctionObject* func = (PyCFunctionObject*)CustomInstanceMethod_GET_FUNCTION(meth);

// the function is globally shared, so set and reset its "self" (ok, b/c of GIL)
    Py_INCREF(self);
    func->m_self = self;
    PyObject* result = CPyCppyy_PyCFunction_Call((PyObject*)func, args, kw);
    func->m_self = nullptr;
    Py_DECREF(self);
    Py_DECREF(args);
    return result;
}

//-----------------------------------------------------------------------------
static PyObject* im_descr_get(PyObject* meth, PyObject* obj, PyObject* pyclass)
{
// from instancemethod: don't rebind an already bound method, or an unbound method
// of a class that's not a base class of pyclass
    if (CustomInstanceMethod_GET_SELF(meth)
#if PY_VERSION_HEX < 0x03000000
         || (CustomInstanceMethod_GET_CLASS(meth) &&
             !PyObject_IsSubclass(pyclass, CustomInstanceMethod_GET_CLASS(meth)))
#endif
            ) {
        Py_INCREF(meth);
        return meth;
    }

    if (obj == Py_None)
        obj = nullptr;

    return CustomInstanceMethod_New(CustomInstanceMethod_GET_FUNCTION(meth), obj, pyclass);
}

//= CPyCppyy custom instance method type =====================================
PyTypeObject CustomInstanceMethod_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.InstanceMethod", // tp_name
    0, 0,
    (destructor)im_dealloc,        // tp_dealloc
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    im_call,                       // tp_call
    0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES |
        Py_TPFLAGS_BASETYPE,       // tp_flags
    (char*)"CPyCppyy custom instance method (internal)",    // tp_doc
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    &PyMethod_Type,                // tp_base
    0,
    im_descr_get,                  // tp_descr_get
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#if PY_VERSION_HEX >= 0x02030000
    , 0                            // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                            // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                            // tp_finalize
#endif
#if PY_VERSION_HEX >= 0x03080000
    , 0                           // tp_vectorcall
#endif
#if PY_VERSION_HEX >= 0x030c0000
    , 0                           // tp_watched
#endif
#if PY_VERSION_HEX >= 0x030d0000
    , 0                           // tp_versions_used
#endif
};


//= CPyCppyy custom iterator for performance =================================
static void indexiter_dealloc(indexiterobject* ii) {
    PyObject_GC_UnTrack(ii);
    Py_XDECREF(ii->ii_container);
    PyObject_GC_Del(ii);
}

static int indexiter_traverse(indexiterobject* ii, visitproc visit, void* arg) {
    Py_VISIT(ii->ii_container);
    return 0;
}

static PyObject* indexiter_iternext(indexiterobject* ii) {
    if (ii->ii_pos >= ii->ii_len)
        return nullptr;

    PyObject* pyindex = PyLong_FromSsize_t(ii->ii_pos);
    PyObject* result = PyObject_CallMethodOneArg((PyObject*)ii->ii_container, PyStrings::gGetItem, pyindex);
    Py_DECREF(pyindex);

    ii->ii_pos += 1;
    return result;
}

PyTypeObject IndexIter_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.indexiter",     // tp_name
    sizeof(indexiterobject),      // tp_basicsize
    0,
    (destructor)indexiter_dealloc,     // tp_dealloc
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_HAVE_GC,       // tp_flags
    0,
    (traverseproc)indexiter_traverse,  // tp_traverse
    0, 0, 0,
    PyObject_SelfIter,            // tp_iter
    (iternextfunc)indexiter_iternext,  // tp_iternext
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#if PY_VERSION_HEX >= 0x02030000
    , 0                           // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                           // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                           // tp_finalize
#endif
#if PY_VERSION_HEX >= 0x03080000
    , 0                           // tp_vectorcall
#endif
#if PY_VERSION_HEX >= 0x030c0000
    , 0                           // tp_watched
#endif
#if PY_VERSION_HEX >= 0x030d0000
    , 0                           // tp_versions_used
#endif
};


static void vectoriter_dealloc(vectoriterobject* vi) {
    if (vi->vi_converter && vi->vi_converter->HasState()) delete vi->vi_converter;
    indexiter_dealloc(vi);
}

static PyObject* vectoriter_iternext(vectoriterobject* vi) {
    if (vi->ii_pos >= vi->ii_len)
        return nullptr;

    PyObject* result = nullptr;

    if (vi->vi_data && vi->vi_converter) {
        void* location = (void*)((ptrdiff_t)vi->vi_data + vi->vi_stride * vi->ii_pos);
        result = vi->vi_converter->FromMemory(location);
    } else if (vi->vi_data && vi->vi_klass) {
    // The CPPInstance::kNoMemReg by-passes the memory regulator; the assumption here is
    // that objects in vectors are simple and thus do not need to maintain object identity
    // (or at least not during the loop anyway). This gains 2x in performance.
        Cppyy::TCppObject_t cppobj = (Cppyy::TCppObject_t)((ptrdiff_t)vi->vi_data + vi->vi_stride * vi->ii_pos);
        if (vi->vi_flags & vectoriterobject::kIsPolymorphic)
            result = CPyCppyy::BindCppObject(*(void**)cppobj, vi->vi_klass, CPyCppyy::CPPInstance::kNoMemReg);
        else
            result = CPyCppyy::BindCppObjectNoCast(cppobj, vi->vi_klass, CPyCppyy::CPPInstance::kNoMemReg);
        if ((vi->vi_flags & vectoriterobject::kNeedLifeLine) && result)
            PyObject_SetAttr(result, PyStrings::gLifeLine, vi->ii_container);
    } else {
        PyObject* pyindex = PyLong_FromSsize_t(vi->ii_pos);
        result = PyObject_CallMethodOneArg((PyObject*)vi->ii_container, PyStrings::gGetNoCheck, pyindex);
        Py_DECREF(pyindex);
    }

    vi->ii_pos += 1;
    return result;
}

PyTypeObject VectorIter_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.vectoriter",    // tp_name
    sizeof(vectoriterobject),     // tp_basicsize
    0,
    (destructor)vectoriter_dealloc,         // tp_dealloc
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_HAVE_GC,       // tp_flags
    0,
    (traverseproc)indexiter_traverse,  // tp_traverse
    0, 0, 0,
    PyObject_SelfIter,            // tp_iter
    (iternextfunc)vectoriter_iternext,      // tp_iternext
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#if PY_VERSION_HEX >= 0x02030000
    , 0                           // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                           // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                           // tp_finalize
#endif
#if PY_VERSION_HEX >= 0x03080000
    , 0                           // tp_vectorcall
#endif
#if PY_VERSION_HEX >= 0x030c0000
    , 0                           // tp_watched
#endif
#if PY_VERSION_HEX >= 0x030d0000
    , 0                           // tp_versions_used
#endif
};

} // namespace CPyCppyy
