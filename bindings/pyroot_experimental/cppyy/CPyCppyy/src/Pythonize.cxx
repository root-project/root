// Bindings
#include "CPyCppyy.h"
#include "Pythonize.h"
#include "Converters.h"
#include "CPPInstance.h"
#include "CPPOverload.h"
#include "ProxyWrappers.h"
#include "PyCallable.h"
#include "PyStrings.h"
#include "Utility.h"

// Standard
#include <stdexcept>
#include <string>
#include <utility>


//- data and local helpers ---------------------------------------------------
namespace CPyCppyy {
    extern PyObject* gThisModule;
    extern std::map<std::string, std::vector<PyObject*>> gPythonizations;
}

namespace {

// for convenience
using namespace CPyCppyy;

//-----------------------------------------------------------------------------
bool HasAttrDirect(PyObject* pyclass, PyObject* pyname, bool mustBeCPyCppyy = false) {
// prevents calls to Py_TYPE(pyclass)->tp_getattr, which is unnecessary for our
// purposes here and could tickle problems w/ spurious lookups into ROOT meta
    PyObject* attr = PyType_Type.tp_getattro(pyclass, pyname);
    if (attr != 0 && (!mustBeCPyCppyy || CPPOverload_Check(attr))) {
        Py_DECREF(attr);
        return true;
    }

    PyErr_Clear();
    return false;
}

//-----------------------------------------------------------------------------
inline bool IsTemplatedSTLClass(const std::string& name, const std::string& klass) {
// Scan the name of the class and determine whether it is a template instantiation.
    const int nsize = (int)name.size();
    const int ksize = (int)klass.size();

    return ((ksize   < nsize && name.substr(0,ksize) == klass) ||
            (ksize+5 < nsize && name.substr(5,ksize) == klass)) &&
            name.find("::", name.find(">")) == std::string::npos;
}

// to prevent compiler warnings about const char* -> char*
inline PyObject* CallPyObjMethod(PyObject* obj, const char* meth)
{
// Helper; call method with signature: obj->meth().
    Py_INCREF(obj);
    PyObject* result = PyObject_CallMethod(obj, const_cast<char*>(meth), const_cast<char*>(""));
    Py_DECREF(obj);
    return result;
}

//-----------------------------------------------------------------------------
inline PyObject* CallPyObjMethod(PyObject* obj, const char* meth, PyObject* arg1)
{
// Helper; call method with signature: obj->meth(arg1).
    Py_INCREF(obj);
    PyObject* result = PyObject_CallMethod(
        obj, const_cast<char*>(meth), const_cast<char*>("O"), arg1);
    Py_DECREF(obj);
    return result;
}


//- helpers --------------------------------------------------------------------
static std::string ExtractNamespace(const std::string& name)
{
// Find the namespace the named class lives in, take care of templates
    int tpl_open = 0;
    for (std::string::size_type pos = name.size()-1; 0 < pos; --pos) {
        std::string::value_type c = name[pos];

    // count '<' and '>' to be able to skip template contents
        if (c == '>')
            ++tpl_open;
        else if (c == '<')
            --tpl_open;

    // collect name up to "::"
        else if (tpl_open == 0 && c == ':' && name[pos-1] == ':') {
        // found the extend of the scope ... done
            return name.substr(0, pos-1);
        }
    }

// no namespace; assume outer scope
    return "";
}

//-----------------------------------------------------------------------------
PyObject* PyStyleIndex(PyObject* self, PyObject* index)
{
// Helper; converts python index into straight C index.
    Py_ssize_t idx = PyInt_AsSsize_t(index);
    if (idx == (Py_ssize_t)-1 && PyErr_Occurred())
        return nullptr;

    Py_ssize_t size = PySequence_Size(self);
    if (idx >= size || (idx < 0 && idx < -size)) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return nullptr;
    }

    PyObject* pyindex = nullptr;
    if (idx >= 0) {
        Py_INCREF(index);
        pyindex = index;
    } else
        pyindex = PyLong_FromLong(size+idx);

    return pyindex;
}

//-----------------------------------------------------------------------------
inline PyObject* CallSelfIndex(CPPInstance* self, PyObject* idx, const char* meth)
{
// Helper; call method with signature: meth(pyindex).
    Py_INCREF((PyObject*)self);
    PyObject* pyindex = PyStyleIndex((PyObject*)self, idx);
    if (!pyindex) {
        Py_DECREF((PyObject*)self);
        return nullptr;
    }

    PyObject* result = CallPyObjMethod((PyObject*)self, meth, pyindex);
    Py_DECREF(pyindex);
    Py_DECREF((PyObject*)self);
    return result;
}

//- "smart pointer" behavior ---------------------------------------------------
PyObject* DeRefGetAttr(PyObject* self, PyObject* name)
{
// Follow operator*() if present (available in python as __deref__), so that
// smart pointers behave as expected.
    if (!CPyCppyy_PyUnicode_Check(name))
        PyErr_SetString(PyExc_TypeError, "getattr(): attribute name must be string");

    PyObject* pyptr = CallPyObjMethod(self, "__deref__");
    if (!pyptr)
        return nullptr;

// prevent a potential infinite loop
    if (Py_TYPE(pyptr) == Py_TYPE(self)) {
        PyObject* val1 = PyObject_Str(self);
        PyObject* val2 = PyObject_Str(name);
        PyErr_Format(PyExc_AttributeError, "%s has no attribute \'%s\'",
            CPyCppyy_PyUnicode_AsString(val1), CPyCppyy_PyUnicode_AsString(val2));
        Py_DECREF(val2);
        Py_DECREF(val1);

        Py_DECREF(pyptr);
        return nullptr;
    }

    PyObject* result = PyObject_GetAttr(pyptr, name);
    Py_DECREF(pyptr);
    return result;
}

//-----------------------------------------------------------------------------
PyObject* FollowGetAttr(PyObject* self, PyObject* name)
{
// Follow operator->() if present (available in python as __follow__), so that
// smart pointers behave as expected.
    if (!CPyCppyy_PyUnicode_Check(name))
        PyErr_SetString(PyExc_TypeError, "getattr(): attribute name must be string");

    PyObject* pyptr = CallPyObjMethod(self, "__follow__");
    if (!pyptr)
         return nullptr;

    PyObject* result = PyObject_GetAttr(pyptr, name);
    Py_DECREF(pyptr);
    return result;
}

//-----------------------------------------------------------------------------
PyObject* GenObjectIsEqual(PyObject* self, PyObject* obj)
{
// Call the C++ operator==() if available, otherwise default.
    PyObject* result = CallPyObjMethod(self, "__cpp_eq__", obj);
    if (result)
        return result;

// failed: fallback like python would do by reversing the arguments
    PyErr_Clear();
    result = CallPyObjMethod(obj, "__cpp_eq__", self);
    if (result)
        return result;

// failed: fallback to generic rich comparison
    PyErr_Clear();
    return CPPInstance_Type.tp_richcompare(self, obj, Py_EQ);
}

//-----------------------------------------------------------------------------
PyObject* GenObjectIsNotEqual(PyObject* self, PyObject* obj)
{
// Reverse of GenObjectIsEqual, if operator!= defined.
    PyObject* result = CallPyObjMethod(self, "__cpp_ne__", obj);
    if (result)
        return result;

// failed: fallback like python would do by reversing the arguments
    PyErr_Clear();
    result = CallPyObjMethod(obj, "__cpp_ne__", self);
    if (result)
        return result;

// failed: fallback to generic rich comparison
    PyErr_Clear();
    return CPPInstance_Type.tp_richcompare(self, obj, Py_NE);
}

//- vector behavior as primitives ----------------------------------------------
typedef struct {
    PyObject_HEAD
    PyObject*                vi_vector;
    void*                    vi_data;
    CPyCppyy::Converter*     vi_converter;
    Py_ssize_t               vi_pos;
    Py_ssize_t               vi_len;
    Py_ssize_t               vi_stride;
} vectoriterobject;

static void vectoriter_dealloc(vectoriterobject* vi) {
    Py_XDECREF(vi->vi_vector);
    delete vi->vi_converter;
    PyObject_GC_Del(vi);
}

static int vectoriter_traverse(vectoriterobject* vi, visitproc visit, void* arg) {
    Py_VISIT(vi->vi_vector);
    return 0;
}

static PyObject* vectoriter_iternext(vectoriterobject* vi) {
    if (vi->vi_pos >= vi->vi_len)
        return nullptr;

    PyObject* result = nullptr;

    if (vi->vi_data && vi->vi_converter) {
        void* location = (void*)((ptrdiff_t)vi->vi_data + vi->vi_stride * vi->vi_pos);
        result = vi->vi_converter->FromMemory(location);
    } else {
        PyObject* pyindex = PyLong_FromLong(vi->vi_pos);
        result = CallPyObjMethod((PyObject*)vi->vi_vector, "_vector__at", pyindex);
        Py_DECREF(pyindex);
    }

    vi->vi_pos += 1;
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
    (traverseproc)vectoriter_traverse,      // tp_traverse
    0, 0, 0,
    PyObject_SelfIter,            // tp_iter
    (iternextfunc)vectoriter_iternext,      // tp_iternext
    0,                            // tp_methods
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#if PY_VERSION_HEX >= 0x02030000
    , 0                           // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                           // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                           // tp_finalize
#endif
};

static PyObject* vector_iter(PyObject* v) {
    vectoriterobject* vi = PyObject_GC_New(vectoriterobject, &VectorIter_Type);
    if (!vi) return nullptr;

    Py_INCREF(v);
    vi->vi_vector = v;

    PyObject* pyvalue_type = PyObject_GetAttrString((PyObject*)Py_TYPE(v), "value_type");
    PyObject* pyvalue_size = PyObject_GetAttrString((PyObject*)Py_TYPE(v), "value_size");

    if (pyvalue_type && pyvalue_size) {
        PyObject* pydata = CallPyObjMethod(v, "data");
        if (!pydata || Utility::GetBuffer(pydata, '*', 1, vi->vi_data, false) == 0)
            vi->vi_data = nullptr;
        Py_XDECREF(pydata);

        vi->vi_converter = CPyCppyy::CreateConverter(CPyCppyy_PyUnicode_AsString(pyvalue_type));
        vi->vi_stride    = PyLong_AsLong(pyvalue_size);
    } else {
        PyErr_Clear();
        vi->vi_data      = nullptr;
        vi->vi_converter = nullptr;
        vi->vi_stride    = 0;
    }

    Py_XDECREF(pyvalue_size);
    Py_XDECREF(pyvalue_type);

    vi->vi_len = vi->vi_pos = 0;
    vi->vi_len = PySequence_Size(v);

    _PyObject_GC_TRACK(vi);
    return (PyObject*)vi;
}

PyObject* VectorGetItem(CPPInstance* self, PySliceObject* index)
{
// Implement python's __getitem__ for std::vector<>s.
    if (PySlice_Check(index)) {
        if (!self->GetObject()) {
            PyErr_SetString(PyExc_TypeError, "unsubscriptable object");
            return nullptr;
        }

        PyObject* pyclass = PyObject_GetAttr((PyObject*)self, PyStrings::gClass);
        PyObject* nseq = PyObject_CallObject(pyclass, nullptr);
        Py_DECREF(pyclass);

        Py_ssize_t start, stop, step;
        PySlice_GetIndices((CPyCppyy_PySliceCast)index, PyObject_Length((PyObject*)self), &start, &stop, &step);
        for (Py_ssize_t i = start; i < stop; i += step) {
            PyObject* pyidx = PyInt_FromSsize_t(i);
            CallPyObjMethod(nseq, "push_back", CallPyObjMethod((PyObject*)self, "_vector__at", pyidx));
            Py_DECREF(pyidx);
        }

        return nseq;
    }

    return CallSelfIndex(self, (PyObject*)index, "_vector__at");
}


static Cppyy::TCppType_t sVectorBoolTypeID = (Cppyy::TCppType_t)0;

PyObject* VectorBoolGetItem(CPPInstance* self, PyObject* idx)
{
// std::vector<bool> is a special-case in C++, and its return type depends on
// the compiler: treat it special here as well
    if (!CPPInstance_Check(self) || self->ObjectIsA() != sVectorBoolTypeID) {
        PyErr_Format(PyExc_TypeError,
            "require object of type std::vector<bool>, but %s given",
            Cppyy::GetScopedFinalName(self->ObjectIsA()).c_str());
        return nullptr;
    }

    if (!self->GetObject()) {
        PyErr_SetString(PyExc_TypeError, "unsubscriptable object");
        return nullptr;
    }

    if (PySlice_Check(idx)) {
        PyObject* pyclass = PyObject_GetAttr((PyObject*)self, PyStrings::gClass);
        PyObject* nseq = PyObject_CallObject(pyclass, nullptr);
        Py_DECREF(pyclass);

        Py_ssize_t start, stop, step;
        PySlice_GetIndices((CPyCppyy_PySliceCast)idx, PyObject_Length((PyObject*)self), &start, &stop, &step);
        for (Py_ssize_t i = start; i < stop; i += step) {
            PyObject* pyidx = PyInt_FromSsize_t(i);
            CallPyObjMethod(nseq, "push_back", CallPyObjMethod((PyObject*)self, "__getitem__", pyidx));
            Py_DECREF(pyidx);
        }

        return nseq;
    }

    PyObject* pyindex = PyStyleIndex((PyObject*)self, idx);
    if (!pyindex)
        return nullptr;

    int index = (int)PyLong_AsLong(pyindex);
    Py_DECREF(pyindex);

// get hold of the actual std::vector<bool> (no cast, as vector is never a base)
    std::vector<bool>* vb = (std::vector<bool>*)self->GetObject();

// finally, return the value
    if (bool((*vb)[index]))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject* VectorBoolSetItem(CPPInstance* self, PyObject* args)
{
// std::vector<bool> is a special-case in C++, and its return type depends on
// the compiler: treat it special here as well
    if (!CPPInstance_Check(self) || self->ObjectIsA() != sVectorBoolTypeID) {
        PyErr_Format(PyExc_TypeError,
            "require object of type std::vector<bool>, but %s given",
            Cppyy::GetScopedFinalName(self->ObjectIsA()).c_str());
        return nullptr;
    }

    if (!self->GetObject()) {
        PyErr_SetString(PyExc_TypeError, "unsubscriptable object");
        return nullptr;
    }

    int bval = 0; PyObject* idx = nullptr;
    if (!PyArg_ParseTuple(args, const_cast<char*>("Oi:__setitem__"), &idx, &bval))
        return nullptr;

    PyObject* pyindex = PyStyleIndex((PyObject*)self, idx);
    if (!pyindex)
        return nullptr;

    int index = (int)PyLong_AsLong(pyindex);
    Py_DECREF(pyindex);

// get hold of the actual std::vector<bool> (no cast, as vector is never a base)
    std::vector<bool>* vb = (std::vector<bool>*)self->GetObject();

// finally, set the value
    (*vb)[index] = (bool)bval;

    Py_RETURN_NONE;
}

//- map behavior as primitives ------------------------------------------------
PyObject* MapContains(PyObject* self, PyObject* obj)
{
// Implement python's __contains__ for std::map<>s.
    PyObject* result = nullptr;

    PyObject* iter = CallPyObjMethod(self, "find", obj);
    if (CPPInstance_Check(iter)) {
        PyObject* end = CallPyObjMethod(self, "end");
        if (CPPInstance_Check(end)) {
            if (!PyObject_RichCompareBool(iter, end, Py_EQ)) {
                Py_INCREF(Py_True);
                result = Py_True;
            }
        }
        Py_XDECREF(end);
    }
    Py_XDECREF(iter);

    if (!result) {
        PyErr_Clear();            // e.g. wrong argument type, which should always lead to False
        Py_INCREF(Py_False);
        result = Py_False;
    }

    return result;
}

//- STL container iterator support --------------------------------------------
PyObject* StlSequenceIter(PyObject* self)
{
// Implement python's __iter__ for std::iterator<>s.
    PyObject* iter = CallPyObjMethod(self, "begin");
    if (iter) {
        PyObject* end = CallPyObjMethod(self, "end");
        if (end)
            PyObject_SetAttr(iter, PyStrings::gEnd, end);
        Py_XDECREF(end);

    // add iterated collection as attribute so its refcount stays >= 1 while it's being iterated over
        PyObject_SetAttr(iter, PyUnicode_FromString("_collection"), self);
    }
    return iter;
}

//- safe indexing for STL-like vector w/o iterator dictionaries ---------------
/*
PyObject* CheckedGetItem(PyObject* self, PyObject* obj)
{
// Implement a generic python __getitem__ for std::vector<>s that are missing
// their std::vector<>::iterator dictionary. This is then used for iteration
// by means of consecutive index.
    bool inbounds = false;
    Py_ssize_t size = PySequence_Size(self);
    Py_ssize_t idx  = PyInt_AsSsize_t(obj);
    if (0 <= idx && 0 <= size && idx < size)
        inbounds = true;

    if (inbounds) {
        return CallPyObjMethod(self, "_getitem__unchecked", obj);
    } else if (PyErr_Occurred()) {
    // argument conversion problem: let method itself resolve anew and report
        PyErr_Clear();
        return CallPyObjMethod(self, "_getitem__unchecked", obj);
    } else {
        PyErr_SetString(PyExc_IndexError, "index out of range");
    }

    return nullptr;
}
*/

//- pair as sequence to allow tuple unpacking --------------------------------
PyObject* PairUnpack(PyObject* self, PyObject* pyindex)
{
// For std::map<> iteration, unpack std::pair<>s into tuples for the loop.
    long idx = PyLong_AsLong(pyindex);
    if (idx == -1 && PyErr_Occurred())
        return nullptr;

    if (!CPPInstance_Check(self) || !((CPPInstance*)self)->GetObject()) {
        PyErr_SetString(PyExc_TypeError, "unsubscriptable object");
        return nullptr;
    }

    if ((int)idx == 0)
        return PyObject_GetAttr(self, PyStrings::gFirst);
    else if ((int)idx == 1)
        return PyObject_GetAttr(self, PyStrings::gSecond);

// still here? Trigger stop iteration
    PyErr_SetString(PyExc_IndexError, "out of bounds");
    return nullptr;
}

//- simplistic len() functions -----------------------------------------------
PyObject* ReturnTwo(CPPInstance*, PyObject*) {
    return PyInt_FromLong(2);
}

//- string behavior as primitives --------------------------------------------
#if PY_VERSION_HEX >= 0x03000000
// TODO: this is wrong, b/c it doesn't order
static int PyObject_Compare(PyObject* one, PyObject* other) {
    return !PyObject_RichCompareBool(one, other, Py_EQ);
}
#endif
static inline PyObject* CPyCppyy_PyString_FromCppString(std::string* s) {
    return CPyCppyy_PyUnicode_FromStringAndSize(s->c_str(), s->size());
}

#define CPPYY_IMPL_STRING_PYTHONIZATION(type, name)                          \
inline PyObject* name##GetData(PyObject* self)                               \
{                                                                            \
    if (CPyCppyy::CPPInstance_Check(self)) {                                 \
        type* obj = ((type*)((CPPInstance*)self)->GetObject());              \
        if (obj) {                                                           \
            return CPyCppyy_PyString_FromCppString(obj);                     \
        } else {                                                             \
            return CPPInstance_Type.tp_str(self);                            \
        }                                                                    \
    }                                                                        \
    PyErr_Format(PyExc_TypeError, "object mismatch (%s expected)", #type);   \
    return nullptr;                                                          \
}                                                                            \
                                                                             \
PyObject* name##StringRepr(PyObject* self)                                   \
{                                                                            \
    PyObject* data = name##GetData(self);                                    \
    if (data) {                                                              \
        PyObject* repr = CPyCppyy_PyUnicode_FromFormat("\'%s\'", CPyCppyy_PyUnicode_AsString(data));\
        Py_DECREF(data);                                                     \
        return repr;                                                         \
    }                                                                        \
    return nullptr;                                                          \
}                                                                            \
                                                                             \
PyObject* name##StringIsEqual(PyObject* self, PyObject* obj)                 \
{                                                                            \
    PyObject* data = name##GetData(self);                                    \
    if (data) {                                                              \
        PyObject* result = PyObject_RichCompare(data, obj, Py_EQ);           \
        Py_DECREF(data);                                                     \
        return result;                                                       \
    }                                                                        \
    return nullptr;                                                          \
}                                                                            \
                                                                             \
PyObject* name##StringIsNotEqual(PyObject* self, PyObject* obj)              \
{                                                                            \
    PyObject* data = name##GetData(self);                                    \
    if (data) {                                                              \
        PyObject* result = PyObject_RichCompare(data, obj, Py_NE);           \
        Py_DECREF(data);                                                     \
        return result;                                                       \
    }                                                                        \
    return nullptr;                                                          \
}

// Only define StlStringCompare:
#define CPPYY_IMPL_STRING_PYTHONIZATION_CMP(type, name)                      \
CPPYY_IMPL_STRING_PYTHONIZATION(type, name)                                  \
PyObject* name##StringCompare(PyObject* self, PyObject* obj)                 \
{                                                                            \
    PyObject* data = name##GetData(self);                                    \
    int result = 0;                                                          \
    if (data) {                                                              \
        result = PyObject_Compare(data, obj);                                \
        Py_DECREF(data);                                                     \
    }                                                                        \
    if (PyErr_Occurred())                                                    \
        return nullptr;                                                      \
    return PyInt_FromLong(result);                                           \
}

CPPYY_IMPL_STRING_PYTHONIZATION_CMP(std::string, Stl)


//- STL iterator behavior ----------------------------------------------------
PyObject* StlIterNext(PyObject* self)
{
// Python iterator protocol __next__ for STL forward iterators.
    PyObject* next = nullptr;
    PyObject* last = PyObject_GetAttr(self, PyStrings::gEnd);

    if (last) {
    // handle special case of empty container (i.e. self is end)
        if (PyObject_RichCompareBool(last, self, Py_EQ)) {
            PyErr_SetString(PyExc_StopIteration, "");
        } else {
            PyObject* dummy = PyInt_FromLong(1l);
            PyObject* iter = CallPyObjMethod(self, "__postinc__", dummy);
            Py_DECREF(dummy);
            if (iter != 0) {
                if (PyObject_RichCompareBool(last, iter, Py_EQ))
                    PyErr_SetString(PyExc_StopIteration, "");
                else
                    next = CallPyObjMethod(iter, "__deref__");
            } else {
                PyErr_SetString(PyExc_StopIteration, "");
            }
            Py_XDECREF(iter);
        }
    } else {
        PyErr_SetString(PyExc_StopIteration, "");
    }

    Py_XDECREF(last);
    return next;
}

//----------------------------------------------------------------------------
PyObject* StlIterIsEqual(PyObject* self, PyObject* other)
{
// Called if operator== not available (e.g. if a global overload as under gcc).
// An exception is raised as the user should fix the dictionary.
    return PyErr_Format(PyExc_LookupError,
        "No operator==(const %s&, const %s&) available in the dictionary!",
        Utility::ClassName(self).c_str(), Utility::ClassName(other).c_str());
}

//----------------------------------------------------------------------------
PyObject* StlIterIsNotEqual(PyObject* self, PyObject* other)
{
// Called if operator!= not available (e.g. if a global overload as under gcc).
// An exception is raised as the user should fix the dictionary.
    return PyErr_Format(PyExc_LookupError,
        "No operator!=(const %s&, const %s&) available in the dictionary!",
        Utility::ClassName(self).c_str(), Utility::ClassName(other).c_str());
}

} // unnamed namespace


//- public functions ---------------------------------------------------------
bool CPyCppyy::Pythonize(PyObject* pyclass, const std::string& name)
{
// Add pre-defined pythonizations (for STL and ROOT) to classes based on their
// signature and/or class name.
    if (!pyclass)
        return false;

//- method name based pythonization ------------------------------------------

// for smart pointer style classes (note fall-through)
    if (HasAttrDirect(pyclass, PyStrings::gDeref)) {
        Utility::AddToClass(pyclass, "__getattr__", (PyCFunction)DeRefGetAttr, METH_O);
    } else if (HasAttrDirect(pyclass, PyStrings::gFollow)) {
        Utility::AddToClass(pyclass, "__getattr__", (PyCFunction)FollowGetAttr, METH_O);
    }

// for STL containers, and user classes modeled after them
    if (HasAttrDirect(pyclass, PyStrings::gSize))
        Utility::AddToClass(pyclass, "__len__", "size");

    if (!IsTemplatedSTLClass(name, "vector")) {       // vector is dealt with below
        if (HasAttrDirect(pyclass, PyStrings::gBegin) && HasAttrDirect(pyclass, PyStrings::gEnd)) {
        // TODO: check whether iterator type is available

        // if yes: install iterator protocol
            ((PyTypeObject*)pyclass)->tp_iter = (getiterfunc)StlSequenceIter;
            Utility::AddToClass(pyclass, "__iter__", (PyCFunction) StlSequenceIter, METH_NOARGS);

        // if not and  if (HasAttrDirect(pyclass, PyStrings::gGetItem) && HasAttrDirect(pyclass, PyStrings::gLen)) {
        // install increment until StopIteration "protocol"
            //Utility::AddToClass(pyclass, "_getitem__unchecked", "__getitem__");
            //Utility::AddToClass(pyclass, "__getitem__", (PyCFunction) CheckedGetItem, METH_O);
        }
    }

// search for global comparator overloads (may fail; not sure whether it isn't better to
// do this lazily just as is done for math operators, but this interplays nicely with the
// generic versions)
    Utility::AddBinaryOperator(pyclass, "==", "__eq__");
    Utility::AddBinaryOperator(pyclass, "!=", "__ne__");

// map operator==() through GenObjectIsEqual to allow comparison to None (true is to
// require that the located method is a CPPOverload; this prevents circular calls as
// GenObjectIsEqual is no CPPOverload)
    if (HasAttrDirect(pyclass, PyStrings::gEq, true)) {
        Utility::AddToClass(pyclass, "__cpp_eq__",  "__eq__");
        Utility::AddToClass(pyclass, "__eq__", (PyCFunction)GenObjectIsEqual, METH_O);
    }

// map operator!=() through GenObjectIsNotEqual to allow comparison to None (see note
// on true above for __eq__)
    if (HasAttrDirect(pyclass, PyStrings::gNe, true)) {
        Utility::AddToClass(pyclass, "__cpp_ne__",  "__ne__");
        Utility::AddToClass(pyclass, "__ne__",  (PyCFunction)GenObjectIsNotEqual, METH_O);
    }


//- class name based pythonization -------------------------------------------

    if (IsTemplatedSTLClass(name, "vector")) {

    // std::vector<bool> is a special case in C++
        if (!sVectorBoolTypeID) sVectorBoolTypeID = (Cppyy::TCppType_t)Cppyy::GetScope("std::vector<bool>");
        if (CPPScope_Check(pyclass) && ((CPPClass*)pyclass)->fCppType == sVectorBoolTypeID) {
            Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)VectorBoolGetItem, METH_O);
            Utility::AddToClass(pyclass, "__setitem__", (PyCFunction)VectorBoolSetItem);
        } else {

            if (HasAttrDirect(pyclass, PyStrings::gLen) && HasAttrDirect(pyclass, PyStrings::gAt)) {
                Utility::AddToClass(pyclass, "_vector__at", "at");
            // remove iterator that was set earlier (checked __getitem__ will do the trick)
                if (HasAttrDirect(pyclass, PyStrings::gIter))
                    PyObject_DelAttr(pyclass, PyStrings::gIter);
            } else if (HasAttrDirect(pyclass, PyStrings::gGetItem)) {
                Utility::AddToClass(pyclass, "_vector__at", "__getitem__");   // unchecked!
            }

       // vector-optimized iterator protocol
            ((PyTypeObject*)pyclass)->tp_iter     = (getiterfunc)vector_iter;

       // helpers for iteration
       /*TODO: remove this use of gInterpreter
            TypedefInfo_t* ti = gInterpreter->TypedefInfo_Factory((name+"::value_type").c_str());
            if (gInterpreter->TypedefInfo_IsValid(ti)) {
                PyObject* pyvalue_size = PyLong_FromLong(gInterpreter->TypedefInfo_Size(ti));
                PyObject_SetAttrString(pyclass, "value_size", pyvalue_size);
                Py_DECREF(pyvalue_size);

                PyObject* pyvalue_type = CPyCppyy_PyUnicode_FromString(gInterpreter->TypedefInfo_TrueName(ti));
                PyObject_SetAttrString(pyclass, "value_type", pyvalue_type);
                Py_DECREF(pyvalue_type);
            }
            gInterpreter->TypedefInfo_Delete(ti);
          */

       // provide a slice-able __getitem__, if possible
            if (HasAttrDirect(pyclass, PyStrings::gVectorAt))
                Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)VectorGetItem, METH_O);
        }
    }

    else if (IsTemplatedSTLClass(name, "map")) {
        Utility::AddToClass(pyclass, "__contains__", (PyCFunction)MapContains, METH_O);
    }

    else if (IsTemplatedSTLClass(name, "pair")) {
        Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)PairUnpack, METH_O);
        Utility::AddToClass(pyclass, "__len__", (PyCFunction)ReturnTwo, METH_NOARGS);
    }

    else if (name.find("iterator") != std::string::npos) {
        ((PyTypeObject*)pyclass)->tp_iternext = (iternextfunc)StlIterNext;
        Utility::AddToClass(pyclass, CPPYY__next__, (PyCFunction)StlIterNext, METH_NOARGS);

    // special case, if operator== is a global overload and included in the dictionary
        if (!HasAttrDirect(pyclass, PyStrings::gCppEq, true))
            Utility::AddToClass(pyclass, "__eq__", (PyCFunction)StlIterIsEqual, METH_O);
        if (!HasAttrDirect(pyclass, PyStrings::gCppNe, true))
            Utility::AddToClass(pyclass, "__ne__", (PyCFunction)StlIterIsNotEqual, METH_O);
    }

    else if (name == "string" || name == "std::string") { // TODO: ask backend as well
        Utility::AddToClass(pyclass, "__repr__", (PyCFunction)StlStringRepr, METH_NOARGS);
        Utility::AddToClass(pyclass, "__str__", "c_str");
        Utility::AddToClass(pyclass, "__cmp__", (PyCFunction)StlStringCompare, METH_O);
        Utility::AddToClass(pyclass, "__eq__",  (PyCFunction)StlStringIsEqual, METH_O);
        Utility::AddToClass(pyclass, "__ne__",  (PyCFunction)StlStringIsNotEqual, METH_O);
    }

    PyObject* args = PyTuple_New(2);
    Py_INCREF(pyclass);
    PyTuple_SET_ITEM(args, 0, pyclass);

    std::string outer_scope = ExtractNamespace(name);

    bool pstatus = true;
    auto p = outer_scope.empty() ? gPythonizations.end() : gPythonizations.find(outer_scope);
    if (p == gPythonizations.end()) {
        p = gPythonizations.find("");
        PyTuple_SET_ITEM(args, 1, CPyCppyy_PyUnicode_FromString(name.c_str()));
    } else {
        PyTuple_SET_ITEM(args, 1, CPyCppyy_PyUnicode_FromString(
                             name.substr(outer_scope.size()+2, std::string::npos).c_str()));
    }

    if (p != gPythonizations.end()) {
        for (auto pythonizor : p->second) {
            PyObject* result = PyObject_CallObject(pythonizor, args);
            if (!result) {
            // TODO: detail error handling for the pythonizors
                pstatus = false;
                break;
            }
            Py_DECREF(result);
        }
    }

    Py_DECREF(args);

// phew! all done ...
    return pstatus;
}
