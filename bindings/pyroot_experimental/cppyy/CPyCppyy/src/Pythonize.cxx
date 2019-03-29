// Bindings
#include "CPyCppyy.h"
#include "Pythonize.h"
#include "Converters.h"
#include "CPPInstance.h"
#include "CPPOverload.h"
#include "CustomPyTypes.h"
#include "LowLevelViews.h"
#include "ProxyWrappers.h"
#include "PyCallable.h"
#include "PyStrings.h"
#include "TypeManip.h"
#include "Utility.h"

// Standard
#include <complex>
#include <stdexcept>
#include <sstream>
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
    PyObject* dct = PyObject_GetAttr(pyclass, PyStrings::gDict);
    if (dct) {
        PyObject* attr = PyObject_GetItem(dct, pyname);
        Py_DECREF(dct);
        if (attr) {
            bool ret = !mustBeCPyCppyy || CPPOverload_Check(attr);
            Py_DECREF(attr);
            return ret;
        }
    }
    PyErr_Clear();
    return false;
}

//-----------------------------------------------------------------------------
inline bool IsTopLevelClass(PyObject* pyclass) {
// determine whether this class directly derives from CPPInstance
    PyObject* bases = PyObject_GetAttr(pyclass, PyStrings::gBases);
    if (!bases)
        return false;

    bool isTopLevel = false;
    if (PyTuple_CheckExact(bases) && PyTuple_GET_SIZE(bases) && \
            (void*)PyTuple_GET_ITEM(bases, 0) == (void*)&CPPInstance_Type) {
        isTopLevel = true;
    }

    Py_DECREF(bases);
    return isTopLevel;
}

//-----------------------------------------------------------------------------
inline bool IsTemplatedSTLClass(const std::string& name, const std::string& klass) {
// Scan the name of the class and determine whether it is a template instantiation.
    auto pos = name.find(klass);
    return (pos == 0 || pos == 5) && name.find("::", name.rfind(">")) == std::string::npos;
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
        pyindex = PyLong_FromSsize_t(size+idx);

    return pyindex;
}

//-----------------------------------------------------------------------------
inline PyObject* CallSelfIndex(CPPInstance* self, PyObject* idx, PyObject* pymeth)
{
// Helper; call method with signature: meth(pyindex).
    Py_INCREF((PyObject*)self);
    PyObject* pyindex = PyStyleIndex((PyObject*)self, idx);
    if (!pyindex) {
        Py_DECREF((PyObject*)self);
        return nullptr;
    }

    PyObject* result = PyObject_CallMethodObjArgs((PyObject*)self, pymeth, pyindex, nullptr);
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

    PyObject* pyptr = PyObject_CallMethodObjArgs(self, PyStrings::gDeref, nullptr);
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

    PyObject* pyptr = PyObject_CallMethodObjArgs(self, PyStrings::gDeref, nullptr);
    if (!pyptr)
         return nullptr;

    PyObject* result = PyObject_GetAttr(pyptr, name);
    Py_DECREF(pyptr);
    return result;
}

//-----------------------------------------------------------------------------
PyObject* GenObjectIsEqualNoCpp(PyObject* self, PyObject* obj)
{
// bootstrap as necessary
    if (obj != Py_None) {
        if (Utility::AddBinaryOperator(self, obj, "==", "__eq__"))
            return PyObject_CallMethodObjArgs(self, PyStrings::gEq, obj, nullptr);

    // drop lazy lookup from future considerations if both types are the same
    // and lookup failed (theoretically, it is possible to write a class that
    // can only compare to other types, but that's very unlikely)
        if (Py_TYPE(self) == Py_TYPE(obj) && \
                HasAttrDirect((PyObject*)Py_TYPE(self), PyStrings::gEq)) {
            if (PyObject_DelAttr((PyObject*)Py_TYPE(self), PyStrings::gEq) != 0)
                PyErr_Clear();
        }
    }

// failed: fallback to generic rich comparison
    return CPPInstance_Type.tp_richcompare(self, obj, Py_EQ);
}

PyObject* GenObjectIsEqual(PyObject* self, PyObject* obj)
{
// Call the C++ operator==() if available, otherwise default.
    PyObject* result = PyObject_CallMethodObjArgs(self, PyStrings::gCppEq, obj, nullptr);
    if (result)
        return result;
    PyErr_Clear();

// failed: fallback like python would do by reversing the arguments
    if (CPPInstance_Check(obj)) {
        result = PyObject_CallMethodObjArgs(obj, PyStrings::gCppEq, self, nullptr);
        if (result)
            return result;
        PyErr_Clear();
    }

// failed, try generic
    return CPPInstance_Type.tp_richcompare(self, obj, Py_EQ);
}

//-----------------------------------------------------------------------------
PyObject* GenObjectIsNotEqualNoCpp(PyObject* self, PyObject* obj)
{
// bootstrap as necessary
    if (obj != Py_None) {
        if (Utility::AddBinaryOperator(self, obj, "!=", "__ne__"))
            return PyObject_CallMethodObjArgs(self, PyStrings::gNe, obj, nullptr);
        PyErr_Clear();

    // drop lazy lookup from future considerations if both types are the same
    // and lookup failed (theoretically, it is possible to write a class that
    // can only compare to other types, but that's very unlikely)
        if (Py_TYPE(self) == Py_TYPE(obj) && \
                HasAttrDirect((PyObject*)Py_TYPE(self), PyStrings::gNe)) {
            if (PyObject_DelAttr((PyObject*)Py_TYPE(self), PyStrings::gNe) != 0)
                PyErr_Clear();
        }
    }

// failed: fallback to generic rich comparison
    return CPPInstance_Type.tp_richcompare(self, obj, Py_NE);
}

PyObject* GenObjectIsNotEqual(PyObject* self, PyObject* obj)
{
// Reverse of GenObjectIsEqual, if operator!= defined.
    PyObject* result = PyObject_CallMethodObjArgs(self, PyStrings::gCppNe, obj, nullptr);
    if (result)
        return result;
    PyErr_Clear();

// failed: fallback like python would do by reversing the arguments
    if (CPPInstance_Check(obj)) {
        result = PyObject_CallMethodObjArgs(obj, PyStrings::gCppNe, self, nullptr);
        if (result)
            return result;
        PyErr_Clear();
    }

// failed, try generic
    return CPPInstance_Type.tp_richcompare(self, obj, Py_NE);
}


//- vector behavior as primitives ----------------------------------------------
PyObject* VectorInit(PyObject* self, PyObject* args, PyObject* /* kwds */)
{
// using initializer_list is possible, but error-prone; since it's so common for
// std::vector, this implements construction from python iterables directly, except
// for arrays, which can be passed wholesale, and strings, which shouldn't be used
    if (PyTuple_GET_SIZE(args) == 1 &&                              \
            (CPyCppyy_PyUnicode_Check(PyTuple_GET_ITEM(args, 0)) || \
             PyBytes_Check(PyTuple_GET_ITEM(args, 0)))) {
        PyErr_SetString(PyExc_TypeError, "can not convert string to vector");
        return 0;
    }

    if (PyTuple_GET_SIZE(args) == 1 && PySequence_Check(PyTuple_GET_ITEM(args, 0)) && \
            !Py_TYPE(PyTuple_GET_ITEM(args, 0))->tp_as_buffer) {
        PyObject* mname = CPyCppyy_PyUnicode_FromString("__real_init");
        PyObject* result = PyObject_CallMethodObjArgs(self, mname, nullptr);
        Py_DECREF(mname);
        if (!result)
            return result;

        PyObject* ll = PyTuple_GET_ITEM(args, 0);
        Py_ssize_t sz = PySequence_Size(ll);
        PyObject* res = PyObject_CallMethod(self, (char*)"reserve", (char*)"n", sz);
        Py_DECREF(res);

        bool fill_ok = true;
        PyObject* pb_call = PyObject_GetAttrString(self, (char*)"push_back");
        if (pb_call) {
            PyObject* pb_args = PyTuple_New(1);
            for (Py_ssize_t i = 0; i < sz; ++i) {
                PyObject* item = PySequence_GetItem(ll, i);
                if (item) {
                    PyTuple_SET_ITEM(pb_args, 0, item);
                    PyObject* pbres = PyObject_CallObject(pb_call, pb_args);
                    Py_DECREF(item);
                    if (!pbres) {
                        fill_ok = false;
                        break;
                    }
                    Py_DECREF(pbres);
                } else {
                    fill_ok = false;
                    break;
                }
            }
            PyTuple_SET_ITEM(pb_args, 0, nullptr);
            Py_DECREF(pb_args);
        }
        Py_DECREF(pb_call);

        if (!fill_ok) {
            Py_DECREF(result);
            return nullptr;
        }

        return result;
    }

    PyObject* realInit = PyObject_GetAttrString(self, "__real_init");
    if (realInit) {
        PyObject* result = PyObject_Call(realInit, args, nullptr);
        Py_DECREF(realInit);
        return result;
    }

    return nullptr;
}

//---------------------------------------------------------------------------
PyObject* VectorData(PyObject* self, PyObject*)
{
    PyObject* pydata = CallPyObjMethod(self, "__real_data");
    if (!LowLevelView_Check(pydata)) return pydata;

    PyObject* pylen = PyObject_CallMethodObjArgs(self, PyStrings::gSize, nullptr);
    if (!pylen) {
        PyErr_Clear();
        return pydata;
    }

    long clen = PyInt_AsLong(pylen);
    Py_DECREF(pylen);

// TODO: should be a LowLevelView helper
    Py_buffer& bi = ((LowLevelView*)pydata)->fBufInfo;
    bi.len = clen * bi.itemsize;
    if (bi.ndim == 1 && bi.shape)
        bi.shape[0] = clen;

    return pydata;
}


//-----------------------------------------------------------------------------
static PyObject* vector_iter(PyObject* v) {
    vectoriterobject* vi = PyObject_GC_New(vectoriterobject, &VectorIter_Type);
    if (!vi) return nullptr;

    Py_INCREF(v);
    vi->ii_container = v;

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

    vi->ii_pos = 0;
    vi->ii_len = PySequence_Size(v);

    PyObject_GC_Track(vi);
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

        PyObject* pyclass = (PyObject*)Py_TYPE((PyObject*)self);
        PyObject* nseq = PyObject_CallObject(pyclass, nullptr);

        Py_ssize_t start, stop, step;
        PySlice_GetIndices((CPyCppyy_PySliceCast)index, PyObject_Length((PyObject*)self), &start, &stop, &step);
        for (Py_ssize_t i = start; i < stop; i += step) {
            PyObject* pyidx = PyInt_FromSsize_t(i);
            PyObject* item = PyObject_CallMethodObjArgs((PyObject*)self, PyStrings::gGetNoCheck, pyidx, nullptr);
            CallPyObjMethod(nseq, "push_back", item);
            Py_DECREF(item);
            Py_DECREF(pyidx);
        }

        return nseq;
    }

    return CallSelfIndex(self, (PyObject*)index, PyStrings::gGetNoCheck);
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
        PyObject* pyclass = (PyObject*)Py_TYPE((PyObject*)self);
        PyObject* nseq = PyObject_CallObject(pyclass, nullptr);

        Py_ssize_t start, stop, step;
        PySlice_GetIndices((CPyCppyy_PySliceCast)idx, PyObject_Length((PyObject*)self), &start, &stop, &step);
        for (Py_ssize_t i = start; i < stop; i += step) {
            PyObject* pyidx = PyInt_FromSsize_t(i);
            PyObject* item = PyObject_CallMethodObjArgs((PyObject*)self, PyStrings::gGetItem, pyidx, nullptr);
            CallPyObjMethod(nseq, "push_back", item);
            Py_DECREF(item);
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
// Implement python's __contains__ for std::map<>s
    PyObject* result = nullptr;

    PyObject* iter = CallPyObjMethod(self, "find", obj);
    if (CPPInstance_Check(iter)) {
        PyObject* end = PyObject_CallMethodObjArgs(self, PyStrings::gEnd, nullptr);
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
// Implement python's __iter__ for std::iterator<>s
    PyObject* iter = PyObject_CallMethodObjArgs(self, PyStrings::gBegin, nullptr);
    if (iter) {
        PyObject* end = PyObject_CallMethodObjArgs(self, PyStrings::gEnd, nullptr);
        if (end)
            PyObject_SetAttr(iter, PyStrings::gEnd, end);
        Py_XDECREF(end);

    // add iterated collection as attribute so its refcount stays >= 1 while it's being iterated over
        PyObject_SetAttr(iter, CPyCppyy_PyUnicode_FromString("_collection"), self);
    }
    return iter;
}

//- generic iterator support over a sequence with operator[] and size ---------
//-----------------------------------------------------------------------------
static PyObject* index_iter(PyObject* c) {
    indexiterobject* ii = PyObject_GC_New(indexiterobject, &IndexIter_Type);
    if (!ii) return nullptr;

    Py_INCREF(c);
    ii->ii_container = c;
    ii->ii_pos = 0;
    ii->ii_len = PySequence_Size(c);

    PyObject_GC_Track(ii);
    return (PyObject*)ii;
}


//- safe indexing for STL-like vector w/o iterator dictionaries ---------------
/* replaced by indexiterobject iteration, but may still have some future use ...
PyObject* CheckedGetItem(PyObject* self, PyObject* obj)
{
// Implement a generic python __getitem__ for STL-like classes that are missing the
// reflection info for their iterators. This is then used for iteration by means of
// consecutive indeces, it such index is of integer type.
    Py_ssize_t size = PySequence_Size(self);
    Py_ssize_t idx  = PyInt_AsSsize_t(obj);
    if ((size == (Py_ssize_t)-1 || idx == (Py_ssize_t)-1) && PyErr_Occurred()) {
    // argument conversion problem: let method itself resolve anew and report
        PyErr_Clear();
        return PyObject_CallMethodObjArgs(self, PyStrings::gGetNoCheck, obj, nullptr);
    }

    bool inbounds = false;
    if (idx < 0) idx += size;
    if (0 <= idx && 0 <= size && idx < size)
        inbounds = true;

    if (inbounds)
        return PyObject_CallMethodObjArgs(self, PyStrings::gGetNoCheck, obj, nullptr);
    else
        PyErr_SetString( PyExc_IndexError, "index out of range" );

    return nullptr;
}*/

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


//- shared_ptr behavior --------------------------------------------------------
PyObject* SharedPtrInit(PyObject* self, PyObject* args, PyObject* /* kwds */)
{
// since the shared pointer will take ownership, we need to relinquish it
    PyObject* realInit = PyObject_GetAttrString(self, "__real_init");
    if (realInit) {
        PyObject* result = PyObject_Call(realInit, args, nullptr);
        Py_DECREF(realInit);
        if (result && PyTuple_GET_SIZE(args) == 1 && CPPInstance_Check(PyTuple_GET_ITEM(args, 0)))
            PyObject_SetAttrString(PyTuple_GET_ITEM(args, 0), "__python_owns__", Py_False);
        return result;
    }
    return nullptr;
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

static inline PyObject* CPyCppyy_PyString_FromCppString(std::wstring* s) {
    return PyUnicode_FromWideChar(s->c_str(), s->size());
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
CPPYY_IMPL_STRING_PYTHONIZATION_CMP(std::wstring, StlW)


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


//- STL complex<T> behavior --------------------------------------------------
#define COMPLEX_METH_GETSET(name, cppname)                                   \
static PyObject* name##ComplexGet(PyObject* self, void*) {                   \
    return PyObject_CallMethodObjArgs(self, cppname, nullptr);               \
}                                                                            \
static int name##ComplexSet(PyObject* self, PyObject* value, void*) {        \
    PyObject* result = PyObject_CallMethodObjArgs(self, cppname, value, nullptr);\
    if (result) {                                                            \
        Py_DECREF(result);                                                   \
        return 0;                                                            \
    }                                                                        \
    return -1;                                                               \
}                                                                            \
PyGetSetDef name##Complex{(char*)#name, (getter)name##ComplexGet, (setter)name##ComplexSet, nullptr, nullptr};

COMPLEX_METH_GETSET(real, PyStrings::gCppReal)
COMPLEX_METH_GETSET(imag, PyStrings::gCppImag)

static PyObject* ComplexComplex(PyObject* self) {
    PyObject* real = PyObject_CallMethodObjArgs(self, PyStrings::gCppReal, nullptr);
    if (!real) return nullptr;
    double r = PyFloat_AsDouble(real);
    Py_DECREF(real);
    if (r == -1. && PyErr_Occurred())
        return nullptr;

    PyObject* imag = PyObject_CallMethodObjArgs(self, PyStrings::gCppImag, nullptr);
    if (!imag) return nullptr;
    double i = PyFloat_AsDouble(imag);
    Py_DECREF(imag);
    if (i == -1. && PyErr_Occurred())
        return nullptr;

    return PyComplex_FromDoubles(r, i);
}

static PyObject* ComplexRepr(PyObject* self) {
    PyObject* real = PyObject_CallMethodObjArgs(self, PyStrings::gCppReal, nullptr);
    if (!real) return nullptr;
    double r = PyFloat_AsDouble(real);
    Py_DECREF(real);
    if (r == -1. && PyErr_Occurred())
        return nullptr;

    PyObject* imag = PyObject_CallMethodObjArgs(self, PyStrings::gCppImag, nullptr);
    if (!imag) return nullptr;
    double i = PyFloat_AsDouble(imag);
    Py_DECREF(imag);
    if (i == -1. && PyErr_Occurred())
        return nullptr;

    std::ostringstream s;
    s << '(' << r << '+' << i << "j)";
    return CPyCppyy_PyUnicode_FromString(s.str().c_str());
}

static PyObject* ComplexDRealGet(CPPInstance* self, void*)
{
    return PyFloat_FromDouble(((std::complex<double>*)self->GetObject())->real());
}

static int ComplexDRealSet(CPPInstance* self, PyObject* value, void*)
{
    double d = PyFloat_AsDouble(value);
    if (d == -1.0 && PyErr_Occurred())
        return -1;
    ((std::complex<double>*)self->GetObject())->real(d);
    return 0;
}

PyGetSetDef ComplexDReal{(char*)"real", (getter)ComplexDRealGet, (setter)ComplexDRealSet, nullptr, nullptr};


static PyObject* ComplexDImagGet(CPPInstance* self, void*)
{
    return PyFloat_FromDouble(((std::complex<double>*)self->GetObject())->imag());
}

static int ComplexDImagSet(CPPInstance* self, PyObject* value, void*)
{
    double d = PyFloat_AsDouble(value);
    if (d == -1.0 && PyErr_Occurred())
        return -1;
    ((std::complex<double>*)self->GetObject())->imag(d);
    return 0;
}

PyGetSetDef ComplexDImag{(char*)"imag", (getter)ComplexDImagGet, (setter)ComplexDImagSet, nullptr, nullptr};

static PyObject* ComplexDComplex(CPPInstance* self)
{
    double r = ((std::complex<double>*)self->GetObject())->real();
    double i = ((std::complex<double>*)self->GetObject())->imag();
    return PyComplex_FromDoubles(r, i);
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

    if (!IsTemplatedSTLClass(name, "vector")  &&      // vector is dealt with below
           !((PyTypeObject*)pyclass)->tp_iter) {
        if (HasAttrDirect(pyclass, PyStrings::gBegin) && HasAttrDirect(pyclass, PyStrings::gEnd)) {
            if (Cppyy::GetScope(name+"::iterator") || Cppyy::GetScope(name+"::const_iterator")) {
            // iterator protocol fully implemented, so use it (TODO: check return type, rather than
            // the existence of these typedefs? I.e. what's the "full protocol"?)
                ((PyTypeObject*)pyclass)->tp_iter = (getiterfunc)StlSequenceIter;
                Utility::AddToClass(pyclass, "__iter__", (PyCFunction)StlSequenceIter, METH_NOARGS);
            }
        }
        if (!((PyTypeObject*)pyclass)->tp_iter &&     // no iterator resolved
                HasAttrDirect(pyclass, PyStrings::gGetItem) && HasAttrDirect(pyclass, PyStrings::gLen)) {
        // Python will iterate over __getitem__ using integers, but C++ operator[] will never raise
        // a StopIteration. A checked getitem (raising IndexError if beyond size()) works in some
        // cases but would mess up if operator[] is meant to implement an associative container. So,
        // this has to be implemented as an interator protocol.
            ((PyTypeObject*)pyclass)->tp_iter = (getiterfunc)index_iter;
            Utility::AddToClass(pyclass, "__iter__", (PyCFunction)index_iter, METH_NOARGS);
        }
    }

// map operator==() through GenObjectIsEqual to allow comparison to None (true is to
// require that the located method is a CPPOverload; this prevents circular calls as
// GenObjectIsEqual is no CPPOverload); this will search lazily for a global overload
    if (HasAttrDirect(pyclass, PyStrings::gEq, true)) {
        Utility::AddToClass(pyclass, "__cpp_eq__",  "__eq__");
        Utility::AddToClass(pyclass, "__eq__", (PyCFunction)GenObjectIsEqual, METH_O);
    } else if (IsTopLevelClass(pyclass))
        Utility::AddToClass(pyclass, "__eq__", (PyCFunction)GenObjectIsEqualNoCpp, METH_O);

// map operator!=() through GenObjectIsNotEqual to allow comparison to None (see note
// on true above for __eq__); this will search lazily for a global overload
    if (HasAttrDirect(pyclass, PyStrings::gNe, true)) {
        Utility::AddToClass(pyclass, "__cpp_ne__",  "__ne__");
        Utility::AddToClass(pyclass, "__ne__",  (PyCFunction)GenObjectIsNotEqual, METH_O);
    } else if (IsTopLevelClass(pyclass))
        Utility::AddToClass(pyclass, "__ne__",  (PyCFunction)GenObjectIsNotEqualNoCpp, METH_O);


//- class name based pythonization -------------------------------------------

    if (IsTemplatedSTLClass(name, "vector")) {

    // std::vector<bool> is a special case in C++
        if (!sVectorBoolTypeID) sVectorBoolTypeID = (Cppyy::TCppType_t)Cppyy::GetScope("std::vector<bool>");
        if (CPPScope_Check(pyclass) && ((CPPClass*)pyclass)->fCppType == sVectorBoolTypeID) {
            Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)VectorBoolGetItem, METH_O);
            Utility::AddToClass(pyclass, "__setitem__", (PyCFunction)VectorBoolSetItem);
        } else {
        // constructor that takes python collections
            Utility::AddToClass(pyclass, "__real_init", "__init__");
            Utility::AddToClass(pyclass, "__init__", (PyCFunction)VectorInit, METH_VARARGS | METH_KEYWORDS);

        // data with size
            Utility::AddToClass(pyclass, "__real_data", "data");
            Utility::AddToClass(pyclass, "data", (PyCFunction)VectorData);

        // checked getitem
            if (HasAttrDirect(pyclass, PyStrings::gLen)) {
                Utility::AddToClass(pyclass, "_getitem__unchecked", "__getitem__");
                Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)VectorGetItem, METH_O);
            }

        // vector-optimized iterator protocol
            ((PyTypeObject*)pyclass)->tp_iter      = (getiterfunc)vector_iter;

        // helpers for iteration
            const std::string& vtype = Cppyy::ResolveName(name+"::value_type");
            size_t typesz = Cppyy::SizeOf(vtype);
            if (typesz) {
                PyObject* pyvalue_size = PyLong_FromSsize_t(typesz);
                PyObject_SetAttrString(pyclass, "value_size", pyvalue_size);
                Py_DECREF(pyvalue_size);

                PyObject* pyvalue_type = CPyCppyy_PyUnicode_FromString(vtype.c_str());
                PyObject_SetAttrString(pyclass, "value_type", pyvalue_type);
                Py_DECREF(pyvalue_type);
            }
        }
    }

    else if (IsTemplatedSTLClass(name, "map")) {
        Utility::AddToClass(pyclass, "__contains__", (PyCFunction)MapContains, METH_O);
    }

    else if (IsTemplatedSTLClass(name, "pair")) {
        Utility::AddToClass(pyclass, "__getitem__", (PyCFunction)PairUnpack, METH_O);
        Utility::AddToClass(pyclass, "__len__", (PyCFunction)ReturnTwo, METH_NOARGS);
    }

    if (IsTemplatedSTLClass(name, "shared_ptr")) {
        Utility::AddToClass(pyclass, "__real_init", "__init__");
        Utility::AddToClass(pyclass, "__init__", (PyCFunction)SharedPtrInit, METH_VARARGS | METH_KEYWORDS);
    }

    else if (name.find("iterator") != std::string::npos) {
        ((PyTypeObject*)pyclass)->tp_iternext = (iternextfunc)StlIterNext;
        Utility::AddToClass(pyclass, CPPYY__next__, (PyCFunction)StlIterNext, METH_NOARGS);
    }

    else if (name == "string" || name == "std::string") { // TODO: ask backend as well
        Utility::AddToClass(pyclass, "__repr__", (PyCFunction)StlStringRepr, METH_NOARGS);
        Utility::AddToClass(pyclass, "__str__", "c_str");
        Utility::AddToClass(pyclass, "__cmp__", (PyCFunction)StlStringCompare, METH_O);
        Utility::AddToClass(pyclass, "__eq__",  (PyCFunction)StlStringIsEqual, METH_O);
        Utility::AddToClass(pyclass, "__ne__",  (PyCFunction)StlStringIsNotEqual, METH_O);
    }

    else if (name == "basic_string<wchar_t,char_traits<wchar_t>,allocator<wchar_t> >" || \
             name == "std::basic_string<wchar_t,char_traits<wchar_t>,allocator<wchar_t> >") {
        Utility::AddToClass(pyclass, "__repr__", (PyCFunction)StlWStringRepr, METH_NOARGS);
        Utility::AddToClass(pyclass, "__str__", "c_str");
        Utility::AddToClass(pyclass, "__cmp__", (PyCFunction)StlWStringCompare, METH_O);
        Utility::AddToClass(pyclass, "__eq__",  (PyCFunction)StlWStringIsEqual, METH_O);
        Utility::AddToClass(pyclass, "__ne__",  (PyCFunction)StlWStringIsNotEqual, METH_O);
    }

    else if (name == "complex<double>" || name == "std::complex<double>") {
        PyObject_SetAttrString(pyclass, "real",  PyDescr_NewGetSet((PyTypeObject*)pyclass, &ComplexDReal));
        PyObject_SetAttrString(pyclass, "imag",  PyDescr_NewGetSet((PyTypeObject*)pyclass, &ComplexDImag));
        Utility::AddToClass(pyclass, "__complex__", (PyCFunction)ComplexDComplex, METH_NOARGS);
        Utility::AddToClass(pyclass, "__repr__", (PyCFunction)ComplexRepr, METH_NOARGS);
    }

    else if (IsTemplatedSTLClass(name, "complex")) {
        Utility::AddToClass(pyclass, "__cpp_real", "real");
        PyObject_SetAttrString(pyclass, "real", PyDescr_NewGetSet((PyTypeObject*)pyclass, &realComplex));
        Utility::AddToClass(pyclass, "__cpp_imag", "imag");
        PyObject_SetAttrString(pyclass, "imag", PyDescr_NewGetSet((PyTypeObject*)pyclass, &imagComplex)); 
        Utility::AddToClass(pyclass, "__complex__", (PyCFunction)ComplexComplex, METH_NOARGS);
        Utility::AddToClass(pyclass, "__repr__", (PyCFunction)ComplexRepr, METH_NOARGS);
    }

    PyObject* args = PyTuple_New(2);
    Py_INCREF(pyclass);
    PyTuple_SET_ITEM(args, 0, pyclass);

    std::string outer_scope = TypeManip::extract_namespace(name);

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
