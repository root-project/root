// Bindings
#include "CPyCppyy.h"
#include "CPPEnum.h"
#include "PyStrings.h"
#include "TypeManip.h"
#include "Utility.h"


//- private helpers ----------------------------------------------------------
static PyObject* pytype_from_enum_type(const std::string& enum_type)
{
    if (enum_type == "char")
        return (PyObject*)&CPyCppyy_PyText_Type;
    else if (enum_type == "bool")
        return (PyObject*)&PyInt_Type;     // can't use PyBool_Type as base
    else if (strstr("long", enum_type.c_str()))
        return (PyObject*)&PyLong_Type;
    return (PyObject*)&PyInt_Type;         // covers most cases
}

//----------------------------------------------------------------------------
static PyObject* pyval_from_enum(const std::string& enum_type, PyObject* pytype,
        PyObject* btype, Cppyy::TCppEnum_t etype, Cppyy::TCppIndex_t idata) {
    long long llval = Cppyy::GetEnumDataValue(etype, idata);

    if (enum_type == "bool") {
        PyObject* result = (bool)llval ? Py_True : Py_False;
        Py_INCREF(result);
        return result;                      // <- immediate return;
    }

    PyObject* bval;
    if (enum_type == "char") {
        char val = (char)llval;
#if PY_VERSION_HEX < 0x03000000
        bval = CPyCppyy_PyText_FromStringAndSize(&val, 1);
#else
        bval = PyUnicode_FromOrdinal((int)val);
#endif
    } else if (enum_type == "int" || enum_type == "unsigned int")
        bval = PyInt_FromLong((long)llval);
    else
        bval = PyLong_FromLongLong(llval);

    if (!bval)
        return nullptr;      // e.g. when out of range for small integers

    PyObject* args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, bval);
    PyObject* result = ((PyTypeObject*)btype)->tp_new((PyTypeObject*)pytype, args, nullptr);
    Py_DECREF(args);
    return result;
}


//- enum methods -------------------------------------------------------------
static int enum_setattro(PyObject* /* pyclass */, PyObject* /* pyname */, PyObject* /* pyval */)
{
// Helper to make enums read-only.
    PyErr_SetString(PyExc_TypeError, "enum values are read-only");
    return -1;
}

//----------------------------------------------------------------------------
static PyObject* enum_repr(PyObject* self)
{
    using namespace CPyCppyy;

    PyObject* kls_cppname = PyObject_GetAttr((PyObject*)Py_TYPE(self), PyStrings::gCppName);
    if (!kls_cppname) PyErr_Clear();
    PyObject* obj_cppname = PyObject_GetAttr(self, PyStrings::gCppName);
    if (!obj_cppname) PyErr_Clear();
    PyObject* obj_str = Py_TYPE(self)->tp_str(self);

    PyObject* repr = nullptr;
    if (kls_cppname && obj_cppname && obj_str) {
        const std::string resolved = Cppyy::ResolveEnum(CPyCppyy_PyText_AsString(kls_cppname));
        repr = CPyCppyy_PyText_FromFormat("(%s::%s) : (%s) %s",
            CPyCppyy_PyText_AsString(kls_cppname), CPyCppyy_PyText_AsString(obj_cppname),
            resolved.c_str(), CPyCppyy_PyText_AsString(obj_str));
    }
    Py_XDECREF(obj_cppname);
    Py_XDECREF(kls_cppname);

    if (repr) {
        Py_DECREF(obj_str);
        return repr;
    }

    return obj_str;
}


//----------------------------------------------------------------------------
// TODO: factor the following lookup with similar codes in Convertes and TemplateProxy.cxx

static std::map<std::string, std::string> gCTypesNames = {
    {"bool", "c_bool"},
    {"char", "c_char"}, {"wchar_t", "c_wchar"},
    {"std::byte", "c_byte"}, {"int8_t", "c_byte"}, {"uint8_t", "c_ubyte"},
    {"short", "c_short"}, {"int16_t", "c_int16"}, {"unsigned short", "c_ushort"}, {"uint16_t", "c_uint16"},
    {"int", "c_int"}, {"unsigned int", "c_uint"},
    {"long", "c_long"}, {"unsigned long", "c_ulong"},
    {"long long", "c_longlong"}, {"unsigned long long", "c_ulonglong"}};

// Both GetCTypesType and GetCTypesPtrType, rely on the ctypes module itself
// caching the types (thus also making them unique), so no ref-count is needed.
// Further, by keeping a ref-count on the module, it won't be off-loaded until
// the 2nd cleanup cycle.
static PyTypeObject* GetCTypesType(const std::string& cppname)
{
    static PyObject* ctmod = PyImport_ImportModule("ctypes");   // ref-count kept
    if (!ctmod)
        return nullptr;

    auto nn = gCTypesNames.find(cppname);
    if (nn == gCTypesNames.end()) {
        PyErr_Format(PyExc_TypeError, "Can not find ctypes type for \"%s\"", cppname.c_str());
        return nullptr;
    }

    return (PyTypeObject*)PyObject_GetAttrString(ctmod, nn->second.c_str());
}

static PyObject* enum_ctype(PyObject* cls, PyObject* args, PyObject* kwds)
{
    PyObject* pyres = PyObject_GetAttr(cls, CPyCppyy::PyStrings::gUnderlying);
    if (!pyres) PyErr_Clear();

    std::string underlying = pyres ? CPyCppyy_PyText_AsString(pyres) : "int";
    PyTypeObject* ct = GetCTypesType(underlying);
    if (!ct)
        return nullptr;

    return PyType_Type.tp_call((PyObject*)ct, args, kwds);
}


//- creation -----------------------------------------------------------------
CPyCppyy::CPPEnum* CPyCppyy::CPPEnum_New(const std::string& name, Cppyy::TCppScope_t scope)
{
// Create a new enum type based on the actual C++ type. Enum values are added to
// the type but may also live in the enclosing scope.

    CPPEnum* pyenum = nullptr;

    const std::string& ename = scope == Cppyy::gGlobalScope ? name : Cppyy::GetScopedFinalName(scope)+"::"+name;
    Cppyy::TCppEnum_t etype = Cppyy::GetEnum(scope, name);
    if (etype) {
    // create new enum type with labeled values in place, with a meta-class
    // to make sure the enum values are read-only
        const std::string& resolved = Cppyy::ResolveEnum(ename);
        PyObject* pyside_type = pytype_from_enum_type(resolved);
        PyObject* pymetabases = PyTuple_New(1);
        PyObject* btype = (PyObject*)Py_TYPE(pyside_type);
        Py_INCREF(btype);
        PyTuple_SET_ITEM(pymetabases, 0, btype);

        PyObject* args = Py_BuildValue((char*)"sO{}", (name+"_meta").c_str(), pymetabases);
        Py_DECREF(pymetabases);
        PyObject* pymeta = PyType_Type.tp_new(Py_TYPE(pyside_type), args, nullptr);
        Py_DECREF(args);

    // prepare the base class
        PyObject* pybases = PyTuple_New(1);
        Py_INCREF(pyside_type);
        PyTuple_SET_ITEM(pybases, 0, (PyObject*)pyside_type);

    // create the __cpp_name__ for templates
        PyObject* dct = PyDict_New();
        PyObject* pycppname = CPyCppyy_PyText_FromString(ename.c_str());
        PyDict_SetItem(dct, PyStrings::gCppName, pycppname);
        Py_DECREF(pycppname);
        PyObject* pyresolved = CPyCppyy_PyText_FromString(resolved.c_str());
        PyDict_SetItem(dct, PyStrings::gUnderlying, pyresolved);
        Py_DECREF(pyresolved);

    // add the __module__ to allow pickling
        std::string modname = TypeManip::extract_namespace(ename);
        TypeManip::cppscope_to_pyscope(modname);      // :: -> .
        if (!modname.empty()) modname = "."+modname;
        PyObject* pymodname = CPyCppyy_PyText_FromString(("cppyy.gbl"+modname).c_str());
        PyDict_SetItem(dct, PyStrings::gModule, pymodname);
        Py_DECREF(pymodname);

    // create the actual enum class
        args = Py_BuildValue((char*)"sOO", name.c_str(), pybases, dct);
        Py_DECREF(pybases);
        Py_DECREF(dct);
        pyenum = ((PyTypeObject*)pymeta)->tp_new((PyTypeObject*)pymeta, args, nullptr);

    // add pythonizations
        Utility::AddToClass(
            (PyObject*)Py_TYPE(pyenum), "__ctype__", (PyCFunction)enum_ctype, METH_VARARGS | METH_KEYWORDS);
        ((PyTypeObject*)pyenum)->tp_repr = enum_repr;
        ((PyTypeObject*)pyenum)->tp_str  = ((PyTypeObject*)pyside_type)->tp_repr;

    // collect the enum values
        Cppyy::TCppIndex_t ndata = Cppyy::GetNumEnumData(etype);
        bool values_ok = true;
        for (Cppyy::TCppIndex_t idata = 0; idata < ndata; ++idata) {
            PyObject* val = pyval_from_enum(resolved, pyenum, pyside_type, etype, idata);
            if (!val) {
                values_ok = false;
                break;
            }
            PyObject* pydname = CPyCppyy_PyText_FromString(Cppyy::GetEnumDataName(etype, idata).c_str());
            PyObject_SetAttr(pyenum, pydname, val);
            PyObject_SetAttr(val, PyStrings::gCppName, pydname);
            Py_DECREF(pydname);
            Py_DECREF(val);
        }

    // disable writing onto enum values
        ((PyTypeObject*)pymeta)->tp_setattro = enum_setattro;

    // final cleanup
        Py_DECREF(args);
        Py_DECREF(pymeta);

        if (!values_ok) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_ValueError, "could not set some of the enum values");
            Py_DECREF(pyenum);
            return nullptr;
        }

    } else {
    // presumably not a class enum; simply pretend int
        Py_INCREF(&PyInt_Type);
        pyenum = (PyObject*)&PyInt_Type;
    }

    return pyenum;
}
