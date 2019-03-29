// Bindings
#include "CPyCppyy.h"
#include "CPyCppyy/TPyReturn.h"

#include "CPPInstance.h"

// Standard
#include <stdexcept>


//______________________________________________________________________________
//                        Python expression eval result
//                        =============================
//
// Transport class for bringing objects from python (dynamically typed) to Cling
// (statically typed). It is best to immediately cast a TPyReturn to the real
// type, either implicitly (for builtin types) or explicitly (through a void*
// cast for pointers to ROOT objects).
//
// Examples:
//
//  root [0] TBrowser* b = (void*)TPython::Eval("ROOT.TBrowser()");
//  root [1] int i = TPython::Eval("1+1");
//  root [2] i
//  (int)2
//  root [3] double d = TPython::Eval("1+3.1415");
//  root [4] d
//  (double)4.14150000000000063e+00


//- constructors/destructor --------------------------------------------------
TPyReturn::TPyReturn()
{
// Construct a TPyReturn object from Py_None.
    Py_INCREF(Py_None);
    fPyObject = Py_None;
}

////////////////////////////////////////////////////////////////////////////////
//----------------------------------------------------------------------------
TPyReturn::TPyReturn(PyObject* pyobject)
{
// Construct a TPyReturn from a python object. The python object may represent
// a ROOT object. Steals reference to given python object.
    if (!pyobject) {
        Py_INCREF(Py_None);
        fPyObject = Py_None;
    } else
        fPyObject = pyobject;           // steals reference
}

//----------------------------------------------------------------------------
TPyReturn::TPyReturn(const TPyReturn& other)
{
// Copy constructor. Applies python object reference counting.
    Py_INCREF(other.fPyObject);
    fPyObject = other.fPyObject;
}

//----------------------------------------------------------------------------
TPyReturn& TPyReturn::operator=(const TPyReturn& other)
{
// Assignment operator. Applies python object reference counting.
    if (this != &other) {
        Py_INCREF(other.fPyObject);
        Py_DECREF(fPyObject);
        fPyObject = other.fPyObject;
    }

    return *this;
}

//----------------------------------------------------------------------------
TPyReturn::~TPyReturn()
{
// Destructor. Reference counting for the held python object is in effect.
    Py_DECREF(fPyObject);
}


//- public members -----------------------------------------------------------
TPyReturn::operator char*() const
{
// Cast python return value to C-style string (may fail).
    return (char*)((const char*)*this);
}

//----------------------------------------------------------------------------
TPyReturn::operator const char*() const
{
// Cast python return value to C-style string (may fail).
    if (fPyObject == Py_None)      // for void returns
        return nullptr;

    const char* s = CPyCppyy_PyUnicode_AsString(fPyObject);
    if (PyErr_Occurred()) {
        PyErr_Print();
        return nullptr;
    }

    return s;
}

//----------------------------------------------------------------------------
TPyReturn::operator char() const
{
// Cast python return value to C++ char (may fail).
    std::string s = operator const char*();
    if (s.size())
        return s[0];

    return '\0';
}

//----------------------------------------------------------------------------
TPyReturn::operator long() const
{
// Cast python return value to C++ long (may fail).
    long l = PyLong_AsLong(fPyObject);

    if (PyErr_Occurred())
        PyErr_Print();

    return l;
}

//----------------------------------------------------------------------------
TPyReturn::operator unsigned long() const
{
// Cast python return value to C++ unsigned long (may fail).
    unsigned long ul = PyLong_AsUnsignedLong(fPyObject);

    if (PyErr_Occurred())
        PyErr_Print();

    return ul;
}

//----------------------------------------------------------------------------
TPyReturn::operator double() const
{
// Cast python return value to C++ double (may fail).
    double d = PyFloat_AsDouble(fPyObject);

    if (PyErr_Occurred())
        PyErr_Print();

    return d;
}

//----------------------------------------------------------------------------
TPyReturn::operator void*() const
{
// Cast python return value to ROOT object with dictionary (may fail; note that
// you have to use the void* converter, as CINT will not call any other).
    if (fPyObject == Py_None)
        return nullptr;

    if (CPyCppyy::CPPInstance_Check(fPyObject)) {
        ((CPyCppyy::CPPInstance*)fPyObject)->CppOwns();
        return ((CPyCppyy::CPPInstance*)fPyObject)->GetObject();
    } else
        return fPyObject;               // borrows reference
}

//----------------------------------------------------------------------------
TPyReturn::operator PyObject*() const
{
// Direct return of the held PyObject; note the new reference.
    if (fPyObject == Py_None)
        return nullptr;

    Py_INCREF(fPyObject);
    return fPyObject;
}
