// Standard
#include <string.h>

// Bindings
#include "CPyCppyy.h"
#define CPYCPPYY_INTERNAL 1
#include "CPyCppyy/PyException.h"
#undef CPYCPPYY_INTERNAL


//______________________________________________________________________________
//                 C++ exception for throwing python exceptions
//                 ============================================
// Purpose: A C++ exception class for throwing python exceptions
//          through C++ code.
// Created: Apr, 2004, Scott Snyder, from the version in D0's python_util.
//
// Note: Don't be tempted to declare the virtual functions defined here
//       as inline.
//       If you do, you may not be able to properly throw these
//       exceptions across shared libraries.


//- constructors/destructor --------------------------------------------------
CPyCppyy::PyException::PyException()
{
#ifdef WITH_THREAD
    PyGILState_STATE state = PyGILState_Ensure();
#endif

#if PY_VERSION_HEX >= 0x030c0000
    PyObject *pyvalue = PyErr_GetRaisedException();
    PyObject *pytype = pyvalue ? (PyObject *)Py_TYPE(pyvalue) : nullptr;
    PyObject* traceback = pyvalue ? PyException_GetTraceback(pyvalue) : nullptr;
#else
    PyObject* pytype = nullptr, *pyvalue = nullptr, *pytrace = nullptr;
    PyErr_Fetch(&pytype, &pyvalue, &pytrace);
    PyObject* traceback = pytrace; // to keep the original unchanged
    Py_XINCREF(traceback);
#endif

    if (pytype && pyvalue) {
        const char* tname = PyExceptionClass_Name(pytype);
        if (tname) {
            char* dot = strrchr((char*)tname, '.');
            if (dot) tname = dot+1;
            fMsg += tname;
            fMsg += ": ";
        }

        PyObject* msg = PyObject_Str(pyvalue);
        if (msg) {
           fMsg += CPyCppyy_PyText_AsString(msg);
           Py_DECREF(msg);
        }
    }

    std::string locName;
    std::string locFile;
    int locLine = 0;

    while (traceback && traceback != Py_None) {
        PyObject* frame = PyObject_GetAttrString(traceback, "tb_frame");
        PyObject* code = PyObject_GetAttrString(frame, "f_code");
        Py_DECREF(frame);

        PyObject* filename = PyObject_GetAttrString(code, "co_filename");
        Py_DECREF(code);

        PyObject* filenameStr = PyObject_Str(filename);
        locFile = CPyCppyy_PyText_AsString(filenameStr);
        Py_DECREF(filenameStr);
        Py_DECREF(filename);

        PyObject* name = PyObject_GetAttrString(code, "co_name");
        PyObject* nameStr = PyObject_Str(name);
        locName = CPyCppyy_PyText_AsString(nameStr);
        Py_DECREF(nameStr);
        Py_DECREF(name);

        PyObject* lineno = PyObject_GetAttrString(traceback, "tb_lineno");
        locLine = PyLong_AsLong(lineno);
        Py_DECREF(lineno);

        if (locFile == "<string>") { // these are not that useful, skipping
            PyObject* nextTraceback = PyObject_GetAttrString(traceback, "tb_next");
            Py_DECREF(traceback);
            traceback = nextTraceback;
            continue;
        }

        break;
    }

    Py_XDECREF(traceback);

#if PY_VERSION_HEX >= 0x030c0000
    PyErr_SetRaisedException(pyvalue);
#else
    PyErr_Restore(pytype, pyvalue, pytrace);
#endif

    if (fMsg.empty())
        fMsg = "python exception";

    if (!locFile.empty()) {

        // only keeping the filename, not the full path
        locFile = locFile.substr(locFile.find_last_of("/\\") + 1);

        fMsg += " (at " + locFile + ":" + std::to_string(locLine);

        if (locName != "<module>")
            fMsg += " in " + locName;

        fMsg += ")";
    }

#ifdef WITH_THREAD
    PyGILState_Release(state);
#endif
}

CPyCppyy::PyException::~PyException() noexcept
{
// destructor
}


//- public members -----------------------------------------------------------
const char* CPyCppyy::PyException::what() const noexcept
{
// Return reason for throwing this exception: a python exception was raised.
    return fMsg.c_str();
}

void CPyCppyy::PyException::clear() const noexcept
{
// clear Python error, to allow full error handling C++ side
    PyErr_Clear();
}
