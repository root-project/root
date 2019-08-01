#ifndef CPYCPPYY_PYOBJECTPTR
#define CPYCPPYY_PYOBJECTPTR

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// PyObjectPtr                                                              //
//                                                                          //
// Smart pointer for reference management when cross-inheriting. If fIsSelf //
// is true, then the given PyObject controls the C++ object of which the    //
// smart pointer is part, and no additional reference should be taken.      //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

namespace CPyCppyy {

class PyObjectPtr {
public:
    PyObjectPtr() : fPyObject(nullptr) {}
    PyObjectPtr(PyObject* pyobj, bool isSelf=false) : fIsSelf(isSelf) {
        if (!fIsSelf) Py_XINCREF(pyobj);
        fPyObject = pyobj;
    }
    PyObjectPtr(const PyObjectPtr& other) : fIsSelf(false) {
        Py_XINCREF(other.fPyObject);
        fPyObject = other.fPyObject;
    }
    PyObjectPtr(PyObjectPtr&& other) : fIsSelf(other.fIsSelf) {
        fPyObject = other.fPyObject;
        other.fPyObject = nullptr;
    }
    PyObjectPtr& operator=(const PyObjectPtr& other) {
        if (this != &other) {
            Py_XINCREF(other.fPyObject);
            if (!fIsSelf && fPyObject) Py_DECREF(fPyObject);
            fIsSelf = false;
            fPyObject = other.fPyObject;
        }
        return *this;
    }
    ~PyObjectPtr() {
        if (!fIsSelf && fPyObject) Py_DECREF(fPyObject);
    }

    operator PyObject*() const {
        return fPyObject;
    }

public:
    PyObject* fPyObject;     // actual python object wrapped
    bool fIsSelf;            // if true, we're self-referencing and don't inc/decref
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_PYOBJECTPTR
