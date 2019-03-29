// Bindings
#include "CPyCppyy.h"
#include "Dispatcher.h"
#include "CPPScope.h"
#include "PyStrings.h"
#include "TypeManip.h"
#include "Utility.h"

// Standard
#include <sstream>


//----------------------------------------------------------------------------
static inline void InjectMethod(Cppyy::TCppMethod_t method, const std::string& mtCppName, std::ostringstream& code)
{
// inject implementation for an overridden method
    using namespace CPyCppyy;

// method declaration
    std::string retType = Cppyy::GetMethodResultType(method);
    code << "  " << retType << " " << mtCppName << "(";

// build out the signature with predictable formal names
    Cppyy::TCppIndex_t nArgs = Cppyy::GetMethodNumArgs(method);
    std::vector<std::string> argtypes; argtypes.reserve(nArgs);
    for (Cppyy::TCppIndex_t i = 0; i < nArgs; ++i) {
        argtypes.push_back(Cppyy::GetMethodArgType(method, i));
        if (i != 0) code << ", ";
        code << argtypes.back() << " arg" << i;
    }
    code << ") ";
    if (Cppyy::IsConstMethod(method)) code << "const ";
    code << "{\n";

// start function body
    Utility::ConstructCallbackPreamble(retType, argtypes, code);

// perform actual method call
#if PY_VERSION_HEX < 0x03000000
    code << "    PyObject* mtPyName = PyString_FromString(\"" << mtCppName << "\");\n" // TODO: intern?
#else
    code << "    PyObject* mtPyName = PyUnicode_FromString(\"" << mtCppName << "\");\n"
#endif
            "    PyObject* pyresult = PyObject_CallMethodObjArgs(m_self, mtPyName";
    for (Cppyy::TCppIndex_t i = 0; i < nArgs; ++i)
        code << ", pyargs[" << i << "]";
    code << ", NULL);\n    Py_DECREF(mtPyName);\n";

// close
    Utility::ConstructCallbackReturn(retType == "void", nArgs, code);
}

//----------------------------------------------------------------------------
bool CPyCppyy::InsertDispatcher(CPPScope* klass, PyObject* dct)
{
// Scan all methods in dct and where it overloads base methods in klass, create
// dispatchers on the C++ side. Then interject the dispatcher class.
    if (Cppyy::IsNamespace(klass->fCppType) || !PyDict_Check(dct))
        return false;

    if (!Utility::IncludePython())
        return false;

    const std::string& baseName       = TypeManip::template_base(Cppyy::GetFinalName(klass->fCppType));
    const std::string& baseNameScoped = Cppyy::GetScopedFinalName(klass->fCppType);

// once classes can be extended, should consider re-use; for now, since derived
// python classes can differ in what they override, simply use different shims
    static int counter = 0;
    std::ostringstream osname;
    osname << "Dispatcher" << ++counter;
    const std::string& derivedName = osname.str();

// generate proxy class with the relevant method dispatchers
    std::ostringstream code;

// start class declaration
    code << "namespace __cppyy_internal {\n"
         << "class " << derivedName << " : public ::" << baseNameScoped << " {\n"
            "  PyObject* m_self;\n"
            "public:\n";

// constructors are simply inherited
    code << "  using " << baseName << "::" << baseName << ";\n";

// methods: first collect all callables, then get overrides from base class, for
// those that are still missing, search the hierarchy
    PyObject* clbs = PyDict_New();
    PyObject* items = PyDict_Items(dct);
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(items); ++i) {
        PyObject* value = PyTuple_GET_ITEM(PyList_GET_ITEM(items, i), 1);
        if (PyCallable_Check(value))
            PyDict_SetItem(clbs, PyTuple_GET_ITEM(PyList_GET_ITEM(items, i), 0), value);
    }
    Py_DECREF(items);
    if (PyDict_DelItem(clbs, PyStrings::gInit) != 0)
        PyErr_Clear();

// simple case: methods from current class
    const Cppyy::TCppIndex_t nMethods = Cppyy::GetNumMethods(klass->fCppType);
    for (Cppyy::TCppIndex_t imeth = 0; imeth < nMethods; ++imeth) {
        Cppyy::TCppMethod_t method = Cppyy::GetMethod(klass->fCppType, imeth);

        std::string mtCppName = Cppyy::GetMethodName(method);
        PyObject* key = CPyCppyy_PyUnicode_FromString(mtCppName.c_str());
        int contains = PyDict_Contains(dct, key);
        if (contains == -1) PyErr_Clear();
        if (contains != 1) {
            Py_DECREF(key);
            continue;
        }

        InjectMethod(method, mtCppName, code);

        if (PyDict_DelItem(clbs, key) != 0)
            PyErr_Clear();        // happens for overloads
        Py_DECREF(key);
    }

// try to locate left-overs in base classes
    if (PyDict_Size(clbs)) {
        size_t nbases = Cppyy::GetNumBases(klass->fCppType);
        for (size_t ibase = 0; ibase < nbases; ++ibase) {
            Cppyy::TCppScope_t tbase = (Cppyy::TCppScope_t)Cppyy::GetScope( \
                Cppyy::GetBaseName(klass->fCppType, ibase));

            PyObject* keys = PyDict_Keys(clbs);
            for (Py_ssize_t i = 0; i < PyList_GET_SIZE(keys); ++i) {
            // TODO: should probably invert this looping; but that makes handling overloads clunky
                PyObject* key = PyList_GET_ITEM(keys, i);
                std::string mtCppName = CPyCppyy_PyUnicode_AsString(key);
                const auto& v = Cppyy::GetMethodIndicesFromName(tbase, mtCppName);
                for (auto idx : v)
                    InjectMethod(Cppyy::GetMethod(tbase, idx), mtCppName, code);
                if (!v.empty()) {
                    if (PyDict_DelItem(clbs, key) != 0) PyErr_Clear();
                }
             }
             Py_DECREF(keys);
        }
    }
    Py_DECREF(clbs);

// finish class declaration
    code << "};\n}";

// finally, compile the code
    if (!Cppyy::Compile(code.str()))
        return false;

// keep track internally of the actual C++ type (this is used in
// CPPConstructor to call the dispatcher's one instead of the base)
    Cppyy::TCppScope_t disp = Cppyy::GetScope("__cppyy_internal::"+derivedName);
    if (!disp) return false;
    klass->fCppType = disp;

    return true;
}
