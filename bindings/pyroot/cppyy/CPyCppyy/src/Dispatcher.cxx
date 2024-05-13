// Bindings
#include "CPyCppyy.h"
#include "Dispatcher.h"
#include "CPPScope.h"
#include "PyStrings.h"
#include "ProxyWrappers.h"
#include "TypeManip.h"
#include "Utility.h"

// Standard
#include <set>
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

// on destruction, the Python object may go first, in which case provide a diagnostic
// warning (raising a PyException may not be possible as this could happen during
// program shutdown); note that this means that the actual result will be the default
// and the caller may need to act on that, but that's still an improvement over a
// possible crash
    code << "    PyObject* iself = (PyObject*)_internal_self;\n"
            "    if (!iself || iself == Py_None) {\n"
            "      PyErr_Warn(PyExc_RuntimeWarning, (char*)\"Call attempted on deleted python-side proxy\");\n"
            "      return";
    if (retType != "void") {
        if (retType.back() != '*')
            code << " " << CPyCppyy::TypeManip::remove_const(retType) << "{}";
        else
            code << " nullptr";
    }
    code << ";\n"
            "    }\n"
            "    Py_INCREF(iself);\n";

// start actual function body
    Utility::ConstructCallbackPreamble(retType, argtypes, code);

// perform actual method call
#if PY_VERSION_HEX < 0x03000000
    code << "    PyObject* mtPyName = PyString_FromString(\"" << mtCppName << "\");\n" // TODO: intern?
#else
    code << "    PyObject* mtPyName = PyUnicode_FromString(\"" << mtCppName << "\");\n"
#endif
            "    PyObject* pyresult = PyObject_CallMethodObjArgs(iself, mtPyName";
    for (Cppyy::TCppIndex_t i = 0; i < nArgs; ++i)
        code << ", pyargs[" << i << "]";
    code << ", NULL);\n    Py_DECREF(mtPyName);\n    Py_DECREF(iself);\n";

// close
    Utility::ConstructCallbackReturn(retType, (int)nArgs, code);
}

//----------------------------------------------------------------------------
namespace {
    struct BaseInfo {
        BaseInfo(Cppyy::TCppType_t t, std::string&& bn, std::string&& bns) :
            btype(t), bname(bn), bname_scoped(bns) {}
        Cppyy::TCppType_t btype;
        std::string bname;
        std::string bname_scoped;
    };

    typedef std::vector<BaseInfo> BaseInfos_t;
    typedef std::vector<Cppyy::TCppMethod_t> Ctors_t;
    typedef std::vector<Ctors_t> AllCtors_t;
    typedef std::vector<std::pair<Cppyy::TCppMethod_t, size_t>> CtorInfos_t;
} // unnamed namespace

static void build_constructors(
    const std::string& derivedName, const BaseInfos_t& base_infos, const AllCtors_t& ctors,
    std::ostringstream& code, const CtorInfos_t& methods = CtorInfos_t{}, size_t idx = 0)
{
    if (idx < ctors.size()) {
        for (const auto& method : ctors[idx]) {
             size_t argsmin = (size_t)Cppyy::GetMethodReqArgs(method);
             size_t argsmax = (size_t)Cppyy::GetMethodNumArgs(method);
             for (size_t i = argsmin; i <= argsmax; ++i) {
                 CtorInfos_t methods1{methods};
                 methods1.emplace_back(method, i);
                 build_constructors(derivedName, base_infos, ctors, code, methods1, idx+1);
             }
        }
    } else {
    // this is as deep as we go; start writing
        code << "  " << derivedName << "(";

    // declare arguments
        std::vector<size_t> arg_tots; arg_tots.reserve(methods.size());
        for (Ctors_t::size_type i = 0; i < methods.size(); ++i) {
            const auto& cinfo = methods[i];
            if (i != 0 && (arg_tots.back() || 1 < arg_tots.size())) code << ", ";
            size_t nArgs = cinfo.second;
            arg_tots.push_back(i == 0 ? nArgs : nArgs+arg_tots.back());

            if (i != 0) code << "__cppyy_internal::Sep*";
            size_t offset = (i != 0 ? arg_tots[i-1] : 0);
            for (size_t j = 0; j < nArgs; ++j) {
                if (i != 0 || j != 0) code << ", ";
                code << Cppyy::GetMethodArgType(cinfo.first, j) << " a" << (j+offset);
            }
        }
        code << ") : ";

    // pass arguments to base constructors
        for (BaseInfos_t::size_type i = 0; i < base_infos.size(); ++i) {
            if (i != 0) code << ", ";
            code << base_infos[i].bname << "(";
            size_t first = (i != 0 ? arg_tots[i-1] : 0);
            for (size_t j = first; j < arg_tots[i]; ++j) {
                if (j != first) code << ", ";
                bool isRValue = CPyCppyy::TypeManip::compound(\
                    Cppyy::GetMethodArgType(methods[i].first, j-first)) == "&&";
                if (isRValue) code << "std::move(";
                code << "a" << j;
                if (isRValue) code << ")";
            }
            code << ")";
        }
        code << " {}\n";
    }
}

namespace {

using namespace Cppyy;

static inline
std::vector<TCppIndex_t> FindBaseMethod(TCppScope_t tbase, const std::string mtCppName)
{
// Recursively walk the inheritance tree to find the overloads of the named method
    std::vector<TCppIndex_t> result;
    result = GetMethodIndicesFromName(tbase, mtCppName);
    if (result.empty()) {
        for (TCppIndex_t ibase = 0; ibase < GetNumBases(tbase); ++ibase) {
            TCppScope_t b = GetScope(GetBaseName(tbase, ibase));
            result = FindBaseMethod(b, mtCppName);
            if (!result.empty())
                break;
        }
    }
    return result;
}

} // unnamed namespace

bool CPyCppyy::InsertDispatcher(CPPScope* klass, PyObject* bases, PyObject* dct, std::ostringstream& err)
{
// Scan all methods in dct and where it overloads base methods in klass, create
// dispatchers on the C++ side. Then interject the dispatcher class.

    if (!PyTuple_Check(bases) || !PyTuple_GET_SIZE(bases) || !dct || !PyDict_Check(dct)) {
        err << "internal error: expected tuple of bases and proper dictionary";
        return false;
    }

    if (!Utility::IncludePython()) {
        err << "failed to include Python.h";
        return false;
    }

// collect all bases, error checking the hierarchy along the way
    const Py_ssize_t nBases = PyTuple_GET_SIZE(bases);
    BaseInfos_t base_infos; base_infos.reserve(nBases);
    for (Py_ssize_t ibase = 0; ibase < nBases; ++ibase) {
        if (!CPPScope_Check(PyTuple_GET_ITEM(bases, ibase)))
            continue;

        Cppyy::TCppType_t basetype = ((CPPScope*)PyTuple_GET_ITEM(bases, ibase))->fCppType;

        if (!basetype) {
            err << "base class is incomplete";
            break;
        }

        if (Cppyy::IsNamespace(basetype)) {
            err << Cppyy::GetScopedFinalName(basetype) << " is a namespace";
            break;
        }

        if (!Cppyy::HasVirtualDestructor(basetype)) {
            const std::string& bname = Cppyy::GetScopedFinalName(basetype);
            PyErr_Warn(PyExc_RuntimeWarning, (char*)("class \""+bname+"\" has no virtual destructor").c_str());
        }

        base_infos.emplace_back(
            basetype, TypeManip::template_base(Cppyy::GetFinalName(basetype)), Cppyy::GetScopedFinalName(basetype));
    }

// TODO: check deep hierarchy for multiple inheritance
    bool isDeepHierarchy = klass->fCppType && base_infos.front().btype != klass->fCppType;

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
         << "class " << derivedName << " : ";
    for (BaseInfos_t::size_type ibase = 0; ibase < base_infos.size(); ++ibase) {
        if (ibase != 0) code << ", ";
        code << "public ::" << base_infos[ibase].bname_scoped;
    }
    code << " {\n";
    if (!isDeepHierarchy)
        code << "protected:\n  CPyCppyy::DispatchPtr _internal_self;\n";
    code << "public:\n";

// add a virtual destructor for good measure, which is allowed to be "overridden" by
// the conventional __destruct__ method (note that __del__ is always called, too, if
// provided, but only when the Python object goes away; furthermore, if the Python
// object goes before the C++ one, only __del__ is called)
    if (PyMapping_HasKeyString(dct, (char*)"__destruct__")) {
        code << "  virtual ~" << derivedName << "() {\n"
                "    PyObject* iself = (PyObject*)_internal_self;\n"
                "    if (!iself || iself == Py_None)\n"
                "      return;\n"      // safe, as destructor always returns void
                "    Py_INCREF(iself);\n"
                "    PyObject* mtPyName = PyUnicode_FromString(\"__destruct__\");\n"
                "    PyObject* pyresult = PyObject_CallMethodObjArgs(iself, mtPyName, NULL);\n"
                "    Py_DECREF(mtPyName);\n    Py_DECREF(iself);\n";

    // this being a destructor, print on exception rather than propagate using the
    // magic C++ exception ...
        code << "      if (!pyresult) PyErr_Print();\n"
                "      else { Py_DECREF(pyresult); }\n"
                "  }\n";
    } else
        code << "  virtual ~" << derivedName << "() {}\n";

// methods: first collect all callables, then get overrides from base classes, for
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

// protected methods and data need their access changed in the C++ trampoline and then
// exposed on the Python side; so, collect their names as we go along
    std::set<std::string> protected_names;

// simple case: methods from current class (collect constructors along the way)
    int has_default = 0, has_cctor = 0, has_ctors = 0, has_tmpl_ctors = 0;
    AllCtors_t ctors{base_infos.size()};
    for (BaseInfos_t::size_type ibase = 0; ibase < base_infos.size(); ++ibase) {
        const auto& binfo = base_infos[ibase];

        const Cppyy::TCppIndex_t nMethods = Cppyy::GetNumMethods(binfo.btype);
        bool cctor_found = false, default_found = false, any_ctor_found = false;
        for (Cppyy::TCppIndex_t imeth = 0; imeth < nMethods; ++imeth) {
            Cppyy::TCppMethod_t method = Cppyy::GetMethod(binfo.btype, imeth);

            if (Cppyy::IsConstructor(method)) {
                any_ctor_found = true;
                if (Cppyy::IsPublicMethod(method) || Cppyy::IsProtectedMethod(method)) {
                    Cppyy::TCppIndex_t nreq = Cppyy::GetMethodReqArgs(method);
                    if (nreq == 0) default_found = true;
                    else if (!cctor_found && nreq == 1) {
                        const std::string& argtype = Cppyy::GetMethodArgType(method, 0);
                        if (TypeManip::compound(argtype) == "&" && TypeManip::clean_type(argtype, false) == binfo.bname_scoped)
                            cctor_found = true;
                    }
                    ctors[ibase].push_back(method);
                }
                continue;
            }

            std::string mtCppName = Cppyy::GetMethodName(method);
            PyObject* key = CPyCppyy_PyText_FromString(mtCppName.c_str());
            int contains = PyDict_Contains(dct, key);
            if (contains == -1) PyErr_Clear();
            if (contains != 1) {
                Py_DECREF(key);

            // if the method is protected, we expose it through re-declaration and forwarding (using
            // does not work here b/c there may be private overloads)
                if (Cppyy::IsProtectedMethod(method)) {
                    protected_names.insert(mtCppName);

                    code << "  " << Cppyy::GetMethodResultType(method) << " " << mtCppName << "(";
                    Cppyy::TCppIndex_t nArgs = Cppyy::GetMethodNumArgs(method);
                    for (Cppyy::TCppIndex_t i = 0; i < nArgs; ++i) {
                        if (i != 0) code << ", ";
                        code << Cppyy::GetMethodArgType(method, i) << " arg" << i;
                    }
                    code << ") ";
                    if (Cppyy::IsConstMethod(method)) code << "const ";
                    code << "{\n    return " << binfo.bname << "::" << mtCppName << "(";
                    for (Cppyy::TCppIndex_t i = 0; i < nArgs; ++i) {
                        if (i != 0) code << ", ";
                        code << "arg" << i;
                    }
                    code << ");\n  }\n";
                }

                continue;
            }

            InjectMethod(method, mtCppName, code);

            if (PyDict_DelItem(clbs, key) != 0)
                PyErr_Clear();        // happens for overloads
            Py_DECREF(key);
        }

    // support for templated ctors in single inheritance (TODO: also multi possible?)
        if (base_infos.size() == 1) {
            const Cppyy::TCppIndex_t nTemplMethods = Cppyy::GetNumTemplatedMethods(binfo.btype);
            for (Cppyy::TCppIndex_t imeth = 0; imeth < nTemplMethods; ++imeth) {
                if (Cppyy::IsTemplatedConstructor(binfo.btype, imeth)) {
                    any_ctor_found = true;
                    has_tmpl_ctors += 1;
                    break;        // one suffices to map as argument packs are used
                }
            }
        }

    // count the cctors and default ctors to determine whether each base has one
        if (cctor_found   || (!cctor_found && !any_ctor_found))   has_cctor   += 1;
        if (default_found || (!default_found && !any_ctor_found)) has_default += 1;
        if (any_ctor_found && !has_tmpl_ctors)                    has_ctors   += 1;
    }

// try to locate left-overs in base classes
    for (const auto& binfo : base_infos) {
        if (PyDict_Size(clbs)) {
            size_t nbases = Cppyy::GetNumBases(binfo.btype);
            for (size_t ibase = 0; ibase < nbases; ++ibase) {
                Cppyy::TCppScope_t tbase = (Cppyy::TCppScope_t)Cppyy::GetScope( \
                    Cppyy::GetBaseName(binfo.btype, ibase));

                PyObject* keys = PyDict_Keys(clbs);
                for (Py_ssize_t i = 0; i < PyList_GET_SIZE(keys); ++i) {
                // TODO: should probably invert this looping; but that makes handling overloads clunky
                    PyObject* key = PyList_GET_ITEM(keys, i);
                    std::string mtCppName = CPyCppyy_PyText_AsString(key);
                    const auto& v = FindBaseMethod(tbase, mtCppName);
                    for (auto idx : v)
                        InjectMethod(Cppyy::GetMethod(tbase, idx), mtCppName, code);
                    if (!v.empty()) {
                        if (PyDict_DelItem(clbs, key) != 0) PyErr_Clear();
                    }
                }
                Py_DECREF(keys);
            }
        }
    }
    Py_DECREF(clbs);

// constructors: build up from the argument types of the base class, for use by the Python
// derived class (inheriting with/ "using" does not work b/c base class constructors may
// have been deleted),
    build_constructors(derivedName, base_infos, ctors, code);

// for working with C++ templates, additional constructors are needed to make
// sure the python object is properly carried, but they can only be generated
// if the base class supports them
    if (1 < nBases && (!has_ctors || has_default == nBases))
        code << "  " << derivedName << "() {}\n";
    if (has_cctor == nBases) {
        code << "  " << derivedName << "(const " << derivedName << "& other) : ";
        for (BaseInfos_t::size_type ibase = 0; ibase < base_infos.size(); ++ibase) {
            if (ibase != 0) code << ", ";
            code << base_infos[ibase].bname << "(other)";
        }
        if (!isDeepHierarchy)
            code << ", _internal_self(other._internal_self, this)";
        code << " {}\n";
    }
    if (has_tmpl_ctors && base_infos.size() == 1) {
    // support for templated ctors in single inheritance (TODO: also multi possible?)
        code << "  template<typename ...Args>\n  " << derivedName << "(Args... args) : "
             << base_infos[0].bname << "(args...) {}\n";
    }

// destructor: default is fine

// pull in data members that are protected
    bool setPublic = false;
    for (const auto& binfo : base_infos) {
        Cppyy::TCppIndex_t nData = Cppyy::GetNumDatamembers(binfo.btype);
        for (Cppyy::TCppIndex_t idata = 0; idata < nData; ++idata) {
            if (Cppyy::IsProtectedData(binfo.btype, idata)) {
                const std::string dm_name = Cppyy::GetDatamemberName(binfo.btype, idata);
                if (dm_name != "_internal_self") {
                    const std::string& dname = Cppyy::GetDatamemberName(binfo.btype, idata);
                    protected_names.insert(dname);
                    if (!setPublic) {
                        code << "public:\n";
                        setPublic = true;
                    }
                    code << "  using " << binfo.bname << "::" << dname << ";\n";
                }
            }
        }
    }

// initialize the dispatch pointer for all direct bases that have one
    BaseInfos_t::size_type disp_inited = 0;
    code << "public:\n  static void _init_dispatchptr(" << derivedName << "* inst, PyObject* self) {\n";
    if (1 < base_infos.size()) {
        for (const auto& binfo : base_infos) {
             if (Cppyy::GetDatamemberIndex(binfo.btype, "_internal_self") != (Cppyy::TCppIndex_t)-1) {
                 code << "    " << binfo.bname << "::_init_dispatchptr(inst, self);\n";
                 disp_inited += 1;
             }
        }
    }
// The dispatch initializer is only used in constructors, and C++ object start out
// as owned by C++, with Python ownership explicitly set only later. To match, the
// dispatch pointer needs to start out with a hard reference, i.e. C++ ownership of
// the dispatch object. If the constructor has __creates__ set to True (default),
// then a call to PythonOwns() will switch the hard ref to a weak ref, preventing
// accidental circular references.
    if (disp_inited != base_infos.size())
       code << "    new ((void*)&inst->_internal_self) CPyCppyy::DispatchPtr{self, true};\n";
    code << "  }";

// provide an accessor to re-initialize after round-tripping from C++ (internal)
    code << "\n  static PyObject* _get_dispatch(" << derivedName << "* inst) {\n"
            "    PyObject* res = (PyObject*)inst->_internal_self;\n"
            "    Py_XINCREF(res); return res;\n  }";

// finish class declaration
    code << "};\n}";

// finally, compile the code
    if (!Cppyy::Compile(code.str())) {
        err << "failed to compile the dispatcher code";
        return false;
    }

// keep track internally of the actual C++ type (this is used in
// CPPConstructor to call the dispatcher's one instead of the base)
    Cppyy::TCppScope_t disp = Cppyy::GetScope("__cppyy_internal::"+derivedName);
    if (!disp) {
        err << "failed to retrieve the internal dispatcher";
        return false;
    }
    klass->fCppType = disp;

// at this point, the dispatcher only lives in C++, as opposed to regular classes
// that are part of the hierarchy in Python, so create it, which will cache it for
// later use by e.g. the MemoryRegulator
    unsigned int flags = (unsigned int)(klass->fFlags & CPPScope::kIsMultiCross);
    PyObject* disp_proxy = CPyCppyy::CreateScopeProxy(disp, flags);
    if (flags) ((CPPScope*)disp_proxy)->fFlags |= CPPScope::kIsMultiCross;
    ((CPPScope*)disp_proxy)->fFlags |= CPPScope::kIsPython;

// finally, to expose protected members, copy them over from the C++ dispatcher base
// to the Python dictionary (the C++ dispatcher's Python proxy is not a base of the
// Python class to keep the inheritance tree intact)
    for (const auto& name : protected_names) {
         PyObject* disp_dct = PyObject_GetAttr(disp_proxy, PyStrings::gDict);
         PyObject* pyf = PyMapping_GetItemString(disp_dct, (char*)name.c_str());
         if (pyf) {
             PyObject_SetAttrString((PyObject*)klass, (char*)name.c_str(), pyf);
             Py_DECREF(pyf);
         }
         Py_DECREF(disp_dct);
    }

    Py_XDECREF(disp_proxy);

    return true;
}
