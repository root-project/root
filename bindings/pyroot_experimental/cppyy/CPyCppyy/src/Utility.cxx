// Bindings
#include "CPyCppyy.h"
#include "Utility.h"
#include "CPPFunction.h"
#include "CPPInstance.h"
#include "CPPOverload.h"
#include "ProxyWrappers.h"
#include "PyCallable.h"
#include "PyStrings.h"
#include "CustomPyTypes.h"
#include "TemplateProxy.h"

// Standard
#include <string.h>
#include <algorithm>
#include <list>
#include <mutex>
#include <sstream>
#include <utility>


//- data _____________________________________________________________________
dict_lookup_func CPyCppyy::gDictLookupOrg = 0;
bool CPyCppyy::gDictLookupActive = false;

typedef std::map<std::string, std::string> TC2POperatorMapping_t;
static TC2POperatorMapping_t gC2POperatorMapping;

namespace {

    using namespace CPyCppyy::Utility;

    struct InitOperatorMapping_t {
    public:
        InitOperatorMapping_t() {
        // Initialize the global map of operator names C++ -> python.

         // gC2POperatorMapping["[]"]  = "__setitem__";     // depends on return type
         // gC2POperatorMapping["+"]   = "__add__";         // depends on # of args (see __pos__)
         // gC2POperatorMapping["-"]   = "__sub__";         // id. (eq. __neg__)
         // gC2POperatorMapping["*"]   = "__mul__";         // double meaning in C++

            gC2POperatorMapping["[]"]  = "__getitem__";
            gC2POperatorMapping["()"]  = "__call__";
            gC2POperatorMapping["/"]   = CPPYY__div__;
            gC2POperatorMapping["%"]   = "__mod__";
            gC2POperatorMapping["**"]  = "__pow__";
            gC2POperatorMapping["<<"]  = "__lshift__";
            gC2POperatorMapping[">>"]  = "__rshift__";
            gC2POperatorMapping["&"]   = "__and__";
            gC2POperatorMapping["|"]   = "__or__";
            gC2POperatorMapping["^"]   = "__xor__";
            gC2POperatorMapping["~"]   = "__inv__";
            gC2POperatorMapping["+="]  = "__iadd__";
            gC2POperatorMapping["-="]  = "__isub__";
            gC2POperatorMapping["*="]  = "__imul__";
            gC2POperatorMapping["/="]  = CPPYY__idiv__;
            gC2POperatorMapping["%="]  = "__imod__";
            gC2POperatorMapping["**="] = "__ipow__";
            gC2POperatorMapping["<<="] = "__ilshift__";
            gC2POperatorMapping[">>="] = "__irshift__";
            gC2POperatorMapping["&="]  = "__iand__";
            gC2POperatorMapping["|="]  = "__ior__";
            gC2POperatorMapping["^="]  = "__ixor__";
            gC2POperatorMapping["=="]  = "__eq__";
            gC2POperatorMapping["!="]  = "__ne__";
            gC2POperatorMapping[">"]   = "__gt__";
            gC2POperatorMapping["<"]   = "__lt__";
            gC2POperatorMapping[">="]  = "__ge__";
            gC2POperatorMapping["<="]  = "__le__";

        // the following type mappings are "exact"
            gC2POperatorMapping["const char*"]  = "__str__";
            gC2POperatorMapping["char*"]        = "__str__";
            gC2POperatorMapping["const char *"] = gC2POperatorMapping[ "const char*" ];
            gC2POperatorMapping["char *"]       = gC2POperatorMapping[ "char*" ];
            gC2POperatorMapping["int"]          = "__int__";
            gC2POperatorMapping["long"]         = CPPYY__long__;
            gC2POperatorMapping["double"]       = "__float__";

        // the following type mappings are "okay"; the assumption is that they
        // are not mixed up with the ones above or between themselves (and if
        // they are, that it is done consistently)
            gC2POperatorMapping["short"]              = "__int__";
            gC2POperatorMapping["unsigned short"]     = "__int__";
            gC2POperatorMapping["unsigned int"]       = CPPYY__long__;
            gC2POperatorMapping["unsigned long"]      = CPPYY__long__;
            gC2POperatorMapping["long long"]          = CPPYY__long__;
            gC2POperatorMapping["unsigned long long"] = CPPYY__long__;
            gC2POperatorMapping["float"]              = "__float__";

            gC2POperatorMapping["->"]  = "__follow__";      // not an actual python operator
            gC2POperatorMapping["="]   = "__assign__";      // id.

#if PY_VERSION_HEX < 0x03000000
            gC2POperatorMapping["bool"] = "__nonzero__";
#else
            gC2POperatorMapping["bool"] = "__bool__";
#endif
        }
    } initOperatorMapping_;

    std::once_flag sOperatorTemplateFlag;
    void InitOperatorTemplate() {
    /* TODO: move to Cppyy.cxx
       gROOT->ProcessLine(
           "namespace _pycppyy_internal { template<class C1, class C2>"
           " bool is_equal(const C1& c1, const C2& c2){ return (bool)(c1 == c2); } }");
       gROOT->ProcessLine(
           "namespace _cpycppyy_internal { template<class C1, class C2>"
           " bool is_not_equal(const C1& c1, const C2& c2){ return (bool)(c1 != c2); } }");
    */
    }

// TODO: this should live with Helpers
    inline void RemoveConst(std::string& cleanName) {
        std::string::size_type spos = std::string::npos;
        while ((spos = cleanName.find("const")) != std::string::npos) {
            cleanName.swap(cleanName.erase(spos, 5));
        }
    }

} // unnamed namespace


//- public functions ---------------------------------------------------------
unsigned long CPyCppyy::PyLongOrInt_AsULong(PyObject* pyobject)
{
// Convert <pybject> to C++ unsigned long, with bounds checking, allow int -> ulong.
    unsigned long ul = PyLong_AsUnsignedLong(pyobject);
    if (PyErr_Occurred() && PyInt_Check(pyobject)) {
        PyErr_Clear();
        long i = PyInt_AS_LONG(pyobject);
        if (0 <= i) {
            ul = (unsigned long)i;
        } else {
            PyErr_SetString(PyExc_ValueError,
                "can\'t convert negative value to unsigned long");
        }
    }

    return ul;
}

//----------------------------------------------------------------------------
ULong64_t CPyCppyy::PyLongOrInt_AsULong64(PyObject* pyobject)
{
// Convert <pyobject> to C++ unsigned long long, with bounds checking.
    ULong64_t ull = PyLong_AsUnsignedLongLong(pyobject);
    if (PyErr_Occurred() && PyInt_Check(pyobject)) {
        PyErr_Clear();
        long i = PyInt_AS_LONG(pyobject);
        if (0 <= i) {
            ull = (ULong64_t)i;
        } else {
            PyErr_SetString(PyExc_ValueError,
                "can\'t convert negative value to unsigned long long");
        }
    }

    return ull;
}

//----------------------------------------------------------------------------
bool CPyCppyy::Utility::AddToClass(
     PyObject* pyclass, const char* label, PyCFunction cfunc, int flags)
{
// Add the given function to the class under name 'label'.

// use list for clean-up (.so's are unloaded only at interpreter shutdown)
    static std::list<PyMethodDef> s_pymeths;

    s_pymeths.push_back(PyMethodDef());
    PyMethodDef* pdef = &s_pymeths.back();
    pdef->ml_name  = const_cast<char*>(label);
    pdef->ml_meth  = cfunc;
    pdef->ml_flags = flags;
    pdef->ml_doc   = nullptr;

    PyObject* func = PyCFunction_New(pdef, nullptr);
    PyObject* method = CustomInstanceMethod_New(func, nullptr, pyclass);
    bool isOk = PyObject_SetAttrString(pyclass, pdef->ml_name, method) == 0;
    Py_DECREF(method);
    Py_DECREF(func);

    if (PyErr_Occurred())
        return false;

    if (!isOk) {
        PyErr_Format(PyExc_TypeError, "could not add method %s", label);
        return false;
    }

    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::Utility::AddToClass(PyObject* pyclass, const char* label, const char* func)
{
// Add the given function to the class under name 'label'.
    PyObject* pyfunc = PyObject_GetAttrString(pyclass, const_cast<char*>(func));
    if (!pyfunc)
        return false;

    bool isOk = PyObject_SetAttrString(pyclass, const_cast<char*>(label), pyfunc) == 0;

    Py_DECREF(pyfunc);
    return isOk;
}

//----------------------------------------------------------------------------
bool CPyCppyy::Utility::AddToClass(PyObject* pyclass, const char* label, PyCallable* pyfunc)
{
// Add the given function to the class under name 'label'.
    CPPOverload* method =
        (CPPOverload*)PyObject_GetAttrString(pyclass, const_cast<char*>(label));

    if (!method || !CPPOverload_Check(method)) {
    // not adding to existing CPPOverload; add callable directly to the class
        if (PyErr_Occurred())
            PyErr_Clear();
        Py_XDECREF((PyObject*)method);
        method = CPPOverload_New(label, pyfunc);
        bool isOk = PyObject_SetAttrString(
            pyclass, const_cast<char*>(label), (PyObject*)method) == 0;
        Py_DECREF(method);
        return isOk;
    }

    method->AddMethod(pyfunc);

    Py_DECREF(method);
    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::Utility::AddUsingToClass(PyObject* pyclass, const char* method)
{
// Helper to add base class methods to the derived class one (this covers the
// 'using' cases, which the dictionary does not provide).
    CPPOverload* derivedMethod =
        (CPPOverload*)PyObject_GetAttrString(pyclass, const_cast<char*>(method));
    if (!CPPOverload_Check(derivedMethod)) {
        Py_XDECREF(derivedMethod);
        return false;
    }

    PyObject* mro = PyObject_GetAttr(pyclass, PyStrings::gMRO);
    if (!mro || ! PyTuple_Check(mro)) {
        Py_XDECREF(mro);
        Py_DECREF(derivedMethod);
        return false;
    }

    CPPOverload* baseMethod = 0;
    for (int i = 1; i < PyTuple_GET_SIZE(mro); ++i) {
        baseMethod = (CPPOverload*)PyObject_GetAttrString(
            PyTuple_GET_ITEM(mro, i), const_cast<char*>(method));

        if (!baseMethod) {
            PyErr_Clear();
            continue;
        }

        if (CPPOverload_Check(baseMethod))
            break;

        Py_DECREF(baseMethod);
        baseMethod = 0;
    }

    Py_DECREF(mro);

    if (!CPPOverload_Check(baseMethod)) {
        Py_XDECREF(baseMethod);
        Py_DECREF(derivedMethod);
        return false;
    }

    derivedMethod->AddMethod(baseMethod);

    Py_DECREF(baseMethod);
    Py_DECREF(derivedMethod);

    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::Utility::AddBinaryOperator(
        PyObject* left, PyObject* right, const char* op, const char* label, const char* alt)
{
// Install the named operator (op) into the left object's class if such a function
// exists as a global overload; a label must be given if the operator is not in
// gC2POperatorMapping (i.e. if it is ambiguous at the member level).

// this should be a given, nevertheless ...
    if (!CPPInstance_Check(left))
        return false;

// retrieve the class names to match the signature of any found global functions
    std::string rcname = ClassName(right);
    std::string lcname = ClassName(left);
    PyObject* pyclass = PyObject_GetAttr(left, PyStrings::gClass);

    bool result = AddBinaryOperator(pyclass, lcname, rcname, op, label, alt);

    Py_DECREF(pyclass);
    return result;
}

//----------------------------------------------------------------------------
bool CPyCppyy::Utility::AddBinaryOperator(
       PyObject* pyclass, const char* op, const char* label, const char* alt)
{
// Install binary operator op in pyclass, working on two instances of pyclass.
    std::string cname;
    if (CPPScope_Check(pyclass))
        cname = Cppyy::GetScopedFinalName(((CPPScope*)pyclass)->fCppType);
    else {
        PyObject* pyname = PyObject_GetAttr(pyclass, PyStrings::gName);
        cname = Cppyy::ResolveName(CPyCppyy_PyUnicode_AsString(pyname));
        Py_DECREF(pyname);
    }

    return AddBinaryOperator(pyclass, cname, cname, op, label, alt);
}

//----------------------------------------------------------------------------
static inline
Cppyy::TCppMethod_t FindAndAddOperator(const std::string& lcname, const std::string& rcname,
        const char* op, Cppyy::TCppScope_t scope = Cppyy::gGlobalScope) {
// Helper to find a function with matching signature in 'funcs'.
    std::string opname = "operator";
    opname += op;

    Cppyy::TCppIndex_t idx = Cppyy::GetGlobalOperator(
        scope, Cppyy::GetScope(lcname), Cppyy::GetScope(rcname), opname);
    if (idx == (Cppyy::TCppIndex_t)-1)
        return (Cppyy::TCppMethod_t)0;

    return Cppyy::GetMethod(scope, idx);
}

bool CPyCppyy::Utility::AddBinaryOperator(PyObject* pyclass, const std::string& lcname,
        const std::string& rcname, const char* op, const char* label, const char* alt)
{
// Find a global function with a matching signature and install the result on pyclass;
// in addition, __gnu_cxx, std::__1, and _cpycppyy_internal are searched pro-actively (as
// there's AFAICS no way to unearth using information).

// For GNU on clang, search the internal __gnu_cxx namespace for binary operators (is
// typically the case for STL iterators operator==/!=.
// TODO: only look in __gnu_cxx for iterators (and more generally: do lookups in the
//       namespace where the class is defined
    static Cppyy::TCppScope_t gnucxx = Cppyy::GetScope("__gnu_cxx");

// Same for clang on Mac. TODO: find proper pre-processor magic to only use those specific
// namespaces that are actually around; although to be sure, this isn't expensive.
    static Cppyy::TCppScope_t std__1 = Cppyy::GetScope("std::__1");

// One more, mostly for Mac, but again not sure whether this is not a general issue. Some
// operators are declared as friends only in classes, so then they're not found in the
// global namespace. That's why there's this little helper.
    std::call_once(sOperatorTemplateFlag, InitOperatorTemplate);
//    static Cppyy::TCppScope_t _pr_int = Cppyy::GetScope("_cpycppyy_internal");

    PyCallable* pyfunc = 0;
    if (gnucxx) {
        Cppyy::TCppMethod_t func = FindAndAddOperator(lcname, rcname, op, gnucxx);
        if (func) pyfunc = new CPPFunction(gnucxx, func);
    }

    if (!pyfunc && std__1) {
        Cppyy::TCppMethod_t func = FindAndAddOperator(lcname, rcname, op, std__1);
        if (func) pyfunc = new CPPFunction(std__1, func);
    }

    if (!pyfunc) {
        std::string::size_type pos = lcname.substr(0, lcname.find('<')).rfind("::");
        if (pos != std::string::npos) {
            Cppyy::TCppScope_t lcscope = Cppyy::GetScope(lcname.substr(0, pos).c_str());
            if (lcscope) {
                Cppyy::TCppMethod_t func = FindAndAddOperator(lcname, rcname, op, lcscope);
                if (func) pyfunc = new CPPFunction(lcscope, func);
            }
        }
    }

    if (!pyfunc) {
        Cppyy::TCppMethod_t func = FindAndAddOperator(lcname, rcname, op);
        if (func) pyfunc = new CPPFunction(Cppyy::gGlobalScope, func);
    }

#if 0
   // TODO: figure out what this was for ...
    if (!pyfunc && _pr_int.GetClass() &&
            lcname.find("iterator") != std::string::npos &&
            rcname.find("iterator") != std::string::npos) {
   // TODO: gets called too often; make sure it's purely lazy calls only; also try to
   // find a better notion for which classes (other than iterators) this is supposed to
   // work; right now it fails for cases where None is passed
        std::stringstream fname;
        if (strncmp(op, "==", 2) == 0) { fname << "is_equal<"; }
        else if (strncmp(op, "!=", 2) == 0) { fname << "is_not_equal<"; }
        else { fname << "not_implemented<"; }
        fname << lcname << ", " << rcname << ">";
        Cppyy::TCppMethod_t func = (Cppyy::TCppMethod_t)Cppyy_pr_int->GetMethodAny(fname.str().c_str());
        if (func) pyfunc = new CPpFunction(Cppyy::GetScope("_cpycppyy_internal"), func);
    }

// last chance: there could be a non-instantiated templated method
    TClass* lc = TClass::GetClass(lcname.c_str());
    if (lc && strcmp(op, "==") != 0 && strcmp(op, "!=") != 0) {
        std::string opname = "operator"; opname += op;
        gInterpreter->LoadFunctionTemplates(lc);
        gInterpreter->GetFunctionTemplate(lc->GetClassInfo(), opname.c_str());
        TFunctionTemplate* f = lc->GetFunctionTemplate(opname.c_str());
        Cppyy::TCppMethod_t func =
            (Cppyy::TCppMethod_t)lc->GetMethodWithPrototype(opname.c_str(), rcname.c_str());
        if (func && f) pyfunc = new CPPMethod(Cppyy::GetScope(lcname), func);
    }
#endif

    if (pyfunc) {  // found a matching overload; add to class
        bool ok = AddToClass(pyclass, label, pyfunc);
        if (ok && alt)
            return AddToClass(pyclass, alt, label);
    }

    return false;
}

//----------------------------------------------------------------------------
std::string CPyCppyy::Utility::ConstructTemplateArgs(PyObject* pyname, PyObject* args, int argoff)
{
// Helper to construct the "<type, type, ...>" part of a templated name (either
// for a class or method lookup
    std::stringstream tmpl_name;
    if (pyname)
        tmpl_name << CPyCppyy_PyUnicode_AsString(pyname);
    tmpl_name << '<';

    Py_ssize_t nArgs = PyTuple_GET_SIZE(args);
    for (int i = argoff; i < nArgs; ++i) {
    // add type as string to name
        PyObject* tn = PyTuple_GET_ITEM(args, i);
        if (CPyCppyy_PyUnicode_Check(tn)) {
            tmpl_name << CPyCppyy_PyUnicode_AsString(tn);
        } else if (CPPScope_Check(tn)) {
            tmpl_name << Cppyy::GetScopedFinalName(((CPPClass*)tn)->fCppType);
        } else if (PyObject_HasAttr(tn, PyStrings::gName)) {
            PyObject* tpName = PyObject_GetAttr(tn, PyStrings::gName);

        // special case for strings
            if (strcmp(CPyCppyy_PyUnicode_AsString(tpName), "str") == 0)
                tmpl_name << "std::string";
            else
                tmpl_name << CPyCppyy_PyUnicode_AsString(tpName);
            Py_DECREF(tpName);
        } else if (PyInt_Check(tn) || PyLong_Check(tn) || PyFloat_Check(tn)) {
        // last ditch attempt, works for things like int values; since this is a
        // source of errors otherwise, it is limited to specific types and not
        // generally used (str(obj) can print anything ...)
            PyObject* pystr = PyObject_Str(tn);
            tmpl_name << CPyCppyy_PyUnicode_AsString(pystr);
            Py_DECREF(pystr);
        } else {
            PyErr_SetString(PyExc_SyntaxError,
                "could not construct C++ name from provided template argument.");
            return "";
        }

    // add a comma, as needed
        if (i != nArgs-1)
            tmpl_name << ", ";
    }

// close template name
    tmpl_name << '>';

    return tmpl_name.str();
}

//----------------------------------------------------------------------------
bool CPyCppyy::Utility::InitProxy(PyObject* module, PyTypeObject* pytype, const char* name)
{
// Initialize a proxy class for use by python, and add it to the module.

// finalize proxy type
    if (PyType_Ready(pytype) < 0)
        return false;

// add proxy type to the given module
    Py_INCREF(pytype);       // PyModule_AddObject steals reference
    if (PyModule_AddObject(module, (char*)name, (PyObject*)pytype) < 0) {
        Py_DECREF(pytype);
        return false;
    }

// declare success
    return true;
}

//----------------------------------------------------------------------------
int CPyCppyy::Utility::GetBuffer(PyObject* pyobject, char tc, int size, void*& buf, bool check)
{
// Retrieve a linear buffer pointer from the given pyobject.

// special case: don't handle character strings here (yes, they're buffers, but not quite)
    if (PyBytes_Check(pyobject))
        return 0;

// new-style buffer interface
    if (PyObject_CheckBuffer(pyobject)) {
        Py_buffer bufinfo;
        memset(&bufinfo, 0, sizeof(Py_buffer));
        if (PyObject_GetBuffer(pyobject, &bufinfo, PyBUF_FORMAT) == 0) {
            if (tc == '*' || strchr(bufinfo.format, tc)) {
                buf = bufinfo.buf;
                if (buf && bufinfo.ndim == 0) {
                    return 1;
                } else if (buf && bufinfo.ndim == 1 && bufinfo.shape) {
                    int size1d = (int)bufinfo.shape[0];
                    PyBuffer_Release(&bufinfo);
                    return size1d;
                }
            } else {
            // have buf, but format mismatch: bail out now, otherwise the old
            // code will return based on itemsize match
                return 0;                
            }
        }
        PyErr_Clear();
    }

// attempt to retrieve pointer through old-style buffer interface
    PyBufferProcs* bufprocs = Py_TYPE(pyobject)->tp_as_buffer;

    PySequenceMethods* seqmeths = Py_TYPE(pyobject)->tp_as_sequence;
    if (seqmeths != 0 && bufprocs != 0
#if  PY_VERSION_HEX < 0x03000000
         && bufprocs->bf_getwritebuffer != 0
         && (*(bufprocs->bf_getsegcount))(pyobject, 0) == 1
#else
         && bufprocs->bf_getbuffer != 0
#endif
        ) {

   // get the buffer
#if PY_VERSION_HEX < 0x03000000
        Py_ssize_t buflen = (*(bufprocs->bf_getwritebuffer))(pyobject, 0, &buf);
#else
        Py_buffer bufinfo;
        (*(bufprocs->bf_getbuffer))(pyobject, &bufinfo, PyBUF_WRITABLE);
        buf = (char*)bufinfo.buf;
        Py_ssize_t buflen = bufinfo.len;
#if PY_VERSION_HEX < 0x03010000
        PyBuffer_Release(pyobject, &bufinfo);
#else
        PyBuffer_Release(&bufinfo);
#endif
#endif

        if (buf && check == true) {
        // determine buffer compatibility (use "buf" as a status flag)
            PyObject* pytc = PyObject_GetAttr(pyobject, PyStrings::gTypeCode);
            if (pytc != 0) {      // for array objects
                if (CPyCppyy_PyUnicode_AsString(pytc)[0] != tc)
                    buf = 0;      // no match
                Py_DECREF(pytc);
            } else if (seqmeths->sq_length &&
                       (int)(buflen/(*(seqmeths->sq_length))(pyobject)) == size) {
            // this is a gamble ... may or may not be ok, but that's for the user
                PyErr_Clear();
            } else if (buflen == size) {
            // also a gamble, but at least 1 item will fit into the buffer, so very likely ok ...
                PyErr_Clear();
            } else {
                buf = 0;                      // not compatible

            // clarify error message
                PyObject* pytype = 0, *pyvalue = 0, *pytrace = 0;
                PyErr_Fetch(&pytype, &pyvalue, &pytrace);
                PyObject* pyvalue2 = CPyCppyy_PyUnicode_FromFormat(
                    (char*)"%s and given element size (%ld) do not match needed (%d)",
                    CPyCppyy_PyUnicode_AsString(pyvalue),
                    seqmeths->sq_length ? (Long_t)(buflen/(*(seqmeths->sq_length))(pyobject)) : (Long_t)buflen,
                    size);
                Py_DECREF(pyvalue);
                PyErr_Restore(pytype, pyvalue2, pytrace);
            }
        }

        return buflen;
    }

    return 0;
}

//----------------------------------------------------------------------------
std::string CPyCppyy::Utility::MapOperatorName(const std::string& name, bool bTakesParams)
{
// Map the given C++ operator name on the python equivalent.
    if (8 < name.size() && name.substr(0, 8) == "operator") {
        std::string op = name.substr(8, std::string::npos);

    // stripping ...
        std::string::size_type start = 0, end = op.size();
        while (start < end && isspace(op[start])) ++start;
        while (start < end && isspace(op[end-1])) --end;
    // TODO: resolve name only if mapping failed
        op = Cppyy::ResolveName(op.substr(start, end - start));

    // map C++ operator to python equivalent, or made up name if no equivalent exists
        TC2POperatorMapping_t::iterator pop = gC2POperatorMapping.find(op);
        if (pop != gC2POperatorMapping.end()) {
            return pop->second;

        } else if (op == "*") {
        // dereference v.s. multiplication of two instances
            return bTakesParams ? "__mul__" : "__deref__";

        } else if (op == "+") {
        // unary positive v.s. addition of two instances
            return bTakesParams ? "__add__" : "__pos__";

        } else if (op == "-") {
        // unary negative v.s. subtraction of two instances
            return bTakesParams ? "__sub__" : "__neg__";

        } else if (op == "++") {
        // prefix v.s. postfix increment
            return bTakesParams ? "__postinc__" : "__preinc__";

        } else if (op == "--") {
        // prefix v.s. postfix decrement
            return bTakesParams ? "__postdec__" : "__predec__";
        }

    }

// might get here, as not all operator methods are handled (new, delete, etc.)
    return name;
}

//----------------------------------------------------------------------------
const std::string CPyCppyy::Utility::Compound(const std::string& name)
{
// TODO: consolidate with other string manipulations in Helpers.cxx
// Break down the compound of a fully qualified type name.
    std::string cleanName = name;
    RemoveConst(cleanName);

    std::string compound = "";
    for (int ipos = (int)cleanName.size()-1; 0 <= ipos; --ipos) {
        char c = cleanName[ipos];
        if (isspace(c)) continue;
        if (isalnum(c) || c == '_' || c == '>' || c == ')') break;

        compound = c + compound;
    }

// for arrays (TODO: deal with the actual size)
    if (compound == "]")
        return "[]";

    return compound;
}

//----------------------------------------------------------------------------
Py_ssize_t CPyCppyy::Utility::ArraySize(const std::string& name)
{
// TODO: consolidate with other string manipulations in Helpers.cxx
// Extract size from an array type, if available.
    std::string cleanName = name;
    RemoveConst(cleanName);

    if (cleanName[cleanName.size()-1] == ']') {
        std::string::size_type idx = cleanName.rfind('[');
        if (idx != std::string::npos) {
            const std::string asize = cleanName.substr(idx+1, cleanName.size()-2);
            return strtoul(asize.c_str(), nullptr, 0);
        }
    }

    return -1;
}

//----------------------------------------------------------------------------
std::string CPyCppyy::Utility::ClassName(PyObject* pyobj)
{
// Retrieve the class name from the given Python instance.
    if (CPPInstance_Check(pyobj))
        return Cppyy::GetScopedFinalName(((CPPInstance*)pyobj)->ObjectIsA());

// generic Python object ...
    std::string clname = "<unknown>";
    PyObject* pyclass = PyObject_GetAttr(pyobj, PyStrings::gClass);
    if (pyclass) {
        PyObject* pyname = PyObject_GetAttr(pyclass, PyStrings::gName);
        if (pyname) {
            clname = CPyCppyy_PyUnicode_AsString(pyname);
            Py_DECREF(pyname);
        } else {
            PyErr_Clear();
        }
        Py_DECREF(pyclass);
    } else {
        PyErr_Clear();
    }

    return clname;
}


//----------------------------------------------------------------------------
PyObject* CPyCppyy::Utility::PyErr_Occurred_WithGIL()
{
// Re-acquire the GIL before calling PyErr_Occurred() in case it has been
// released; note that the p2.2 code assumes that there are no callbacks in
// C++ to python (or at least none returning errors).
#if PY_VERSION_HEX >= 0x02030000
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* e = PyErr_Occurred();
    PyGILState_Release(gstate);
#else
    if (PyThreadState_GET())
        return PyErr_Occurred();
    PyObject* e = 0;
#endif

    return e;
}


//----------------------------------------------------------------------------
size_t CPyCppyy::Utility::FetchError(std::vector<PyError_t>& errors)
{
// Fetch the current python error, if any, and store it for future use.
    if (PyErr_Occurred()) {
        PyError_t e;
        PyErr_Fetch(&e.fType, &e.fValue, &e.fTrace);
        errors.push_back(e);
    }
    return errors.size();
}

//----------------------------------------------------------------------------
void CPyCppyy::Utility::SetDetailedException(std::vector<PyError_t>& errors, PyObject* topmsg, PyObject* defexc)
{
// Use the collected exceptions to build up a detailed error log.
    if (errors.empty()) {
    // should not happen ...
        Py_DECREF(topmsg);
        return;
    }

// add the details to the topmsg
    PyObject* separator = CPyCppyy_PyUnicode_FromString("\n  ");

    PyObject* exc_type = nullptr;
    for (auto& e : errors) {
        if (!exc_type) exc_type = e.fType;
        else if (exc_type != e.fType) exc_type = defexc;
        CPyCppyy_PyUnicode_Append(&topmsg, separator);
        CPyCppyy_PyUnicode_Append(&topmsg, e.fValue);
    }

    Py_DECREF(separator);
    std::for_each(errors.begin(), errors.end(), PyError_t::Clear);

// set the python exception
    PyErr_SetObject(exc_type, topmsg);
    Py_DECREF(topmsg);
}
