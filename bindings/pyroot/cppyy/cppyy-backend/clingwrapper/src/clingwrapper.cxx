#ifndef WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
// silence warnings about getenv, strncpy, etc.
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

// Bindings
#include "precommondefs.h"
#include "cpp_cppyy.h"
#include "callcontext.h"

// ROOT
#include "TBaseClass.h"
#include "TClass.h"
#include "TClassRef.h"
#include "TClassTable.h"
#include "TClassEdit.h"
#include "TCollection.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TEnum.h"
#include "TEnumConstant.h"
#include "TEnv.h"
#include "TError.h"
#include "TException.h"
#include "TFunction.h"
#include "TFunctionTemplate.h"
#include "TGlobal.h"
#include "THashList.h"
#include "TInterpreter.h"
#include "TList.h"
#include "TListOfDataMembers.h"
#include "TListOfEnums.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TThread.h"

// Standard
#include <cassert>
#include <algorithm>     // for std::count, std::remove
#include <climits>
#include <stdexcept>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <csignal>
#include <cstdlib>      // for getenv
#include <cstring>
#include <typeinfo>

// temp
#include <iostream>
typedef CPyCppyy::Parameter Parameter;
// --temp


// small number that allows use of stack for argument passing
const int SMALL_ARGS_N = 8;

// convention to pass flag for direct calls (similar to Python's vector calls)
#define DIRECT_CALL ((size_t)1 << (8 * sizeof(size_t) - 1))
static inline size_t CALL_NARGS(size_t nargs) {
    return nargs & ~DIRECT_CALL;
}

// data for life time management ---------------------------------------------
typedef std::vector<TClassRef> ClassRefs_t;
static ClassRefs_t g_classrefs(1);
static const ClassRefs_t::size_type GLOBAL_HANDLE = 1;
static const ClassRefs_t::size_type STD_HANDLE = GLOBAL_HANDLE + 1;

typedef std::map<std::string, ClassRefs_t::size_type> Name2ClassRefIndex_t;
static Name2ClassRefIndex_t g_name2classrefidx;
static std::map<std::string, std::string> resolved_enum_types;

namespace {

static inline
Cppyy::TCppType_t find_memoized_scope(const std::string& name)
{
    auto icr = g_name2classrefidx.find(name);
    if (icr != g_name2classrefidx.end())
        return (Cppyy::TCppType_t)icr->second;
    return (Cppyy::TCppType_t)0;
}

static inline
std::string find_memoized_resolved_name(const std::string& name)
{
// resolved class types
    Cppyy::TCppType_t klass = find_memoized_scope(name);
    if (klass) return Cppyy::GetScopedFinalName(klass);

// resolved enum types
    auto res = resolved_enum_types.find(name);
    if (res != resolved_enum_types.end())
        return res->second;

// unknown ...
    return "";
}

class CallWrapper {
public:
    typedef const void* DeclId_t;

public:
    CallWrapper(TFunction* f) : fDecl(f->GetDeclId()), fName(f->GetName()), fTF(new TFunction(*f)) {}
    CallWrapper(DeclId_t fid, const std::string& n) : fDecl(fid), fName(n), fTF(nullptr) {}
    ~CallWrapper() {
        delete fTF;
    }

public:
    TInterpreter::CallFuncIFacePtr_t fFaceptr;
    DeclId_t      fDecl;
    std::string   fName;
    TFunction*    fTF;
};

}

static std::vector<CallWrapper*> gWrapperHolder;

static inline
CallWrapper* new_CallWrapper(TFunction* f)
{
    CallWrapper* wrap = new CallWrapper(f);
    gWrapperHolder.push_back(wrap);
    return wrap;
}

static inline
CallWrapper* new_CallWrapper(CallWrapper::DeclId_t fid, const std::string& n)
{
    CallWrapper* wrap = new CallWrapper(fid, n);
    gWrapperHolder.push_back(wrap);
    return wrap;
}

typedef std::vector<TGlobal*> GlobalVars_t;
typedef std::map<TGlobal*, GlobalVars_t::size_type> GlobalVarsIndices_t;

static GlobalVars_t g_globalvars;
static GlobalVarsIndices_t g_globalidx;

static std::set<std::string> gSTLNames;


// data ----------------------------------------------------------------------
Cppyy::TCppScope_t Cppyy::gGlobalScope = GLOBAL_HANDLE;

// builtin types (including a few common STL templates as long as they live in
// the global namespace b/c of choices upstream)
static std::set<std::string> g_builtins =
    {"bool", "char", "signed char", "unsigned char", "wchar_t", "short", "unsigned short",
     "int", "unsigned int", "long", "unsigned long", "long long", "unsigned long long",
     "float", "double", "long double", "void",
     "allocator", "array", "basic_string", "complex", "initializer_list", "less", "list",
     "map", "pair", "set", "vector"};

// smart pointer types
static std::set<std::string> gSmartPtrTypes =
    {"auto_ptr", "std::auto_ptr", "shared_ptr", "std::shared_ptr",
     "unique_ptr", "std::unique_ptr", "weak_ptr", "std::weak_ptr"};

// to filter out ROOT names
static std::set<std::string> gInitialNames;
static std::set<std::string> gRootSOs;

// configuration
static bool gEnableFastPath = true;


// global initialization -----------------------------------------------------
namespace {

// names copied from TUnixSystem
#ifdef WIN32
const int SIGBUS   = 0;      // simple placeholders for ones that don't exist
const int SIGSYS   = 0;
const int SIGPIPE  = 0;
const int SIGQUIT  = 0;
const int SIGWINCH = 0;
const int SIGALRM  = 0;
const int SIGCHLD  = 0;
const int SIGURG   = 0;
const int SIGUSR1  = 0;
const int SIGUSR2  = 0;
#endif

static struct Signalmap_t {
   int               fCode;
   const char       *fSigName;
} gSignalMap[kMAXSIGNALS] = {       // the order of the signals should be identical
   { SIGBUS,   "bus error" }, // to the one in TSysEvtHandler.h
   { SIGSEGV,  "segmentation violation" },
   { SIGSYS,    "bad argument to system call" },
   { SIGPIPE,   "write on a pipe with no one to read it" },
   { SIGILL,    "illegal instruction" },
   { SIGABRT,   "abort" },
   { SIGQUIT,   "quit" },
   { SIGINT,    "interrupt" },
   { SIGWINCH,  "window size change" },
   { SIGALRM,   "alarm clock" },
   { SIGCHLD,   "death of a child" },
   { SIGURG,    "urgent data arrived on an I/O channel" },
   { SIGFPE,    "floating point exception" },
   { SIGTERM,   "termination signal" },
   { SIGUSR1,   "user-defined signal 1" },
   { SIGUSR2,   "user-defined signal 2" }
};

static void inline do_trace(int sig) {
    std::cerr << " *** Break *** " << (sig < kMAXSIGNALS ? gSignalMap[sig].fSigName : "") << std::endl;
    gSystem->StackTrace();
}

class TExceptionHandlerImp : public TExceptionHandler {
public:
    void HandleException(Int_t sig) override {
        if (TROOT::Initialized()) {
            if (gException) {
                gInterpreter->RewindDictionary();
                gInterpreter->ClearFileBusy();
            }

            if (!std::getenv("CPPYY_CRASH_QUIET"))
                do_trace(sig);

        // jump back, if catch point set
            Throw(sig);
        }

        do_trace(sig);
        gSystem->Exit(128 + sig);
    }
};

class ApplicationStarter {
public:
    ApplicationStarter() {
    // initialize ROOT early to guarantee proper order of shutdown later on (gROOT is a
    // macro that resolves to the ROOT::GetROOT() function call)
        (void)gROOT;

    // setup dummy holders for global and std namespaces
        assert(g_classrefs.size() == GLOBAL_HANDLE);
        g_name2classrefidx[""]     = GLOBAL_HANDLE;
        g_classrefs.push_back(TClassRef(""));

    // aliases for std (setup already in pythonify)
        g_name2classrefidx["std"]   = STD_HANDLE;
        g_name2classrefidx["::std"] = g_name2classrefidx["std"];
        g_classrefs.push_back(TClassRef("std"));

    // add a dummy global to refer to as null at index 0
        g_globalvars.push_back(nullptr);
        g_globalidx[nullptr] = 0;

    // disable fast path if requested
        if (std::getenv("CPPYY_DISABLE_FASTPATH")) gEnableFastPath = false;

    // fill the set of STL names
        const char* stl_names[] = {"allocator", "auto_ptr", "bad_alloc", "bad_cast",
            "bad_exception", "bad_typeid", "basic_filebuf", "basic_fstream", "basic_ifstream",
            "basic_ios", "basic_iostream", "basic_istream", "basic_istringstream",
            "basic_ofstream", "basic_ostream", "basic_ostringstream", "basic_streambuf",
            "basic_string", "basic_stringbuf", "basic_stringstream", "binary_function",
            "binary_negate", "bitset", "byte", "char_traits", "codecvt_byname", "codecvt", "collate",
            "collate_byname", "compare", "complex", "ctype_byname", "ctype", "default_delete",
            "deque", "divides", "domain_error", "equal_to", "exception", "forward_list", "fpos",
            "function", "greater_equal", "greater", "gslice_array", "gslice", "hash", "indirect_array",
            "integer_sequence", "invalid_argument", "ios_base", "istream_iterator", "istreambuf_iterator",
            "istrstream", "iterator_traits", "iterator", "length_error", "less_equal", "less",
            "list", "locale", "localedef utility", "locale utility", "logic_error", "logical_and",
            "logical_not", "logical_or", "map", "mask_array", "mem_fun", "mem_fun_ref", "messages",
            "messages_byname", "minus", "modulus", "money_get", "money_put", "moneypunct",
            "moneypunct_byname", "multimap", "multiplies", "multiset", "negate", "not_equal_to",
            "num_get", "num_put", "numeric_limits", "numpunct", "numpunct_byname",
            "ostream_iterator", "ostreambuf_iterator", "ostrstream", "out_of_range",
            "overflow_error", "pair", "plus", "pointer_to_binary_function",
            "pointer_to_unary_function", "priority_queue", "queue", "range_error",
            "raw_storage_iterator", "reverse_iterator", "runtime_error", "set", "shared_ptr",
            "slice_array", "slice", "stack", "string", "strstream", "strstreambuf",
            "time_get_byname", "time_get", "time_put_byname", "time_put", "unary_function",
            "unary_negate", "unique_ptr", "underflow_error", "unordered_map", "unordered_multimap",
            "unordered_multiset", "unordered_set", "valarray", "vector", "weak_ptr", "wstring",
            "__hash_not_enabled"};
        for (auto& name : stl_names)
            gSTLNames.insert(name);

    // set opt level (default to 2 if not given; Cling itself defaults to 0)
        int optLevel = 2;
        if (std::getenv("CPPYY_OPT_LEVEL")) optLevel = atoi(std::getenv("CPPYY_OPT_LEVEL"));
        if (optLevel != 0) {
            std::ostringstream s;
            s << "#pragma cling optimize " << optLevel;
            gInterpreter->ProcessLine(s.str().c_str());
        }

    // load frequently used headers
        const char* code =
               "#include <iostream>\n"
               "#include <string>\n"
               "#include <DllImport.h>\n"     // defines R__EXTERN
               "#include <vector>\n"
               "#include <utility>";
        gInterpreter->ProcessLine(code);

    // create helpers for comparing thingies
        gInterpreter->Declare(
            "namespace __cppyy_internal { template<class C1, class C2>"
            " bool is_equal(const C1& c1, const C2& c2) { return (bool)(c1 == c2); } }");
        gInterpreter->Declare(
            "namespace __cppyy_internal { template<class C1, class C2>"
            " bool is_not_equal(const C1& c1, const C2& c2) { return (bool)(c1 != c2); } }");

    // helper for multiple inheritance
        gInterpreter->Declare("namespace __cppyy_internal { struct Sep; }");

    // retrieve all initial (ROOT) C++ names in the global scope to allow filtering later
        if (!std::getenv("CPPYY_NO_ROOT_FILTER")) {
            gROOT->GetListOfGlobals(true);             // force initialize
            gROOT->GetListOfGlobalFunctions(true);     // id.
            std::set<std::string> initial;
            Cppyy::GetAllCppNames(GLOBAL_HANDLE, initial);
            gInitialNames = initial;

#ifndef WIN32
            gRootSOs.insert("libCore.so ");
            gRootSOs.insert("libRIO.so ");
            gRootSOs.insert("libThread.so ");
            gRootSOs.insert("libMathCore.so ");
#else
            gRootSOs.insert("libCore.dll ");
            gRootSOs.insert("libRIO.dll ");
            gRootSOs.insert("libThread.dll ");
            gRootSOs.insert("libMathCore.dll ");
#endif
        }

    // start off with a reasonable size placeholder for wrappers
        gWrapperHolder.reserve(1024);

    // create an exception handler to process signals
        gExceptionHandler = new TExceptionHandlerImp{};
    }

    ~ApplicationStarter() {
        for (auto wrap : gWrapperHolder)
            delete wrap;
        delete gExceptionHandler; gExceptionHandler = nullptr;
    }
} _applicationStarter;

} // unnamed namespace


// local helpers -------------------------------------------------------------
static inline
TClassRef& type_from_handle(Cppyy::TCppScope_t scope)
{
    assert((ClassRefs_t::size_type)scope < g_classrefs.size());
    return g_classrefs[(ClassRefs_t::size_type)scope];
}

static inline
TFunction* m2f(Cppyy::TCppMethod_t method) {
    CallWrapper* wrap = ((CallWrapper*)method);
    if (!wrap->fTF) {
        MethodInfo_t* mi = gInterpreter->MethodInfo_Factory(wrap->fDecl);
        wrap->fTF = new TFunction(mi);
    }
    return wrap->fTF;
}

/*
static inline
CallWrapper::DeclId_t m2d(Cppyy::TCppMethod_t method) {
    CallWrapper* wrap = ((CallWrapper*)method);
    if (!wrap->fTF || wrap->fTF->GetDeclId() != wrap->fDecl) {
        MethodInfo_t* mi = gInterpreter->MethodInfo_Factory(wrap->fDecl);
        wrap->fTF = new TFunction(mi);
    }
    return wrap->fDecl;
}
*/

static inline
char* cppstring_to_cstring(const std::string& cppstr)
{
    char* cstr = (char*)malloc(cppstr.size()+1);
    memcpy(cstr, cppstr.c_str(), cppstr.size()+1);
    return cstr;
}

static inline
bool match_name(const std::string& tname, const std::string fname)
{
// either match exactly, or match the name as template
    if (fname.rfind(tname, 0) == 0) {
        if ((tname.size() == fname.size()) ||
              (tname.size() < fname.size() && fname[tname.size()] == '<'))
           return true;
    }
    return false;
}

static inline
bool is_missclassified_stl(const std::string& name)
{
    std::string::size_type pos = name.find('<');
    if (pos != std::string::npos)
        return gSTLNames.find(name.substr(0, pos)) != gSTLNames.end();
    return gSTLNames.find(name) != gSTLNames.end();
}


// direct interpreter access -------------------------------------------------
bool Cppyy::Compile(const std::string& code, bool /*silent*/)
{
    return gInterpreter->Declare(code.c_str());
}

std::string Cppyy::ToString(TCppType_t klass, TCppObject_t obj)
{
    if (klass && obj && !IsNamespace((TCppScope_t)klass))
        return gInterpreter->ToString(GetScopedFinalName(klass).c_str(), (void*)obj);
    return "";
}


// name to opaque C++ scope representation -----------------------------------
std::string Cppyy::ResolveName(const std::string& cppitem_name)
{
// Fully resolve the given name to the final type name.

// try memoized type cache, in case seen before
    std::string memoized = find_memoized_resolved_name(cppitem_name);
    if (!memoized.empty()) return memoized;

// remove global scope '::' if present
    std::string tclean = cppitem_name.compare(0, 2, "::") == 0 ?
        cppitem_name.substr(2, std::string::npos) : cppitem_name;

// classes (most common)
    tclean = TClassEdit::CleanType(tclean.c_str());
    if (tclean.empty() /* unknown, eg. an operator */) return cppitem_name;

// reduce [N] to []
    if (tclean[tclean.size()-1] == ']')
        tclean = tclean.substr(0, tclean.rfind('[')) + "[]";

    if (tclean.rfind("byte", 0) == 0 || tclean.rfind("std::byte", 0) == 0)
        return tclean;

// remove __restrict and __restrict__
    auto pos = tclean.rfind("__restrict");
    if (pos != std::string::npos)
        tclean = tclean.substr(0, pos);

    if (tclean.compare(0, 9, "std::byte") == 0)
        return tclean;


// check data types list (accept only builtins as typedefs will
// otherwise not be resolved)
    if (IsBuiltin(tclean)) return tclean;
// special case for enums
    if (IsEnum(cppitem_name))
        return ResolveEnum(cppitem_name);

// special case for clang's builtin __type_pack_element (which does not resolve)
    pos = cppitem_name.size() > 20 ? \
              cppitem_name.rfind("__type_pack_element", 5) : std::string::npos;
    if (pos != std::string::npos) {
    // shape is "[std::]__type_pack_element<index,type1,type2,...,typeN>cpd": extract
    // first the index, and from there the indexed type; finally, restore the
    // qualifiers
        const char* str = cppitem_name.c_str();
        char* endptr = nullptr;
        unsigned long index = strtoul(str+20+pos, &endptr, 0);

        std::string tmplvars{endptr};
        auto start = tmplvars.find(',') + 1;
        auto end = tmplvars.find(',', start);
        while (index != 0) {
            start = end+1;
            end = tmplvars.find(',', start);
            if (end == std::string::npos) end = tmplvars.rfind('>');
            --index;
        }

        std::string resolved = tmplvars.substr(start, end-start);
        auto cpd = tmplvars.rfind('>');
        if (cpd != std::string::npos && cpd+1 != tmplvars.size())
            return resolved + tmplvars.substr(cpd+1, std::string::npos);
        return resolved;
    }

// typedefs etc. (and a couple of hacks around TClassEdit-isms, fixing of which
// in ResolveTypedef itself is a TODO ...)
    tclean = TClassEdit::ResolveTypedef(tclean.c_str(), true);
    pos = 0;
    while ((pos = tclean.find("::::", pos)) != std::string::npos) {
        tclean.replace(pos, 4, "::");
        pos += 2;
    }

    if (tclean.compare(0, 6, "const ") != 0)
        return TClassEdit::ShortType(tclean.c_str(), 2);
    return "const " + TClassEdit::ShortType(tclean.c_str(), 2);
}


std::string Cppyy::ResolveEnum(const std::string& enum_type)
{
// The underlying type of a an enum may be any kind of integer.
// Resolve that type via a workaround (note: this function assumes
// that the enum_type name is a valid enum type name)
    auto res = resolved_enum_types.find(enum_type);
    if (res != resolved_enum_types.end())
        return res->second;

// desugar the type before resolving
    std::string et_short = TClassEdit::ShortType(enum_type.c_str(), 1);
    if (et_short.find("(unnamed") == std::string::npos) {
        std::ostringstream decl;
    // TODO: now presumed fixed with https://sft.its.cern.ch/jira/browse/ROOT-6988
        for (auto& itype : {"unsigned int"}) {
            decl << "std::is_same<"
                 << itype
                 << ", std::underlying_type<"
                 << et_short
                 << ">::type>::value;";
            if (gInterpreter->ProcessLine(decl.str().c_str())) {
            // TODO: "re-sugaring" like this is brittle, but the top
            // should be re-translated into AST-based code anyway
                std::string resugared;
                if (et_short.size() != enum_type.size()) {
                    auto pos = enum_type.find(et_short);
                    if (pos != std::string::npos) {
                        resugared = enum_type.substr(0, pos) + itype;
                        if (pos+et_short.size() < enum_type.size())
                            resugared += enum_type.substr(pos+et_short.size(), std::string::npos);
                    }
                }
                if (resugared.empty()) resugared = itype;
                resolved_enum_types[enum_type] = resugared;
                return resugared;
            }
        }
    }

// failed or anonymous ... signal upstream to special case this
    int ipos = (int)enum_type.size()-1;
    for (; 0 <= ipos; --ipos) {
        char c = enum_type[ipos];
        if (isspace(c)) continue;
        if (isalnum(c) || c == '_' || c == '>' || c == ')') break;
    }
    bool isConst = enum_type.find("const ", 6) != std::string::npos;
    std::string restype = isConst ? "const " : "";
    restype += "internal_enum_type_t"+enum_type.substr((std::string::size_type)ipos+1, std::string::npos);
    resolved_enum_types[enum_type] = restype;
    return restype;     // should default to some int variant
}

Cppyy::TCppScope_t Cppyy::GetScope(const std::string& sname)
{
// First, try cache
    TCppType_t result = find_memoized_scope(sname);
    if (result) return result;

// Second, skip builtins before going through the more expensive steps of resolving
// typedefs and looking up TClass
    if (g_builtins.find(sname) != g_builtins.end())
        return (TCppScope_t)0;

// TODO: scope_name should always be final already?
// Resolve name fully before lookup to make sure all aliases point to the same scope
    std::string scope_name = ResolveName(sname);
    bool bHasAlias1 = sname != scope_name;
    if (bHasAlias1) {
        result = find_memoized_scope(scope_name);
        if (result) {
            g_name2classrefidx[sname] = result;
            return result;
        }
    }

// both failed, but may be STL name that's missing 'std::' now, but didn't before
    bool b_scope_name_missclassified = is_missclassified_stl(scope_name);
    if (b_scope_name_missclassified) {
        result = find_memoized_scope("std::"+scope_name);
        if (result) g_name2classrefidx["std::"+scope_name] = (ClassRefs_t::size_type)result;
    }
    bool b_sname_missclassified = bHasAlias1 ? is_missclassified_stl(sname) : false;
    if (b_sname_missclassified) {
        if (!result) result = find_memoized_scope("std::"+sname);
        if (result) g_name2classrefidx["std::"+sname] = (ClassRefs_t::size_type)result;
    }

    if (result) return result;

// use TClass directly, to enable auto-loading; class may be stubbed (eg. for
// function returns) or forward declared, leading to a non-null TClass that is
// otherwise invalid/unusable
    TClassRef cr(TClass::GetClass(scope_name.c_str(), true /* load */, true /* silent */));
    if (!cr.GetClass())
        return (TCppScope_t)0;

// memoize found/created TClass
    bool bHasAlias2 = cr->GetName() != scope_name;
    if (bHasAlias2) {
        result = find_memoized_scope(cr->GetName());
        if (result) {
            g_name2classrefidx[scope_name] = result;
            if (bHasAlias1) g_name2classrefidx[sname] = result;
            return result;
        }
    }

    ClassRefs_t::size_type sz = g_classrefs.size();
    g_name2classrefidx[scope_name] = sz;
    if (bHasAlias1) g_name2classrefidx[sname] = sz;
    if (bHasAlias2) g_name2classrefidx[cr->GetName()] = sz;
// TODO: make ROOT/meta NOT remove std :/
    if (b_scope_name_missclassified)
        g_name2classrefidx["std::"+scope_name] = sz;
    if (b_sname_missclassified)
        g_name2classrefidx["std::"+sname] = sz;

    g_classrefs.push_back(TClassRef(scope_name.c_str()));

    return (TCppScope_t)sz;
}

bool Cppyy::IsTemplate(const std::string& template_name)
{
    return (bool)gInterpreter->CheckClassTemplate(template_name.c_str());
}

namespace {
    class AutoCastRTTI {
    public:
        virtual ~AutoCastRTTI() {}
    };
}

Cppyy::TCppType_t Cppyy::GetActualClass(TCppType_t klass, TCppObject_t obj)
{
    TClassRef& cr = type_from_handle(klass);
    if (!cr.GetClass() || !obj) return klass;

    if (!(cr->ClassProperty() & kClassHasVirtual))
        return klass;   // not polymorphic: no RTTI info available

// TODO: ios class casting (ostream, streambuf, etc.) fails with a crash in GetActualClass()
// below on Mac ARM (it's likely that the found actual class was replaced, maybe because
// there are duplicates from pcm/pch?); filter them out for now as it's usually unnecessary
// anyway to autocast these
    std::string clName = cr->GetName();
    if (clName.find("std::", 0, 5) == 0 && clName.find("stream") != std::string::npos)
        return klass;

#ifdef _WIN64
// Cling does not provide a consistent ImageBase address for calculating relative addresses
// as used in Windows 64b RTTI. So, check for our own RTTI extension instead. If that fails,
// see whether the unmangled raw_name is available (e.g. if this is an MSVC compiled rather
// than JITed class) and pass on if it is.
    volatile const char* raw = nullptr;     // to prevent too aggressive reordering
    try {
    // this will filter those objects that do not have RTTI to begin with (throws)
        AutoCastRTTI* pcst = (AutoCastRTTI*)obj;
        raw = typeid(*pcst).raw_name();

    // check the signature id (0 == absolute, 1 == relative, 2 == ours)
        void* vfptr = *(void**)((intptr_t)obj);
        void* meta = (void*)((intptr_t)*((void**)((intptr_t)vfptr-sizeof(void*))));
        if (*(intptr_t*)meta == 2) {
        // access the extra data item which is an absolute pointer to the RTTI
            void* ptdescr = (void*)((intptr_t)meta + 4*sizeof(unsigned long)+sizeof(void*));
            if (ptdescr && *(void**)ptdescr) {
                auto rtti = *(std::type_info**)ptdescr;
                raw = rtti->raw_name();
                if (raw && raw[0] != '\0')    // likely unnecessary
                    return (TCppType_t)GetScope(rtti->name());
            }

            return klass;        // do not fall through if no RTTI info available
        }

    // if the raw name is the empty string (no guarantees that this is so as truly, the
    // address is corrupt, but it is common to be empty), then there is no accessible RTTI
    // and getting the unmangled name will crash ...
        if (!raw)
            return klass;
    } catch (std::bad_typeid) {
        return klass;        // can't risk passing to ROOT/meta as it may do RTTI
    }
#endif

    TClass* clActual = cr->GetActualClass((void*)obj);
    // The additional check using TClass::GetClassInfo is to prevent returning classes of which the Interpreter has no info
    if (clActual && clActual != cr.GetClass() && clActual->GetClassInfo()) {
        auto itt = g_name2classrefidx.find(clActual->GetName());
        if (itt != g_name2classrefidx.end())
            return (TCppType_t)itt->second;
        return (TCppType_t)GetScope(clActual->GetName());
    }

    return klass;
}

size_t Cppyy::SizeOf(TCppType_t klass)
{
    TClassRef& cr = type_from_handle(klass);
    if (cr.GetClass() && cr->GetClassInfo())
        return (size_t)gInterpreter->ClassInfo_Size(cr->GetClassInfo());
    return (size_t)0;
}

size_t Cppyy::SizeOf(const std::string& type_name)
{
    TDataType* dt = gROOT->GetType(type_name.c_str());
    if (dt) return dt->Size();
    return SizeOf(GetScope(type_name));
}


bool Cppyy::IsBuiltin(const std::string& type_name)
{
    if (g_builtins.find(type_name) != g_builtins.end())
        return true;

    const std::string& tclean = TClassEdit::CleanType(type_name.c_str(), 1);
    if (g_builtins.find(tclean) != g_builtins.end())
        return true;

    if (strstr(tclean.c_str(), "std::complex"))
        return true;

    return false;
}

bool Cppyy::IsComplete(const std::string& type_name)
{
// verify whether the dictionary of this class is fully available
    bool b = false;

    int oldEIL = gErrorIgnoreLevel;
    gErrorIgnoreLevel = 3000;
    TClass* klass = TClass::GetClass(TClassEdit::ShortType(type_name.c_str(), 1).c_str());
    if (klass && klass->GetClassInfo())     // works for normal case w/ dict
        b = gInterpreter->ClassInfo_IsLoaded(klass->GetClassInfo());
    else {    // special case for forward declared classes
        ClassInfo_t* ci = gInterpreter->ClassInfo_Factory(type_name.c_str());
        if (ci) {
            b = gInterpreter->ClassInfo_IsLoaded(ci);
            gInterpreter->ClassInfo_Delete(ci);  // we own the fresh class info
        }
    }
    gErrorIgnoreLevel = oldEIL;
    return b;
}

// memory management ---------------------------------------------------------
Cppyy::TCppObject_t Cppyy::Allocate(TCppType_t type)
{
    TClassRef& cr = type_from_handle(type);
    return (TCppObject_t)::operator new(gInterpreter->ClassInfo_Size(cr->GetClassInfo()));
}

void Cppyy::Deallocate(TCppType_t /* type */, TCppObject_t instance)
{
    ::operator delete(instance);
}

Cppyy::TCppObject_t Cppyy::Construct(TCppType_t type, void* arena)
{
    TClassRef& cr = type_from_handle(type);
    if (arena)
        return (TCppObject_t)cr->New(arena, TClass::kRealNew);
    return (TCppObject_t)cr->New(TClass::kRealNew);
}

static std::map<Cppyy::TCppType_t, bool> sHasOperatorDelete;
void Cppyy::Destruct(TCppType_t type, TCppObject_t instance)
{
    TClassRef& cr = type_from_handle(type);
    if (cr->ClassProperty() & (kClassHasExplicitDtor | kClassHasImplicitDtor))
        cr->Destructor((void*)instance);
    else {
        ROOT::DelFunc_t fdel = cr->GetDelete();
        if (fdel) fdel((void*)instance);
        else {
            auto ib = sHasOperatorDelete.find(type);
            if (ib == sHasOperatorDelete.end()) {
               TFunction *f = (TFunction *)cr->GetMethodAllAny("operator delete");
               sHasOperatorDelete[type] = (bool)(f && (f->Property() & kIsPublic));
               ib = sHasOperatorDelete.find(type);
            }
            ib->second ? cr->Destructor((void *)instance) : ::operator delete((void *)instance);
        }
    }
}


// method/function dispatching -----------------------------------------------
static TInterpreter::CallFuncIFacePtr_t GetCallFunc(Cppyy::TCppMethod_t method)
{
// TODO: method should be a callfunc, so that no mapping would be needed.
    CallWrapper* wrap = (CallWrapper*)method;

    CallFunc_t* callf = gInterpreter->CallFunc_Factory();
    MethodInfo_t* meth = gInterpreter->MethodInfo_Factory(wrap->fDecl);
    gInterpreter->CallFunc_SetFunc(callf, meth);
    gInterpreter->MethodInfo_Delete(meth);

    if (!(callf && gInterpreter->CallFunc_IsValid(callf))) {
    // TODO: propagate this error to caller w/o use of Python C-API
    /*
        PyErr_Format(PyExc_RuntimeError, "could not resolve %s::%s(%s)",
            const_cast<TClassRef&>(klass).GetClassName(),
            wrap.fName, callString.c_str()); */
        std::cerr << "TODO: report unresolved function error to Python\n";
        if (callf) gInterpreter->CallFunc_Delete(callf);
        return TInterpreter::CallFuncIFacePtr_t{};
    }

// generate the wrapper and JIT it; ignore wrapper generation errors (will simply
// result in a nullptr that is reported upstream if necessary; often, however,
// there is a different overload available that will do)
    auto oldErrLvl = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kFatal;
    wrap->fFaceptr = gInterpreter->CallFunc_IFacePtr(callf);
    gErrorIgnoreLevel = oldErrLvl;

    gInterpreter->CallFunc_Delete(callf);   // does not touch IFacePtr
    return wrap->fFaceptr;
}

static inline
bool copy_args(Parameter* args, size_t nargs, void** vargs)
{
    bool runRelease = false;
    for (size_t i = 0; i < nargs; ++i) {
        switch (args[i].fTypeCode) {
        case 'X':       /* (void*)type& with free */
            runRelease = true;
        case 'V':       /* (void*)type& */
            vargs[i] = args[i].fValue.fVoidp;
            break;
        case 'r':       /* const type& */
            vargs[i] = args[i].fRef;
            break;
        default:        /* all other types in union */
            vargs[i] = (void*)&args[i].fValue.fVoidp;
            break;
        }
    }
    return runRelease;
}

static inline
void release_args(Parameter* args, size_t nargs) {
    for (size_t i = 0; i < nargs; ++i) {
        if (args[i].fTypeCode == 'X')
            free(args[i].fValue.fVoidp);
    }
}

static inline bool WrapperCall(Cppyy::TCppMethod_t method, size_t nargs, void* args_, void* self, void* result)
{
    Parameter* args = (Parameter*)args_;
    //bool is_direct = nargs & DIRECT_CALL;
    nargs = CALL_NARGS(nargs);

    CallWrapper* wrap = (CallWrapper*)method;
    const TInterpreter::CallFuncIFacePtr_t& faceptr = wrap->fFaceptr.fGeneric ? wrap->fFaceptr : GetCallFunc(method);
    if (!faceptr.fGeneric)
        return false;        // happens with compilation error

    if (faceptr.fKind == TInterpreter::CallFuncIFacePtr_t::kGeneric) {
        bool runRelease = false;
        if (nargs <= SMALL_ARGS_N) {
            void* smallbuf[SMALL_ARGS_N];
            if (nargs) runRelease = copy_args(args, nargs, smallbuf);
            faceptr.fGeneric(self, (int)nargs, smallbuf, result);
        } else {
            std::vector<void*> buf(nargs);
            runRelease = copy_args(args, nargs, buf.data());
            faceptr.fGeneric(self, (int)nargs, buf.data(), result);
        }
        if (runRelease) release_args(args, nargs);
        return true;
    }

    if (faceptr.fKind == TInterpreter::CallFuncIFacePtr_t::kCtor) {
        bool runRelease = false;
        if (nargs <= SMALL_ARGS_N) {
            void* smallbuf[SMALL_ARGS_N];
            if (nargs) runRelease = copy_args(args, nargs, (void**)smallbuf);
            faceptr.fCtor((void**)smallbuf, result, (unsigned long)nargs);
        } else {
            std::vector<void*> buf(nargs);
            runRelease = copy_args(args, nargs, buf.data());
            faceptr.fCtor(buf.data(), result, (unsigned long)nargs);
        }
        if (runRelease) release_args(args, nargs);
        return true;
    }

    if (faceptr.fKind == TInterpreter::CallFuncIFacePtr_t::kDtor) {
        std::cerr << " DESTRUCTOR NOT IMPLEMENTED YET! " << std::endl;
        return false;
    }

    return false;
}

template<typename T>
static inline
T CallT(Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, size_t nargs, void* args)
{
    T t{};
    if (WrapperCall(method, nargs, args, (void*)self, &t))
        return t;
    return (T)-1;
}

#define CPPYY_IMP_CALL(typecode, rtype)                                      \
rtype Cppyy::Call##typecode(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args)\
{                                                                            \
    return CallT<rtype>(method, self, nargs, args);                          \
}

void Cppyy::CallV(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args)
{
    if (!WrapperCall(method, nargs, args, (void*)self, nullptr))
        return /* TODO ... report error */;
}

CPPYY_IMP_CALL(B,  unsigned char)
CPPYY_IMP_CALL(C,  char         )
CPPYY_IMP_CALL(H,  short        )
CPPYY_IMP_CALL(I,  int          )
CPPYY_IMP_CALL(L,  long         )
CPPYY_IMP_CALL(LL, Long64_t     )
CPPYY_IMP_CALL(F,  float        )
CPPYY_IMP_CALL(D,  double       )
CPPYY_IMP_CALL(LD, LongDouble_t )

void* Cppyy::CallR(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args)
{
    void* r = nullptr;
    if (WrapperCall(method, nargs, args, (void*)self, &r))
        return r;
    return nullptr;
}

char* Cppyy::CallS(
    TCppMethod_t method, TCppObject_t self, size_t nargs, void* args, size_t* length)
{
    char* cstr = nullptr;
    TClassRef cr("std::string");
    std::string* cppresult = (std::string*)malloc(sizeof(std::string));
    if (WrapperCall(method, nargs, args, self, (void*)cppresult)) {
        cstr = cppstring_to_cstring(*cppresult);
        *length = cppresult->size();
        cppresult->std::string::~basic_string();
    } else
        *length = 0;
    free((void*)cppresult);
    return cstr;
}

Cppyy::TCppObject_t Cppyy::CallConstructor(
    TCppMethod_t method, TCppType_t /* klass */, size_t nargs, void* args)
{
    void* obj = nullptr;
    if (WrapperCall(method, nargs, args, nullptr, &obj))
        return (TCppObject_t)obj;
    return (TCppObject_t)0;
}

void Cppyy::CallDestructor(TCppType_t type, TCppObject_t self)
{
    TClassRef& cr = type_from_handle(type);
    cr->Destructor((void*)self, true);
}

Cppyy::TCppObject_t Cppyy::CallO(TCppMethod_t method,
    TCppObject_t self, size_t nargs, void* args, TCppType_t result_type)
{
    TClassRef& cr = type_from_handle(result_type);
    void* obj = ::operator new(gInterpreter->ClassInfo_Size(cr->GetClassInfo()));
    if (WrapperCall(method, nargs, args, self, obj))
        return (TCppObject_t)obj;
    ::operator delete(obj);
    return (TCppObject_t)0;
}

Cppyy::TCppFuncAddr_t Cppyy::GetFunctionAddress(TCppMethod_t method, bool check_enabled)
{
    if (check_enabled && !gEnableFastPath) return (TCppFuncAddr_t)nullptr;
    TFunction* f = m2f(method);

    TCppFuncAddr_t pf = (TCppFuncAddr_t)gInterpreter->FindSym(f->GetMangledName());
    if (pf) return pf;

    int ierr = 0;
    const char* fn = TClassEdit::DemangleName(f->GetMangledName(), ierr);
    if (ierr || !fn)
        return pf;

    // TODO: the following attempts are all brittle and leak transactions, but
    // each properly exposes the symbol so subsequent lookups will succeed
    if (strstr(f->GetName(), "<")) {
    // force explicit instantiation and try again
        std::ostringstream sig;
        sig << "template " << fn << ";";
        gInterpreter->ProcessLine(sig.str().c_str());
    } else {
        std::ostringstream sig;

        std::string sfn = fn;
        std::string::size_type pos = sfn.find('(');
        if (pos != std::string::npos) sfn = sfn.substr(0, pos);

    // start cast
        sig << '(' << f->GetReturnTypeName() << " (";

    // add scope for methods
        pos = sfn.rfind(':');
        if (pos != std::string::npos) {
            std::string scope_name = sfn.substr(0, pos-1);
            TCppScope_t scope = GetScope(scope_name);
            if (scope && !IsNamespace(scope))
                sig << scope_name << "::";
        }

    // finalize cast
        sig << "*)" << GetMethodSignature(method, false)
                    << ((f->Property() & kIsConstMethod) ? " const" : "")
            << ')';

    // load address
        sig << '&' << sfn;
        gInterpreter->Calc(sig.str().c_str());
    }

    return (TCppFuncAddr_t)gInterpreter->FindSym(f->GetMangledName());
}


// handling of function argument buffer --------------------------------------
void* Cppyy::AllocateFunctionArgs(size_t nargs)
{
    return new Parameter[nargs];
}

void Cppyy::DeallocateFunctionArgs(void* args)
{
    delete [] (Parameter*)args;
}

size_t Cppyy::GetFunctionArgSizeof()
{
    return sizeof(Parameter);
}

size_t Cppyy::GetFunctionArgTypeoffset()
{
    return offsetof(Parameter, fTypeCode);
}


// scope reflection information ----------------------------------------------
bool Cppyy::IsNamespace(TCppScope_t scope)
{
// Test if this scope represents a namespace.
    if (scope == GLOBAL_HANDLE)
        return true;
    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass())
        return cr->Property() & kIsNamespace;
    return false;
}

bool Cppyy::IsAbstract(TCppType_t klass)
{
// Test if this type may not be instantiated.
    TClassRef& cr = type_from_handle(klass);
    if (cr.GetClass())
        return cr->Property() & kIsAbstract;
    return false;
}

bool Cppyy::IsEnum(const std::string& type_name)
{
    if (type_name.empty()) return false;
    std::string tn_short = TClassEdit::ShortType(type_name.c_str(), 1);
    if (tn_short.empty()) return false;
    return gInterpreter->ClassInfo_IsEnum(tn_short.c_str());
}

bool Cppyy::IsAggregate(TCppType_t klass)
{
// Test if this type is an aggregate type
    TClassRef& cr = type_from_handle(klass);
    if (cr.GetClass())
        return cr->ClassProperty() & kClassIsAggregate;
    return false;
}

bool Cppyy::IsDefaultConstructable(TCppType_t type)
{
// Test if this type has a default constructor or is a "plain old data" type
    TClassRef& cr = type_from_handle(type);
    if (cr.GetClass())
        return cr->HasDefaultConstructor() || (cr->ClassProperty() & kClassIsAggregate);
    return true;
}

// helpers for stripping scope names
static
std::string outer_with_template(const std::string& name)
{
// Cut down to the outer-most scope from <name>, taking proper care of templates.
    int tpl_open = 0;
    for (std::string::size_type pos = 0; pos < name.size(); ++pos) {
        std::string::value_type c = name[pos];

    // count '<' and '>' to be able to skip template contents
        if (c == '<')
            ++tpl_open;
        else if (c == '>')
            --tpl_open;

    // collect name up to "::"
        else if (tpl_open == 0 && \
                 c == ':' && pos+1 < name.size() && name[pos+1] == ':') {
        // found the extend of the scope ... done
            return name.substr(0, pos-1);
        }
    }

// whole name is apparently a single scope
    return name;
}

static
std::string outer_no_template(const std::string& name)
{
// Cut down to the outer-most scope from <name>, drop templates
    std::string::size_type first_scope = name.find(':');
    if (first_scope == std::string::npos)
        return name.substr(0, name.find('<'));
    std::string::size_type first_templ = name.find('<');
    if (first_templ == std::string::npos)
        return name.substr(0, first_scope);
    return name.substr(0, std::min(first_templ, first_scope));
}

#define FILL_COLL(type, filter) {                                             \
    TIter itr{coll};                                                          \
    type* obj = nullptr;                                                      \
    while ((obj = (type*)itr.Next())) {                                       \
        const char* nm = obj->GetName();                                      \
        if (nm && nm[0] != '_' && !(obj->Property() & (filter))) {            \
            if (gInitialNames.find(nm) == gInitialNames.end())                \
                cppnames.insert(nm);                                          \
    }}}

static inline
void cond_add(Cppyy::TCppScope_t scope, const std::string& ns_scope,
    std::set<std::string>& cppnames, const char* name, bool nofilter = false)
{
    if (!name || name[0] == '_' || strstr(name, ".h") != 0 || strncmp(name, "operator", 8) == 0)
        return;

    if (scope == GLOBAL_HANDLE) {
        std::string to_add = outer_no_template(name);
        if ((nofilter || gInitialNames.find(to_add) == gInitialNames.end()) && !is_missclassified_stl(name))
            cppnames.insert(outer_no_template(name));
    } else if (scope == STD_HANDLE) {
        if (strncmp(name, "std::", 5) == 0) {
            name += 5;
#ifdef __APPLE__
            if (strncmp(name, "__1::", 5) == 0) name += 5;
#endif
        } else if (!is_missclassified_stl(name))
            return;
        cppnames.insert(outer_no_template(name));
    } else {
        if (strncmp(name, ns_scope.c_str(), ns_scope.size()) == 0)
            cppnames.insert(outer_with_template(name + ns_scope.size()));
    }
}

void Cppyy::GetAllCppNames(TCppScope_t scope, std::set<std::string>& cppnames)
{
// Collect all known names of C++ entities under scope. This is useful for IDEs
// employing tab-completion, for example. Note that functions names need not be
// unique as they can be overloaded.
    TClassRef& cr = type_from_handle(scope);
    if (scope != GLOBAL_HANDLE && !(cr.GetClass() && cr->Property()))
        return;

    std::string ns_scope = GetFinalName(scope);
    if (scope != GLOBAL_HANDLE) ns_scope += "::";

// add existing values from read rootmap files if within this scope
    TCollection* coll = gInterpreter->GetMapfile()->GetTable();
    {
        TIter itr{coll};
        TEnvRec* ev = nullptr;
        while ((ev = (TEnvRec*)itr.Next())) {
        // TEnv contains rootmap entries and user-side rootmap files may be already
        // loaded on startup. Thus, filter on file name rather than load time.
            if (gRootSOs.find(ev->GetValue()) == gRootSOs.end())
                cond_add(scope, ns_scope, cppnames, ev->GetName(), true);
        }
    }

// do we care about the class table or are the rootmap and list of types enough?
/*
    gClassTable->Init();
    const int N = gClassTable->Classes();
    for (int i = 0; i < N; ++i)
        cond_add(scope, ns_scope, cppnames, gClassTable->Next());
*/

// any other types (e.g. that may have come from parsing headers)
    coll = gROOT->GetListOfTypes();
    {
        TIter itr{coll};
        TDataType* dt = nullptr;
        while ((dt = (TDataType*)itr.Next())) {
            if (!(dt->Property() & kIsFundamental)) {
                cond_add(scope, ns_scope, cppnames, dt->GetName());
            }
        }
    }

// add functions
    coll = (scope == GLOBAL_HANDLE) ?
        gROOT->GetListOfGlobalFunctions() : cr->GetListOfMethods();
    {
        TIter itr{coll};
        TFunction* obj = nullptr;
        while ((obj = (TFunction*)itr.Next())) {
            const char* nm = obj->GetName();
        // skip templated functions, adding only the un-instantiated ones
            if (nm && nm[0] != '_' && strstr(nm, "<") == 0 && strncmp(nm, "operator", 8) != 0) {
                if (gInitialNames.find(nm) == gInitialNames.end())
                    cppnames.insert(nm);
            }
        }
    }

// add uninstantiated templates
    coll = (scope == GLOBAL_HANDLE) ?
        gROOT->GetListOfFunctionTemplates() : cr->GetListOfFunctionTemplates();
    FILL_COLL(TFunctionTemplate, kIsPrivate | kIsProtected)

// add (global) data members
    if (scope == GLOBAL_HANDLE) {
        coll = gROOT->GetListOfGlobals();
        FILL_COLL(TGlobal, kIsEnum | kIsPrivate | kIsProtected)
    } else {
        coll = cr->GetListOfDataMembers();
        FILL_COLL(TDataMember, kIsEnum | kIsPrivate | kIsProtected)
        coll = cr->GetListOfUsingDataMembers();
        FILL_COLL(TDataMember, kIsEnum | kIsPrivate | kIsProtected)
    }

// add enums values only for user classes/namespaces
    if (scope != GLOBAL_HANDLE && scope != STD_HANDLE) {
        coll = cr->GetListOfEnums();
        FILL_COLL(TEnum, kIsPrivate | kIsProtected)
    }

#ifdef __APPLE__
// special case for Apple, add version namespace '__1' entries to std
    if (scope == STD_HANDLE)
        GetAllCppNames(GetScope("std::__1"), cppnames);
#endif
}


// class reflection information ----------------------------------------------
std::vector<Cppyy::TCppScope_t> Cppyy::GetUsingNamespaces(TCppScope_t scope)
{
    std::vector<Cppyy::TCppScope_t> res;
    if (!IsNamespace(scope))
        return res;

#ifdef __APPLE__
    if (scope == STD_HANDLE) {
        res.push_back(GetScope("__1"));
        return res;
    }
#endif

    TClassRef& cr = type_from_handle(scope);
    if (!cr.GetClass() || !cr->GetClassInfo())
        return res;

    const std::vector<std::string>& v = gInterpreter->GetUsingNamespaces(cr->GetClassInfo());
    res.reserve(v.size());
    for (const auto& uid : v) {
        Cppyy::TCppScope_t uscope = GetScope(uid);
        if (uscope) res.push_back(uscope);
    }

    return res;
}


// class reflection information ----------------------------------------------
std::string Cppyy::GetFinalName(TCppType_t klass)
{
    if (klass == GLOBAL_HANDLE)
        return "";
    TClassRef& cr = type_from_handle(klass);
    std::string clName = cr->GetName();
// TODO: why is this template splitting needed?
    std::string::size_type pos = clName.substr(0, clName.find('<')).rfind("::");
    if (pos != std::string::npos)
        return clName.substr(pos+2, std::string::npos);
    return clName;
}

std::string Cppyy::GetScopedFinalName(TCppType_t klass)
{
    if (klass == GLOBAL_HANDLE)
        return "";
    TClassRef& cr = type_from_handle(klass);
    if (cr.GetClass()) {
        std::string name = cr->GetName();
        if (is_missclassified_stl(name))
            return std::string("std::")+cr->GetName();
        return cr->GetName();
    }
    return "";
}

bool Cppyy::HasVirtualDestructor(TCppType_t klass)
{
    TClassRef& cr = type_from_handle(klass);
    if (!cr.GetClass())
        return false;

    TFunction* f = cr->GetMethod(("~"+GetFinalName(klass)).c_str(), "");
    if (f && (f->Property() & kIsVirtual))
        return true;

    return false;
}

bool Cppyy::HasComplexHierarchy(TCppType_t klass)
{
    int is_complex = 1;
    size_t nbases = 0;

    TClassRef& cr = type_from_handle(klass);
    if (cr.GetClass() && cr->GetListOfBases() != 0)
        nbases = GetNumBases(klass);

    if (1 < nbases)
        is_complex = 1;
    else if (nbases == 0)
        is_complex = 0;
    else {         // one base class only
        TBaseClass* base = (TBaseClass*)cr->GetListOfBases()->At(0);
        if (base->Property() & kIsVirtualBase)
            is_complex = 1;       // TODO: verify; can be complex, need not be.
        else
            is_complex = HasComplexHierarchy(GetScope(base->GetName()));
    }

    return is_complex;
}

Cppyy::TCppIndex_t Cppyy::GetNumBases(TCppType_t klass)
{
// Get the total number of base classes that this class has.
    TClassRef& cr = type_from_handle(klass);
    if (cr.GetClass() && cr->GetListOfBases() != 0)
        return (TCppIndex_t)cr->GetListOfBases()->GetSize();
    return (TCppIndex_t)0;
}

////////////////////////////////////////////////////////////////////////////////
/// \fn Cppyy::TCppIndex_t GetLongestInheritancePath(TClass *klass)
/// \brief Retrieve number of base classes in the longest branch of the
///        inheritance tree of the input class.
/// \param[in] klass The class to start the retrieval process from.
///
/// This is a helper function for Cppyy::GetNumBasesLongestBranch.
/// Given an inheritance tree, the function assigns weight 1 to each class that
/// has at least one base. Starting from the input class, the function is
/// called recursively on all the bases. For each base the return value is one
/// (the weight of the base itself) plus the maximum value retrieved for their
/// bases in turn. For example, given the following inheritance tree:
///
/// ~~~{.cpp}
/// class A {}; class B: public A {};
/// class X {}; class Y: public X {}; class Z: public Y {};
/// class C: public B, Z {};
/// ~~~
///
/// calling this function on an instance of `C` will return 3, the steps
/// required to go from C to X.
Cppyy::TCppIndex_t GetLongestInheritancePath(TClass *klass)
{

   auto directbases = klass->GetListOfBases();
   if (!directbases) {
      // This is a leaf with no bases
      return 0;
   }
   auto ndirectbases = directbases->GetSize();
   if (ndirectbases == 0) {
      // This is a leaf with no bases
      return 0;
   } else {
      // If there is at least one direct base
      std::vector<Cppyy::TCppIndex_t> nbases_branches;
      nbases_branches.reserve(ndirectbases);

      // Traverse all direct bases of the current class and call the function
      // recursively
      for (auto baseclass : TRangeDynCast<TBaseClass>(directbases)) {
         if (!baseclass)
            continue;
         if (auto baseclass_tclass = baseclass->GetClassPointer()) {
            nbases_branches.emplace_back(GetLongestInheritancePath(baseclass_tclass));
         }
      }

      // Get longest path among the direct bases of the current class
      auto longestbranch = std::max_element(std::begin(nbases_branches), std::end(nbases_branches));

      // Add 1 to include the current class in the count
      return 1 + *longestbranch;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \fn Cppyy::TCppIndex_t Cppyy::GetNumBasesLongest(TCppType_t klass)
/// \brief Retrieve number of base classes in the longest branch of the
///        inheritance tree.
/// \param[in] klass The class to start the retrieval process from.
///
/// The function converts the input class to a `TClass *` and calls
/// GetLongestInheritancePath.
Cppyy::TCppIndex_t Cppyy::GetNumBasesLongestBranch(TCppType_t klass)
{

   const auto &cr = type_from_handle(klass);

   if (auto klass_tclass = cr.GetClass()) {
      return GetLongestInheritancePath(klass_tclass);
   }

   // In any other case, return zero
   return 0;
}

std::string Cppyy::GetBaseName(TCppType_t klass, TCppIndex_t ibase)
{
    TClassRef& cr = type_from_handle(klass);
    return ((TBaseClass*)cr->GetListOfBases()->At((int)ibase))->GetName();
}

bool Cppyy::IsSubtype(TCppType_t derived, TCppType_t base)
{
    if (derived == base)
        return true;
    TClassRef& derived_type = type_from_handle(derived);
    TClassRef& base_type = type_from_handle(base);
    if (derived_type.GetClass() && base_type.GetClass())
        return derived_type->GetBaseClass(base_type) != 0;
    return false;
}

bool Cppyy::IsSmartPtr(TCppType_t klass)
{
    TClassRef& cr = type_from_handle(klass);
    const std::string& tn = cr->GetName();
    if (gSmartPtrTypes.find(tn.substr(0, tn.find("<"))) != gSmartPtrTypes.end())
        return true;
    return false;
}

bool Cppyy::GetSmartPtrInfo(
    const std::string& tname, TCppType_t* raw, TCppMethod_t* deref)
{
    const std::string& rn = ResolveName(tname);
    if (gSmartPtrTypes.find(rn.substr(0, rn.find("<"))) != gSmartPtrTypes.end()) {
        if (!raw && !deref) return true;

        TClassRef& cr = type_from_handle(GetScope(tname));
        if (cr.GetClass()) {
            TFunction* func = cr->GetMethod("operator->", "");
            if (!func) {
                gInterpreter->UpdateListOfMethods(cr.GetClass());
                func = cr->GetMethod("operator->", "");
            }
            if (func) {
               if (deref) *deref = (TCppMethod_t)new_CallWrapper(func);
               if (raw) *raw = GetScope(TClassEdit::ShortType(
                   func->GetReturnTypeNormalizedName().c_str(), 1));
               return (!deref || *deref) && (!raw || *raw);
            }
        }
    }

    return false;
}

void Cppyy::AddSmartPtrType(const std::string& type_name)
{
    gSmartPtrTypes.insert(ResolveName(type_name));
}

void Cppyy::AddTypeReducer(const std::string& /*reducable*/, const std::string& /*reduced*/)
{
    // This function is deliberately left empty, because it is not used in
    // PyROOT, and synchronizing it with cppyy-backend upstream would require
    // patches to ROOT meta.
}


// type offsets --------------------------------------------------------------
ptrdiff_t Cppyy::GetBaseOffset(TCppType_t derived, TCppType_t base,
    TCppObject_t address, int direction, bool rerror)
{
// calculate offsets between declared and actual type, up-cast: direction > 0; down-cast: direction < 0
    if (derived == base || !(base && derived))
        return (ptrdiff_t)0;

    TClassRef& cd = type_from_handle(derived);
    TClassRef& cb = type_from_handle(base);

    if (!cd.GetClass() || !cb.GetClass())
        return (ptrdiff_t)0;

    ptrdiff_t offset = -1;
    if (!(cd->GetClassInfo() && cb->GetClassInfo())) {     // gInterpreter requirement
    // would like to warn, but can't quite determine error from intentional
    // hiding by developers, so only cover the case where we really should have
    // had a class info, but apparently don't:
        if (cd->IsLoaded()) {
        // warn to allow diagnostics
            std::ostringstream msg;
            msg << "failed offset calculation between " << cb->GetName() << " and " << cd->GetName();
            // TODO: propagate this warning to caller w/o use of Python C-API
            // PyErr_Warn(PyExc_RuntimeWarning, const_cast<char*>(msg.str().c_str()));
            std::cerr << "Warning: " << msg.str() << '\n';
        }

    // return -1 to signal caller NOT to apply offset
        return rerror ? (ptrdiff_t)offset : 0;
    }

    offset = gInterpreter->ClassInfo_GetBaseOffset(
        cd->GetClassInfo(), cb->GetClassInfo(), (void*)address, direction > 0);
    if (offset == -1)   // Cling error, treat silently
        return rerror ? (ptrdiff_t)offset : 0;

    return (ptrdiff_t)(direction < 0 ? -offset : offset);
}


// method/function reflection information ------------------------------------
Cppyy::TCppIndex_t Cppyy::GetNumMethods(TCppScope_t scope, bool accept_namespace)
{
    if (!accept_namespace && IsNamespace(scope))
        return (TCppIndex_t)0;     // enforce lazy

    if (scope == GLOBAL_HANDLE)
        return gROOT->GetListOfGlobalFunctions(true)->GetSize();

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass() && cr->GetListOfMethods(true)) {
        Cppyy::TCppIndex_t nMethods = (TCppIndex_t)cr->GetListOfMethods(false)->GetSize();
        if (nMethods == (TCppIndex_t)0) {
            std::string clName = GetScopedFinalName(scope);
            if (clName.find('<') != std::string::npos) {
            // chicken-and-egg problem: TClass does not know about methods until
            // instantiation, so force it
                std::ostringstream stmt;
                stmt << "template class " << clName << ";";
                gInterpreter->Declare(stmt.str().c_str()/*, silent = true*/);

            // now reload the methods
                return (TCppIndex_t)cr->GetListOfMethods(true)->GetSize();
            }
        }
        return nMethods;
    }

    return (TCppIndex_t)0;         // unknown class?
}

std::vector<Cppyy::TCppIndex_t> Cppyy::GetMethodIndicesFromName(
    TCppScope_t scope, const std::string& name)
{
    std::vector<TCppIndex_t> indices;
    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        gInterpreter->UpdateListOfMethods(cr.GetClass());
        int imeth = 0;
        TFunction* func = nullptr;
        TIter next(cr->GetListOfMethods());
        while ((func = (TFunction*)next())) {
            if (match_name(name, func->GetName())) {
            // C++ functions should be public to allow access; C functions have no access
            // specifier and should always be accepted
                auto prop = func->Property();
                if ((prop & kIsPublic) || !(prop & (kIsPrivate | kIsProtected | kIsPublic)))
                    indices.push_back((TCppIndex_t)imeth);
            }
            ++imeth;
        }
    } else if (scope == GLOBAL_HANDLE) {
        TCollection* funcs = gROOT->GetListOfGlobalFunctions(true);

    // tickle deserialization
        if (!funcs->FindObject(name.c_str()))
            return indices;

        TFunction* func = nullptr;
        TIter ifunc(funcs);
        while ((func = (TFunction*)ifunc.Next())) {
            if (match_name(name, func->GetName()))
                indices.push_back((TCppIndex_t)new_CallWrapper(func));
        }
    }

    return indices;
}

Cppyy::TCppMethod_t Cppyy::GetMethod(TCppScope_t scope, TCppIndex_t idx)
{
    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TFunction* f = (TFunction*)cr->GetListOfMethods(false)->At((int)idx);
        if (f) return (Cppyy::TCppMethod_t)new_CallWrapper(f);
        return (Cppyy::TCppMethod_t)nullptr;
    }

    assert(klass == (Cppyy::TCppType_t)GLOBAL_HANDLE);
    return (Cppyy::TCppMethod_t)idx;
}

std::string Cppyy::GetMethodName(TCppMethod_t method)
{
    if (method) {
        const std::string& name = ((CallWrapper*)method)->fName;

        if (name.compare(0, 8, "operator") != 0)
        // strip template instantiation part, if any
            return name.substr(0, name.find('<'));
        return name;
    }
    return "<unknown>";
}

std::string Cppyy::GetMethodFullName(TCppMethod_t method)
{
    if (method) {
        std::string name = ((CallWrapper*)method)->fName;
        name.erase(std::remove(name.begin(), name.end(), ' '), name.end());
        return name;
    }
    return "<unknown>";
}

std::string Cppyy::GetMethodMangledName(TCppMethod_t method)
{
    if (method)
        return m2f(method)->GetMangledName();
    return "<unknown>";
}

std::string Cppyy::GetMethodResultType(TCppMethod_t method)
{
    if (method) {
        TFunction* f = m2f(method);
        if (f->ExtraProperty() & kIsConstructor)
            return "constructor";
        std::string restype = f->GetReturnTypeName();
        // TODO: this is ugly; GetReturnTypeName() keeps typedefs, but may miss scopes
        // for some reason; GetReturnTypeNormalizedName() has been modified to return
        // the canonical type to guarantee correct namespaces. Sometimes typedefs look
        // better, sometimes not, sometimes it's debatable (e.g. vector<int>::size_type).
        // So, for correctness sake, GetReturnTypeNormalizedName() is used, except for a
        // special case of uint8_t/int8_t that must propagate as their typedefs.
        if (restype.find("int8_t") != std::string::npos)
            return restype;
        restype = f->GetReturnTypeNormalizedName();
        if (restype == "(lambda)") {
            std::ostringstream s;
            // TODO: what if there are parameters to the lambda?
            s << "__cling_internal::FT<decltype("
              << GetMethodFullName(method) << "(";
            for (Cppyy::TCppIndex_t i = 0; i < Cppyy::GetMethodNumArgs(method); ++i) {
                if (i != 0) s << ", ";
                s << Cppyy::GetMethodArgType(method, i) << "{}";
            }
            s << "))>::F";
            TClass* cl = TClass::GetClass(s.str().c_str());
            if (cl) return cl->GetName();
            // TODO: signal some type of error (or should that be upstream?
        }
        return restype;
    }
    return "<unknown>";
}

Cppyy::TCppIndex_t Cppyy::GetMethodNumArgs(TCppMethod_t method)
{
    if (method)
        return m2f(method)->GetNargs();
    return 0;
}

Cppyy::TCppIndex_t Cppyy::GetMethodReqArgs(TCppMethod_t method)
{
    if (method) {
        TFunction* f = m2f(method);
        return (TCppIndex_t)(f->GetNargs() - f->GetNargsOpt());
    }
    return (TCppIndex_t)0;
}

std::string Cppyy::GetMethodArgName(TCppMethod_t method, TCppIndex_t iarg)
{
    if (method) {
        TFunction* f = m2f(method);
        TMethodArg* arg = (TMethodArg*)f->GetListOfMethodArgs()->At((int)iarg);
        return arg->GetName();
    }
    return "<unknown>";
}

std::string Cppyy::GetMethodArgType(TCppMethod_t method, TCppIndex_t iarg)
{
    if (method) {
        TFunction* f = m2f(method);
        TMethodArg* arg = (TMethodArg*)f->GetListOfMethodArgs()->At((int)iarg);
        std::string ft = arg->GetFullTypeName();
        if (ft.rfind("enum ", 0) != std::string::npos) {   // special case to preserve 'enum' tag
            std::string arg_type = arg->GetTypeNormalizedName();
            return arg_type.insert(arg_type.rfind("const ", 0) == std::string::npos ? 0 : 6, "enum ");
        } else if (g_builtins.find(ft) != g_builtins.end() || ft.find("int8_t") != std::string::npos)
            return ft;       // do not resolve int8_t and uint8_t typedefs

        return arg->GetTypeNormalizedName();
    }
    return "<unknown>";
}

Cppyy::TCppIndex_t Cppyy::CompareMethodArgType(TCppMethod_t method, TCppIndex_t iarg, const std::string &req_type)
{
    if (method) {
        TFunction* f = m2f(method);
        TMethodArg* arg = (TMethodArg *)f->GetListOfMethodArgs()->At((int)iarg);
        void *argqtp = gInterpreter->TypeInfo_QualTypePtr(arg->GetTypeInfo());

        TypeInfo_t *reqti = gInterpreter->TypeInfo_Factory(req_type.c_str());
        void *reqqtp = gInterpreter->TypeInfo_QualTypePtr(reqti);

        // This scoring is not based on any particular rules
        if (gInterpreter->IsSameType(argqtp, reqqtp))
            return 0; // Best match
        else if ((gInterpreter->IsSignedIntegerType(argqtp) && gInterpreter->IsSignedIntegerType(reqqtp)) || 
                 (gInterpreter->IsUnsignedIntegerType(argqtp) && gInterpreter->IsUnsignedIntegerType(reqqtp)) ||
                 (gInterpreter->IsFloatingType(argqtp) && gInterpreter->IsFloatingType(reqqtp)))
            return 1;
        else if ((gInterpreter->IsSignedIntegerType(argqtp) && gInterpreter->IsUnsignedIntegerType(reqqtp)) ||
                 (gInterpreter->IsFloatingType(argqtp) && gInterpreter->IsUnsignedIntegerType(reqqtp)))
            return 2;
        else if ((gInterpreter->IsIntegerType(argqtp) && gInterpreter->IsIntegerType(reqqtp)))
            return 3;
        else if ((gInterpreter->IsVoidPointerType(argqtp) && gInterpreter->IsPointerType(reqqtp)))
            return 4;
        else 
            return 10; // Penalize heavily for no possible match
    }
    return INT_MAX; // Method is not valid
}

std::string Cppyy::GetMethodArgDefault(TCppMethod_t method, TCppIndex_t iarg)
{
    if (method) {
        TFunction* f = m2f(method);
        TMethodArg* arg = (TMethodArg*)f->GetListOfMethodArgs()->At((int)iarg);
        const char* def = arg->GetDefault();
        if (def)
            return def;
    }

    return "";
}

std::string Cppyy::GetMethodSignature(TCppMethod_t method, bool show_formalargs, TCppIndex_t maxargs)
{
    TFunction* f = m2f(method);
    if (f) {
        std::ostringstream sig;
        sig << "(";
        int nArgs = f->GetNargs();
        if (maxargs != (TCppIndex_t)-1) nArgs = std::min(nArgs, (int)maxargs);
        for (int iarg = 0; iarg < nArgs; ++iarg) {
            TMethodArg* arg = (TMethodArg*)f->GetListOfMethodArgs()->At(iarg);
            sig << arg->GetFullTypeName();
            if (show_formalargs) {
                const char* argname = arg->GetName();
                if (argname && argname[0] != '\0') sig << " " << argname;
                const char* defvalue = arg->GetDefault();
                if (defvalue && defvalue[0] != '\0') sig << " = " << defvalue;
            }
            if (iarg != nArgs-1) sig << (show_formalargs ? ", " : ",");
        }
        sig << ")";
        return sig.str();
    }
    return "<unknown>";
}

std::string Cppyy::GetMethodPrototype(TCppScope_t scope, TCppMethod_t method, bool show_formalargs)
{
    std::string scName = GetScopedFinalName(scope);
    TFunction* f = m2f(method);
    if (f) {
        std::ostringstream sig;
        sig << f->GetReturnTypeName() << " "
            << scName << "::" << f->GetName();
        sig << GetMethodSignature(method, show_formalargs);
        return sig.str();
    }
    return "<unknown>";
}

bool Cppyy::IsConstMethod(TCppMethod_t method)
{
    if (method) {
        TFunction* f = m2f(method);
        return f->Property() & kIsConstMethod;
    }
    return false;
}

Cppyy::TCppIndex_t Cppyy::GetNumTemplatedMethods(TCppScope_t scope, bool accept_namespace)
{
    if (!accept_namespace && IsNamespace(scope))
        return (TCppIndex_t)0;     // enforce lazy

    if (scope == GLOBAL_HANDLE) {
        TCollection* coll = gROOT->GetListOfFunctionTemplates();
        if (coll) return (TCppIndex_t)coll->GetSize();
    } else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass()) {
            TCollection* coll = cr->GetListOfFunctionTemplates(true);
            if (coll) return (TCppIndex_t)coll->GetSize();
        }
    }

// failure ...
    return (TCppIndex_t)0;
}

std::string Cppyy::GetTemplatedMethodName(TCppScope_t scope, TCppIndex_t imeth)
{
    if (scope == (TCppScope_t)GLOBAL_HANDLE)
        return ((THashList*)gROOT->GetListOfFunctionTemplates())->At((int)imeth)->GetName();
    else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass())
            return cr->GetListOfFunctionTemplates(false)->At((int)imeth)->GetName();
    }

// failure ...
    assert(!"should not be called unless GetNumTemplatedMethods() succeeded");
    return "";
}

bool Cppyy::IsTemplatedConstructor(TCppScope_t scope, TCppIndex_t imeth)
{
    if (scope == (TCppScope_t)GLOBAL_HANDLE)
        return false;

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TFunctionTemplate* f = (TFunctionTemplate*)cr->GetListOfFunctionTemplates(false)->At((int)imeth);
        return f->ExtraProperty() & kIsConstructor;
    }

    return false;
}

bool Cppyy::ExistsMethodTemplate(TCppScope_t scope, const std::string& name)
{
    if (scope == (TCppScope_t)GLOBAL_HANDLE)
        return (bool)gROOT->GetFunctionTemplate(name.c_str());
    else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass())
            return (bool)cr->GetFunctionTemplate(name.c_str());
    }

// failure ...
    return false;
}

bool Cppyy::IsStaticTemplate(TCppScope_t scope, const std::string& name)
{
    TFunctionTemplate* tf = nullptr;
    if (scope == (TCppScope_t)GLOBAL_HANDLE)
        tf = gROOT->GetFunctionTemplate(name.c_str());
    else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass())
            tf = cr->GetFunctionTemplate(name.c_str());
    }

    if (!tf) return false;

    return (bool)(tf->Property() & kIsStatic);
}

bool Cppyy::IsMethodTemplate(TCppScope_t scope, TCppIndex_t idx)
{
    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TFunction* f = (TFunction*)cr->GetListOfMethods(false)->At((int)idx);
        if (f && strstr(f->GetName(), "<")) return true;
        return false;
    }

    assert(scope == (Cppyy::TCppType_t)GLOBAL_HANDLE);
    if (((CallWrapper*)idx)->fName.find('<') != std::string::npos) return true;
    return false;
}

// helpers for Cppyy::GetMethodTemplate()
static std::map<TDictionary::DeclId_t, CallWrapper*> gMethodTemplates;

static inline
void remove_space(std::string& n) {
   std::string::iterator pos = std::remove_if(n.begin(), n.end(), isspace);
   n.erase(pos, n.end());
}

static inline
bool template_compare(std::string n1, std::string n2) {
    if (n1.back() == '>') n1 = n1.substr(0, n1.size()-1);
    remove_space(n1);
    remove_space(n2);
    return n2.compare(0, n1.size(), n1) == 0;
}

Cppyy::TCppMethod_t Cppyy::GetMethodTemplate(
    TCppScope_t scope, const std::string& name, const std::string& proto)
{
// There is currently no clean way of extracting a templated method out of ROOT/meta
// for a variety of reasons, none of them fundamental. The game played below is to
// first get any pre-existing functions already managed by ROOT/meta, but if that fails,
// to do an explicit lookup that ignores the prototype (i.e. the full name should be
// enough), and finally to ignore the template arguments part of the name as this fails
// in cling if there are default parameters.
    TFunction* func = nullptr; ClassInfo_t* cl = nullptr;
    if (scope == (TCppScope_t)GLOBAL_HANDLE) {
        func = gROOT->GetGlobalFunctionWithPrototype(name.c_str(), proto.c_str());
        if (func && name.back() == '>') {
        // make sure that all template parameters match (more are okay, e.g. defaults or
        // ones derived from the arguments or variadic templates)
            if (!template_compare(name, func->GetName()))
                func = nullptr;  // happens if implicit conversion matches the overload
        }
    } else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass()) {
            func = cr->GetMethodWithPrototype(name.c_str(), proto.c_str());
            if (!func) {
                cl = cr->GetClassInfo();
            // try base classes to cover a common 'using' case (TODO: this is stupid and misses
            // out on base classes; fix that with improved access to Cling)
                TCppIndex_t nbases = GetNumBases(scope);
                for (TCppIndex_t i = 0; i < nbases; ++i) {
                    TClassRef& base = type_from_handle(GetScope(GetBaseName(scope, i)));
                    if (base.GetClass()) {
                        func = base->GetMethodWithPrototype(name.c_str(), proto.c_str());
                        if (func) break;
                    }
                }
            }
        }
    }

    if (!func && name.back() == '>' && (cl || scope == (TCppScope_t)GLOBAL_HANDLE)) {
    // try again, ignoring proto in case full name is complete template
        auto declid = gInterpreter->GetFunction(cl, name.c_str());
        if (declid) {
             auto existing = gMethodTemplates.find(declid);
             if (existing == gMethodTemplates.end()) {
                 auto cw = new_CallWrapper(declid, name);
                 existing = gMethodTemplates.insert(std::make_pair(declid, cw)).first;
             }
             return (TCppMethod_t)existing->second;
        }
    }

    if (func) {
    // make sure we didn't match a non-templated overload
        if (func->ExtraProperty() & kIsTemplateSpec)
            return (TCppMethod_t)new_CallWrapper(func);

    // disregard this non-templated method as it will be considered when appropriate
        return (TCppMethod_t)nullptr;
    }

// try again with template arguments removed from name, if applicable
    if (name.back() == '>') {
        auto pos = name.find('<');
        if (pos != std::string::npos) {
            TCppMethod_t cppmeth = GetMethodTemplate(scope, name.substr(0, pos), proto);
            if (cppmeth) {
            // allow if requested template names match up to the result
                const std::string& alt = GetMethodFullName(cppmeth);
                if (name.size() < alt.size() && alt.find('<') == pos) {
                    if (template_compare(name, alt))
                        return cppmeth;
                }
            }
        }
    }

// failure ...
    return (TCppMethod_t)nullptr;
}

static inline
std::string type_remap(const std::string& n1, const std::string& n2)
{
// Operator lookups of (C++ string, Python str) should succeed for the combos of
// string/str, wstring/str, string/unicode and wstring/unicode; since C++ does not have a
// operator+(std::string, std::wstring), we'll have to look up the same type and rely on
// the converters in CPyCppyy/_cppyy.
    if (n1 == "str" || n1 == "unicode") {
        if (n2 == "std::basic_string<wchar_t,std::char_traits<wchar_t>,std::allocator<wchar_t> >")
            return n2;                      // match like for like
        return "std::string";               // probably best bet
    } else if (n1 == "float") {
        return "double";                    // debatable, but probably intended
    } else if (n1 == "complex") {
        return "std::complex<double>";
    }
    return n1;
}

Cppyy::TCppIndex_t Cppyy::GetGlobalOperator(
    TCppType_t scope, const std::string& lc, const std::string& rc, const std::string& opname)
{
// Find a global operator function with a matching signature; prefer by-ref, but
// fall back on by-value if that fails.
    std::string lcname1 = TClassEdit::CleanType(lc.c_str());
    const std::string& rcname = rc.empty() ? rc : type_remap(TClassEdit::CleanType(rc.c_str()), lcname1);
    const std::string& lcname = type_remap(lcname1, rcname);

    std::string proto = lcname + "&" + (rc.empty() ? rc : (", " + rcname + "&"));
    if (scope == (TCppScope_t)GLOBAL_HANDLE) {
        TFunction* func = gROOT->GetGlobalFunctionWithPrototype(opname.c_str(), proto.c_str());
        if (func) return (TCppIndex_t)new_CallWrapper(func);
        proto = lcname + (rc.empty() ? rc : (", " + rcname));
        func = gROOT->GetGlobalFunctionWithPrototype(opname.c_str(), proto.c_str());
        if (func) return (TCppIndex_t)new_CallWrapper(func);
    } else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass()) {
            TFunction* func = cr->GetMethodWithPrototype(opname.c_str(), proto.c_str());
            if (func) return (TCppIndex_t)cr->GetListOfMethods()->IndexOf(func);
            proto = lcname + (rc.empty() ? rc : (", " + rcname));
            func = cr->GetMethodWithPrototype(opname.c_str(), proto.c_str());
            if (func) return (TCppIndex_t)cr->GetListOfMethods()->IndexOf(func);
        }
    }

// failure ...
    return (TCppIndex_t)-1;
}

// method properties ---------------------------------------------------------

static inline bool testMethodProperty(Cppyy::TCppMethod_t method, EProperty prop)
{
    if (!method)
        return false;
    TFunction *f = m2f(method);
    return f->Property() & prop;
}

static inline bool testMethodExtraProperty(Cppyy::TCppMethod_t method, EFunctionProperty prop)
{
    if (!method)
        return false;
    TFunction *f = m2f(method);
    return f->ExtraProperty() & prop;
}

bool Cppyy::IsPublicMethod(TCppMethod_t method)
{
    return testMethodProperty(method, kIsPublic);
}

bool Cppyy::IsProtectedMethod(TCppMethod_t method)
{
    return testMethodProperty(method, kIsProtected);
}

bool Cppyy::IsConstructor(TCppMethod_t method)
{
    return testMethodExtraProperty(method, kIsConstructor);
}

bool Cppyy::IsDestructor(TCppMethod_t method)
{
    return testMethodExtraProperty(method, kIsDestructor);
}

bool Cppyy::IsStaticMethod(TCppMethod_t method)
{
    return testMethodProperty(method, kIsStatic);
}

bool Cppyy::IsExplicit(TCppMethod_t method)
{
    return testMethodProperty(method, kIsExplicit);
}

// data member reflection information ----------------------------------------
Cppyy::TCppIndex_t Cppyy::GetNumDatamembers(TCppScope_t scope, bool accept_namespace)
{
    if (!accept_namespace && IsNamespace(scope))
        return (TCppIndex_t)0;     // enforce lazy

    if (scope == GLOBAL_HANDLE)
        return gROOT->GetListOfGlobals(true)->GetSize();

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass() && cr->GetListOfDataMembers())
        return cr->GetListOfDataMembers()->GetSize();

    return (TCppIndex_t)0;         // unknown class?
}

std::string Cppyy::GetDatamemberName(TCppScope_t scope, TCppIndex_t idata)
{
    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
        return m->GetName();
    }
    assert(scope == GLOBAL_HANDLE);
    TGlobal* gbl = g_globalvars[idata];
    return gbl->GetName();
}

static inline
int count_scopes(const std::string& tpname)
{
    int count = 0;
    std::string::size_type pos = tpname.find("::", 0);
    while (pos != std::string::npos) {
        count++;
        pos = tpname.find("::", pos+1);
    }
    return count;
}

std::string Cppyy::GetDatamemberType(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gbl = g_globalvars[idata];
        std::string fullType = gbl->GetFullTypeName();

        if ((int)gbl->GetArrayDim()) {
            std::ostringstream s;
            for (int i = 0; i < (int)gbl->GetArrayDim(); ++i)
                s << '[' << gbl->GetMaxIndex(i) << ']';
            fullType.append(s.str());
        }
        return fullType;
    }

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass())  {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
    // TODO: fix this upstream ... Usually, we want m->GetFullTypeName(), because it
    // does not resolve typedefs, but it looses scopes for inner classes/structs, it
    // doesn't resolve constexpr (leaving unresolved names), leaves spurious "struct"
    // or "union" in the name, and can not handle anonymous unions. In that case
    // m->GetTrueTypeName() should be used. W/o clear criteria to determine all these
    // cases, the general rules are to prefer the true name if the full type does not
    // exist as a type for classes, and the most scoped name otherwise.
        const char* ft = m->GetFullTypeName(); std::string fullType = ft ? ft : "";
        const char* tn = m->GetTrueTypeName(); std::string trueName = tn ? tn : "";
        if (!trueName.empty() && fullType != trueName && !IsBuiltin(trueName)) {
            if ( (!TClass::GetClass(fullType.c_str()) && TClass::GetClass(trueName.c_str())) || \
                 (count_scopes(trueName) > count_scopes(fullType)) ) {
                bool is_enum_tag = fullType.rfind("enum ", 0) != std::string::npos;
                fullType = trueName;
                if (is_enum_tag)
                   fullType.insert(fullType.rfind("const ", 0) == std::string::npos ? 0 : 6, "enum ");
            }
        }

        if ((int)m->GetArrayDim()) {
            std::ostringstream s;
            for (int i = 0; i < (int)m->GetArrayDim(); ++i)
                s << '[' << m->GetMaxIndex(i) << ']';
            fullType.append(s.str());
        }
        return fullType;
    }

    return "<unknown>";
}

intptr_t Cppyy::GetDatamemberOffset(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gbl = g_globalvars[idata];
        if (!gbl->GetAddress() || gbl->GetAddress() == (void*)-1) {
        // CLING WORKAROUND: make sure variable is loaded
            intptr_t addr = (intptr_t)gInterpreter->ProcessLine((std::string("&")+gbl->GetName()+";").c_str());
            if (gbl->GetAddress() && gbl->GetAddress() != (void*)-1)
                return (intptr_t)gbl->GetAddress();        // now loaded!
            return addr;                                   // last resort ...
        }
        return (intptr_t)gbl->GetAddress();
    }

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
    // CLING WORKAROUND: the following causes templates to be instantiated first within the proper
    // scope, making the lookup succeed and preventing spurious duplicate instantiations later. Also,
    // if the variable is not yet loaded, pull it in through gInterpreter.
        intptr_t offset = (intptr_t)-1;
        if (m->Property() & kIsStatic) {
            if (strchr(cr->GetName(), '<'))
                gInterpreter->ProcessLine(((std::string)cr->GetName()+"::"+m->GetName()+";").c_str());
            offset = (intptr_t)m->GetOffsetCint();    // yes, CINT (GetOffset() is both wrong
                                                      // and caches that wrong result!
            if (offset == (intptr_t)-1)
                return (intptr_t)gInterpreter->ProcessLine((std::string("&")+cr->GetName()+"::"+m->GetName()+";").c_str());
        } else
            offset = (intptr_t)m->GetOffsetCint();    // yes, CINT, see above
        return offset;
    }

    return (intptr_t)-1;
}

static inline
Cppyy::TCppIndex_t gb2idx(TGlobal* gb)
{
    if (!gb) return (Cppyy::TCppIndex_t)-1;

    auto pidx = g_globalidx.find(gb);
    if (pidx == g_globalidx.end()) {
        auto idx = g_globalvars.size();
        g_globalvars.push_back(gb);
        g_globalidx[gb] = idx;
        return (Cppyy::TCppIndex_t)idx;
    }
    return (Cppyy::TCppIndex_t)pidx->second;
}

Cppyy::TCppIndex_t Cppyy::GetDatamemberIndex(TCppScope_t scope, const std::string& name)
{
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gb = (TGlobal*)gROOT->GetListOfGlobals(false /* load */)->FindObject(name.c_str());
        if (!gb) gb = (TGlobal*)gROOT->GetListOfGlobals(true  /* load */)->FindObject(name.c_str());
        if (!gb) {
        // some enums are not loaded as they are not considered part of
        // the global scope, but of the enum scope; get them w/o checking
            TDictionary::DeclId_t did = gInterpreter->GetDataMember(nullptr, name.c_str());
            if (did) {
                DataMemberInfo_t* t = gInterpreter->DataMemberInfo_Factory(did, nullptr);
                ((TListOfDataMembers*)gROOT->GetListOfGlobals())->Get(t, true);
                gb = (TGlobal*)gROOT->GetListOfGlobals(false /* load */)->FindObject(name.c_str());
            }
        }

        if (gb && strcmp(gb->GetFullTypeName(), "(lambda)") == 0) {
        // lambdas use a compiler internal closure type, so we wrap
        // them, then return the wrapper's type
        // TODO: this current leaks the std::function; also, if possible,
        //       should instantiate through TClass rather then ProcessLine
            std::ostringstream s;
            s << "auto __cppyy_internal_wrap_" << name << " = "
                 "new __cling_internal::FT<decltype(" << name << ")>::F"
                 "{" << name << "};";
            gInterpreter->ProcessLine(s.str().c_str());
            TGlobal* wrap = (TGlobal*)gROOT->GetListOfGlobals(true)->FindObject(
                ("__cppyy_internal_wrap_"+name).c_str());
            if (wrap && wrap->GetAddress()) gb = wrap;
        }

        return gb2idx(gb);

    } else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass()) {
            TDataMember* dm =
                (TDataMember*)cr->GetListOfDataMembers()->FindObject(name.c_str());
            // TODO: turning this into an index is silly ...
            if (dm) return (TCppIndex_t)cr->GetListOfDataMembers()->IndexOf(dm);
        }
    }

    return (TCppIndex_t)-1;
}

Cppyy::TCppIndex_t Cppyy::GetDatamemberIndexEnumerated(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gb = (TGlobal*)((THashList*)gROOT->GetListOfGlobals(false /* load */))->At((int)idata);
        return gb2idx(gb);
    }

    return idata;
}


// data member properties ----------------------------------------------------
bool Cppyy::IsPublicData(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE)
        return true;
    TClassRef& cr = type_from_handle(scope);
    if (cr->Property() & kIsNamespace)
        return true;
    TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
    return m->Property() & kIsPublic;
}

bool Cppyy::IsProtectedData(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE)
        return true;
    TClassRef& cr = type_from_handle(scope);
    if (cr->Property() & kIsNamespace)
        return true;
    TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
    return m->Property() & kIsProtected;
}

bool Cppyy::IsStaticData(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE)
        return true;
    TClassRef& cr = type_from_handle(scope);
    if (cr->Property() & kIsNamespace)
        return true;
    TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
    return m->Property() & kIsStatic;
}

bool Cppyy::IsConstData(TCppScope_t scope, TCppIndex_t idata)
{
    Long_t property = 0;
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gbl = g_globalvars[idata];
        property = gbl->Property();
    }
    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
        property = m->Property();
    }

// if the data type is const, but the data member is a pointer/array, the data member
// itself is not const; alternatively it is a pointer that is constant
    return ((property & kIsConstant) && !(property & (kIsPointer | kIsArray))) || (property & kIsConstPointer);
}

bool Cppyy::IsEnumData(TCppScope_t scope, TCppIndex_t idata)
{
// TODO: currently, ROOT/meta does not properly distinguish between variables of enum
// type, and values of enums. The latter are supposed to be const. This code relies on
// odd features (bugs?) to figure out the difference, but this should really be fixed
// upstream and/or deserves a new API.

    if (scope == GLOBAL_HANDLE) {
        TGlobal* gbl = g_globalvars[idata];

    // make use of an oddity: enum global variables do not have their kIsStatic bit
    // set, whereas enum global values do
        return (gbl->Property() & kIsEnum) && (gbl->Property() & kIsStatic);
    }

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
        std::string ti = m->GetTypeName();

    // can't check anonymous enums by type name, so just accept them as enums
        if (ti.rfind("(anonymous)") != std::string::npos || ti.rfind("(unnamed)") != std::string::npos)
            return m->Property() & kIsEnum;

    // since there seems to be no distinction between data of enum type and enum values,
    // check the list of constants for the type to see if there's a match
        if (ti.rfind(cr->GetName(), 0) != std::string::npos) {
            std::string::size_type s = strlen(cr->GetName())+2;
            if (s < ti.size()) {
                TEnum* ee = ((TListOfEnums*)cr->GetListOfEnums())->GetObject(ti.substr(s, std::string::npos).c_str());
                if (ee) return ee->GetConstant(m->GetName());
            }
        }
    }

// this default return only means that the data will be writable, not that it will
// be unreadable or otherwise misrepresented
    return false;
}

int Cppyy::GetDimensionSize(TCppScope_t scope, TCppIndex_t idata, int dimension)
{
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gbl = g_globalvars[idata];
        return gbl->GetMaxIndex(dimension);
    }
    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
        return m->GetMaxIndex(dimension);
    }
    return -1;
}


// enum properties -----------------------------------------------------------
Cppyy::TCppEnum_t Cppyy::GetEnum(TCppScope_t scope, const std::string& enum_name)
{
    if (scope == GLOBAL_HANDLE)
        return (TCppEnum_t)gROOT->GetListOfEnums(kTRUE)->FindObject(enum_name.c_str());

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass())
        return (TCppEnum_t)cr->GetListOfEnums(kTRUE)->FindObject(enum_name.c_str());

    return (TCppEnum_t)0;
}

Cppyy::TCppIndex_t Cppyy::GetNumEnumData(TCppEnum_t etype)
{
    return (TCppIndex_t)((TEnum*)etype)->GetConstants()->GetSize();
}

std::string Cppyy::GetEnumDataName(TCppEnum_t etype, TCppIndex_t idata)
{
    return ((TEnumConstant*)((TEnum*)etype)->GetConstants()->At((int)idata))->GetName();
}

long long Cppyy::GetEnumDataValue(TCppEnum_t etype, TCppIndex_t idata)
{
     TEnumConstant* ecst = (TEnumConstant*)((TEnum*)etype)->GetConstants()->At((int)idata);
     return (long long)ecst->GetValue();
}
