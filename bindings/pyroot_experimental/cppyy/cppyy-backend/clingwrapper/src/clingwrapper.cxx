// Bindings
#include "capi.h"
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
#include "TEnv.h"
#include "TError.h"
#include "TFunction.h"
#include "TFunctionTemplate.h"
#include "TGlobal.h"
#include "TInterpreter.h"
#include "TList.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TROOT.h"
#include "TSystem.h"

// Standard
#include <assert.h>
#include <algorithm>     // for std::count
#include <dlfcn.h>
#include <stdexcept>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <stdlib.h>      // for getenv
#include <string.h>

// temp
#include <iostream>
typedef CPyCppyy::Parameter Parameter;
// --temp


// small number that allows use of stack for argument passing
const int SMALL_ARGS_N = 8;

// data for life time management ---------------------------------------------
typedef std::vector<TClassRef> ClassRefs_t;
static ClassRefs_t g_classrefs(1);
static const ClassRefs_t::size_type GLOBAL_HANDLE = 1;
static const ClassRefs_t::size_type STD_HANDLE = GLOBAL_HANDLE + 1;

typedef std::map<std::string, ClassRefs_t::size_type> Name2ClassRefIndex_t;
static Name2ClassRefIndex_t g_name2classrefidx;

struct CallWrapper {
    CallWrapper(TFunction* f) : fMetaFunction(f), fWrapper(nullptr) {}
    TFunction*  fMetaFunction;
    CallFunc_t* fWrapper;
};
static std::vector<CallWrapper*> gWrapperHolder;
static inline CallWrapper* new_CallWrapper(TFunction* f) {
    CallWrapper* wrap = new CallWrapper(f);
    gWrapperHolder.push_back(wrap);
    return wrap;
}


typedef std::vector<TGlobal*> GlobalVars_t;
static GlobalVars_t g_globalvars;

static std::set<std::string> gSTLNames;

// data ----------------------------------------------------------------------
Cppyy::TCppScope_t Cppyy::gGlobalScope = GLOBAL_HANDLE;

// smart pointer types
static std::set<std::string> gSmartPtrTypes =
    {"auto_ptr", "shared_ptr", "weak_ptr", "unique_ptr"};

// configuration
static bool gEnableFastPath = true;


// global initialization -----------------------------------------------------
namespace {

class ApplicationStarter {
public:
    ApplicationStarter() {
    // setup dummy holders for global and std namespaces
        assert(g_classrefs.size() == GLOBAL_HANDLE);
        g_name2classrefidx[""]      = GLOBAL_HANDLE;
        g_classrefs.push_back(TClassRef(""));

    // aliases for std (setup already in pythonify)
        g_name2classrefidx["std"]   = STD_HANDLE;
        g_name2classrefidx["::std"] = g_name2classrefidx["std"];
        g_classrefs.push_back(TClassRef("std"));

    // add a dummy global to refer to as null at index 0
        g_globalvars.push_back(nullptr);

    // disable fast path if requested
        if (getenv("CPPYY_DISABLE_FASTPATH")) gEnableFastPath = false;

    // fill the set of STL names
        const char* stl_names[] = {"string", "bitset", "pair", "allocator",
            "auto_ptr", "shared_ptr", "unique_ptr", "weak_ptr",
            "vector", "list", "forward_list", "deque", "map", "multimap",
            "set", "multiset", "unordered_set", "unordered_multiset",
            "unordered_map", "unordered_multimap",
            "iterator", "reverse_iterator", "basic_string",
            "complex", "valarray"};
        for (auto& name : stl_names)
            gSTLNames.insert(name);

    // create a helper for wrapping lambdas
        gInterpreter->Declare(
            "namespace __cppyy_internal { template <typename F>"
            "struct FT : public FT<decltype(&F::operator())> {};"
            "template <typename C, typename R, typename... Args>"
            "struct FT<R(C::*)(Args...) const> { typedef std::function<R(Args...)> F; };}"
        );

    // start off with a reasonable size placeholder for wrappers
        gWrapperHolder.reserve(1024);
    }

    ~ApplicationStarter() {
        for (auto wrap : gWrapperHolder) {
            if (wrap->fWrapper)
                gInterpreter->CallFunc_Delete(wrap->fWrapper);
            delete wrap;
        }
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

// type_from_handle to go here
static inline
TFunction* type_get_method(Cppyy::TCppType_t klass, Cppyy::TCppIndex_t idx)
{
    TClassRef& cr = type_from_handle(klass);
    if (cr.GetClass())
        return (TFunction*)cr->GetListOfMethods(false)->At(idx);
    assert(klass == (Cppyy::TCppType_t)GLOBAL_HANDLE);
    return ((CallWrapper*)idx)->fMetaFunction;
}

static inline
TFunction* m2f(Cppyy::TCppMethod_t method) {
    return ((CallWrapper*)method)->fMetaFunction;
}

static inline
Cppyy::TCppScope_t declaring_scope(Cppyy::TCppMethod_t method)
{
    if (method) {
        TMethod* m = dynamic_cast<TMethod*>(m2f(method));
        if (m) return Cppyy::GetScope(m->GetClass()->GetName());
    }
    return (Cppyy::TCppScope_t)GLOBAL_HANDLE;
}

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
bool is_stl(const std::string& name)
{
    std::string w = name;
    if (w.compare(0, 5, "std::") == 0)
        w = w.substr(5, std::string::npos);
    std::string::size_type pos = name.find('<');
    if (pos != std::string::npos)
        w = w.substr(0, pos);
    return gSTLNames.find(w) != gSTLNames.end();
}

static inline
bool is_missclassified_stl(const std::string& name)
{
    std::string::size_type pos = name.find('<');
    if (pos != std::string::npos)
        return gSTLNames.find(name.substr(0, pos)) != gSTLNames.end();
    return gSTLNames.find(name) != gSTLNames.end();
}


// name to opaque C++ scope representation -----------------------------------
std::string Cppyy::ResolveName(const std::string& cppitem_name)
{
// Fully resolve the given name to the final type name.
    std::string tclean = cppitem_name.compare(0, 2, "::") == 0 ?
        cppitem_name.substr(2, std::string::npos) : cppitem_name;

// classes (most common)
    tclean = TClassEdit::CleanType(tclean.c_str());
    if (tclean.empty() /* unknown, eg. an operator */) return cppitem_name;

// reduce [N] to []
    if (tclean[tclean.size()-1] == ']')
        tclean = tclean.substr(0, tclean.rfind('[')) + "[]";

// data types (such as builtins)
    TDataType* dt = gROOT->GetType(tclean.c_str());
    if (dt) return dt->GetFullTypeName();

// special case for enums
    if (IsEnum(cppitem_name))
        return ResolveEnum(cppitem_name);

// typedefs
    return TClassEdit::ResolveTypedef(tclean.c_str(), true);
}

static std::map<std::string, std::string> resolved_enum_types;
std::string Cppyy::ResolveEnum(const std::string& enum_type)
{
// The underlying type of a an enum may be any kind of integer.
// Resolve that type via a workaround (note: this function assumes
// that the enum_type name is a valid enum type name)
    auto res = resolved_enum_types.find(enum_type);
    if (res != resolved_enum_types.end())
        return res->second;

    if (enum_type.find("(anonymous") == std::string::npos) {
        std::ostringstream decl;
        for (auto& itype : {"unsigned int"}) {
            decl << "std::is_same<"
                 << itype
                 << ", std::underlying_type<"
                 << enum_type
                 << ">::type>::value;";
            if (gInterpreter->ProcessLine(decl.str().c_str())) {
                resolved_enum_types[enum_type] = itype;
                return itype;
            }
        }
    }

// failed or anonymous ... signal up stream to special case this
    resolved_enum_types[enum_type] = "internal_enum_type_t";
    return "internal_enum_type_t";      // should default to int
}

Cppyy::TCppScope_t Cppyy::GetScope(const std::string& sname)
{
// TODO: scope_name should always be final already
    std::string scope_name = ResolveName(sname);
    auto icr = g_name2classrefidx.find(scope_name);
    if (icr != g_name2classrefidx.end())
        return (TCppType_t)icr->second;

// use TClass directly, to enable auto-loading; class may be stubbed (eg. for
// function returns) leading to a non-null TClass that is otherwise invalid
    TClassRef cr(TClass::GetClass(scope_name.c_str(), true /* load */, true /* silent */));
    if (!cr.GetClass() || !cr->Property())
        return (TCppScope_t)0;

    // no check for ClassInfo as forward declared classes are okay (fragile)

    ClassRefs_t::size_type sz = g_classrefs.size();
    g_name2classrefidx[scope_name] = sz;
    g_classrefs.push_back(TClassRef(scope_name.c_str()));
    return (TCppScope_t)sz;
}

bool Cppyy::IsTemplate(const std::string& template_name)
{
    return (bool)gInterpreter->CheckClassTemplate(template_name.c_str());
}

Cppyy::TCppType_t Cppyy::GetActualClass(TCppType_t klass, TCppObject_t obj)
{
    TClassRef& cr = type_from_handle(klass);
    TClass* clActual = cr->GetActualClass((void*)obj);
    if (clActual && clActual != cr.GetClass()) {
    // TODO: lookup through name should not be needed
        return (TCppType_t)GetScope(clActual->GetName());
    }
    return klass;
}

size_t Cppyy::SizeOf(TCppType_t klass)
{
    TClassRef& cr = type_from_handle(klass);
    if (cr.GetClass()) return (size_t)cr->Size();
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
    TDataType* dt = gROOT->GetType(TClassEdit::CleanType(type_name.c_str(), 1).c_str());
    if (dt) return dt->GetType() != kOther_t;
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
    return (TCppObject_t)malloc(cr->Size());
}

void Cppyy::Deallocate(TCppType_t /* type */, TCppObject_t instance)
{
    ::operator delete(instance);
}

Cppyy::TCppObject_t Cppyy::Construct(TCppType_t type)
{
    TClassRef& cr = type_from_handle(type);
    return (TCppObject_t)cr->New();
}

void Cppyy::Destruct(TCppType_t type, TCppObject_t instance)
{
    TClassRef& cr = type_from_handle(type);
    cr->Destructor((void*)instance);
}


// method/function dispatching -----------------------------------------------
static inline CallFunc_t* GetCallFunc(Cppyy::TCppMethod_t method)
{
// TODO: method should be a callfunc, so that no mapping would be needed.
    CallWrapper* wrap = (CallWrapper*)method;
    if (wrap->fWrapper) return wrap->fWrapper;

    TFunction* func = wrap->fMetaFunction;

    CallFunc_t* callf = gInterpreter->CallFunc_Factory();
    MethodInfo_t* meth = gInterpreter->MethodInfo_Factory(func->GetDeclId());
    gInterpreter->CallFunc_SetFunc(callf, meth);
    gInterpreter->MethodInfo_Delete(meth);

    if (!(callf && gInterpreter->CallFunc_IsValid(callf))) {
    // TODO: propagate this error to caller w/o use of Python C-API
    /*
        PyErr_Format(PyExc_RuntimeError, "could not resolve %s::%s(%s)",
            const_cast<TClassRef&>(klass).GetClassName(),
            func ? func->GetName() : const_cast<TClassRef&>(klass).GetClassName(),
            callString.c_str()); */
        std::cerr << "TODO: report unresolved function error to Python\n";
        if (callf) gInterpreter->CallFunc_Delete(callf);
        return nullptr;
    }

    wrap->fWrapper = callf;
    return callf;
}

static inline
bool copy_args(void* args_, void** vargs)
{
    bool runRelease = false;
    std::vector<Parameter>& args = *(std::vector<Parameter>*)args_;
    for (std::vector<Parameter>::size_type i = 0; i < args.size(); ++i) {
        switch (args[i].fTypeCode) {
        case 'b':       /* bool */
            vargs[i] = (void*)&args[i].fValue.fBool;
            break;
        case 'h':       /* short */
            vargs[i] = (void*)&args[i].fValue.fShort;
            break;
        case 'H':       /* unsigned short */
            vargs[i] = (void*)&args[i].fValue.fUShort;
           break;
        case 'i':       /* int */
            vargs[i] = (void*)&args[i].fValue.fInt;
           break;
        case 'I':       /* unsigned int */
            vargs[i] = (void*)&args[i].fValue.fUInt;
            break;
        case 'l':       /* long */
            vargs[i] = (void*)&args[i].fValue.fLong;
            break;
        case 'L':       /* unsigned long */
            vargs[i] = (void*)&args[i].fValue.fULong;
            break;
        case 'q':       /* long long */
            vargs[i] = (void*)&args[i].fValue.fLongLong;
            break;
        case 'Q':       /* unsigned long long */
            vargs[i] = (void*)&args[i].fValue.fULongLong;
            break;
        case 'f':       /* float */
            vargs[i] = (void*)&args[i].fValue.fFloat;
            break;
        case 'd':       /* double */
            vargs[i] = (void*)&args[i].fValue.fDouble;
            break;
        case 'g':       /* long double */
            vargs[i] = (void*)&args[i].fValue.fLongDouble;
            break;
        case 'a':
        case 'o':
        case 'p':       /* void* */
            vargs[i] = (void*)&args[i].fValue.fVoidp;
            break;
        case 'X':       /* (void*)type& with free */
            runRelease = true;
        case 'V':       /* (void*)type& */
            vargs[i] = args[i].fValue.fVoidp;
            break;
        case 'r':       /* const type& */
            vargs[i] = args[i].fRef;
            break;
        default:
            std::cerr << "unknown type code: " << args[i].fTypeCode << std::endl;
            break;
        }
    }
    return runRelease;
}

static inline
void release_args(const std::vector<Parameter>& args) {
    for (std::vector<Parameter>::size_type i = 0; i < args.size(); ++i) {
        if (args[i].fTypeCode == 'X')
            free(args[i].fValue.fVoidp);
    }
}

static bool FastCall(Cppyy::TCppMethod_t method, void* args_, void* self, void* result)
{
    const std::vector<Parameter>& args = *(std::vector<Parameter>*)args_;

    CallFunc_t* callf = GetCallFunc(method);
    if (!callf)
        return false;

    TInterpreter::CallFuncIFacePtr_t faceptr = gCling->CallFunc_IFacePtr(callf);
    if (faceptr.fKind == TInterpreter::CallFuncIFacePtr_t::kGeneric) {
        bool runRelease = false;
        if (args.size() <= SMALL_ARGS_N) {
            void* smallbuf[SMALL_ARGS_N];
            runRelease = copy_args(args_, smallbuf);
            faceptr.fGeneric(self, args.size(), smallbuf, result);
        } else {
            std::vector<void*> buf(args.size());
            runRelease = copy_args(args_, buf.data());
            faceptr.fGeneric(self, args.size(), buf.data(), result);
        }
        if (runRelease) release_args(args);
        return true;
    }

    if (faceptr.fKind == TInterpreter::CallFuncIFacePtr_t::kCtor) {
        bool runRelease = false;
        if (args.size() <= SMALL_ARGS_N) {
            void* smallbuf[SMALL_ARGS_N];
            runRelease = copy_args(args_, (void**)smallbuf);
            faceptr.fCtor((void**)smallbuf, result, args.size());
        } else {
            std::vector<void*> buf(args.size());
            runRelease = copy_args(args_, buf.data());
            faceptr.fCtor(buf.data(), result, args.size());
        }
        if (runRelease) release_args(args);
        return true;
    }

    if (faceptr.fKind == TInterpreter::CallFuncIFacePtr_t::kDtor) {
        std::cerr << " DESTRUCTOR NOT IMPLEMENTED YET! " << std::endl;
        return false;
    }

    return false;
}

template< typename T >
static inline
T CallT(Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, void* args)
{
    T t{};
    if (FastCall(method, args, (void*)self, &t))
        return t;
    return (T)-1;
}

#define CPPYY_IMP_CALL(typecode, rtype)                                      \
rtype Cppyy::Call##typecode(TCppMethod_t method, TCppObject_t self, void* args)\
{                                                                            \
    return CallT<rtype>(method, self, args);                                 \
}

void Cppyy::CallV(TCppMethod_t method, TCppObject_t self, void* args)
{
   if (!FastCall(method, args, (void*)self, nullptr))
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

void* Cppyy::CallR(TCppMethod_t method, TCppObject_t self, void* args)
{
    void* r = nullptr;
    if (FastCall(method, args, (void*)self, &r))
        return r;
    return nullptr;
}

char* Cppyy::CallS(
    TCppMethod_t method, TCppObject_t self, void* args, size_t* length)
{
    char* cstr = nullptr;
    TClassRef cr("std::string");
    std::string* cppresult = (std::string*)malloc(sizeof(std::string));
    if (FastCall(method, args, self, (void*)cppresult)) {
        cstr = cppstring_to_cstring(*cppresult);
        *length = cppresult->size();
        cppresult->std::string::~basic_string();
    } else
        *length = 0;
    free((void*)cppresult); 
    return cstr;
}

Cppyy::TCppObject_t Cppyy::CallConstructor(
    TCppMethod_t method, TCppType_t /* klass */, void* args)
{
    void* obj = nullptr;
    if (FastCall(method, args, nullptr, &obj))
        return (TCppObject_t)obj;
    return (TCppObject_t)0;
}

void Cppyy::CallDestructor(TCppType_t type, TCppObject_t self)
{
    TClassRef& cr = type_from_handle(type);
    cr->Destructor((void*)self, true);
}

Cppyy::TCppObject_t Cppyy::CallO(TCppMethod_t method,
    TCppObject_t self, void* args, TCppType_t result_type)
{
    TClassRef& cr = type_from_handle(result_type);
    void* obj = ::operator new(cr->Size());
    if (FastCall(method, args, self, obj))
        return (TCppObject_t)obj;
    return (TCppObject_t)0;
}

Cppyy::TCppFuncAddr_t Cppyy::GetFunctionAddress(TCppScope_t scope, TCppIndex_t idx)
{
    if (!gEnableFastPath) return (TCppFuncAddr_t)nullptr;
    TFunction* f = type_get_method(scope, idx);
    return (TCppFuncAddr_t)dlsym(RTLD_DEFAULT, f->GetMangledName());
}

Cppyy::TCppFuncAddr_t Cppyy::GetFunctionAddress(TCppMethod_t method)
{
    if (!gEnableFastPath) return (TCppFuncAddr_t)nullptr;
    TFunction* f = m2f(method);
    return (TCppFuncAddr_t)dlsym(RTLD_DEFAULT, f->GetMangledName());
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
    return gInterpreter->ClassInfo_IsEnum(type_name.c_str());
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
        if (nm && nm[0] != '_' && !(obj->Property() & (filter)))              \
            cppnames.insert(nm);                                              \
    }}

static inline
void cond_add(Cppyy::TCppScope_t scope, const std::string& ns_scope,
    std::set<std::string>& cppnames, const char* name)
{
    if (!name || name[0] == '_' || strstr(name, ".h") != 0 || strncmp(name, "operator", 8) == 0)
        return;

    if (scope == GLOBAL_HANDLE) {
        if (!is_missclassified_stl(name))
            cppnames.insert(outer_no_template(name));
    } else if (scope == STD_HANDLE) {
        if (strncmp(name, "std::", 5) == 0) name += 5;
        else if (!is_missclassified_stl(name)) return;
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
        while ((ev = (TEnvRec*)itr.Next()))
            cond_add(scope, ns_scope, cppnames, ev->GetName());
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
            if (!(dt->Property() & kIsFundamental))
                cond_add(scope, ns_scope, cppnames, dt->GetName());
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
            if (nm && nm[0] != '_' && strstr(nm, "<") == 0 && strncmp(nm, "operator", 8) != 0)
                cppnames.insert(nm);
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
    }

// add enums values only for user classes/namespaces
    if (scope != GLOBAL_HANDLE && scope != STD_HANDLE) {
        coll = cr->GetListOfEnums();
        FILL_COLL(TEnum, kIsPrivate | kIsProtected)
    }
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
    TClassRef& cr = type_from_handle(klass);
    if (cr.GetClass()) {
        std::string name = cr->GetName();
        if (is_missclassified_stl(name))
            return std::string("std::")+cr->GetName();
        return cr->GetName();
    }
    return "";
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

std::string Cppyy::GetBaseName(TCppType_t klass, TCppIndex_t ibase)
{
    TClassRef& cr = type_from_handle(klass);
    return ((TBaseClass*)cr->GetListOfBases()->At(ibase))->GetName();
}

bool Cppyy::IsSubtype(TCppType_t derived, TCppType_t base)
{
    if (derived == base)
        return true;
    TClassRef& derived_type = type_from_handle(derived);
    TClassRef& base_type = type_from_handle(base);
    return derived_type->GetBaseClass(base_type) != 0;
}

bool Cppyy::GetSmartPtrInfo(
    const std::string& tname, TCppType_t& raw, TCppMethod_t& deref)
{
    const std::string& rn = ResolveName(tname);
    if (gSmartPtrTypes.find(rn.substr(0, rn.find("<"))) != gSmartPtrTypes.end()) {
        TClassRef& cr = type_from_handle(GetScope(tname));
        if (cr.GetClass()) {
            gInterpreter->UpdateListOfMethods(cr.GetClass());
            TFunction* func = nullptr;
            TIter next(cr->GetListOfAllPublicMethods()); 
            while ((func = (TFunction*)next())) {
                if (strstr(func->GetName(), "operator->")) {
                    deref = (TCppMethod_t)new_CallWrapper(func);
                    raw = GetScope(TClassEdit::ShortType(
                        func->GetReturnTypeNormalizedName().c_str(), 1));
                    return deref && raw;
                }
            }
        }
    }

    return false;
}

void Cppyy::AddSmartPtrType(const std::string& type_name)
{
    gSmartPtrTypes.insert(ResolveName(type_name));
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
Cppyy::TCppIndex_t Cppyy::GetNumMethods(TCppScope_t scope)
{
    if (IsNamespace(scope))
        return (TCppIndex_t)0;     // enforce lazy

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass() && cr->GetListOfMethods(true)) {
        Cppyy::TCppIndex_t nMethods = (TCppIndex_t)cr->GetListOfMethods(false)->GetSize();
        if (nMethods == (TCppIndex_t)0) {
            std::string clName = GetScopedFinalName(scope);
            if (clName.find('<') != std::string::npos) {
            // chicken-and-egg problem: TClass does not know about methods until instantiation: force it
                if (TClass::GetClass(("std::"+clName).c_str())) // TODO: this doesn't work for templates
                    clName = "std::" + clName;
                std::ostringstream stmt;
                stmt << "template class " << clName << ";";
                gInterpreter->Declare(stmt.str().c_str());

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
                if (func->Property() & kIsPublic)
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

Cppyy::TCppMethod_t Cppyy::GetMethod(TCppScope_t scope, TCppIndex_t imeth)
{
    TFunction* func = type_get_method(scope, imeth);
    if (func)
        return (Cppyy::TCppMethod_t)new_CallWrapper(func);
    return (Cppyy::TCppMethod_t)nullptr;
}

std::string Cppyy::GetMethodName(TCppMethod_t method)
{
    if (method) {
        std::string name = m2f(method)->GetName();

        if (name.compare(0, 8, "operator") != 0)
        // strip template instantiation part, if any
            return name.substr(0, name.find('<'));
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
        return f->GetReturnTypeNormalizedName();
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

std::string Cppyy::GetMethodArgName(TCppMethod_t method, int iarg)
{
    if (method) {
        TFunction* f = m2f(method);
        TMethodArg* arg = (TMethodArg*)f->GetListOfMethodArgs()->At(iarg);
        return arg->GetName();
    }
    return "<unknown>";
}

std::string Cppyy::GetMethodArgType(TCppMethod_t method, int iarg)
{
    if (method) {
        TFunction* f = m2f(method);
        TMethodArg* arg = (TMethodArg*)f->GetListOfMethodArgs()->At(iarg);
        return arg->GetTypeNormalizedName();
    }
    return "<unknown>";
}

std::string Cppyy::GetMethodArgDefault(TCppMethod_t method, int iarg)
{
    if (method) {
        TFunction* f = m2f(method);
        TMethodArg* arg = (TMethodArg*)f->GetListOfMethodArgs()->At(iarg);
        const char* def = arg->GetDefault();
        if (def)
            return def;
    }

    return "";
}

std::string Cppyy::GetMethodSignature(TCppScope_t scope, TCppIndex_t imeth, bool show_formalargs)
{
    TFunction* f = type_get_method(scope, imeth);
    if (f) {
        std::ostringstream sig;
        sig << "(";
        int nArgs = f->GetNargs();
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

std::string Cppyy::GetMethodPrototype(TCppScope_t scope, TCppIndex_t imeth, bool show_formalargs)
{
    std::string scName = GetScopedFinalName(scope);
    TFunction* f = type_get_method(scope, imeth);
    if (f) {
        std::ostringstream sig;
        sig << f->GetReturnTypeName() << " "
            << scName << "::" << f->GetName();
        sig << GetMethodSignature(scope, imeth, show_formalargs);
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

bool Cppyy::ExistsMethodTemplate(TCppScope_t scope, const std::string& name)
{
    if (scope == (cppyy_scope_t)GLOBAL_HANDLE)
        return (bool)gROOT->GetFunctionTemplate(name.c_str());
    else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass())
            return (bool)cr->GetFunctionTemplate(name.c_str());
    }

// failure ...
    return false;
}

bool Cppyy::IsMethodTemplate(TCppScope_t scope, TCppIndex_t imeth)
{
    TFunction* f = type_get_method(scope, imeth);
    if (!f) return false;

    if (scope == (Cppyy::TCppType_t)GLOBAL_HANDLE) {
    // TODO: figure this one out ...
        return false;
    } else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass()) {
            return (bool)cr->GetFunctionTemplate(f->GetName());
        }
    }
    return false;
}

Cppyy::TCppIndex_t Cppyy::GetMethodNumTemplateArgs(
    TCppScope_t scope, TCppIndex_t imeth)
{
// this is dumb, but the fact that Cling can instantiate template
// methods on-the-fly means that there is some vast reworking TODO
// in interp_cppyy.py, so this is just to make the original tests
// pass that worked in the Reflex era ...
    const std::string name = GetMethodName(GetMethod(scope, imeth));
    return (TCppIndex_t)(std::count(name.begin(), name.end(), ',')+1);
}

std::string Cppyy::GetMethodTemplateArgName(
    TCppScope_t scope, TCppIndex_t imeth, TCppIndex_t /* iarg */)
{
// TODO: like above, given Cling's instantiation capability, this
// is just dumb ...
    TFunction* f = type_get_method(scope, imeth);
    std::string name = f->GetName();
    std::string::size_type pos = name.find('<');
// TODO: left as-is, this should loop over arguments, but what is here
// suffices to pass the Reflex-based tests (need more tests :))
    return cppstring_to_cstring(
        ResolveName(name.substr(pos+1, name.size()-pos-2)));
}

Cppyy::TCppMethod_t Cppyy::GetMethodTemplate(
    TCppScope_t scope, const std::string& name, const std::string& proto)
{
    TFunction* func = nullptr;
    if (scope == (cppyy_scope_t)GLOBAL_HANDLE) {
        func = gROOT->GetGlobalFunctionWithPrototype(name.c_str(), proto.c_str());
    } else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass())
            func = cr->GetMethodWithPrototype(name.c_str(), proto.c_str());
    }

    if (func)
        return (TCppMethod_t)new_CallWrapper(func);

// failure ...
    return (TCppMethod_t)nullptr;
}

Cppyy::TCppIndex_t Cppyy::GetGlobalOperator(
    TCppScope_t scope, TCppType_t lc, TCppType_t rc, const std::string& opname)
{
// Find a global operator function with a matching signature
    std::string proto = GetScopedFinalName(lc) + ", " + GetScopedFinalName(rc);
    if (scope == (cppyy_scope_t)GLOBAL_HANDLE) {
        TFunction* func = gROOT->GetGlobalFunctionWithPrototype(opname.c_str(), proto.c_str());
        if (func) return (TCppIndex_t)new_CallWrapper(func);
    } else {
        TClassRef& cr = type_from_handle(scope);
        if (cr.GetClass()) {
            TFunction* func = cr->GetMethodWithPrototype(opname.c_str(), proto.c_str());
            if (func) return (TCppIndex_t)cr->GetListOfMethods()->IndexOf(func);
        }
    }

// failure ...
    return (TCppIndex_t)-1;
}

// method properties ---------------------------------------------------------
bool Cppyy::IsPublicMethod(TCppMethod_t method)
{
    if (method) {
        TFunction* f = m2f(method);
        return f->Property() & kIsPublic;
    }
    return false;
}

bool Cppyy::IsConstructor(TCppMethod_t method)
{
    if (method) {
        TFunction* f = m2f(method);
        return f->ExtraProperty() & kIsConstructor;
    }
    return false;
}

bool Cppyy::IsDestructor(TCppMethod_t method)
{
    if (method) {
        TFunction* f = m2f(method);
        return f->ExtraProperty() & kIsDestructor;
    }
    return false;
}

bool Cppyy::IsStaticMethod(TCppMethod_t method)
{
    if (method) {
        TFunction* f = m2f(method);
        return f->Property() & kIsStatic;
    }
    return false;
}

// data member reflection information ----------------------------------------
Cppyy::TCppIndex_t Cppyy::GetNumDatamembers(TCppScope_t scope)
{
    if (IsNamespace(scope))
        return (TCppIndex_t)0;     // enforce lazy

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass() && cr->GetListOfDataMembers())
        return cr->GetListOfDataMembers()->GetSize();

    return (TCppIndex_t)0;         // unknown class?
}

std::string Cppyy::GetDatamemberName(TCppScope_t scope, TCppIndex_t idata)
{
    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(idata);
        return m->GetName();
    }
    assert(scope == GLOBAL_HANDLE);
    TGlobal* gbl = g_globalvars[idata];
    return gbl->GetName();
}

std::string Cppyy::GetDatamemberType(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gbl = g_globalvars[idata];
        std::string fullType = gbl->GetFullTypeName();
        if (!strcmp(gbl->GetName(), "gInterpreter"))
            return fullType;

        if (fullType[fullType.size()-1] == '*' && \
              fullType.compare(0, 4, "char") != 0)
            fullType.append("*");
        else if ((int)gbl->GetArrayDim() > 1)
            fullType.append("*");
        else if ((int)gbl->GetArrayDim() == 1) {
            std::ostringstream s;
            s << '[' << gbl->GetMaxIndex(0) << ']' << std::ends;
            fullType.append(s.str());
        }
        return fullType;
    }

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass())  {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(idata);
        std::string fullType = m->GetTrueTypeName();
        if ((int)m->GetArrayDim() > 1 || (!m->IsBasic() && m->IsaPointer()))
            fullType.append("*");
        else if ((int)m->GetArrayDim() == 1) {
            std::ostringstream s;
            s << '[' << m->GetMaxIndex(0) << ']' << std::ends;
            fullType.append(s.str());
        }
        return fullType;
    }

    return "<unknown>";
}

ptrdiff_t Cppyy::GetDatamemberOffset(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gbl = g_globalvars[idata];
        return (ptrdiff_t)gbl->GetAddress();
    }

    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(idata);
    // CLING WORKAROUND: the following causes templates to be instantiated first
    // in the proper scope, making the lookup succeed and preventing spurious
    // duplicate instantiations later.
        if ((m->Property() & kIsStatic) && strchr(cr->GetName(), '<'))
            gInterpreter->ProcessLine(((std::string)cr->GetName()+"::"+m->GetName()+";").c_str());
        return (ptrdiff_t)m->GetOffsetCint();    // yes, CINT (GetOffset() is both wrong
                                                 // and caches that wrong result!
    }

    return (ptrdiff_t)-1;
}

Cppyy::TCppIndex_t Cppyy::GetDatamemberIndex(TCppScope_t scope, const std::string& name)
{
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gb = (TGlobal*)gROOT->GetListOfGlobals(true)->FindObject(name.c_str());
        if (gb && gb->GetAddress()) {
            if (gb->GetAddress() == (void*)-1) {
            // name known, but variable not in loaded by Cling yet ... force it
            // TODO: figure out a less hackish way (problem is that the metaProcessor
            // is hidden in TCling)
                gInterpreter->ProcessLine((name+";").c_str());
            }
            if (gb->GetAddress() != (void*)-1) {
                if (strcmp(gb->GetFullTypeName(), "(lambda)") == 0) {
                // lambdas use a compiler internal closure type, so we wrap
                // them, then return the wrapper's type
                // TODO: this current leaks the std::function; also, if possible,
                //       should instantiate through TClass rather then ProcessLine
                    std::ostringstream s;
                    s << "auto __cppyy_internal_wrap_" << name << " = "
                        "new __cppyy_internal::FT<decltype(" << name << ")>::F"
                        "{" << name << "};";
                    gInterpreter->ProcessLine(s.str().c_str());
                    TGlobal* wrap = (TGlobal*)gROOT->GetListOfGlobals(true)->FindObject(
                        ("__cppyy_internal_wrap_"+name).c_str());
                    if (wrap && wrap->GetAddress()) gb = wrap;
                }

                g_globalvars.push_back(gb);
                return g_globalvars.size() - 1;
            }
        }

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


// data member properties ----------------------------------------------------
bool Cppyy::IsPublicData(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE)
        return true;
    TClassRef& cr = type_from_handle(scope);
    if (cr->Property() & kIsNamespace)
        return true;
    TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(idata);
    return m->Property() & kIsPublic;
}

bool Cppyy::IsStaticData(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE)
        return true;
    TClassRef& cr = type_from_handle(scope);
    if (cr->Property() & kIsNamespace)
        return true;
    TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(idata);
    return m->Property() & kIsStatic;
}

bool Cppyy::IsConstData(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gbl = g_globalvars[idata];
        return gbl->Property() & kIsConstant;
    }
    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(idata);
        return m->Property() & kIsConstant;
    }
    return false;
}

bool Cppyy::IsEnumData(TCppScope_t scope, TCppIndex_t idata)
{
    if (scope == GLOBAL_HANDLE) {
        TGlobal* gbl = g_globalvars[idata];
        return gbl->Property() & kIsEnum;
    }
    TClassRef& cr = type_from_handle(scope);
    if (cr.GetClass()) {
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(idata);
        return m->Property() & kIsEnum;
    }
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
        TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(idata);
        return m->GetMaxIndex(dimension);
    }
    return -1;
}


//- C-linkage wrappers -------------------------------------------------------
static inline
std::vector<Parameter> vsargs_to_parvec(void* args, int nargs)
{
    std::vector<Parameter> v;
    v.reserve(nargs);
    for (int i=0; i<nargs; ++i)
        v.push_back(((Parameter*)args)[i]);
    return v;
}

extern "C" {
/* name to opaque C++ scope representation -------------------------------- */
char* cppyy_resolve_name(const char* cppitem_name) {
    return cppstring_to_cstring(Cppyy::ResolveName(cppitem_name));
}

char* cppyy_resolve_enum(const char* enum_type) {
    return cppstring_to_cstring(Cppyy::ResolveEnum(enum_type));
}

cppyy_scope_t cppyy_get_scope(const char* scope_name) {
    return cppyy_scope_t(Cppyy::GetScope(scope_name));
}

cppyy_type_t cppyy_actual_class(cppyy_type_t klass, cppyy_object_t obj) {
    return cppyy_type_t(Cppyy::GetActualClass(klass, (void*)obj));
}

size_t cppyy_size_of_klass(cppyy_type_t klass) {
    return Cppyy::SizeOf(klass);
}

size_t cppyy_size_of_type(const char* type_name) {
    return Cppyy::SizeOf(type_name);
}


/* memory management ------------------------------------------------------ */
cppyy_object_t cppyy_allocate(cppyy_type_t type) {
    return cppyy_object_t(Cppyy::Allocate(type));
}

void cppyy_deallocate(cppyy_type_t type, cppyy_object_t self) {
    Cppyy::Deallocate(type, (void*)self);
}

cppyy_object_t cppyy_construct(cppyy_type_t type) {
    return (cppyy_object_t)Cppyy::Construct(type);
}

void cppyy_destruct(cppyy_type_t type, cppyy_object_t self) {
    Cppyy::Destruct(type, (void*)self);
}


/* method/function dispatching -------------------------------------------- */
/* Exception types:
    1: default (unknown exception)
    2: standard exception
*/
#define CPPYY_HANDLE_EXCEPTION                                               \
    catch (std::exception& e) {                                              \
        cppyy_exctype_t* etype = (cppyy_exctype_t*)((Parameter*)args+nargs); \
        *etype = (cppyy_exctype_t)2;                                         \
        *((char**)(etype+1)) = cppstring_to_cstring(e.what());               \
    }                                                                        \
    catch (...) {                                                            \
        cppyy_exctype_t* etype = (cppyy_exctype_t*)((Parameter*)args+nargs); \
        *etype = (cppyy_exctype_t)1;                                         \
        *((char**)(etype+1)) =                                               \
            cppstring_to_cstring("unhandled, unknown C++ exception");        \
    }

void cppyy_call_v(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        Cppyy::CallV(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
}

unsigned char cppyy_call_b(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return (unsigned char)Cppyy::CallB(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
    return (unsigned char)-1;
}

char cppyy_call_c(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return (char)Cppyy::CallC(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
    return (char)-1;
}

short cppyy_call_h(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return (short)Cppyy::CallH(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
    return (short)-1;
}

int cppyy_call_i(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return (int)Cppyy::CallI(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
    return (int)-1;
}

long cppyy_call_l(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return (long)Cppyy::CallL(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
    return (long)-1;
}

long long cppyy_call_ll(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return (long long)Cppyy::CallLL(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
    return (long long)-1;
}

float cppyy_call_f(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return (float)Cppyy::CallF(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
    return (float)-1;
}

double cppyy_call_d(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return (double)Cppyy::CallD(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
    return (double)-1;
}

long double cppyy_call_ld(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return (long double)Cppyy::CallLD(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
    return (long double)-1;
}

void* cppyy_call_r(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return (void*)Cppyy::CallR(method, (void*)self, &parvec);
    } CPPYY_HANDLE_EXCEPTION
    return (void*)nullptr;
}

char* cppyy_call_s(
        cppyy_method_t method, cppyy_object_t self, int nargs, void* args, size_t* lsz) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return Cppyy::CallS(method, (void*)self, &parvec, lsz);
    } CPPYY_HANDLE_EXCEPTION
    return (char*)nullptr;
}

cppyy_object_t cppyy_constructor(
        cppyy_method_t method, cppyy_type_t klass, int nargs, void* args) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return cppyy_object_t(Cppyy::CallConstructor(method, klass, &parvec));
    } CPPYY_HANDLE_EXCEPTION
    return (cppyy_object_t)0;
}

void cppyy_destructor(cppyy_type_t klass, cppyy_object_t self) {
    Cppyy::CallDestructor(klass, self);
}

cppyy_object_t cppyy_call_o(cppyy_method_t method, cppyy_object_t self,
        int nargs, void* args, cppyy_type_t result_type) {
    try {
        std::vector<Parameter> parvec = vsargs_to_parvec(args, nargs);
        return cppyy_object_t(Cppyy::CallO(method, (void*)self, &parvec, result_type));
    } CPPYY_HANDLE_EXCEPTION
    return (cppyy_object_t)0;
}

cppyy_funcaddr_t cppyy_function_address_from_index(cppyy_scope_t scope, cppyy_index_t idx) {
    return cppyy_funcaddr_t(Cppyy::GetFunctionAddress(scope, idx));
}

cppyy_funcaddr_t cppyy_function_address_from_method(cppyy_method_t method) {
    return cppyy_funcaddr_t(Cppyy::GetFunctionAddress(method));
}


/* handling of function argument buffer ----------------------------------- */
void* cppyy_allocate_function_args(int nargs) {
// for calls through C interface, require extra space for reporting exceptions
    return malloc(nargs*sizeof(Parameter)+sizeof(cppyy_exctype_t)+sizeof(char**));
}

void cppyy_deallocate_function_args(void* args) {
    free(args);
}

size_t cppyy_function_arg_sizeof() {
    return (size_t)Cppyy::GetFunctionArgSizeof();
}

size_t cppyy_function_arg_typeoffset() {
    return (size_t)Cppyy::GetFunctionArgTypeoffset();
}


/* scope reflection information ------------------------------------------- */
int cppyy_is_namespace(cppyy_scope_t scope) {
    return (int)Cppyy::IsNamespace(scope);
}

int cppyy_is_template(const char* template_name) {
    return (int)Cppyy::IsTemplate(template_name);
}

int cppyy_is_abstract(cppyy_type_t type) {
    return (int)Cppyy::IsAbstract(type);
}

int cppyy_is_enum(const char* type_name) {
    return (int)Cppyy::IsEnum(type_name);
}

const char** cppyy_get_all_cpp_names(cppyy_scope_t scope, size_t* count) {
    std::set<std::string> cppnames;
    Cppyy::GetAllCppNames(scope, cppnames);
    const char** c_cppnames = (const char**)malloc(cppnames.size()*sizeof(const char*));
    int i = 0;
    for (const auto& name : cppnames) {
        c_cppnames[i] = cppstring_to_cstring(name);
        ++i;
    }
    *count = cppnames.size();
    return c_cppnames;
}


/* class reflection information ------------------------------------------- */
char* cppyy_final_name(cppyy_type_t type) {
    return cppstring_to_cstring(Cppyy::GetFinalName(type));
}

char* cppyy_scoped_final_name(cppyy_type_t type) {
    return cppstring_to_cstring(Cppyy::GetScopedFinalName(type));
}

int cppyy_has_complex_hierarchy(cppyy_type_t type) {
    return (int)Cppyy::HasComplexHierarchy(type);
}

int cppyy_num_bases(cppyy_type_t type) {
    return (int)Cppyy::GetNumBases(type);
}

char* cppyy_base_name(cppyy_type_t type, int base_index) {
    return cppstring_to_cstring(Cppyy::GetBaseName (type, base_index));
}

int cppyy_is_subtype(cppyy_type_t derived, cppyy_type_t base) {
    return (int)Cppyy::IsSubtype(derived, base);
}

int cppyy_smartptr_info(const char* name, cppyy_type_t* raw, cppyy_method_t* deref) {
    Cppyy::TCppScope_t r2 = *raw;
    Cppyy::TCppMethod_t d2 = *deref;
    int result = (int)Cppyy::GetSmartPtrInfo(name, r2, d2);
    *raw = r2; *deref = d2;
    return result;
}

void cppyy_add_smartptr_type(const char* type_name) {
    Cppyy::AddSmartPtrType(type_name);
}


/* calculate offsets between declared and actual type, up-cast: direction > 0; down-cast: direction < 0 */
ptrdiff_t cppyy_base_offset(cppyy_type_t derived, cppyy_type_t base, cppyy_object_t address, int direction) {
    return (ptrdiff_t)Cppyy::GetBaseOffset(derived, base, (void*)address, direction, 0);
}


/* method/function reflection information --------------------------------- */
int cppyy_num_methods(cppyy_scope_t scope) {
    return (int)Cppyy::GetNumMethods(scope);
}

cppyy_index_t* cppyy_method_indices_from_name(cppyy_scope_t scope, const char* name)
{
    std::vector<cppyy_index_t> result = Cppyy::GetMethodIndicesFromName(scope, name);

    if (result.empty())
        return (cppyy_index_t*)nullptr;

    cppyy_index_t* llresult = (cppyy_index_t*)malloc(sizeof(cppyy_index_t)*(result.size()+1));
    for (int i = 0; i < (int)result.size(); ++i) llresult[i] = result[i];
    llresult[result.size()] = -1;
    return llresult;
}

char* cppyy_method_name(cppyy_scope_t scope, cppyy_index_t idx) {
    CallWrapper wrap{type_get_method(scope, idx)};
    return cppstring_to_cstring(Cppyy::GetMethodName((Cppyy::TCppMethod_t)&wrap));
}

char* cppyy_method_mangled_name(cppyy_scope_t scope, cppyy_index_t idx) {
    CallWrapper wrap{type_get_method(scope, idx)};
    return cppstring_to_cstring(Cppyy::GetMethodMangledName((Cppyy::TCppMethod_t)&wrap));
}

char* cppyy_method_result_type(cppyy_scope_t scope, cppyy_index_t idx) {
    CallWrapper wrap{type_get_method(scope, idx)};
    return cppstring_to_cstring(Cppyy::GetMethodResultType((Cppyy::TCppMethod_t)&wrap));
}

int cppyy_method_num_args(cppyy_scope_t scope, cppyy_index_t idx) {
    CallWrapper wrap{type_get_method(scope, idx)};
    return (int)Cppyy::GetMethodNumArgs((Cppyy::TCppMethod_t)&wrap);
}

int cppyy_method_req_args(cppyy_scope_t scope, cppyy_index_t idx) {
    CallWrapper wrap{type_get_method(scope, idx)};
    return (int)Cppyy::GetMethodReqArgs((Cppyy::TCppMethod_t)&wrap);
}

char* cppyy_method_arg_type(cppyy_scope_t scope, cppyy_index_t idx, int arg_index) {
    CallWrapper wrap{type_get_method(scope, idx)};
    return cppstring_to_cstring(Cppyy::GetMethodArgType((Cppyy::TCppMethod_t)&wrap, arg_index));
}

char* cppyy_method_arg_default(cppyy_scope_t scope, cppyy_index_t idx, int arg_index) {
    CallWrapper wrap{type_get_method(scope, idx)};
    return cppstring_to_cstring(Cppyy::GetMethodArgDefault((Cppyy::TCppMethod_t)&wrap, arg_index));
}

char* cppyy_method_signature(cppyy_scope_t scope, cppyy_index_t idx, int show_formalargs) {
    return cppstring_to_cstring(Cppyy::GetMethodSignature(scope, idx, (bool)show_formalargs));
}

char* cppyy_method_prototype(cppyy_scope_t scope, cppyy_index_t idx, int show_formalargs) {
    return cppstring_to_cstring(Cppyy::GetMethodPrototype(scope, idx, (bool)show_formalargs));
}

int cppyy_is_const_method(cppyy_method_t method) {
    return (int)Cppyy::IsConstMethod(method);
}

int cppyy_exists_method_template(cppyy_scope_t scope, const char* name) {
    return (int)Cppyy::ExistsMethodTemplate(scope, name);
}

int cppyy_method_is_template(cppyy_scope_t scope, cppyy_index_t idx) {
    return (int)Cppyy::IsMethodTemplate(scope, idx);
}

int cppyy_method_num_template_args(cppyy_scope_t scope, cppyy_index_t idx) {
    return (int)Cppyy::GetMethodNumTemplateArgs(scope, idx);
}

char* cppyy_method_template_arg_name(cppyy_scope_t scope, cppyy_index_t idx, cppyy_index_t iarg) {
    return cppstring_to_cstring(Cppyy::GetMethodTemplateArgName(scope, idx, iarg));
}

cppyy_method_t cppyy_get_method(cppyy_scope_t scope, cppyy_index_t idx) {
    return cppyy_method_t(Cppyy::GetMethod(scope, idx));
}

cppyy_index_t cppyy_get_global_operator(cppyy_scope_t scope, cppyy_scope_t lc, cppyy_scope_t rc, const char* op) {
    return cppyy_index_t(Cppyy::GetGlobalOperator(scope, lc, rc, op));
}


/* method properties ------------------------------------------------------ */
int cppyy_is_publicmethod(cppyy_type_t type, cppyy_index_t idx) {
    CallWrapper wrap{type_get_method(type, idx)};
    return (int)Cppyy::IsPublicMethod((Cppyy::TCppMethod_t)&wrap);
}

int cppyy_is_constructor(cppyy_type_t type, cppyy_index_t idx) {
    CallWrapper wrap{type_get_method(type, idx)};
    return (int)Cppyy::IsConstructor((Cppyy::TCppMethod_t)&wrap);
}

int cppyy_is_destructor(cppyy_type_t type, cppyy_index_t idx) {
    CallWrapper wrap{type_get_method(type, idx)};
    return (int)Cppyy::IsDestructor((Cppyy::TCppMethod_t)&wrap);
}

int cppyy_is_staticmethod(cppyy_type_t type, cppyy_index_t idx) {
    CallWrapper wrap{type_get_method(type, idx)};
    return (int)Cppyy::IsStaticMethod((Cppyy::TCppMethod_t)&wrap);
}


/* data member reflection information ------------------------------------- */
int cppyy_num_datamembers(cppyy_scope_t scope) {
    return (int)Cppyy::GetNumDatamembers(scope);
}

char* cppyy_datamember_name(cppyy_scope_t scope, int datamember_index) {
    return cppstring_to_cstring(Cppyy::GetDatamemberName(scope, datamember_index));
}

char* cppyy_datamember_type(cppyy_scope_t scope, int datamember_index) {
    return cppstring_to_cstring(Cppyy::GetDatamemberType(scope, datamember_index));
}

ptrdiff_t cppyy_datamember_offset(cppyy_scope_t scope, int datamember_index) {
    return ptrdiff_t(Cppyy::GetDatamemberOffset(scope, datamember_index));
}

int cppyy_datamember_index(cppyy_scope_t scope, const char* name) {
    return (int)Cppyy::GetDatamemberIndex(scope, name);
}



/* data member properties ------------------------------------------------- */
int cppyy_is_publicdata(cppyy_type_t type, cppyy_index_t datamember_index) {
    return (int)Cppyy::IsPublicData(type, datamember_index);
}

int cppyy_is_staticdata(cppyy_type_t type, cppyy_index_t datamember_index) {
    return (int)Cppyy::IsStaticData(type, datamember_index);
}

int cppyy_is_const_data(cppyy_scope_t scope, cppyy_index_t idata) {
    return (int)Cppyy::IsConstData(scope, idata);
}

int cppyy_is_enum_data(cppyy_scope_t scope, cppyy_index_t idata) {
    return (int)Cppyy::IsEnumData(scope, idata);
}

int cppyy_get_dimension_size(cppyy_scope_t scope, cppyy_index_t idata, int dimension) {
    return Cppyy::GetDimensionSize(scope, idata, dimension);
}


/* misc helpers ----------------------------------------------------------- */
RPY_EXTERN
void* cppyy_load_dictionary(const char* lib_name) {
    int result = gSystem->Load(lib_name);
    return (void*)(result == 0 /* success */ || result == 1 /* already loaded */);
}

#if defined(_MSC_VER)
long long cppyy_strtoll(const char* str) {
    return _strtoi64(str, NULL, 0);
}

extern "C" {
unsigned long long cppyy_strtoull(const char* str) {
    return _strtoui64(str, NULL, 0);
}
}
#else
long long cppyy_strtoll(const char* str) {
    return strtoll(str, NULL, 0);
}

extern "C" {
unsigned long long cppyy_strtoull(const char* str) {
    return strtoull(str, NULL, 0);
}
}
#endif

void cppyy_free(void* ptr) {
    free(ptr);
}

cppyy_object_t cppyy_charp2stdstring(const char* str, size_t sz) {
    return (cppyy_object_t)new std::string(str, sz);
}

const char* cppyy_stdstring2charp(cppyy_object_t ptr, size_t* lsz) {
    *lsz = ((std::string*)ptr)->size();
    return ((std::string*)ptr)->data();
}

cppyy_object_t cppyy_stdstring2stdstring(cppyy_object_t ptr) {
    return (cppyy_object_t)new std::string(*(std::string*)ptr);
}

const char* cppyy_stdvector_valuetype(const char* clname)
{
    const char* result = nullptr;
    std::string name = clname;
    TypedefInfo_t* ti = gInterpreter->TypedefInfo_Factory((name+"::value_type").c_str());
    if (gInterpreter->TypedefInfo_IsValid(ti))
        result = cppstring_to_cstring(gInterpreter->TypedefInfo_TrueName(ti));
    gInterpreter->TypedefInfo_Delete(ti);
    return result;
}

size_t cppyy_stdvector_valuesize(const char* clname)
{
    size_t result = 0;
    std::string name = clname;
    TypedefInfo_t* ti = gInterpreter->TypedefInfo_Factory((name+"::value_type").c_str());
    if (gInterpreter->TypedefInfo_IsValid(ti))
       result = (size_t)gInterpreter->TypedefInfo_Size(ti);
    gInterpreter->TypedefInfo_Delete(ti);
    return result;
}
   
} // end C-linkage wrappers
