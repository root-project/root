#ifndef WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
// silence warnings about getenv, strncpy, etc.
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

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

#ifndef WIN32
#include <dlfcn.h>
#endif

// Standard
#include <assert.h>
#include <algorithm>     // for std::count, std::remove
#include <stdexcept>
#include <map>
#include <new>
#include <regex>
#include <set>
#include <sstream>
#include <signal.h>
#include <stdlib.h>      // for getenv
#include <string.h>
#include <typeinfo>
#include <iostream>
#include <vector>


// temp
#include <iostream>
// --temp

// data for life time management ---------------------------------------------
// typedef std::vector<TClassRef> ClassRefs_t;
// static ClassRefs_t g_classrefs(1);
// static const ClassRefs_t::size_type GLOBAL_HANDLE = 1;
// static const ClassRefs_t::size_type STD_HANDLE = GLOBAL_HANDLE + 1;

// typedef std::map<std::string, ClassRefs_t::size_type> Name2ClassRefIndex_t;
// static Name2ClassRefIndex_t g_name2classrefidx;

// namespace {

// static inline
// Cppyy::TCppType_t find_memoized(const std::string& name)
// {
//     auto icr = g_name2classrefidx.find(name);
//     if (icr != g_name2classrefidx.end())
//         return (Cppyy::TCppType_t)icr->second;
//     return (Cppyy::TCppType_t)0;
// }
//
// } // namespace
//
// static inline
// CallWrapper* new_CallWrapper(CppyyLegacy::TFunction* f)
// {
//     CallWrapper* wrap = new CallWrapper(f);
//     gWrapperHolder.push_back(wrap);
//     return wrap;
// }
//

// typedef std::vector<TGlobal*> GlobalVars_t;
// typedef std::map<TGlobal*, GlobalVars_t::size_type> GlobalVarsIndices_t;

// static GlobalVars_t g_globalvars;
// static GlobalVarsIndices_t g_globalidx;


// builtin types
static std::set<std::string> g_builtins =
    {"bool", "char", "signed char", "unsigned char", "wchar_t", "short", "unsigned short",
     "int", "unsigned int", "long", "unsigned long", "long long", "unsigned long long",
     "float", "double", "long double", "void"};

// to filter out ROOT names
static std::set<std::string> gInitialNames;
static std::set<std::string> gRootSOs;

// configuration
static bool gEnableFastPath = true;


// global initialization -----------------------------------------------------
namespace {

const int kMAXSIGNALS = 16;

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

// static void inline do_trace(int sig) {
//     std::cerr << " *** Break *** " << (sig < kMAXSIGNALS ? gSignalMap[sig].fSigName : "") << std::endl;
//     gSystem->StackTrace();
// }

// class TExceptionHandlerImp : public TExceptionHandler {
// public:
//     virtual void HandleException(Int_t sig) {
//         if (TROOT::Initialized()) {
//             if (gException) {
//                 gInterpreter->RewindDictionary();
//                 gInterpreter->ClearFileBusy();
//             }
//
//             if (!getenv("CPPYY_CRASH_QUIET"))
//                 do_trace(sig);
//
//         // jump back, if catch point set
//             Throw(sig);
//         }
//
//         do_trace(sig);
//         gSystem->Exit(128 + sig);
//     }
// };

static inline
void push_tokens_from_string(char *s, std::vector <const char*> &tokens) {
    char *token = strtok(s, " ");

    while (token) {
        tokens.push_back(token);
        token = strtok(NULL, " ");
    }
}

static inline
bool is_integral(std::string& s)
{
    if (s == "false") { s = "0"; return true; }
    else if (s == "true") { s = "1"; return true; }
    return !s.empty() && std::find_if(s.begin(), 
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

class ApplicationStarter {
  Cpp::TInterp_t Interp;
public:
    ApplicationStarter() {


        (void)gROOT;
        char *libcling = gSystem->DynamicPathName("libCling");

        if (!libcling) {
            std::cerr << "[cppyy-backend] Failed to find libCling" << std::endl;
            return;
        }
        if (!Cpp::LoadDispatchAPI(libcling)) {
            std::cerr << "[cppyy-backend] Failed to load CppInterOp" << std::endl;
            return;
        }

        // Check if somebody already loaded CppInterOp and created an
        // interpreter for us.
        if (auto * existingInterp = Cpp::GetInterpreter()) {
            Interp = existingInterp;
        }
        else {
#ifdef __arm64__
#ifdef __APPLE__
            // If on apple silicon don't use -march=native
            std::vector<const char *> InterpArgs({"-std=c++17"});
#else
            std::vector<const char *> InterpArgs(
                {"-std=c++17", "-march=native"});
#endif
#else
            std::vector <const char *> InterpArgs({"-std=c++17", "-march=native"});
#endif
            char *InterpArgString = getenv("CPPINTEROP_EXTRA_INTERPRETER_ARGS");

            if (InterpArgString)
              push_tokens_from_string(InterpArgString, InterpArgs);

#ifdef __arm64__
#ifdef __APPLE__
            // If on apple silicon don't use -march=native
            Interp = Cpp::CreateInterpreter({"-std=c++17"}, /*GpuArgs=*/{});
#else
            Interp = Cpp::CreateInterpreter({"-std=c++17", "-march=native"},
                                            /*GpuArgs=*/{});
#endif
#else
            Interp = Cpp::CreateInterpreter({"-std=c++17", "-march=native"},
                                            /*GpuArgs=*/{});
#endif
        }

        // fill out the builtins
        std::set<std::string> bi{g_builtins};
        for (const auto& name : bi) {
            for (const char* a : {"*", "&", "*&", "[]", "*[]"})
                g_builtins.insert(name+a);
        }

    // disable fast path if requested
        if (getenv("CPPYY_DISABLE_FASTPATH")) gEnableFastPath = false;

    // set opt level (default to 2 if not given; Cling itself defaults to 0)
        int optLevel = 2;

        if (getenv("CPPYY_OPT_LEVEL")) optLevel = atoi(getenv("CPPYY_OPT_LEVEL"));

        if (optLevel != 0) {
            std::ostringstream s;
            s << "#pragma cling optimize " << optLevel;
            Cpp::Process(s.str().c_str());
        }

        // This would give us something like:
        // /home/vvassilev/workspace/builds/scratch/cling-build/builddir/lib/clang/13.0.0
        const char * ResourceDir = Cpp::GetResourceDir();
        std::string ClingSrc = std::string(ResourceDir) + "/../../../../cling-src";
        std::string ClingBuildDir = std::string(ResourceDir) + "/../../../";
        Cpp::AddIncludePath((ClingSrc + "/tools/cling/include").c_str());
        Cpp::AddIncludePath((ClingSrc + "/include").c_str());
        Cpp::AddIncludePath((ClingBuildDir + "/include").c_str());
        Cpp::AddIncludePath((std::string(CPPINTEROP_DIR) + "/include").c_str());
        Cpp::LoadLibrary("libstdc++", /* lookup= */ true);

        // load frequently used headers
        const char* code =
            "#include <algorithm>\n"
            "#include <numeric>\n"
            "#include <complex>\n"
            "#include <iostream>\n"
            "#include <string.h>\n" // for strcpy
            "#include <string>\n"
            //    "#include <DllImport.h>\n"     // defines R__EXTERN
            "#include <vector>\n"
            "#include <utility>\n"
            "#include <memory>\n"
            "#include <functional>\n" // for the dispatcher code to use
                                      // std::function
            "#include <map>\n"        // FIXME: Replace with modules
            "#include <sstream>\n"    // FIXME: Replace with modules
            "#include <array>\n"      // FIXME: Replace with modules
            "#include <list>\n"       // FIXME: Replace with modules
            "#include <deque>\n"      // FIXME: Replace with modules
            "#include <tuple>\n"      // FIXME: Replace with modules
            "#include <set>\n"        // FIXME: Replace with modules
            "#include <chrono>\n"     // FIXME: Replace with modules
            "#include <cmath>\n"      // FIXME: Replace with modules
            "#if __has_include(<optional>)\n"
            "#include <optional>\n"
            "#endif\n"
            "#include <CppInterOp/Dispatch.h>\n";
        Cpp::Process(code);

    // create helpers for comparing thingies
        Cpp::Declare("namespace __cppyy_internal { template<class C1, class C2>"
                     " bool is_equal(const C1& c1, const C2& c2) { return "
                     "(bool)(c1 == c2); } }",
                     /*silent=*/false);
        Cpp::Declare("namespace __cppyy_internal { template<class C1, class C2>"
                     " bool is_not_equal(const C1& c1, const C2& c2) { return "
                     "(bool)(c1 != c2); } }",
                     /*silent=*/false);

        // Define gCling when we run with clang-repl.
        // FIXME: We should get rid of all the uses of gCling as this seems to
        // break encapsulation.
        std::stringstream InterpPtrSS;
        InterpPtrSS << "#ifndef __CLING__\n"
                    << "namespace cling { namespace runtime {\n"
                    << "void* gCling=(void*)" << static_cast<void*>(Interp)
                    << ";\n }}\n"
                    << "#endif \n";
        Cpp::Process(InterpPtrSS.str().c_str());

    // helper for multiple inheritance
        Cpp::Declare("namespace __cppyy_internal { struct Sep; }",
                     /*silent=*/false);

        // std::string libInterOp = I->getDynamicLibraryManager()->lookupLibrary("libcling");
        // void *interopDL = dlopen(libInterOp.c_str(), RTLD_LAZY);
        // if (!interopDL) {
        //     std::cerr << "libInterop could not be opened!\n";
        //     exit(1);
        // }

    // start off with a reasonable size placeholder for wrappers
        // gWrapperHolder.reserve(1024);

    // create an exception handler to process signals
        // gExceptionHandler = new TExceptionHandlerImp{};
    }

    ~ApplicationStarter() {
      //Cpp::DeleteInterpreter(Interp);
        // for (auto wrap : gWrapperHolder)
        //     delete wrap;
        // delete gExceptionHandler; gExceptionHandler = nullptr;
    }
} _applicationStarter;

} // unnamed namespace


// // local helpers -------------------------------------------------------------
// static inline
// TClassRef& type_from_handle(Cppyy::TCppScope_t scope)
// {
//     assert((ClassRefs_t::size_type)scope < g_classrefs.size());
//     return g_classrefs[(ClassRefs_t::size_type)scope];
// }
//
// static inline
// TFunction* m2f(Cppyy::TCppMethod_t method) {
//     CallWrapper *wrap = (CallWrapper *)method;
//
//     if (!wrap->fTF) {
//         MethodInfo_t* mi = gInterpreter->MethodInfo_Factory(wrap->fDecl);
//         wrap->fTF = new TFunction(mi);
//     }
//     return (TFunction *) wrap->fTF;
// }
//
static inline
char* cppstring_to_cstring(const std::string& cppstr)
{
    char* cstr = (char*)malloc(cppstr.size()+1);
    memcpy(cstr, cppstr.c_str(), cppstr.size()+1);
    return cstr;
}
//
// static inline
// bool match_name(const std::string& tname, const std::string fname)
// {
// // either match exactly, or match the name as template
//     if (fname.rfind(tname, 0) == 0) {
//         if ((tname.size() == fname.size()) ||
//               (tname.size() < fname.size() && fname[tname.size()] == '<'))
//            return true;
//     }
//     return false;
// }
//
//
// // direct interpreter access -------------------------------------------------
// Returns false on failure and true on success
bool Cppyy::Compile(const std::string& code, bool silent)
{
    // Declare returns an enum which equals 0 on success
    return !Cpp::Declare(code.c_str(), silent);
}

std::string Cppyy::ToString(TCppType_t klass, TCppObject_t obj)
{
    if (klass && obj && !Cpp::IsNamespace((TCppScope_t)klass))
        return Cpp::ObjToString(Cpp::GetQualifiedCompleteName(klass).c_str(),
                                    (void*)obj);
    return "";
}

// // name to opaque C++ scope representation -----------------------------------
std::string Cppyy::ResolveName(const std::string& name) {
  if (!name.empty()) {
    if (Cppyy::TCppType_t type =
            Cppyy::GetType(name, /*enable_slow_lookup=*/true))
      return Cppyy::GetTypeAsString(Cppyy::ResolveType(type));
    return name;
  }
  return "";
}
// // Fully resolve the given name to the final type name.
//
// // try memoized type cache, in case seen before
//     TCppType_t klass = find_memoized(cppitem_name);
//     if (klass) return GetScopedFinalName(klass);
//
// // remove global scope '::' if present
//     std::string tclean = cppitem_name.compare(0, 2, "::") == 0 ?
//         cppitem_name.substr(2, std::string::npos) : cppitem_name;
//
// // classes (most common)
//     tclean = TClassEdit::CleanType(tclean.c_str());
//     if (tclean.empty() [> unknown, eg. an operator <]) return cppitem_name;
//
// // reduce [N] to []
//     if (tclean[tclean.size()-1] == ']')
//         tclean = tclean.substr(0, tclean.rfind('[')) + "[]";
//
// // remove __restrict and __restrict__
//     auto pos = tclean.rfind("__restrict");
//     if (pos != std::string::npos)
//         tclean = tclean.substr(0, pos);
//
//     if (tclean.compare(0, 9, "std::byte") == 0)
//         return tclean;
//
// // check data types list (accept only builtins as typedefs will
// // otherwise not be resolved)
//     if (IsBuiltin(tclean)) return tclean;
//
// // special case for enums
//     if (IsEnum(cppitem_name))
//         return ResolveEnum(cppitem_name);
//
// // special case for clang's builtin __type_pack_element (which does not resolve)
//     pos = cppitem_name.size() > 20 ? \
//               cppitem_name.rfind("__type_pack_element", 5) : std::string::npos;
//     if (pos != std::string::npos) {
//     // shape is "[std::]__type_pack_element<index,type1,type2,...,typeN>cpd": extract
//     // first the index, and from there the indexed type; finally, restore the
//     // qualifiers
//         const char* str = cppitem_name.c_str();
//         char* endptr = nullptr;
//         unsigned long index = strtoul(str+20+pos, &endptr, 0);
//
//         std::string tmplvars{endptr};
//         auto start = tmplvars.find(',') + 1;
//         auto end = tmplvars.find(',', start);
//         while (index != 0) {
//             start = end+1;
//             end = tmplvars.find(',', start);
//             if (end == std::string::npos) end = tmplvars.rfind('>');
//             --index;
//         }
//
//         std::string resolved = tmplvars.substr(start, end-start);
//         auto cpd = tmplvars.rfind('>');
//         if (cpd != std::string::npos && cpd+1 != tmplvars.size())
//             return resolved + tmplvars.substr(cpd+1, std::string::npos);
//         return resolved;
//     }
//
// // typedefs etc. (and a couple of hacks around TClassEdit-isms, fixing of which
// // in ResolveTypedef itself is a TODO ...)
//     tclean = TClassEdit::ResolveTypedef(tclean.c_str(), true);
//     pos = 0;
//     while ((pos = tclean.find("::::", pos)) != std::string::npos) {
//         tclean.replace(pos, 4, "::");
//         pos += 2;
//     }
//
//     if (tclean.compare(0, 6, "const ") != 0)
//         return TClassEdit::ShortType(tclean.c_str(), 2);
//     return "const " + TClassEdit::ShortType(tclean.c_str(), 2);
// }

Cppyy::TCppType_t Cppyy::ResolveEnumReferenceType(TCppType_t type) {
    if (Cpp::GetValueKind(type) != Cpp::ValueKind::LValue)
        return type;

    TCppType_t nonReferenceType = Cpp::GetNonReferenceType(type);
    if (Cpp::IsEnumType(nonReferenceType)) {
        TCppType_t underlying_type =  Cpp::GetIntegerTypeFromEnumType(nonReferenceType);
        return Cpp::GetReferencedType(underlying_type, /*rvalue=*/false);
    }
    return type;
}

Cppyy::TCppType_t Cppyy::ResolveEnumPointerType(TCppType_t type) {
    if (!Cpp::IsPointerType(type))
        return type;

    TCppType_t PointeeType = Cpp::GetPointeeType(type);
    if (Cpp::IsEnumType(PointeeType)) {
        TCppType_t underlying_type =  Cpp::GetIntegerTypeFromEnumType(PointeeType);
        return Cpp::GetPointerType(underlying_type);
    }
    return type;
}

Cppyy::TCppType_t int_like_type(Cppyy::TCppType_t type) {
    Cppyy::TCppType_t check_int_typedefs = type;
    if (Cpp::IsPointerType(check_int_typedefs))
        check_int_typedefs = Cpp::GetPointeeType(check_int_typedefs);
    if (Cpp::IsReferenceType(check_int_typedefs))
      check_int_typedefs =
          Cpp::GetReferencedType(check_int_typedefs, /*rvalue=*/false);

    if (Cpp::GetTypeAsString(check_int_typedefs) == "int8_t" || Cpp::GetTypeAsString(check_int_typedefs) == "uint8_t")
        return check_int_typedefs;
    return nullptr;
}

Cppyy::TCppType_t Cppyy::ResolveType(TCppType_t type) {
    if (!type) return type;

    TCppType_t check_int_typedefs = int_like_type(type);
    if (check_int_typedefs)
        return type;

    Cppyy::TCppType_t canonType = Cpp::GetCanonicalType(type);

    if (Cpp::IsEnumType(canonType)) {
        if (Cppyy::GetTypeAsString(type) != "std::byte")
            return Cpp::GetIntegerTypeFromEnumType(canonType);
    }
    if (Cpp::HasTypeQualifier(canonType, Cpp::QualKind::Restrict)) {
        return Cpp::RemoveTypeQualifier(canonType, Cpp::QualKind::Restrict);
    }

    return canonType;
}

Cppyy::TCppType_t Cppyy::GetRealType(TCppType_t type) {
    TCppType_t check_int_typedefs = int_like_type(type);
    if (check_int_typedefs)
        return check_int_typedefs;
    return Cpp::GetUnderlyingType(type);
}

Cppyy::TCppType_t Cppyy::GetPointerType(TCppType_t type) {
  return Cpp::GetPointerType(type);
}

Cppyy::TCppType_t Cppyy::GetReferencedType(TCppType_t type, bool rvalue) {
  return Cpp::GetReferencedType(type, rvalue);
}

bool Cppyy::IsRValueReferenceType(TCppType_t type) {
    return Cpp::GetValueKind(type) == Cpp::ValueKind::RValue;
}

bool Cppyy::IsLValueReferenceType(TCppType_t type) {
    return Cpp::GetValueKind(type) == Cpp::ValueKind::LValue;
}

bool Cppyy::IsClassType(TCppType_t type) {
    return Cpp::IsRecordType(type);
}

bool Cppyy::IsPointerType(TCppType_t type) {
    return Cpp::IsPointerType(type);
}

bool Cppyy::IsFunctionPointerType(TCppType_t type) {
    return Cpp::IsFunctionPointerType(type);
}

std::string trim(const std::string& line)
{
    if (line.empty()) return "";
    const char* WhiteSpace = " \t\v\r\n";
    std::size_t start = line.find_first_not_of(WhiteSpace);
    std::size_t end = line.find_last_not_of(WhiteSpace);
    return line.substr(start, end - start + 1);
}

// returns false of angular brackets dont match, else true
bool split_comma_saparated_types(const std::string& name,
                                 std::vector<std::string>& types) {
  std::string trimed_name = trim(name);
  size_t start_pos = 0;
  size_t end_pos = 0;
  size_t appended_count = 0;
  int matching_angular_brackets = 0;
  while (end_pos < trimed_name.size()) {
    switch (trimed_name[end_pos]) {
    case ',': {
      if (!matching_angular_brackets) {
        types.push_back(
            trim(trimed_name.substr(start_pos, end_pos - start_pos)));
        start_pos = end_pos + 1;
      }
      break;
    }
    case '<': {
      matching_angular_brackets++;
      break;
    }
    case '>': {
      if (matching_angular_brackets > 0) {
        types.push_back(
            trim(trimed_name.substr(start_pos, end_pos - start_pos + 1)));
        start_pos = end_pos + 1;
      } else if (matching_angular_brackets < 1) {
        types.clear();
        return false;
      }
      start_pos++;
      end_pos++;
      matching_angular_brackets--;
      break;
    }
    }
    end_pos++;
  }
  if (start_pos < trimed_name.size())
    types.push_back(trim(trimed_name.substr(start_pos, end_pos - start_pos)));
  return true;
}

Cpp::TCppScope_t GetEnumFromCompleteName(const std::string &name) {
  std::string delim = "::";
  size_t start = 0;
  size_t end = name.find(delim);
  Cpp::TCppScope_t curr_scope = 0;
  while (end != std::string::npos) {
    curr_scope = Cpp::GetNamed(name.substr(start, end - start), curr_scope);
    start = end + delim.length();
    end = name.find(delim, start);
  }
  return Cpp::GetNamed(name.substr(start, end), curr_scope);
}

// returns true if no new type was added.
bool Cppyy::AppendTypesSlow(const std::string& name,
                            std::vector<Cpp::TemplateArgInfo>& types, Cppyy::TCppScope_t parent) {

  // Add no new type if string is empty
  if (name.empty())
    return true;

  auto replace_all = [](std::string& str, const std::string& from, const std::string& to) {
      if(from.empty())
        return;
      size_t start_pos = 0;
      while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
  };

  std::string resolved_name = name;
  replace_all(resolved_name, "std::initializer_list<", "std::vector<"); // replace initializer_list with vector

  // We might have an entire expression such as int, double.
  static unsigned long long struct_count = 0;
  std::string code = "template<typename ...T> struct __Cppyy_AppendTypesSlow {};\n";
  if (!struct_count)
    Cpp::Declare(code.c_str(), /*silent=*/false); // initialize the trampoline

  std::string var = "__Cppyy_s" + std::to_string(struct_count++);
  // FIXME: We cannot use silent because it erases our error code from Declare!
  if (!Cpp::Declare(("__Cppyy_AppendTypesSlow<" + resolved_name + "> " + var +";\n").c_str(), /*silent=*/false)) {
    TCppType_t varN =
        Cpp::GetVariableType(Cpp::GetNamed(var.c_str(), /*parent=*/nullptr));
    TCppScope_t instance_class = Cpp::GetScopeFromType(varN);
    size_t oldSize = types.size();
    Cpp::GetClassTemplateInstantiationArgs(instance_class, types);
    return oldSize == types.size();
  }

  // We split each individual types based on , and resolve it
  // FIXME: see discussion on should we support template instantiation with string:
  //   https://github.com/compiler-research/cppyy-backend/pull/137#discussion_r2079357491
  //   We should consider eliminating the `split_comma_saparated_types` and `is_integral`
  //   string parsing.
  std::vector<std::string> individual_types;
  if (!split_comma_saparated_types(resolved_name, individual_types))
    return true;

  for (std::string& i : individual_types) {
    // Try going via Cppyy::GetType first.
    const char* integral_value = nullptr;
    Cppyy::TCppType_t type = nullptr;

    type = GetType(i, /*enable_slow_lookup=*/true);
    if (!type && parent && (Cpp::IsNamespace(parent) || Cpp::IsClass(parent))) {
        type = Cppyy::GetTypeFromScope(Cppyy::GetNamed(resolved_name, parent));
    }

    if (!type) {
      types.clear();
      return true;
    }

    if (is_integral(i))
        integral_value = strdup(i.c_str());
    if (Cpp::TCppScope_t scope = GetEnumFromCompleteName(i))
      if (Cpp::IsEnumConstant(scope))
        integral_value =
            strdup(std::to_string(Cpp::GetEnumConstantValue(scope)).c_str());
    types.emplace_back(type, integral_value);
  }
  return false;
}

Cppyy::TCppType_t Cppyy::GetType(const std::string &name, bool enable_slow_lookup /* = false */) {
    static unsigned long long var_count = 0;

    if (auto type = Cpp::GetType(name))
        return type;

    if (!enable_slow_lookup) {
        if (name.find("::") != std::string::npos)
            throw std::runtime_error("Calling Cppyy::GetType with qualified name '"
                                + name + "'\n");
        return nullptr;
    }

    // Here we might need to deal with integral types such as 3.14.

    std::string id = "__Cppyy_GetType_" + std::to_string(var_count++);
    std::string using_clause = "using " + id + " = __typeof__(" + name + ");\n";

    if (!Cpp::Declare(using_clause.c_str(), /*silent=*/false)) {
      TCppScope_t lookup = Cpp::GetNamed(id, 0);
      TCppType_t lookup_ty = Cpp::GetTypeFromScope(lookup);
      return Cpp::GetCanonicalType(lookup_ty);
    }
    return nullptr;
}


Cppyy::TCppType_t Cppyy::GetComplexType(const std::string &name) {
    return Cpp::GetComplexType(Cpp::GetType(name));
}


// //----------------------------------------------------------------------------
// static std::string extract_namespace(const std::string& name)
// {
// // Find the namespace the named class lives in, take care of templates
// // Note: this code also lives in CPyCppyy (TODO: refactor?)
//     if (name.empty())
//         return name;
//
//     int tpl_open = 0;
//     for (std::string::size_type pos = name.size()-1; 0 < pos; --pos) {
//         std::string::value_type c = name[pos];
//
//     // count '<' and '>' to be able to skip template contents
//         if (c == '>')
//             ++tpl_open;
//         else if (c == '<')
//             --tpl_open;
//
//     // collect name up to "::"
//         else if (tpl_open == 0 && c == ':' && name[pos-1] == ':') {
//         // found the extend of the scope ... done
//             return name.substr(0, pos-1);
//         }
//     }
//
// // no namespace; assume outer scope
//     return "";
// }
//

std::string Cppyy::ResolveEnum(TCppScope_t handle)
{
    std::string type = Cpp::GetTypeAsString(
        Cpp::GetIntegerTypeFromEnumScope(handle));
    if (type == "signed char")
        return "char";
    return type;
}

Cppyy::TCppScope_t Cppyy::GetUnderlyingScope(TCppScope_t scope)
{
    return Cpp::GetUnderlyingScope(scope);
}

Cppyy::TCppScope_t Cppyy::GetScope(const std::string& name,
                                   TCppScope_t parent_scope)
{
    if (Cppyy::TCppScope_t scope = Cpp::GetScope(name, parent_scope))
      return scope;
    if (!parent_scope || parent_scope == Cpp::GetGlobalScope())
      if (Cppyy::TCppScope_t scope = Cpp::GetScopeFromCompleteName(name))
        return scope;

    // FIXME: avoid string parsing here
    if (name.find('<') != std::string::npos) {
      // Templated Type; May need instantiation
      size_t start = name.find('<');
      size_t end = name.rfind('>');
      std::string params = name.substr(start + 1, end - start - 1);

      std::string pure_name = name.substr(0, start);
      Cppyy::TCppScope_t scope = Cpp::GetScope(pure_name, parent_scope);
      if (!scope && (!parent_scope || parent_scope == Cpp::GetGlobalScope()))
        scope = Cpp::GetScopeFromCompleteName(pure_name);

      if (Cppyy::IsTemplate(scope)) {
        std::vector<Cpp::TemplateArgInfo> templ_params;
        if (!Cppyy::AppendTypesSlow(params, templ_params))
          return Cpp::InstantiateTemplate(scope, templ_params.data(),
                                          templ_params.size(),
                                          /*instantiate_body=*/false);
      }
    }
    return nullptr;
}

Cppyy::TCppScope_t Cppyy::GetFullScope(const std::string& name)
{
  return Cppyy::GetScope(name);
}

Cppyy::TCppScope_t Cppyy::GetTypeScope(TCppScope_t var)
{
    return Cpp::GetScopeFromType(
        Cpp::GetVariableType(var));
}

Cppyy::TCppScope_t Cppyy::GetNamed(const std::string& name,
                                   TCppScope_t parent_scope)
{
    return Cpp::GetNamed(name, parent_scope);
}

Cppyy::TCppScope_t Cppyy::GetParentScope(TCppScope_t scope)
{
    return Cpp::GetParentScope(scope);
}

Cppyy::TCppScope_t Cppyy::GetScopeFromType(TCppType_t type)
{
    return Cpp::GetScopeFromType(type);
}

Cppyy::TCppType_t Cppyy::GetTypeFromScope(TCppScope_t klass)
{
    return Cpp::GetTypeFromScope(klass);
}

Cppyy::TCppScope_t Cppyy::GetGlobalScope()
{
    return Cpp::GetGlobalScope();
}

bool Cppyy::IsTemplate(TCppScope_t handle)
{
    return Cpp::IsTemplate(handle);
}

bool Cppyy::IsTemplateInstantiation(TCppScope_t handle)
{
    return Cpp::IsTemplateSpecialization(handle);
}

bool Cppyy::IsTypedefed(TCppScope_t handle)
{
    return Cpp::IsTypedefed(handle);
}

namespace {
class AutoCastRTTI {
public:
  virtual ~AutoCastRTTI() {}
};
} // namespace

Cppyy::TCppScope_t Cppyy::GetActualClass(TCppScope_t klass, TCppObject_t obj) {
    if (!Cpp::IsClassPolymorphic(klass))
        return klass;

    const std::type_info *typ = &typeid(*(AutoCastRTTI *)obj);

    std::string mangled_name = typ->name();
    std::string demangled_name = Cpp::Demangle(mangled_name);

    if (TCppScope_t scope = Cppyy::GetScope(demangled_name))
        return scope;

    return klass;
}

size_t Cppyy::SizeOf(TCppScope_t klass)
{
    return Cpp::SizeOf(klass);
}

size_t Cppyy::SizeOfType(TCppType_t klass)
{
    return Cpp::GetSizeOfType(klass);
}

// size_t Cppyy::SizeOf(const std::string& type_name)
// {
//     TDataType* dt = gROOT->GetType(type_name.c_str());
//     if (dt) return dt->Size();
//     return SizeOf(GetScope(type_name));
// }

bool Cppyy::IsBuiltin(const std::string& type_name)
{
    static std::set<std::string> s_builtins =
       {"bool", "char", "signed char", "unsigned char", "wchar_t", "short",
        "unsigned short", "int", "unsigned int", "long", "unsigned long",
        "long long", "unsigned long long", "float", "double", "long double",
        "void"};
     if (s_builtins.find(trim(type_name)) != s_builtins.end())
         return true;

    if (strstr(type_name.c_str(), "std::complex"))
        return true;

    return false;
}

bool Cppyy::IsBuiltin(TCppType_t type)
{
    return  Cpp::IsBuiltin(type);
    
}

bool Cppyy::IsComplete(TCppScope_t scope)
{
    return Cpp::IsComplete(scope);
}

// // memory management ---------------------------------------------------------
Cppyy::TCppObject_t Cppyy::Allocate(TCppScope_t scope)
{
  return Cpp::Allocate(scope, /*count=*/1);
}

void Cppyy::Deallocate(TCppScope_t scope, TCppObject_t instance)
{
  Cpp::Deallocate(scope, instance, /*count=*/1);
}

Cppyy::TCppObject_t Cppyy::Construct(TCppScope_t scope, void* arena/*=nullptr*/)
{
  return Cpp::Construct(scope, arena, /*count=*/1);
}

void Cppyy::Destruct(TCppScope_t scope, TCppObject_t instance)
{
  Cpp::Destruct(instance, scope, true, /*count=*/0);
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

// static inline
// bool is_ready(CallWrapper* wrap, bool is_direct) {
//     return (!is_direct && wrap->fFaceptr.fGeneric) || (is_direct && wrap->fFaceptr.fDirect);
// }

static inline
bool WrapperCall(Cppyy::TCppMethod_t method, size_t nargs, void* args_, void* self, void* result)
{
    Parameter* args = (Parameter*)args_;
    bool is_direct = nargs & DIRECT_CALL;
    nargs = CALL_NARGS(nargs);

    // if (!is_ready(wrap, is_direct))
    //     return false;        // happens with compilation error

    if (Cpp::JitCall JC = Cpp::MakeFunctionCallable(method)) {
        bool runRelease = false;
        //const auto& fgen = /* is_direct ? faceptr.fDirect : */ faceptr;
        if (nargs <= SMALL_ARGS_N) {
            void* smallbuf[SMALL_ARGS_N];
            if (nargs) runRelease = copy_args(args, nargs, smallbuf);
            // CLING_CATCH_UNCAUGHT_
            JC.Invoke(result, {smallbuf, nargs}, self);
            // _CLING_CATCH_UNCAUGHT
        } else {
            std::vector<void*> buf(nargs);
            runRelease = copy_args(args, nargs, buf.data());
            // CLING_CATCH_UNCAUGHT_
            JC.Invoke(result, {buf.data(), nargs}, self);
            // _CLING_CATCH_UNCAUGHT
        }
        if (runRelease) release_args(args, nargs);
        return true;
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
    throw std::runtime_error("failed to resolve function");
    return (T)-1;
}

#ifdef PRINT_DEBUG
    #define _IMP_CALL_PRINT_STMT(type)                                       \
        printf("IMP CALL with type: %s\n", #type);
#else
    #define _IMP_CALL_PRINT_STMT(type)
#endif

#define CPPYY_IMP_CALL(typecode, rtype)                                      \
rtype Cppyy::Call##typecode(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args)\
{                                                                            \
    _IMP_CALL_PRINT_STMT(rtype)                                              \
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
CPPYY_IMP_CALL(LL, long long    )
CPPYY_IMP_CALL(F,  float        )
CPPYY_IMP_CALL(D,  double       )
CPPYY_IMP_CALL(LD, long double  )

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
    // TClassRef cr("std::string"); // TODO: Why is this required?
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
    TCppMethod_t method, TCppScope_t klass, size_t nargs, void* args)
{
    void* obj = nullptr;
    WrapperCall(method, nargs, args, nullptr, &obj);
    return (TCppObject_t)obj;
}

void Cppyy::CallDestructor(TCppScope_t scope, TCppObject_t self)
{
  Cpp::Destruct(self, scope, /*withFree=*/false, /*count=*/0);
}

Cppyy::TCppObject_t Cppyy::CallO(TCppMethod_t method,
    TCppObject_t self, size_t nargs, void* args, TCppType_t result_type)
{
    void* obj = ::operator new(Cpp::GetSizeOfType(result_type));
    if (WrapperCall(method, nargs, args, self, obj))
        return (TCppObject_t)obj;
    ::operator delete(obj);
    return (TCppObject_t)0;
}

Cppyy::TCppFuncAddr_t Cppyy::GetFunctionAddress(TCppMethod_t method, bool check_enabled)
{
    return (TCppFuncAddr_t) Cpp::GetFunctionAddress(method);
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
    if (!scope)
      return false;

    // Test if this scope represents a namespace.
    return Cpp::IsNamespace(scope) || Cpp::GetGlobalScope() == scope;
}

bool Cppyy::IsClass(TCppScope_t scope)
{
    // Test if this scope represents a namespace.
    return Cpp::IsClass(scope);
}
//
bool Cppyy::IsAbstract(TCppScope_t scope)
{
    // Test if this type may not be instantiated.
    return Cpp::IsAbstract(scope);
}

bool Cppyy::IsEnumScope(TCppScope_t scope)
{
    return Cpp::IsEnumScope(scope);
}

bool Cppyy::IsEnumConstant(TCppScope_t scope)
{
  return Cpp::IsEnumConstant(Cpp::GetUnderlyingScope(scope));
}

bool Cppyy::IsEnumType(TCppType_t type)
{
    return Cpp::IsEnumType(type);
}

bool Cppyy::IsAggregate(TCppType_t type)
{
  // Test if this type is a "plain old data" type
  return Cpp::IsAggregate(type);
}

bool Cppyy::IsDefaultConstructable(TCppScope_t scope)
{
// Test if this type has a default constructor or is a "plain old data" type
    return Cpp::HasDefaultConstructor(scope);
}

bool Cppyy::IsVariable(TCppScope_t scope)
{
    return Cpp::IsVariable(scope);
}

// // helpers for stripping scope names
// static
// std::string outer_with_template(const std::string& name)
// {
// // Cut down to the outer-most scope from <name>, taking proper care of templates.
//     int tpl_open = 0;
//     for (std::string::size_type pos = 0; pos < name.size(); ++pos) {
//         std::string::value_type c = name[pos];
//
//     // count '<' and '>' to be able to skip template contents
//         if (c == '<')
//             ++tpl_open;
//         else if (c == '>')
//             --tpl_open;
//
//     // collect name up to "::"
//         else if (tpl_open == 0 && \
//                  c == ':' && pos+1 < name.size() && name[pos+1] == ':') {
//         // found the extend of the scope ... done
//             return name.substr(0, pos-1);
//         }
//     }
//
// // whole name is apparently a single scope
//     return name;
// }
//
// static
// std::string outer_no_template(const std::string& name)
// {
// // Cut down to the outer-most scope from <name>, drop templates
//     std::string::size_type first_scope = name.find(':');
//     if (first_scope == std::string::npos)
//         return name.substr(0, name.find('<'));
//     std::string::size_type first_templ = name.find('<');
//     if (first_templ == std::string::npos)
//         return name.substr(0, first_scope);
//     return name.substr(0, std::min(first_templ, first_scope));
// }
//
// #define FILL_COLL(type, filter) {                                             \
//     TIter itr{coll};                                                          \
//     type* obj = nullptr;                                                      \
//     while ((obj = (type*)itr.Next())) {                                       \
//         const char* nm = obj->GetName();                                      \
//         if (nm && nm[0] != '_' && !(obj->Property() & (filter))) {            \
//             if (gInitialNames.find(nm) == gInitialNames.end())                \
//                 cppnames.insert(nm);                                          \
//     }}}
//
// static inline
// void cond_add(Cppyy::TCppScope_t scope, const std::string& ns_scope,
//     std::set<std::string>& cppnames, const char* name, bool nofilter = false)
// {
//     if (!name || name[0] == '_' || strstr(name, ".h") != 0 || strncmp(name, "operator", 8) == 0)
//         return;
//
//     if (scope == GLOBAL_HANDLE) {
//         std::string to_add = outer_no_template(name);
//         if (nofilter || gInitialNames.find(to_add) == gInitialNames.end())
//             cppnames.insert(outer_no_template(name));
//     } else if (scope == STD_HANDLE) {
//         if (strncmp(name, "std::", 5) == 0) {
//             name += 5;
// #ifdef __APPLE__
//             if (strncmp(name, "__1::", 5) == 0) name += 5;
// #endif
//         }
//         cppnames.insert(outer_no_template(name));
//     } else {
//         if (strncmp(name, ns_scope.c_str(), ns_scope.size()) == 0)
//             cppnames.insert(outer_with_template(name + ns_scope.size()));
//     }
// }

void Cppyy::GetAllCppNames(TCppScope_t scope, std::set<std::string>& cppnames)
{
// Collect all known names of C++ entities under scope. This is useful for IDEs
// employing tab-completion, for example. Note that functions names need not be
// unique as they can be overloaded.
    Cpp::GetAllCppNames(scope, cppnames);
}

//
// // class reflection information ----------------------------------------------
std::vector<Cppyy::TCppScope_t> Cppyy::GetUsingNamespaces(TCppScope_t scope)
{
    return Cpp::GetUsingNamespaces(scope);
}

// // class reflection information ----------------------------------------------
std::string Cppyy::GetFinalName(TCppType_t klass)
{
  return Cpp::GetCompleteName(Cpp::GetUnderlyingScope(klass));
}

std::string Cppyy::GetScopedFinalName(TCppType_t klass)
{
    return Cpp::GetQualifiedCompleteName(klass);
}

bool Cppyy::HasVirtualDestructor(TCppScope_t scope)
{
    TCppMethod_t func = Cpp::GetDestructor(scope);
    return Cpp::IsVirtualMethod(func);
}

// bool Cppyy::HasComplexHierarchy(TCppType_t klass)
// {
//     int is_complex = 1;
//     size_t nbases = 0;
//
//     TClassRef& cr = type_from_handle(klass);
//     if (cr.GetClass() && cr->GetListOfBases() != 0)
//         nbases = GetNumBases(klass);
//
//     if (1 < nbases)
//         is_complex = 1;
//     else if (nbases == 0)
//         is_complex = 0;
//     else {         // one base class only
//         TBaseClass* base = (TBaseClass*)cr->GetListOfBases()->At(0);
//         if (base->Property() & kIsVirtualBase)
//             is_complex = 1;       // TODO: verify; can be complex, need not be.
//         else
//             is_complex = HasComplexHierarchy(GetScope(base->GetName()));
//     }
//
//     return is_complex;
// }

Cppyy::TCppIndex_t Cppyy::GetNumBases(TCppScope_t klass)
{
// Get the total number of base classes that this class has.
    return Cpp::GetNumBases(klass);
}

////////////////////////////////////////////////////////////////////////////////
/// \fn Cppyy::TCppIndex_t Cppyy::GetNumBasesLongestBranch(TCppScope_t klass)
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
Cppyy::TCppIndex_t Cppyy::GetNumBasesLongestBranch(TCppScope_t klass) {
    std::vector<size_t> num;
    for (TCppIndex_t ibase = 0; ibase < GetNumBases(klass); ++ibase)
        num.push_back(GetNumBasesLongestBranch(Cppyy::GetBaseScope(klass, ibase)));
    if (num.empty())
        return 0;
    return *std::max_element(num.begin(), num.end()) + 1;
}

std::string Cppyy::GetBaseName(TCppType_t klass, TCppIndex_t ibase)
{
    return Cpp::GetName(Cpp::GetBaseClass(klass, ibase));
}

Cppyy::TCppScope_t Cppyy::GetBaseScope(TCppScope_t klass, TCppIndex_t ibase)
{
    return Cpp::GetBaseClass(klass, ibase);
}

bool Cppyy::IsSubclass(TCppScope_t derived, TCppScope_t base)
{
    return Cpp::IsSubclass(derived, base);
}

static std::set<std::string> gSmartPtrTypes =
    {"std::auto_ptr", "std::shared_ptr", "std::unique_ptr", "std::weak_ptr"};

bool Cppyy::IsSmartPtr(TCppScope_t klass)
{
    const std::string& rn = Cppyy::GetScopedFinalName(klass);
    if (gSmartPtrTypes.find(rn.substr(0, rn.find("<"))) != gSmartPtrTypes.end())
        return true;
    return false;
}

bool Cppyy::GetSmartPtrInfo(
    const std::string& tname, TCppScope_t* raw, TCppMethod_t* deref)
{
    // TODO: We can directly accept scope instead of name
    const std::string& rn = ResolveName(tname);
    if (gSmartPtrTypes.find(rn.substr(0, rn.find("<"))) == gSmartPtrTypes.end())
        return false;

    if (!raw && !deref) return true;

    TCppScope_t scope = Cppyy::GetScope(rn);
    if (!scope)
        return false;

    std::vector<TCppMethod_t> ops;
    Cpp::GetOperator(scope, Cpp::Operator::OP_Arrow, ops,
                     /*kind=*/Cpp::OperatorArity::kBoth);
    if (ops.size() != 1)
        return false;

    if (deref) *deref = ops[0];
    if (raw) *raw = Cppyy::GetScopeFromType(Cpp::GetFunctionReturnType(ops[0]));
    return (!deref || *deref) && (!raw || *raw);
}

// void Cppyy::AddSmartPtrType(const std::string& type_name)
// {
//     gSmartPtrTypes.insert(ResolveName(type_name));
// }
//
// void Cppyy::AddTypeReducer(const std::string& reducable, const std::string& reduced)
// {
//     gInterpreter->AddTypeReducer(reducable, reduced);
// }


// type offsets --------------------------------------------------------------
ptrdiff_t Cppyy::GetBaseOffset(TCppScope_t derived, TCppScope_t base,
    TCppObject_t address, int direction, bool rerror)
{
    intptr_t offset = Cpp::GetBaseClassOffset(derived, base);
    
    if (offset == -1)   // Cling error, treat silently
        return rerror ? (ptrdiff_t)offset : 0;

    return (ptrdiff_t)(direction < 0 ? -offset : offset);
}


// // method/function reflection information ------------------------------------
// Cppyy::TCppIndex_t Cppyy::GetNumMethods(TCppScope_t scope, bool accept_namespace)
// {
//     if (!accept_namespace && IsNamespace(scope))
//         return (TCppIndex_t)0;     // enforce lazy
//
//     if (scope == GLOBAL_HANDLE)
//         return gROOT->GetListOfGlobalFunctions(true)->GetSize();
//
//     TClassRef& cr = type_from_handle(scope);
//     if (cr.GetClass() && cr->GetListOfMethods(true)) {
//         Cppyy::TCppIndex_t nMethods = (TCppIndex_t)cr->GetListOfMethods(false)->GetSize();
//         if (nMethods == (TCppIndex_t)0) {
//             std::string clName = GetScopedFinalName(scope);
//             if (clName.find('<') != std::string::npos) {
//             // chicken-and-egg problem: TClass does not know about methods until
//             // instantiation, so force it
//                 std::ostringstream stmt;
//                 stmt << "template class " << clName << ";";
//                 gInterpreter->Declare(stmt.str().c_str(), true [> silent <]);
//
//             // now reload the methods
//                 return (TCppIndex_t)cr->GetListOfMethods(true)->GetSize();
//             }
//         }
//         return nMethods;
//     }
//
//     return (TCppIndex_t)0;         // unknown class?
// }

void Cppyy::GetClassMethods(TCppScope_t scope, std::vector<Cppyy::TCppMethod_t> &methods)
{
    Cpp::GetClassMethods(scope, methods);
}

std::vector<Cppyy::TCppScope_t> Cppyy::GetMethodsFromName(
    TCppScope_t scope, const std::string& name)
{
    return Cpp::GetFunctionsUsingName(scope, name);
}

// Cppyy::TCppMethod_t Cppyy::GetMethod(TCppScope_t scope, TCppIndex_t idx)
// {
//     TClassRef& cr = type_from_handle(scope);
//     if (cr.GetClass()) {
//         TFunction* f = (TFunction*)cr->GetListOfMethods(false)->At((int)idx);
//         if (f) return (Cppyy::TCppMethod_t)new_CallWrapper(f);
//         return (Cppyy::TCppMethod_t)nullptr;
//     }
//
//     assert(klass == (Cppyy::TCppType_t)GLOBAL_HANDLE);
//     return (Cppyy::TCppMethod_t)idx;
// }
//
std::string Cppyy::GetMethodName(TCppMethod_t method)
{
    return Cpp::GetName(method);
}

std::string Cppyy::GetMethodFullName(TCppMethod_t method)
{
    return Cpp::GetCompleteName(method);
}

// std::string Cppyy::GetMethodMangledName(TCppMethod_t method)
// {
//     if (method)
//         return m2f(method)->GetMangledName();
//     return "<unknown>";
// }

Cppyy::TCppType_t Cppyy::GetMethodReturnType(TCppMethod_t method)
{
    return Cpp::GetFunctionReturnType(method);
}

std::string Cppyy::GetMethodReturnTypeAsString(TCppMethod_t method)
{
    return 
    Cpp::GetTypeAsString(
        Cpp::GetCanonicalType(
            Cpp::GetFunctionReturnType(method)));
}

Cppyy::TCppIndex_t Cppyy::GetMethodNumArgs(TCppMethod_t method)
{
    return Cpp::GetFunctionNumArgs(method);
}

Cppyy::TCppIndex_t Cppyy::GetMethodReqArgs(TCppMethod_t method)
{
    return Cpp::GetFunctionRequiredArgs(method);
}

std::string Cppyy::GetMethodArgName(TCppMethod_t method, TCppIndex_t iarg)
{
    if (!method)
        return "<unknown>";

    return Cpp::GetFunctionArgName(method, iarg);
}

Cppyy::TCppType_t Cppyy::GetMethodArgType(TCppMethod_t method, TCppIndex_t iarg)
{
    return Cpp::GetFunctionArgType(method, iarg);
}

std::string Cppyy::GetMethodArgTypeAsString(TCppMethod_t method, TCppIndex_t iarg)
{
  return Cpp::GetTypeAsString(Cpp::RemoveTypeQualifier(
      Cpp::GetFunctionArgType(method, iarg), Cpp::QualKind::Const));
}

std::string Cppyy::GetMethodArgCanonTypeAsString(TCppMethod_t method, TCppIndex_t iarg)
{
    return
    Cpp::GetTypeAsString(
        Cpp::GetCanonicalType(
            Cpp::GetFunctionArgType(method, iarg)));
}

std::string Cppyy::GetMethodArgDefault(TCppMethod_t method, TCppIndex_t iarg)
{
    if (!method)
       return "";
    return Cpp::GetFunctionArgDefault(method, iarg);
}

Cppyy::TCppIndex_t Cppyy::CompareMethodArgType(TCppMethod_t method, TCppIndex_t iarg, const std::string &req_type)
{
    // if (method) {
    //     TFunction* f = m2f(method);
    //     TMethodArg* arg = (TMethodArg *)f->GetListOfMethodArgs()->At((int)iarg);
    //     void *argqtp = gInterpreter->TypeInfo_QualTypePtr(arg->GetTypeInfo());

    //     TypeInfo_t *reqti = gInterpreter->TypeInfo_Factory(req_type.c_str());
    //     void *reqqtp = gInterpreter->TypeInfo_QualTypePtr(reqti);

    //     if (ArgSimilarityScore(argqtp, reqqtp) < 10) {
    //         return ArgSimilarityScore(argqtp, reqqtp);
    //     }
    //     else { // Match using underlying types
    //         if(gInterpreter->IsPointerType(argqtp))
    //             argqtp = gInterpreter->TypeInfo_QualTypePtr(gInterpreter->GetPointerType(argqtp));

    //         // Handles reference types and strips qualifiers
    //         TypeInfo_t *arg_ul = gInterpreter->GetNonReferenceType(argqtp);
    //         TypeInfo_t *req_ul = gInterpreter->GetNonReferenceType(reqqtp);
    //         argqtp = gInterpreter->TypeInfo_QualTypePtr(gInterpreter->GetUnqualifiedType(gInterpreter->TypeInfo_QualTypePtr(arg_ul)));
    //         reqqtp = gInterpreter->TypeInfo_QualTypePtr(gInterpreter->GetUnqualifiedType(gInterpreter->TypeInfo_QualTypePtr(req_ul)));

    //         return ArgSimilarityScore(argqtp, reqqtp);
    //     }
    // }
    return 0; // Method is not valid
}

std::string Cppyy::GetMethodSignature(TCppMethod_t method, bool show_formal_args, TCppIndex_t max_args)
{
    std::ostringstream sig;
    sig << "(";
    int nArgs = GetMethodNumArgs(method);
    if (max_args != (TCppIndex_t)-1) nArgs = std::min(nArgs, (int)max_args);
    for (int iarg = 0; iarg < nArgs; ++iarg) {
        sig << Cppyy::GetMethodArgTypeAsString(method, iarg);
        if (show_formal_args) {
            std::string argname = Cppyy::GetMethodArgName(method, iarg);
            if (!argname.empty()) sig << " " << argname;
            std::string defvalue = Cppyy::GetMethodArgDefault(method, iarg);
            if (!defvalue.empty()) sig << " = " << defvalue;
        }
        if (iarg != nArgs-1) sig << ", ";
    }
    sig << ")";
    return sig.str();
}

std::string Cppyy::GetMethodPrototype(TCppMethod_t method, bool show_formal_args)
{
  assert(0 && "Unused");
  return ""; // return Cpp::GetFunctionPrototype(method, show_formal_args);
}

std::string Cppyy::GetDoxygenComment(TCppScope_t scope, bool strip_markers)
{
    return Cpp::GetDoxygenComment(scope, strip_markers);
}

bool Cppyy::IsConstMethod(TCppMethod_t method)
{
    if (!method)
        return false;
    return Cpp::IsConstMethod(method);
}

void Cppyy::GetTemplatedMethods(TCppScope_t scope, std::vector<Cppyy::TCppMethod_t> &methods)
{
    Cpp::GetFunctionTemplatedDecls(scope, methods);
}

Cppyy::TCppIndex_t Cppyy::GetNumTemplatedMethods(TCppScope_t scope, bool accept_namespace)
{
    std::vector<Cppyy::TCppMethod_t> mc;
    Cpp::GetFunctionTemplatedDecls(scope, mc);
    return mc.size();
}

std::string Cppyy::GetTemplatedMethodName(TCppScope_t scope, TCppIndex_t imeth)
{
    std::vector<Cppyy::TCppMethod_t> mc;
    Cpp::GetFunctionTemplatedDecls(scope, mc);

    if (imeth < mc.size()) return GetMethodName(mc[imeth]);

    return "";
}

bool Cppyy::ExistsMethodTemplate(TCppScope_t scope, const std::string& name)
{
    return Cpp::ExistsFunctionTemplate(name, scope);
}

bool Cppyy::IsTemplatedMethod(TCppMethod_t method)
{
    return Cpp::IsTemplatedFunction(method);
}

bool Cppyy::IsStaticTemplate(TCppScope_t scope, const std::string& name)
{
    if (Cpp::TCppFunction_t tf = GetMethodTemplate(scope, name, ""))
        return Cpp::IsStaticMethod(tf);
    return false;
}

Cppyy::TCppMethod_t Cppyy::GetMethodTemplate(
    TCppScope_t scope, const std::string& name, const std::string& proto)
{
    std::string pureName;
    std::string explicit_params;

    if ((name.find("operator<") != 0) &&
        (name.find('<') != std::string::npos)) {
        pureName = name.substr(0, name.find('<'));
        size_t start = name.find('<');
        size_t end = name.rfind('>');
        explicit_params = name.substr(start + 1, end - start - 1);
    } else {
        pureName = name;
    }

    std::vector<Cppyy::TCppMethod_t> unresolved_candidate_methods;
    Cpp::GetClassTemplatedMethods(pureName, scope,
                                  unresolved_candidate_methods);
    if (unresolved_candidate_methods.empty() && name.find("operator") == 0) {
        // try operators
        Cppyy::GetClassOperators(scope, pureName, unresolved_candidate_methods);
    }

    // CPyCppyy assumes that we attempt instantiation here
    std::vector<Cpp::TemplateArgInfo> arg_types;
    std::vector<Cpp::TemplateArgInfo> templ_params;
    Cppyy::AppendTypesSlow(proto, arg_types, scope);
    Cppyy::AppendTypesSlow(explicit_params, templ_params, scope);

    Cppyy::TCppMethod_t cppmeth = Cpp::BestOverloadFunctionMatch(
        unresolved_candidate_methods, templ_params, arg_types);

    if (!cppmeth && unresolved_candidate_methods.size() == 1 &&
        !templ_params.empty())
      cppmeth = Cpp::InstantiateTemplate(
          unresolved_candidate_methods[0], templ_params.data(),
          templ_params.size(), /*instantiate_body=*/false);

    return cppmeth;

    // if it fails, use Sema to propogate info about why it failed (DeductionInfo)

}

static inline std::string type_remap(const std::string& n1,
                                     const std::string& n2) {
    // Operator lookups of (C++ string, Python str) should succeed for the
    // combos of string/str, wstring/str, string/unicode and wstring/unicode;
    // since C++ does not have a operator+(std::string, std::wstring), we'll
    // have to look up the same type and rely on the converters in
    // CPyCppyy/_cppyy.
    if (n1 == "str" || n1 == "unicode" || n1 == "std::basic_string<char>") {
        if (n2 == "std::basic_string<wchar_t>")
            return "std::basic_string<wchar_t>&";                      // match like for like
        return "std::basic_string<char>&"; // probably best bet
    } else if (n1 == "std::basic_string<wchar_t>") {
        return "std::basic_string<wchar_t>&";
    } else if (n1 == "float") {
        return "double"; // debatable, but probably intended
    } else if (n1 == "complex") {
        return "std::complex<double>";
    }
    return n1;
}

void Cppyy::GetClassOperators(Cppyy::TCppScope_t klass,
                              const std::string& opname,
                              std::vector<TCppScope_t>& operators) {
    std::string op = opname.substr(8);
    Cpp::GetOperator(klass, Cpp::GetOperatorFromSpelling(op), operators,
                     /*kind=*/Cpp::OperatorArity::kBoth);
}

Cppyy::TCppMethod_t Cppyy::GetGlobalOperator(
    TCppType_t scope, const std::string& lc, const std::string& rc, const std::string& opname)
{
    std::string rc_type = type_remap(rc, lc);
    std::string lc_type = type_remap(lc, rc);
    bool is_templated = false;
    if ((lc_type.find('<') != std::string::npos) ||
        (rc_type.find('<') != std::string::npos)) {
        is_templated = true;
    }

    std::vector<TCppScope_t> overloads;
    Cpp::GetOperator(scope, Cpp::GetOperatorFromSpelling(opname), overloads,
                     /*kind=*/Cpp::OperatorArity::kBoth);

    std::vector<Cppyy::TCppMethod_t> unresolved_candidate_methods;
    for (auto overload: overloads) {
        if (Cpp::IsTemplatedFunction(overload)) {
            unresolved_candidate_methods.push_back(overload);
            continue;
        } else {
            TCppType_t lhs_type = Cpp::GetFunctionArgType(overload, 0);
            if (lc_type !=
                Cpp::GetTypeAsString(Cpp::GetUnderlyingType(lhs_type)))
                continue;

            if (!rc_type.empty()) {
                if (Cpp::GetFunctionNumArgs(overload) != 2)
                    continue;
                TCppType_t rhs_type = Cpp::GetFunctionArgType(overload, 1);
                if (rc_type !=
                    Cpp::GetTypeAsString(Cpp::GetUnderlyingType(rhs_type)))
                    continue;
            }
            return overload;
        }
    }
    if (is_templated) {
        std::string lc_template = lc_type.substr(
            lc_type.find("<") + 1, lc_type.rfind(">") - lc_type.find("<") - 1);
        std::string rc_template = rc_type.substr(
            rc_type.find("<") + 1, rc_type.rfind(">") - rc_type.find("<") - 1);

        std::vector<Cpp::TemplateArgInfo> arg_types;
        if (auto l = Cppyy::GetType(lc_type, true))
            arg_types.emplace_back(l);
        else
            return nullptr;

        if (!rc_type.empty()) {
            if (auto r = Cppyy::GetType(rc_type, true))
                arg_types.emplace_back(r);
            else
                return nullptr;
        }
        Cppyy::TCppMethod_t cppmeth = Cpp::BestOverloadFunctionMatch(
            unresolved_candidate_methods, {}, arg_types);
        if (cppmeth)
            return cppmeth;
    }
    {
        // we are trying to do a madeup IntegralToFloating implicit cast emulating clang
        bool flag = false;
        if (rc_type == "int") {
            rc_type = "double";
            flag = true;
        }
        if (lc_type == "int") {
            lc_type = "double";
            flag = true;
        }
        if (flag)
            return GetGlobalOperator(scope, lc_type, rc_type, opname);
    }
    return nullptr;
}

// // method properties ---------------------------------------------------------
bool Cppyy::IsDeletedMethod(TCppMethod_t method)
{
    return Cpp::IsFunctionDeleted(method);
}

bool Cppyy::IsPublicMethod(TCppMethod_t method)
{
    return Cpp::IsPublicMethod(method);
}

bool Cppyy::IsProtectedMethod(TCppMethod_t method)
{
    return Cpp::IsProtectedMethod(method);
}

bool Cppyy::IsPrivateMethod(TCppMethod_t method)
{
    return Cpp::IsPrivateMethod(method);
}

bool Cppyy::IsConstructor(TCppMethod_t method)
{
    return Cpp::IsConstructor(method);
}

bool Cppyy::IsDestructor(TCppMethod_t method)
{
    return Cpp::IsDestructor(method);
}

bool Cppyy::IsStaticMethod(TCppMethod_t method)
{
    return Cpp::IsStaticMethod(method);
}

bool Cppyy::IsExplicit(TCppMethod_t method)
{
    return Cpp::IsExplicit(method);
}

//
// // data member reflection information ----------------------------------------
// Cppyy::TCppIndex_t Cppyy::GetNumDatamembers(TCppScope_t scope, bool accept_namespace)
// {
//     if (!accept_namespace && IsNamespace(scope))
//         return (TCppIndex_t)0;     // enforce lazy
//
//     if (scope == GLOBAL_HANDLE)
//         return gROOT->GetListOfGlobals(true)->GetSize();
//
//     TClassRef& cr = type_from_handle(scope);
//     if (cr.GetClass() && cr->GetListOfDataMembers())
//         return cr->GetListOfDataMembers()->GetSize();
//
//     return (TCppIndex_t)0;         // unknown class?
// }

void Cppyy::GetDatamembers(TCppScope_t scope, std::vector<TCppScope_t>& datamembers)
{
    Cpp::GetDatamembers(scope, datamembers);
    Cpp::GetStaticDatamembers(scope, datamembers);
    Cpp::GetEnumConstantDatamembers(scope, datamembers, false);
}

bool Cppyy::CheckDatamember(TCppScope_t scope, const std::string& name) {
    return (bool) Cpp::LookupDatamember(name, scope);
}

bool Cppyy::IsLambdaClass(TCppType_t type) {
    return Cpp::IsLambdaClass(type);
}

Cppyy::TCppScope_t Cppyy::WrapLambdaFromVariable(TCppScope_t var) {
    std::ostringstream code;
    std::string name = Cppyy::GetFinalName(var);
    code << "namespace __cppyy_internal_wrap_g {\n"
      << "  " << "std::function " << name << " = ::" << Cpp::GetQualifiedName(var) << ";\n"
      << "}\n";
    
    if (Cppyy::Compile(code.str().c_str())) {
      TCppScope_t res = Cpp::GetNamed(
          name, Cpp::GetScope("__cppyy_internal_wrap_g", /*parent=*/nullptr));
      if (res) return res;
    }
    return var;
}

Cppyy::TCppScope_t Cppyy::AdaptFunctionForLambdaReturn(TCppScope_t fn) {
    std::string fn_name = Cpp::GetQualifiedCompleteName(fn);
    std::string signature = Cppyy::GetMethodSignature(fn, true);

    std::ostringstream call;
    call << "(";
    for (size_t i = 0, n = Cppyy::GetMethodNumArgs(fn); i < n; i++) {
        call << Cppyy::GetMethodArgName(fn, i);
        if (i != n - 1)
            call << ", ";
    }
    call << ")";
    
    std::ostringstream code;
    static int i = 0;
    std::string name = "lambda_return_convert_" + std::to_string(++i);
        code << "namespace __cppyy_internal_wrap_g {\n"
         << "auto " << name << signature << "{" << "return std::function(" << fn_name << call.str() << "); }\n"
         << "}\n";
    if (Cppyy::Compile(code.str().c_str())) {
      TCppScope_t res = Cpp::GetNamed(
          name, Cpp::GetScope("__cppyy_internal_wrap_g", /*parent=*/nullptr));
      if (res) return res;
    }
    return fn;
}

// std::string Cppyy::GetDatamemberName(TCppScope_t scope, TCppIndex_t idata)
// {
//     TClassRef& cr = type_from_handle(scope);
//     if (cr.GetClass()) {
//         TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
//         return m->GetName();
//     }
//     assert(scope == GLOBAL_HANDLE);
//     TGlobal* gbl = g_globalvars[idata];
//     return gbl->GetName();
// }
//
// static inline
// int count_scopes(const std::string& tpname)
// {
//     int count = 0;
//     std::string::size_type pos = tpname.find("::", 0);
//     while (pos != std::string::npos) {
//         count++;
//         pos = tpname.find("::", pos+1);
//     }
//     return count;
// }

Cppyy::TCppType_t Cppyy::GetDatamemberType(TCppScope_t var)
{
  return Cpp::GetVariableType(Cpp::GetUnderlyingScope(var));
}

std::string Cppyy::GetDatamemberTypeAsString(TCppScope_t scope)
{
  return Cpp::GetTypeAsString(
      Cpp::GetVariableType(Cpp::GetUnderlyingScope(scope)));
}

std::string Cppyy::GetTypeAsString(TCppType_t type)
{
    return Cpp::GetTypeAsString(type);
}

intptr_t Cppyy::GetDatamemberOffset(TCppScope_t var, TCppScope_t klass)
{
  return Cpp::GetVariableOffset(Cpp::GetUnderlyingScope(var), klass);
}

// static inline
// Cppyy::TCppIndex_t gb2idx(TGlobal* gb)
// {
//     if (!gb) return (Cppyy::TCppIndex_t)-1;
//
//     auto pidx = g_globalidx.find(gb);
//     if (pidx == g_globalidx.end()) {
//         auto idx = g_globalvars.size();
//         g_globalvars.push_back(gb);
//         g_globalidx[gb] = idx;
//         return (Cppyy::TCppIndex_t)idx;
//     }
//     return (Cppyy::TCppIndex_t)pidx->second;
// }
//
// Cppyy::TCppIndex_t Cppyy::GetDatamemberIndex(TCppScope_t scope, const std::string& name)
// {
//     if (scope == GLOBAL_HANDLE) {
//         TGlobal* gb = (TGlobal*)gROOT->GetListOfGlobals(false [> load <])->FindObject(name.c_str());
//         if (!gb) gb = (TGlobal*)gROOT->GetListOfGlobals(true  [> load <])->FindObject(name.c_str());
//         if (!gb) {
//         // some enums are not loaded as they are not considered part of
//         // the global scope, but of the enum scope; get them w/o checking
//             TDictionary::DeclId_t did = gInterpreter->GetDataMember(nullptr, name.c_str());
//             if (did) {
//                 DataMemberInfo_t* t = gInterpreter->DataMemberInfo_Factory(did, nullptr);
//                 ((TListOfDataMembers*)gROOT->GetListOfGlobals())->Get(t, true);
//                 gb = (TGlobal*)gROOT->GetListOfGlobals(false [> load <])->FindObject(name.c_str());
//             }
//         }
//
//         if (gb && strcmp(gb->GetFullTypeName(), "(lambda)") == 0) {
//         // lambdas use a compiler internal closure type, so we wrap
//         // them, then return the wrapper's type
//         // TODO: this current leaks the std::function; also, if possible,
//         //       should instantiate through TClass rather then ProcessLine
//             std::ostringstream s;
//             s << "auto __cppyy_internal_wrap_" << name << " = "
//                  "new __cling_internal::FT<decltype(" << name << ")>::F"
//                  "{" << name << "};";
//             gInterpreter->ProcessLine(s.str().c_str());
//             TGlobal* wrap = (TGlobal*)gROOT->GetListOfGlobals(true)->FindObject(
//                 ("__cppyy_internal_wrap_"+name).c_str());
//             if (wrap && wrap->GetAddress()) gb = wrap;
//         }
//
//         return gb2idx(gb);
//
//     } else {
//         TClassRef& cr = type_from_handle(scope);
//         if (cr.GetClass()) {
//             TDataMember* dm =
//                 (TDataMember*)cr->GetListOfDataMembers()->FindObject(name.c_str());
//             // TODO: turning this into an index is silly ...
//             if (dm) return (TCppIndex_t)cr->GetListOfDataMembers()->IndexOf(dm);
//         }
//     }
//
//     return (TCppIndex_t)-1;
// }
//
// Cppyy::TCppIndex_t Cppyy::GetDatamemberIndexEnumerated(TCppScope_t scope, TCppIndex_t idata)
// {
//     if (scope == GLOBAL_HANDLE) {
//         TGlobal* gb = (TGlobal*)((THashList*)gROOT->GetListOfGlobals(false [> load <]))->At((int)idata);
//         return gb2idx(gb);
//     }
//
//     return idata;
// }

// data member properties ----------------------------------------------------
bool Cppyy::IsPublicData(TCppScope_t datamem)
{
    return Cpp::IsPublicVariable(datamem);
}

bool Cppyy::IsProtectedData(TCppScope_t datamem)
{
    return Cpp::IsProtectedVariable(datamem);
}

bool Cppyy::IsPrivateData(TCppScope_t datamem)
{
    return Cpp::IsPrivateVariable(datamem);
}

bool Cppyy::IsStaticDatamember(TCppScope_t var)
{
  return Cpp::IsStaticVariable(Cpp::GetUnderlyingScope(var));
}

bool Cppyy::IsConstVar(TCppScope_t var)
{
    return Cpp::IsConstVariable(var);
}

Cppyy::TCppScope_t Cppyy::ReduceReturnType(TCppScope_t fn, TCppType_t reduce) {
    std::string fn_name = Cpp::GetQualifiedCompleteName(fn);
    std::string signature = Cppyy::GetMethodSignature(fn, true);
    std::string result_type = Cppyy::GetTypeAsString(reduce);

    std::ostringstream call;
    call << "(";
    for (size_t i = 0, n = Cppyy::GetMethodNumArgs(fn); i < n; i++) {
        call << Cppyy::GetMethodArgName(fn, i);
        if (i != n - 1)
            call << ", ";
    }
    call << ")";
    
    std::ostringstream code;
    static int i = 0;
    std::string name = "reduced_function_" + std::to_string(++i);
        code << "namespace __cppyy_internal_wrap_g {\n"
         << result_type << " " << name << signature << "{" << "return (" << result_type << ")::" << fn_name << call.str() << "; }\n"
         << "}\n";
    if (Cppyy::Compile(code.str().c_str())) {
      TCppScope_t res = Cpp::GetNamed(
          name, Cpp::GetScope("__cppyy_internal_wrap_g", /*parent=*/nullptr));
      if (res) return res;
    }
    return fn;
}

// bool Cppyy::IsEnumData(TCppScope_t scope, TCppIndex_t idata)
// {
// // TODO: currently, ROOT/meta does not properly distinguish between variables of enum
// // type, and values of enums. The latter are supposed to be const. This code relies on
// // odd features (bugs?) to figure out the difference, but this should really be fixed
// // upstream and/or deserves a new API.

//     if (scope == GLOBAL_HANDLE) {
//         TGlobal* gbl = g_globalvars[idata];

//     // make use of an oddity: enum global variables do not have their kIsStatic bit
//     // set, whereas enum global values do
//         return (gbl->Property() & kIsEnum) && (gbl->Property() & kIsStatic);
//     }

//     TClassRef& cr = type_from_handle(scope);
//     if (cr.GetClass()) {
//         TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At((int)idata);
//         std::string ti = m->GetTypeName();

//     // can't check anonymous enums by type name, so just accept them as enums
//         if (ti.rfind("(anonymous)") != std::string::npos)
//             return m->Property() & kIsEnum;

//     // since there seems to be no distinction between data of enum type and enum values,
//     // check the list of constants for the type to see if there's a match
//         if (ti.rfind(cr->GetName(), 0) != std::string::npos) {
//             std::string::size_type s = strlen(cr->GetName())+2;
//             if (s < ti.size()) {
//                 TEnum* ee = ((TListOfEnums*)cr->GetListOfEnums())->GetObject(ti.substr(s, std::string::npos).c_str());
//                 if (ee) return ee->GetConstant(m->GetName());
//             }
//         }
//     }

// // this default return only means that the data will be writable, not that it will
// // be unreadable or otherwise misrepresented
//     return false;
// }

std::vector<long int>  Cppyy::GetDimensions(TCppType_t type)
{
    return Cpp::GetDimensions(type);
}

// enum properties -----------------------------------------------------------
std::vector<Cppyy::TCppScope_t> Cppyy::GetEnumConstants(TCppScope_t scope)
{
    return Cpp::GetEnumConstants(scope);
}

Cppyy::TCppType_t Cppyy::GetEnumConstantType(TCppScope_t scope)
{
  return Cpp::GetEnumConstantType(Cpp::GetUnderlyingScope(scope));
}

Cppyy::TCppIndex_t Cppyy::GetEnumDataValue(TCppScope_t scope)
{
    return Cpp::GetEnumConstantValue(scope);
}

// std::string Cppyy::GetEnumDataName(TCppEnum_t etype, TCppIndex_t idata)
// {
//     return ((TEnumConstant*)((TEnum*)etype)->GetConstants()->At((int)idata))->GetName();
// }
//
// long long Cppyy::GetEnumDataValue(TCppEnum_t etype, TCppIndex_t idata)
// {
//      TEnumConstant* ecst = (TEnumConstant*)((TEnum*)etype)->GetConstants()->At((int)idata);
//      return (long long)ecst->GetValue();
// }

Cppyy::TCppScope_t Cppyy::InstantiateTemplate(
             TCppScope_t tmpl, Cpp::TemplateArgInfo* args, size_t args_size)
{
  return Cpp::InstantiateTemplate(tmpl, args, args_size,
                                  /*instantiate_body=*/false);
}

void Cppyy::DumpScope(TCppScope_t scope)
{
    Cpp::DumpScope(scope);
}

//- C-linkage wrappers -------------------------------------------------------

extern "C" {
// direct interpreter access ----------------------------------------------
int cppyy_compile(const char* code) {
    return Cppyy::Compile(code);
}

int cppyy_compile_silent(const char* code) {
    return Cppyy::Compile(code, true /* silent */);
}

char* cppyy_to_string(cppyy_type_t klass, cppyy_object_t obj) {
    return cppstring_to_cstring(Cppyy::ToString((Cppyy::TCppType_t) klass, obj));
}


// name to opaque C++ scope representation --------------------------------
// char* cppyy_resolve_name(const char* cppitem_name) {
//     return cppstring_to_cstring(Cppyy::ResolveName(cppitem_name));
// }

// char* cppyy_resolve_enum(const char* enum_type) {
//     return cppstring_to_cstring(Cppyy::ResolveEnum(enum_type));
// }

cppyy_scope_t cppyy_get_scope(const char* scope_name) {
    return cppyy_scope_t(Cppyy::GetScope(scope_name));
}
//
// cppyy_type_t cppyy_actual_class(cppyy_type_t klass, cppyy_object_t obj) {
//     return cppyy_type_t(Cppyy::GetActualClass(klass, (void*)obj));
// }

size_t cppyy_size_of_klass(cppyy_type_t klass) {
    return Cppyy::SizeOf((Cppyy::TCppType_t) klass);
}

// size_t cppyy_size_of_type(const char* type_name) {
//     return Cppyy::SizeOf(type_name);
// }
//
// int cppyy_is_builtin(const char* type_name) {
//     return (int)Cppyy::IsBuiltin(type_name);
// }
//
// int cppyy_is_complete(const char* type_name) {
//     return (int)Cppyy::IsComplete(type_name);
// }
//
//
// [> memory management ------------------------------------------------------ <]
// cppyy_object_t cppyy_allocate(cppyy_scope_t type) {
//     return cppyy_object_t(Cppyy::Allocate(type));
// }
//
// void cppyy_deallocate(cppyy_scope_t type, cppyy_object_t self) {
//     Cppyy::Deallocate(type, (void*)self);
// }
//
// cppyy_object_t cppyy_construct(cppyy_type_t type) {
//     return (cppyy_object_t)Cppyy::Construct(type);
// }
//
// void cppyy_destruct(cppyy_type_t type, cppyy_object_t self) {
//     Cppyy::Destruct(type, (void*)self);
// }
//
//
// [> method/function dispatching -------------------------------------------- <]
// [> Exception types:
//     1: default (unknown exception)
//     2: standard exception
// */
// #define CPPYY_HANDLE_EXCEPTION                                               \
//     catch (std::exception& e) {                                              \
//         cppyy_exctype_t* etype = (cppyy_exctype_t*)((Parameter*)args+nargs); \
//         *etype = (cppyy_exctype_t)2;                                         \
//         *((char**)(etype+1)) = cppstring_to_cstring(e.what());               \
//     }                                                                        \
//     catch (...) {                                                            \
//         cppyy_exctype_t* etype = (cppyy_exctype_t*)((Parameter*)args+nargs); \
//         *etype = (cppyy_exctype_t)1;                                         \
//         *((char**)(etype+1)) =                                               \
//             cppstring_to_cstring("unhandled, unknown C++ exception");        \
//     }
//
// void cppyy_call_v(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         Cppyy::CallV(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
// }
//
// unsigned char cppyy_call_b(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         return (unsigned char)Cppyy::CallB(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
//     return (unsigned char)-1;
// }
//
// char cppyy_call_c(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         return (char)Cppyy::CallC(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
//     return (char)-1;
// }
//
// short cppyy_call_h(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         return (short)Cppyy::CallH(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
//     return (short)-1;
// }
//
// int cppyy_call_i(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         return (int)Cppyy::CallI(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
//     return (int)-1;
// }
//
// long cppyy_call_l(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         return (long)Cppyy::CallL(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
//     return (long)-1;
// }
//
// long long cppyy_call_ll(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         return (long long)Cppyy::CallLL(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
//     return (long long)-1;
// }
//
// float cppyy_call_f(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         return (float)Cppyy::CallF(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
//     return (float)-1;
// }
//
// double cppyy_call_d(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         return (double)Cppyy::CallD(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
//     return (double)-1;
// }
//
// long double cppyy_call_ld(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         return (long double)Cppyy::CallLD(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
//     return (long double)-1;
// }
//
// double cppyy_call_nld(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     return (double)cppyy_call_ld(method, self, nargs, args);
// }
//
// void* cppyy_call_r(cppyy_method_t method, cppyy_object_t self, int nargs, void* args) {
//     try {
//         return (void*)Cppyy::CallR(method, (void*)self, nargs, args);
//     } CPPYY_HANDLE_EXCEPTION
//     return (void*)nullptr;
// }
//
// char* cppyy_call_s(
//         cppyy_method_t method, cppyy_object_t self, int nargs, void* args, size_t* lsz) {
//     try {
//         return Cppyy::CallS(method, (void*)self, nargs, args, lsz);
//     } CPPYY_HANDLE_EXCEPTION
//     return (char*)nullptr;
// }
//
// cppyy_object_t cppyy_constructor(
//         cppyy_method_t method, cppyy_type_t klass, int nargs, void* args) {
//     try {
//         return cppyy_object_t(Cppyy::CallConstructor(method, klass, nargs, args));
//     } CPPYY_HANDLE_EXCEPTION
//     return (cppyy_object_t)0;
// }
//
// void cppyy_destructor(cppyy_type_t klass, cppyy_object_t self) {
//     Cppyy::CallDestructor(klass, self);
// }
//
// cppyy_object_t cppyy_call_o(cppyy_method_t method, cppyy_object_t self,
//         int nargs, void* args, cppyy_type_t result_type) {
//     try {
//         return cppyy_object_t(Cppyy::CallO(method, (void*)self, nargs, args, result_type));
//     } CPPYY_HANDLE_EXCEPTION
//     return (cppyy_object_t)0;
// }
//
// cppyy_funcaddr_t cppyy_function_address(cppyy_method_t method) {
//     return cppyy_funcaddr_t(Cppyy::GetFunctionAddress(method, true));
// }
//
//
// [> handling of function argument buffer ----------------------------------- <]
// void* cppyy_allocate_function_args(int nargs) {
// // for calls through C interface, require extra space for reporting exceptions
//     return malloc(nargs*sizeof(Parameter)+sizeof(cppyy_exctype_t)+sizeof(char**));
// }
//
// void cppyy_deallocate_function_args(void* args) {
//     free(args);
// }
//
// size_t cppyy_function_arg_sizeof() {
//     return (size_t)Cppyy::GetFunctionArgSizeof();
// }
//
// size_t cppyy_function_arg_typeoffset() {
//     return (size_t)Cppyy::GetFunctionArgTypeoffset();
// }


// scope reflection information ------------------------------------------- 
int cppyy_is_namespace(cppyy_scope_t scope) {
    return (int)Cppyy::IsNamespace((Cppyy::TCppScope_t) scope);
}

// int cppyy_is_template(const char* template_name) {
//     return (int)Cppyy::IsTemplate(template_name);
// }
//
// int cppyy_is_abstract(cppyy_type_t type) {
//     return (int)Cppyy::IsAbstract(type);
// }
//
// int cppyy_is_enum(const char* type_name) {
//     return (int)Cppyy::IsEnum(type_name);
// }
//
// int cppyy_is_aggregate(cppyy_type_t type) {
//     return (int)Cppyy::IsAggregate(type);
// }
//
// int cppyy_is_default_constructable(cppyy_type_t type) {
//     return (int)Cppyy::IsDefaultConstructable(type);
// }
//
// const char** cppyy_get_all_cpp_names(cppyy_scope_t scope, size_t* count) {
//     std::set<std::string> cppnames;
//     Cppyy::GetAllCppNames(scope, cppnames);
//     const char** c_cppnames = (const char**)malloc(cppnames.size()*sizeof(const char*));
//     int i = 0;
//     for (const auto& name : cppnames) {
//         c_cppnames[i] = cppstring_to_cstring(name);
//         ++i;
//     }
//     *count = cppnames.size();
//     return c_cppnames;
// }
//
//
// [> namespace reflection information --------------------------------------- <]
// cppyy_scope_t* cppyy_get_using_namespaces(cppyy_scope_t scope) {
//     const std::vector<Cppyy::TCppScope_t>& uv = Cppyy::GetUsingNamespaces((Cppyy::TCppScope_t)scope);
//
//     if (uv.empty())
//         return (cppyy_index_t*)nullptr;
//
//     cppyy_scope_t* llresult = (cppyy_scope_t*)malloc(sizeof(cppyy_scope_t)*(uv.size()+1));
//     for (int i = 0; i < (int)uv.size(); ++i) llresult[i] = uv[i];
//     llresult[uv.size()] = (cppyy_scope_t)0;
//     return llresult;
// }
//
//
// [> class reflection information ------------------------------------------- <]
// char* cppyy_final_name(cppyy_type_t type) {
//     return cppstring_to_cstring(Cppyy::GetFinalName(type));
// }
//
// char* cppyy_scoped_final_name(cppyy_type_t type) {
//     return cppstring_to_cstring(Cppyy::GetScopedFinalName(type));
// }
//
// int cppyy_has_virtual_destructor(cppyy_type_t type) {
//     return (int)Cppyy::HasVirtualDestructor(type);
// }
//
// int cppyy_has_complex_hierarchy(cppyy_type_t type) {
//     return (int)Cppyy::HasComplexHierarchy(type);
// }
//
// int cppyy_num_bases(cppyy_type_t type) {
//     return (int)Cppyy::GetNumBases(type);
// }
//
// char* cppyy_base_name(cppyy_type_t type, int base_index) {
//     return cppstring_to_cstring(Cppyy::GetBaseName (type, base_index));
// }
//
// int cppyy_is_subtype(cppyy_type_t derived, cppyy_type_t base) {
//     return (int)Cppyy::IsSubclass(derived, base);
// }

  int cppyy_is_smartptr(cppyy_type_t type) {
      return (int)Cppyy::IsSmartPtr((Cppyy::TCppType_t)type);
  }

// int cppyy_smartptr_info(const char* name, cppyy_type_t* raw, cppyy_method_t* deref) {
//     return (int)Cppyy::GetSmartPtrInfo(name, raw, deref);
// }
//
// void cppyy_add_smartptr_type(const char* type_name) {
//     Cppyy::AddSmartPtrType(type_name);
// }
//
// void cppyy_add_type_reducer(const char* reducable, const char* reduced) {
//     Cppyy::AddTypeReducer(reducable, reduced);
// }
//
//
// [> calculate offsets between declared and actual type, up-cast: direction > 0; down-cast: direction < 0 <]
// ptrdiff_t cppyy_base_offset(cppyy_type_t derived, cppyy_type_t base, cppyy_object_t address, int direction) {
//     return (ptrdiff_t)Cppyy::GetBaseOffset(derived, base, (void*)address, direction, 0);
// }
//
//
// [> method/function reflection information --------------------------------- <]
// int cppyy_num_methods(cppyy_scope_t scope) {
//     return (int)Cppyy::GetNumMethods(scope);
// }
//
// int cppyy_num_methods_ns(cppyy_scope_t scope) {
//     return (int)Cppyy::GetNumMethods(scope, true);
// }
//
// cppyy_index_t* cppyy_method_indices_from_name(cppyy_scope_t scope, const char* name)
// {
//     std::vector<cppyy_index_t> result = Cppyy::GetMethodIndicesFromName(scope, name);
//
//     if (result.empty())
//         return (cppyy_index_t*)nullptr;
//
//     cppyy_index_t* llresult = (cppyy_index_t*)malloc(sizeof(cppyy_index_t)*(result.size()+1));
//     for (int i = 0; i < (int)result.size(); ++i) llresult[i] = result[i];
//     llresult[result.size()] = -1;
//     return llresult;
// }
//
// cppyy_method_t cppyy_get_method(cppyy_scope_t scope, cppyy_index_t idx) {
//     return cppyy_method_t(Cppyy::GetMethod(scope, idx));
// }
//
// char* cppyy_method_name(cppyy_method_t method) {
//     return cppstring_to_cstring(Cppyy::GetMethodName((Cppyy::TCppMethod_t)method));
// }
//
// char* cppyy_method_full_name(cppyy_method_t method) {
//     return cppstring_to_cstring(Cppyy::GetMethodFullName((Cppyy::TCppMethod_t)method));
// }
//
// char* cppyy_method_mangled_name(cppyy_method_t method) {
//     return cppstring_to_cstring(Cppyy::GetMethodMangledName((Cppyy::TCppMethod_t)method));
// }
//
// char* cppyy_method_result_type(cppyy_method_t method) {
//     return cppstring_to_cstring(Cppyy::GetMethodReturnTypeAsString((Cppyy::TCppMethod_t)method));
// }
//
// int cppyy_method_num_args(cppyy_method_t method) {
//     return (int)Cppyy::GetMethodNumArgs((Cppyy::TCppMethod_t)method);
// }
//
// int cppyy_method_req_args(cppyy_method_t method) {
//     return (int)Cppyy::GetMethodReqArgs((Cppyy::TCppMethod_t)method);
// }
//
// char* cppyy_method_arg_name(cppyy_method_t method, int arg_index) {
//     return cppstring_to_cstring(Cppyy::GetMethodArgName((Cppyy::TCppMethod_t)method, (Cppyy::TCppIndex_t)arg_index));
// }
//
// char* cppyy_method_arg_type(cppyy_method_t method, int arg_index) {
//     return cppstring_to_cstring(Cppyy::GetMethodArgType((Cppyy::TCppMethod_t)method, (Cppyy::TCppIndex_t)arg_index));
// }
//
// char* cppyy_method_arg_default(cppyy_method_t method, int arg_index) {
//     return cppstring_to_cstring(Cppyy::GetMethodArgDefault((Cppyy::TCppMethod_t)method, (Cppyy::TCppIndex_t)arg_index));
// }
//
// char* cppyy_method_signature(cppyy_method_t method, int show_formalargs) {
//     return cppstring_to_cstring(Cppyy::GetMethodSignature((Cppyy::TCppMethod_t)method, (bool)show_formalargs));
// }
//
// char* cppyy_method_signature_max(cppyy_method_t method, int show_formalargs, int maxargs) {
//     return cppstring_to_cstring(Cppyy::GetMethodSignature((Cppyy::TCppMethod_t)method, (bool)show_formalargs, (Cppyy::TCppIndex_t)maxargs));
// }
//
// char* cppyy_method_prototype(cppyy_scope_t scope, cppyy_method_t method, int show_formalargs) {
//     return cppstring_to_cstring(Cppyy::GetMethodPrototype(
//         (Cppyy::TCppScope_t)scope, (Cppyy::TCppMethod_t)method, (bool)show_formalargs));
// }
//
// int cppyy_is_const_method(cppyy_method_t method) {
//     return (int)Cppyy::IsConstMethod((Cppyy::TCppMethod_t)method);
// }
//
// int cppyy_get_num_templated_methods(cppyy_scope_t scope) {
//     return (int)Cppyy::GetNumTemplatedMethods((Cppyy::TCppScope_t)scope);
// }
//
// int cppyy_get_num_templated_methods_ns(cppyy_scope_t scope) {
//     return (int)Cppyy::GetNumTemplatedMethods((Cppyy::TCppScope_t)scope, true);
// }
//
// char* cppyy_get_templated_method_name(cppyy_scope_t scope, cppyy_index_t imeth) {
//     return cppstring_to_cstring(Cppyy::GetTemplatedMethodName((Cppyy::TCppScope_t)scope, (Cppyy::TCppIndex_t)imeth));
// }
//
// int cppyy_is_templated_constructor(cppyy_scope_t scope, cppyy_index_t imeth) {
//     return Cppyy::IsTemplatedConstructor((Cppyy::TCppScope_t)scope, (Cppyy::TCppIndex_t)imeth);
// }
//
// int cppyy_exists_method_template(cppyy_scope_t scope, const char* name) {
//     return (int)Cppyy::ExistsMethodTemplate((Cppyy::TCppScope_t)scope, name);
// }
//
// int cppyy_method_is_template(cppyy_scope_t scope, cppyy_index_t idx) {
//     return (int)Cppyy::IsMethodTemplate((Cppyy::TCppScope_t)scope, idx);
// }
//
// cppyy_method_t cppyy_get_method_template(cppyy_scope_t scope, const char* name, const char* proto) {
//     return cppyy_method_t(Cppyy::GetMethodTemplate((Cppyy::TCppScope_t)scope, name, proto));
// }
//
// cppyy_index_t cppyy_get_global_operator(cppyy_scope_t scope, cppyy_scope_t lc, cppyy_scope_t rc, const char* op) {
//     return cppyy_index_t(Cppyy::GetGlobalOperator(scope, Cppyy::GetScopedFinalName(lc), Cppyy::GetScopedFinalName(rc), op));
// }
//
//
// [> method properties ------------------------------------------------------ <]
// int cppyy_is_publicmethod(cppyy_method_t method) {
//     return (int)Cppyy::IsPublicMethod((Cppyy::TCppMethod_t)method);
// }
//
// int cppyy_is_protectedmethod(cppyy_method_t method) {
//     return (int)Cppyy::IsProtectedMethod((Cppyy::TCppMethod_t)method);
// }
//
// int cppyy_is_constructor(cppyy_method_t method) {
//     return (int)Cppyy::IsConstructor((Cppyy::TCppMethod_t)method);
// }
//
// int cppyy_is_destructor(cppyy_method_t method) {
//     return (int)Cppyy::IsDestructor((Cppyy::TCppMethod_t)method);
// }
//
// int cppyy_is_staticmethod(cppyy_method_t method) {
//     return (int)Cppyy::IsStaticMethod((Cppyy::TCppMethod_t)method);
// }
//
//
// [> data member reflection information ------------------------------------- <]
// int cppyy_num_datamembers(cppyy_scope_t scope) {
//     return (int)Cppyy::GetNumDatamembers(scope);
// }
//
// int cppyy_num_datamembers_ns(cppyy_scope_t scope) {
//     return (int)Cppyy::GetNumDatamembers(scope, true);
// }
//
// char* cppyy_datamember_name(cppyy_scope_t scope, int datamember_index) {
//     return cppstring_to_cstring(Cppyy::GetDatamemberName(scope, datamember_index));
// }
//
// char* cppyy_datamember_type(cppyy_scope_t scope, int datamember_index) {
//     return cppstring_to_cstring(Cppyy::GetDatamemberType(scope, datamember_index));
// }
//
// intptr_t cppyy_datamember_offset(cppyy_scope_t scope, int datamember_index) {
//     return intptr_t(Cppyy::GetDatamemberOffset(scope, datamember_index));
// }
//
// int cppyy_datamember_index(cppyy_scope_t scope, const char* name) {
//     return (int)Cppyy::GetDatamemberIndex(scope, name);
// }
//
// int cppyy_datamember_index_enumerated(cppyy_scope_t scope, int datamember_index) {
//     return (int)Cppyy::GetDatamemberIndexEnumerated(scope, datamember_index);
// }
//
//
// [> data member properties ------------------------------------------------- <]
// int cppyy_is_publicdata(cppyy_type_t type, cppyy_index_t datamember_index) {
//     return (int)Cppyy::IsPublicData(type, datamember_index);
// }
//
// int cppyy_is_protecteddata(cppyy_type_t type, cppyy_index_t datamember_index) {
//     return (int)Cppyy::IsProtectedData(type, datamember_index);
// }
//
// int cppyy_is_staticdata(cppyy_type_t type, cppyy_index_t datamember_index) {
//     return (int)Cppyy::IsStaticData(type, datamember_index);
// }
//
// int cppyy_is_const_data(cppyy_scope_t scope, cppyy_index_t idata) {
//     return (int)Cppyy::IsConstData(scope, idata);
// }
//
// int cppyy_is_enum_data(cppyy_scope_t scope, cppyy_index_t idata) {
//     return (int)Cppyy::IsEnumData(scope, idata);
// }
//
// int cppyy_get_dimension_size(cppyy_scope_t scope, cppyy_index_t idata, int dimension) {
//     return Cppyy::GetDimensionSize(scope, idata, dimension);
// }
//
//
// [> enum properties -------------------------------------------------------- <]
// cppyy_enum_t cppyy_get_enum(cppyy_scope_t scope, const char* enum_name) {
//     return Cppyy::GetEnum(scope, enum_name);
// }
//
// cppyy_index_t cppyy_get_num_enum_data(cppyy_enum_t e) {
//     return Cppyy::GetNumEnumData(e);
// }
//
// const char* cppyy_get_enum_data_name(cppyy_enum_t e, cppyy_index_t idata) {
//     return cppstring_to_cstring(Cppyy::GetEnumDataName(e, idata));
// }
//
// long long cppyy_get_enum_data_value(cppyy_enum_t e, cppyy_index_t idata) {
//     return Cppyy::GetEnumDataValue(e, idata);
// }
//
//
// [> misc helpers ----------------------------------------------------------- <]
// RPY_EXTERN
// void* cppyy_load_dictionary(const char* lib_name) {
//     int result = gSystem->Load(lib_name);
//     return (void*)(result == 0 [> success */ || result == 1 /* already loaded <]);
// }
//
// #if defined(_MSC_VER)
// long long cppyy_strtoll(const char* str) {
//     return _strtoi64(str, NULL, 0);
// }
//
// extern "C" {
// unsigned long long cppyy_strtoull(const char* str) {
//     return _strtoui64(str, NULL, 0);
// }
// }
// #else
// long long cppyy_strtoll(const char* str) {
//     return strtoll(str, NULL, 0);
// }
//
// extern "C" {
// unsigned long long cppyy_strtoull(const char* str) {
//     return strtoull(str, NULL, 0);
// }
// }
// #endif
//
// void cppyy_free(void* ptr) {
//     free(ptr);
// }
//
// cppyy_object_t cppyy_charp2stdstring(const char* str, size_t sz) {
//     return (cppyy_object_t)new std::string(str, sz);
// }
//
// const char* cppyy_stdstring2charp(cppyy_object_t ptr, size_t* lsz) {
//     *lsz = ((std::string*)ptr)->size();
//     return ((std::string*)ptr)->data();
// }
//
// cppyy_object_t cppyy_stdstring2stdstring(cppyy_object_t ptr) {
//     return (cppyy_object_t)new std::string(*(std::string*)ptr);
// }
//
// double cppyy_longdouble2double(void* p) {
//     return (double)*(long double*)p;
// }
//
// void cppyy_double2longdouble(double d, void* p) {
//     *(long double*)p = d;
// }
//
// int cppyy_vectorbool_getitem(cppyy_object_t ptr, int idx) {
//     return (int)(*(std::vector<bool>*)ptr)[idx];
// }
//
// void cppyy_vectorbool_setitem(cppyy_object_t ptr, int idx, int value) {
//     (*(std::vector<bool>*)ptr)[idx] = (bool)value;
// }

} // end C-linkage wrappers
