#ifndef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
// silence warnings about getenv, strncpy, etc.
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

// Bindings
#include "cpp_cppyy.h"
#include "callcontext.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

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

#ifndef _WIN32
#include <dlfcn.h>
#endif

// Standard
#include <cassert>
#include <algorithm>     // for std::count, std::remove
#include <stdexcept>
#include <map>
#include <new>
#include <regex>
#include <set>
#include <sstream>
#include <csignal>
#include <cstdlib>      // for getenv
#include <cstring>
#include <typeinfo>
#include <iostream>
#include <vector>
#include <mutex>

// The backend must share one lock domain with the rest of ROOT: with
// ROOT::EnableThreadSafety(), code jitted through TCling (e.g. an RDataFrame
// event loop running in a different thread) serializes interpreter access via
// gInterpreterMutex. Guarding the CppInterOp calls only with a private mutex
// would still let them race with such threads on the one underlying
// cling::Interpreter. Therefore, also acquire gInterpreterMutex if it exists.
// The private recursive mutex still serializes concurrent Python-side callers
// when ROOT thread safety is not enabled. Lock order is global-then-local; no
// deadlock is possible since threads coming through TCling only ever take the
// global mutex.
class RInterOpMutex {
public:
    void lock()
    {
        // gInterpreterMutex may get created at any point (by
        // ROOT::EnableThreadSafety), so remember for each acquisition what was
        // actually locked, to release exactly that in unlock().
        TVirtualMutex* global = gInterpreterMutex;
        if (global) global->Lock();
        fLocal.lock();
        fGlobalLocked.push_back(global);
    }

    void unlock()
    {
        TVirtualMutex* global = fGlobalLocked.back();
        fGlobalLocked.pop_back();
        fLocal.unlock();
        if (global) global->UnLock();
    }

private:
    std::recursive_mutex fLocal;
    // Lock/unlock pairs are properly nested per thread, so a per-thread stack
    // suffices to match each unlock() to its lock().
    static thread_local std::vector<TVirtualMutex*> fGlobalLocked;
};

thread_local std::vector<TVirtualMutex*> RInterOpMutex::fGlobalLocked;

RInterOpMutex InterOpMutex;

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

//const int kMAXSIGNALS = 16;

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

#if 0
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
#endif

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
  Cppyy::TInterp_t Interp;
public:
    ApplicationStarter() {
        std::lock_guard<RInterOpMutex> Lock(InterOpMutex);

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
        if (auto existingInterp = Cpp::GetInterpreter()) {
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
        Cpp::AddIncludePath(CPPINTEROP_DIR);
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
                    << "void* gCling=(void*)" << Interp.data
                    << ";\n }}\n"
                    << "#endif \n";
        Cpp::Process(InterpPtrSS.str().c_str());

    // helper for multiple inheritance
        Cpp::Declare("namespace __cppyy_internal { struct Sep; }",
                     /*silent=*/false);

                         // retrieve all initial (ROOT) C++ names in the global scope to allow filtering later
        gROOT->GetListOfGlobals(true);             // force initialize
        gROOT->GetListOfGlobalFunctions(true);     // id.
        std::set<std::string> initial;
        Cppyy::GetAllCppNames(Cppyy::GetGlobalScope(), initial);
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


// local helpers -------------------------------------------------------------
static inline
char* cppstring_to_cstring(const std::string& cppstr)
{
    char* cstr = (char*)malloc(cppstr.size()+1);
    memcpy(cstr, cppstr.c_str(), cppstr.size()+1);
    return cstr;
}

// direct interpreter access -------------------------------------------------
// Returns false on failure and true on success
bool Cppyy::Compile(const std::string& code, bool silent)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    // Declare returns an enum which equals 0 on success
    return !Cpp::Declare(code.c_str(), silent);
}

std::string Cppyy::ToString(TCppScope_t klass, TCppObject_t obj)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    if (klass && obj && !Cpp::IsNamespace(klass))
        return Cpp::ObjToString(Cpp::GetQualifiedCompleteName(klass).c_str(), obj.data);
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

Cppyy::TCppType_t Cppyy::ResolveEnumReferenceType(TCppType_t type) {
std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
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
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
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

    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);

    TCppType_t check_int_typedefs = int_like_type(type);
    if (check_int_typedefs)
        return type;

    Cppyy::TCppType_t canonType = Cpp::GetCanonicalType(type);

    if (Cpp::IsEnumType(canonType)) {
        if (Cpp::GetTypeAsString(type) != "std::byte")
            return Cpp::GetIntegerTypeFromEnumType(canonType);
    }
    if (Cpp::HasTypeQualifier(canonType, Cpp::QualKind::Restrict)) {
        return Cpp::RemoveTypeQualifier(canonType, Cpp::QualKind::Restrict);
    }

    return canonType;
}

Cppyy::TCppType_t Cppyy::GetRealType(TCppType_t type) {
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    TCppType_t check_int_typedefs = int_like_type(type);
    if (check_int_typedefs)
        return check_int_typedefs;
    return Cpp::GetUnderlyingType(type);
}

Cppyy::TCppType_t Cppyy::GetPointerType(TCppType_t type) {
  std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
  return Cpp::GetPointerType(type);
}

Cppyy::TCppType_t Cppyy::GetReferencedType(TCppType_t type, bool rvalue) {
  std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
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

bool Cppyy::IsIntegerType(TCppType_t type, bool* is_signed /*= nullptr*/) {
  if (is_signed) {
    Cpp::Signedness sign;
    bool res = Cpp::IsIntegerType(type, &sign);
    *is_signed = (sign == Cpp::Signedness::kSigned);
    return res;
  }
  return Cpp::IsIntegerType(type, nullptr);
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
  int matching_angular_brackets = 0;
  while (end_pos < trimed_name.size()) {
    switch (trimed_name[end_pos]) {
    case ',': {
      if (!matching_angular_brackets) {
        if(end_pos > start_pos)
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

Cppyy::TCppScope_t GetEnumFromCompleteName(const std::string &name) {
  std::string delim = "::";
  size_t start = 0;
  size_t end = name.find(delim);
  Cppyy::TCppScope_t curr_scope;
  while (end != std::string::npos) {
    curr_scope = Cpp::GetNamed(name.substr(start, end - start), curr_scope);
    start = end + delim.length();
    end = name.find(delim, start);
  }
  return Cpp::GetNamed(name.substr(start, end), curr_scope);
}
static bool is_identifier(std::string_view s) {
  if (s.empty()) return false;
  auto is_valid_start = [](unsigned char c) {
    return std::isalpha(c) || c == '_';
  };
  auto is_valid_body = [](unsigned char c) {
    return std::isalnum(c) || c == '_';
  };
  return is_valid_start(s[0]) &&
    std::all_of(s.begin() + 1, s.end(), is_valid_body);
};

// returns true if no new type was added.
bool Cppyy::AppendTypesSlow(const std::string& name,
                            std::vector<Cpp::TemplateArgInfo>& types, Cppyy::TCppScope_t parent) {

  // Add no new type if string is empty
  if (name.empty())
    return true;

  // The ast printer gave us garbage.
  if (name == "<unnamed>")
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

  // If we have a single identifier, we don't need anything complicated.
  // Try scoped lookup first (catches type aliases / nested types declared
  // inside `parent`), then fall back to TU (catches typedefs declared
  // outside the query scope, e.g. `typedef Foo Bar;` at TU consulted
  // from a method on Foo).
  if (is_identifier(name)) {
    TCppType_t type = parent ? Cpp::GetType(name, parent) : nullptr;
    if (!type)
      type = Cpp::GetType(name);
    if (type) {
      types.emplace_back(type.data);
      return false;
    }
    if (!parent || parent == Cpp::GetGlobalScope())
      return true;
    // Fall through: the name may live in a scope enclosing `parent` (e.g.
    // "RVecF" from within ROOT::RDF::RInterface<...> resolves to
    // ROOT::RVecF); the trampoline below applies real unqualified lookup.
  }

  std::lock_guard<RInterOpMutex> Lock(InterOpMutex);

  // We might have an entire expression such as int, double.
  static unsigned long long struct_count = 0;
  std::string code = "template<typename ...T> struct __Cppyy_AppendTypesSlow {};\n";
  if (!struct_count)
    Cpp::Declare(code.c_str(), /*silent=*/true); // initialize the trampoline

  // Perform the lookup from within the innermost (reopenable) namespace
  // enclosing `parent` -- `parent` itself if it is a namespace, else the
  // namespace its class chain lives in -- by reopening that namespace around
  // the trampoline declaration: the compiler then applies real C++ name
  // lookup, walking outward with proper shadowing of outer names, which a
  // global declaration with a parent-qualified fallback cannot emulate (e.g.
  // "RVecF" from within ROOT::RDF::RInterface<...> resolves to ROOT::RVecF).
  std::string lookup_ns;
  TCppScope_t lookup_scope = nullptr;
  for (TCppScope_t s = parent; s && s != Cpp::GetGlobalScope();
       s = Cpp::GetParentScope(s)) {
    if (!Cpp::IsNamespace(s))
      continue;
    std::string qname = Cpp::GetQualifiedName(s);
    if (!qname.empty() && qname.find('(') == std::string::npos) {  // unnamed?
      lookup_ns = qname;
      lookup_scope = s;
    }
    break;
  }

  // The reopened namespace doesn't see names nested in a class parent, so a
  // name written relative to it (e.g. a nested type) won't resolve. Try the
  // name as given, then qualified by the parent.
  std::vector<std::string> candidates = {resolved_name};
  if (parent && parent != Cpp::GetGlobalScope() && parent != lookup_scope &&
      (Cppyy::IsNamespace(parent) || Cppyy::IsClass(parent)))
    candidates.push_back(Cpp::GetQualifiedCompleteName(parent) + "::" + resolved_name);

  for (const std::string& candidate : candidates) {
    std::string var = "__Cppyy_s" + std::to_string(struct_count++);
    std::string decl = "__Cppyy_AppendTypesSlow<" + candidate + "> " + var + ";\n";
    if (!lookup_ns.empty())
      decl = "namespace " + lookup_ns + " { " + decl + "}\n";
    if (!Cpp::Declare(decl.c_str(), /*silent=*/true)) {
      TCppType_t varN = Cpp::GetVariableType(Cpp::GetNamed(
          var.c_str(), lookup_ns.empty() ? nullptr : lookup_scope));
      TCppScope_t instance_class = Cpp::GetScopeFromType(varN);
      if (!instance_class)
        continue; // recovered-but-broken decl; try next candidate or split path
      size_t oldSize = types.size();
      Cpp::GetClassTemplateInstantiationArgs(instance_class, types);
      return oldSize == types.size();
    }
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

    if (!lookup_ns.empty()) {
      // resolve from within the parent namespace, honouring C++ name lookup
      // and shadowing (see the trampoline declaration above)
      std::string id = "__Cppyy_s" + std::to_string(struct_count++);
      if (!Cpp::Declare(("namespace " + lookup_ns + " { using " + id +
                         " = __typeof__(" + i + "); }\n").c_str(),
                        /*silent=*/true))
        type = Cpp::GetCanonicalType(
            Cpp::GetTypeFromScope(Cpp::GetNamed(id, lookup_scope)));
    } else {
      type = GetType(i, /*enable_slow_lookup=*/true);
      if (!type && parent && (Cppyy::IsNamespace(parent) || Cppyy::IsClass(parent))) {
        type = Cppyy::GetTypeFromScope(Cppyy::GetNamed(resolved_name, parent));
      }
    }

    if (!type) {
      types.clear();
      return true;
    }

    if (is_integral(i))
        integral_value = strdup(i.c_str());
    if (TCppScope_t scope = GetEnumFromCompleteName(i))
      if (Cpp::IsEnumConstant(scope))
        integral_value =
            strdup(std::to_string(Cpp::GetEnumConstantValue(scope)).c_str());
    types.emplace_back(type.data, integral_value);
  }
  return false;
}

Cppyy::TCppType_t Cppyy::GetType(const std::string &name, bool enable_slow_lookup /* = false */) {
    // The ast printer gave us garbage.
    if (name == "<unnamed>")
      return nullptr;
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);

    if (auto type = Cpp::GetType(name))
        return type;

    // Plain identifiers don't need the heavy __typeof__ trampoline:
    // Cpp::GetType above already covers builtin types and named
    // scopes. Exception: the three identifier-shaped C++ value-
    // literals -- `true`, `false`, `nullptr` -- aren't reachable by
    // name (no type called "false") but appear as non-type template
    // args in libstdc++ types like _Node_iterator<..., false, false>;
    // map them to their underlying type directly so the per-chunk
    // fallback in AppendTypesSlow gets a real type without paying
    // the trampoline cost.
    if (is_identifier(name)) {
      if (name == "true" || name == "false")
        return Cpp::GetType("bool");
      if (name == "nullptr")
        return Cpp::GetType("nullptr_t", Cpp::GetNamed("std"));
      return nullptr;
    }

    if (!enable_slow_lookup) {
        if (name.find("::") != std::string::npos)
            throw std::runtime_error("Calling Cppyy::GetType with qualified name '"
                                + name + "'\n");
        return nullptr;
    }

    // Here we might need to deal with integral types such as 3.14.

    static unsigned long long var_count = 0;
    std::string id = "__Cppyy_GetType_" + std::to_string(var_count++);
    std::string using_clause = "using " + id + " = __typeof__(" + name + ");\n";

    if (!Cpp::Declare(using_clause.c_str(), /*silent=*/true)) {
      TCppScope_t lookup = Cpp::GetNamed(id);
      TCppType_t lookup_ty = Cpp::GetTypeFromScope(lookup);
      return Cpp::GetCanonicalType(lookup_ty);
    }
    return nullptr;
}


Cppyy::TCppType_t Cppyy::GetComplexType(const std::string &name) {
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
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
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    std::string type = Cpp::GetTypeAsString(
        Cpp::GetIntegerTypeFromEnumScope(handle));
    if (type == "signed char")
        return "char";
    return type;
}

Cppyy::TCppScope_t Cppyy::GetUnderlyingScope(TCppScope_t scope)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetUnderlyingScope(scope);
}

Cppyy::TCppScope_t Cppyy::GetScope(const std::string& name,
                                   TCppScope_t parent_scope)
{
    std::unique_lock<RInterOpMutex> Lock(InterOpMutex);
// CppInterOp directly looks at the AST which is not enough.
// We require lazy module loading that ROOT relies on, so we do it here first.
// Use TClass::GetClass to trigger auto-loading of dictionaries and modules.
    if (!parent_scope || parent_scope == Cpp::GetGlobalScope())
        TClass::GetClass(name.c_str(), true /* load */, true /* silent */);

    if (Cppyy::TCppScope_t scope = Cpp::GetScope(name, parent_scope))
      return scope;
    if (!parent_scope || parent_scope == Cpp::GetGlobalScope()) {
      if (Cppyy::TCppScope_t scope = Cpp::GetScopeFromCompleteName(name))
        return scope;
    } else if (name.find('<') == std::string::npos) {
      // Qualified name relative to a non-global parent (e.g. "B::C" looked up
      // in namespace A): Cpp::GetScope only handles single identifiers, so
      // walk the components. Skip templated names: "::" inside template
      // arguments cannot be split naively (they take the branch below).
      TCppScope_t curr = parent_scope;
      size_t start = 0, end;
      while (curr && (end = name.find("::", start)) != std::string::npos) {
        curr = Cpp::GetScope(name.substr(start, end - start), curr);
        start = end + 2;
      }
      if (curr && curr != parent_scope)
        if (Cppyy::TCppScope_t scope = Cpp::GetScope(name.substr(start), curr))
          return scope;
    }

    // FIXME: avoid string parsing here
    if (name.find('<') != std::string::npos) {
      // Templated type; may need instantiation. Resolve the whole type
      // expression (e.g. "std::array<float, 3>") and read back its scope.
      // Splitting off the argument list and resolving it directly cannot
      // represent non-type arguments such as the `3` in std::array<float, 3>.
      std::vector<Cpp::TemplateArgInfo> types;
      Lock.unlock(); // unlock to allow AppendTypesSlow
      bool added_new_type = !Cppyy::AppendTypesSlow(name, types, /*parent=*/parent_scope);
      Lock.lock();
      if (added_new_type && types.size() == 1) {
        TCppScope_t scope = Cpp::GetScopeFromType(types[0].m_Type);
        // Naming the type as a template argument above does not instantiate
        // it, so the specialization may still be declared-but-undefined.
        // Force its definition: callers expect a complete scope, e.g. to
        // walk its base classes.
        if (scope)
          Cpp::IsComplete(scope);
        return scope;
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
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetScopeFromType(
        Cpp::GetVariableType(var));
}

Cppyy::TCppScope_t Cppyy::GetNamed(const std::string& name,
                                   TCppScope_t parent_scope)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    if (TCppScope_t named = Cpp::GetNamed(name, parent_scope))
        return named;

    // "gROOT" is a macro expanding to ROOT::GetROOT(); ROOT master exposed it
    // to cppyy as a TGlobalMappedFunction, a mechanism this backend does not
    // consult. Resolve it to the underlying variable instead (making sure it
    // has been initialized first).
    if (name == "gROOT" &&
        (!parent_scope || parent_scope == Cpp::GetGlobalScope())) {
        ROOT::GetROOT();
        return Cpp::GetNamed("gROOTLocal",
            Cpp::GetNamed("Internal", Cpp::GetNamed("ROOT", nullptr)));
    }

    return nullptr;
}

Cppyy::TCppScope_t Cppyy::GetParentScope(TCppScope_t scope)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetParentScope(scope);
}

Cppyy::TCppScope_t Cppyy::GetScopeFromType(TCppType_t type)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetScopeFromType(type);
}

Cppyy::TCppType_t Cppyy::GetTypeFromScope(TCppScope_t klass)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetTypeFromScope(klass);
}

Cppyy::TCppScope_t Cppyy::GetGlobalScope()
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
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
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);

    if (!obj || !Cpp::IsClassPolymorphic(klass))
        return klass;

// Do not autocast iostream classes (ostream, streambuf, etc.); it is usually
// unnecessary, and the RTTI probe below crashes on them: on MSVC they use
// virtual inheritance, which puts the vbptr - not a vfptr - at offset 0
// (seen with std::cout on the Windows CI); the old backend had the same
// filter, originally for a crash on Mac ARM.
    const std::string& clName = Cppyy::GetScopedFinalName(klass);
    if (clName.compare(0, 5, "std::") == 0 &&
            clName.find("stream") != std::string::npos)
        return klass;

    const std::type_info *typ = &typeid(*(AutoCastRTTI *)obj.data);
    if (!typ)
        return klass;

#ifdef _WIN32
    // MSVC's type_info::name() is already human-readable, but prefixed with
    // the tag kind (e.g. "class TWinNTSystem"), which name lookup would trip
    // over. Strip the prefix; template arguments keep their tags, which is
    // harmless as templated names go through type resolution instead of
    // plain lookup.
    std::string demangled_name = typ->name();
    for (const char* prefix : {"class ", "struct ", "union ", "enum "}) {
        if (demangled_name.compare(0, strlen(prefix), prefix) == 0) {
            demangled_name = demangled_name.substr(strlen(prefix));
            break;
        }
    }
#else
    std::string mangled_name = typ->name();
    std::string demangled_name = Cpp::Demangle(mangled_name);
#endif

    if (TCppScope_t scope = Cppyy::GetScope(demangled_name)) {
    // Only return the derived type if theres a complete definition in the
    // interpreter. internal classes like TCling have no public header and
    // no dictionary, so their CXXRecordDecl has no DefinitionData.
    // returning them crashes when querying offsets. Fall back to the base
    // type if the derived type is incomplete.
        if (Cpp::IsComplete(scope))
            return scope;
    }

    return klass;
}

size_t Cppyy::SizeOf(TCppScope_t klass)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::SizeOf(klass);
}

size_t Cppyy::SizeOfType(TCppType_t klass)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetSizeOfType(klass);
}

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
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::IsComplete(scope);
}

// // memory management ---------------------------------------------------------
Cppyy::TCppObject_t Cppyy::Allocate(TCppScope_t scope)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::Allocate(scope, /*count=*/1);
}

void Cppyy::Deallocate(TCppScope_t scope, TCppObject_t instance)
{
  std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
  Cpp::Deallocate(scope, instance, /*count=*/1);
}

Cppyy::TCppObject_t Cppyy::Construct(TCppScope_t scope, void* arena/*=nullptr*/)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex); // TODO: this shouldn't locks the JIT call
    return Cpp::Construct(scope, arena, /*count=*/1);
}

void Cppyy::Destruct(TCppScope_t scope, TCppObject_t instance)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);  // TODO: this shouldn't locks the JIT call
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

static inline
bool WrapperCall(Cppyy::TCppMethod_t method, size_t nargs, void* args_, void* self, void* result)
{
    Parameter* args = (Parameter*)args_;
    //bool is_direct = nargs & DIRECT_CALL;
    nargs = CALL_NARGS(nargs);

    static const bool traceCalls = getenv("CPPYY_TRACE_CALLS") != nullptr;
    if (traceCalls) {
        static std::atomic<int> callCounter{0};
        fprintf(stderr, "[JITCALL %d] %s%s\n", ++callCounter,
                Cppyy::GetScopedFinalName(Cppyy::TCppScope_t(method.data)).c_str(),
                Cppyy::GetMethodSignature(method, false).c_str());
        fflush(stderr);
    }

    // if (!is_ready(wrap, is_direct))
    //     return false;        // happens with compilation error
    InterOpMutex.lock();
    if (Cpp::JitCall JC = Cpp::MakeFunctionCallable(method)) {
        InterOpMutex.unlock();
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
    InterOpMutex.unlock();
    return false;
}

template<typename T>
static inline
T CallT(Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, size_t nargs, void* args)
{
    T t{};
    if (WrapperCall(method, nargs, args, self.data, &t))
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
    if (!WrapperCall(method, nargs, args, self.data, nullptr))
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
    if (WrapperCall(method, nargs, args, self.data, &r))
        return r;
    return nullptr;
}

char* Cppyy::CallS(
    TCppMethod_t method, TCppObject_t self, size_t nargs, void* args, size_t* length)
{
    char* cstr = nullptr;
    // TClassRef cr("std::string"); // TODO: Why is this required?
    std::string* cppresult = (std::string*)malloc(sizeof(std::string));
    if (WrapperCall(method, nargs, args, self.data, (void*)cppresult)) {
        cstr = cppstring_to_cstring(*cppresult);
        *length = cppresult->size();
        cppresult->std::string::~basic_string();
    } else
        *length = 0;
    free((void*)cppresult);
    return cstr;
}

Cppyy::TCppObject_t Cppyy::CallConstructor(
    TCppMethod_t method, TCppScope_t /*klass*/, size_t nargs, void* args)
{
    void* obj = nullptr;
    WrapperCall(method, nargs, args, nullptr, &obj);
    return (TCppObject_t)obj;
}

void Cppyy::CallDestructor(TCppScope_t scope, TCppObject_t self)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex); // TODO: this shouldn't locks the JIT call
    Cpp::Destruct(self, scope, /*withFree=*/false, /*count=*/0);
}

Cppyy::TCppObject_t Cppyy::CallO(TCppMethod_t method,
    TCppObject_t self, size_t nargs, void* args, TCppType_t result_type)
{
    void* obj = ::operator new(Cppyy::SizeOfType(result_type));
    if (WrapperCall(method, nargs, args, self.data, obj))
        return (TCppObject_t)obj;
    ::operator delete(obj);
    return TCppObject_t{};
}

Cppyy::TCppFuncAddr_t Cppyy::GetFunctionAddress(TCppMethod_t method, bool /*check_enabled*/)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetFunctionAddress(method);
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
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
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
  return Cpp::IsEnumConstant(Cppyy::GetUnderlyingScope(scope));
}

bool Cppyy::IsEnumType(TCppType_t type)
{
    return Cpp::IsEnumType(type);
}

bool Cppyy::IsAggregate(TCppScope_t type)
{
  // Test if this type is a "plain old data" type
  return Cpp::IsAggregate(type);
}

bool Cppyy::IsDefaultConstructable(TCppScope_t scope)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
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
//         else if (tpl_open == 0 && c == ':' && pos+1 < name.size() && name[pos+1] == ':') {
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
#if 0
#define FILL_COLL(type, filter) {                                             \
    TIter itr{coll};                                                          \
    type* obj = nullptr;                                                      \
    while ((obj = (type*)itr.Next())) {                                       \
        const char* nm = obj->GetName();                                      \
        if (nm && nm[0] != '_' && !(obj->Property() & (filter))) {            \
            if (gInitialNames.find(nm) == gInitialNames.end())                \
                cppnames.insert(nm);                                          \
    }}}
#endif
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
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    Cpp::GetAllCppNames(scope, cppnames);
}

// class reflection information ----------------------------------------------
std::vector<Cppyy::TCppScope_t> Cppyy::GetUsingNamespaces(TCppScope_t scope)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetUsingNamespaces(scope);
}

// Normalize a type or scope name to cppyy's canonical form: no space after
// the commas separating template arguments, and pointers/references attached
// to the type. This is the form CPyCppyy itself constructs (e.g. when
// looking up cached template instantiations by name, see
// Utility::ConstructTemplateArgs) and the convention that user code and the
// test suite inherited from upstream cppyy; clang's printer instead emits
// "a, b", "T *" and "T &".
static std::string cppyy_normalize_name(std::string name)
{
    std::string::size_type pos = 0;
    while ((pos = name.find(", ", pos)) != std::string::npos)
        name.erase(pos + 1, 1);
    pos = 0;
    while ((pos = name.find(" *", pos)) != std::string::npos)
        name.erase(pos, 1);
    pos = 0;
    while ((pos = name.find(" &", pos)) != std::string::npos)
        name.erase(pos, 1);
    return name;
}

// class reflection information ----------------------------------------------
std::string Cppyy::GetFinalName(TCppScope_t klass)
{
  std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
  return cppyy_normalize_name(
      Cpp::GetCompleteName(Cpp::GetUnderlyingScope(klass)));
}

std::string Cppyy::GetScopedFinalName(TCppScope_t klass)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return cppyy_normalize_name(Cpp::GetQualifiedCompleteName(klass));
}

bool Cppyy::HasVirtualDestructor(TCppScope_t scope)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    TCppMethod_t func = Cpp::GetDestructor(scope);
    return Cpp::IsVirtualMethod(func);
}

Cppyy::TCppIndex_t Cppyy::GetNumBases(TCppScope_t klass)
{
// Get the total number of base classes that this class has.
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
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

std::string Cppyy::GetBaseName(TCppScope_t klass, TCppIndex_t ibase)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetName(Cpp::GetBaseClass(klass, ibase));
}

Cppyy::TCppScope_t Cppyy::GetBaseScope(TCppScope_t klass, TCppIndex_t ibase)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetBaseClass(klass, ibase);
}

bool Cppyy::IsSubclass(TCppScope_t derived, TCppScope_t base)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
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
    {
        std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
        Cpp::GetOperator(scope, Cpp::Operator::OP_Arrow, ops,
                         /*kind=*/Cpp::OperatorArity::kBoth);
    }
    if (ops.size() != 1)
        return false;

    if (deref) *deref = ops[0];
    if (raw) *raw = Cppyy::GetScopeFromType(Cppyy::GetMethodReturnType(ops[0]));
    return (!deref || *deref) && (!raw || *raw);
}

// type offsets --------------------------------------------------------------
ptrdiff_t Cppyy::GetBaseOffset(TCppScope_t derived, TCppScope_t base,
    TCppObject_t /*address*/, int direction, bool rerror)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    // Either base or derived class is incomplete, treat silently
    if (!Cpp::IsComplete(derived) || !Cpp::IsComplete(base))
        return rerror ? (ptrdiff_t)-1 : 0;

    intptr_t offset = Cpp::GetBaseClassOffset(derived, base);
    
    if (offset == -1)   // Cling error, treat silently
        return rerror ? (ptrdiff_t)offset : 0;

    return (ptrdiff_t)(direction < 0 ? -offset : offset);
}

// method/function reflection information ------------------------------------
// deleted functions are not callable and must not enter Python-visible
// overload sets: a deleted overload would add a bogus conversion error to
// every failed call report and break the "all failures raised the same C++
// exception" consolidation in Utility::SetDetailedException
static void remove_deleted_methods(std::vector<Cppyy::TCppMethod_t>& methods)
{
    methods.erase(std::remove_if(methods.begin(), methods.end(),
        [](Cppyy::TCppMethod_t m) { return Cpp::IsFunctionDeleted(m); }),
        methods.end());
}

void Cppyy::GetClassMethods(TCppScope_t scope, std::vector<Cppyy::TCppMethod_t> &methods)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    Cpp::GetClassMethods(scope, methods);
    remove_deleted_methods(methods);
}

std::vector<Cppyy::TCppMethod_t> Cppyy::GetMethodsFromName(
    TCppScope_t scope, const std::string& name)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    std::vector<Cppyy::TCppMethod_t> methods =
        Cpp::GetFunctionsUsingName(scope, name);
    remove_deleted_methods(methods);
    return methods;
}

std::string Cppyy::GetName(TCppScope_t method)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetName(method);
}

std::string Cppyy::GetFullName(TCppScope_t method)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return cppyy_normalize_name(Cpp::GetCompleteName(method));
}

Cppyy::TCppType_t Cppyy::GetMethodReturnType(TCppMethod_t method)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetFunctionReturnType(method);
}

std::string Cppyy::GetMethodReturnTypeAsString(TCppMethod_t method)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return 
    Cpp::GetTypeAsString(
        Cpp::GetCanonicalType(
            Cpp::GetFunctionReturnType(method)));
}

Cppyy::TCppIndex_t Cppyy::GetMethodNumArgs(TCppMethod_t method)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetFunctionNumArgs(method);
}

Cppyy::TCppIndex_t Cppyy::GetMethodReqArgs(TCppMethod_t method)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetFunctionRequiredArgs(method);
}

std::string Cppyy::GetMethodArgName(TCppMethod_t method, TCppIndex_t iarg)
{
    if (!method)
        return "<unknown>";

    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetFunctionArgName(method, iarg);
}

Cppyy::TCppType_t Cppyy::GetMethodArgType(TCppMethod_t method, TCppIndex_t iarg)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetFunctionArgType(method, iarg);
}

std::string Cppyy::GetMethodArgTypeAsString(TCppMethod_t method, TCppIndex_t iarg)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetTypeAsString(Cpp::RemoveTypeQualifier(
      Cpp::GetFunctionArgType(method, iarg), Cpp::QualKind::Const));
}

std::string Cppyy::GetMethodArgCanonTypeAsString(TCppMethod_t method, TCppIndex_t iarg)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return
    Cpp::GetTypeAsString(
        Cpp::GetCanonicalType(
            Cpp::GetFunctionArgType(method, iarg)));
}

std::string Cppyy::GetMethodArgDefault(TCppMethod_t method, TCppIndex_t iarg)
{
    if (!method)
       return "";

    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetFunctionArgDefault(method, iarg);
}

Cppyy::TCppIndex_t Cppyy::CompareMethodArgType(TCppMethod_t /*method*/, TCppIndex_t /*iarg*/, const std::string & /*req_type*/)
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
        sig << cppyy_normalize_name(Cppyy::GetMethodArgTypeAsString(method, iarg));
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

Cppyy::TCppType_t Cppyy::GetFnTypeFromStdFn(TCppType_t fn_type) {
    fn_type = Cpp::IsReferenceType(fn_type) ? Cpp::GetNonReferenceType(fn_type) : fn_type;
    fn_type = Cpp::IsPointerType(fn_type) ? Cpp::GetPointeeType(fn_type) : fn_type;
    TCppScope_t scope = Cpp::GetScopeFromType(fn_type);
    std::vector<Cpp::TemplateArgInfo> args;
    Cpp::GetClassTemplateArgs(scope, args);
    assert(args.size() == 1);
    if (args.size() == 1)
        return args[0].m_Type;
    return nullptr;
}

void Cppyy::GetFnTypeSig(TCppType_t fn_type, std::vector<TCppType_t>& arg_types) {
    fn_type = Cpp::IsReferenceType(fn_type) ? Cpp::GetNonReferenceType(fn_type) : fn_type;
    fn_type = Cpp::IsPointerType(fn_type) ? Cpp::GetPointeeType(fn_type) : fn_type;
    Cpp::GetFnTypeSignature(fn_type, arg_types);
}

bool Cppyy::IsSameType(TCppType_t typ1, TCppType_t typ2) {
    return Cpp::IsSameType(typ1, typ2);
}

bool Cppyy::IsFunctionType(TCppType_t typ) {
    typ = Cpp::IsReferenceType(typ) ? Cpp::GetNonReferenceType(typ) : typ;
    typ = Cpp::IsPointerType(typ) ? Cpp::GetPointeeType(typ) : typ;
    return Cpp::IsFunctionProtoType(typ);
}

bool Cppyy::IsSimilarFnTypes(TCppType_t typ1, TCppType_t typ2) {
    typ1 = Cpp::IsReferenceType(typ1) ? Cpp::GetNonReferenceType(typ1) : typ1;
    typ2 = Cpp::IsReferenceType(typ2) ? Cpp::GetNonReferenceType(typ2) : typ2;
    typ1 = Cpp::IsPointerType(typ1) ? Cpp::GetPointeeType(typ1) : typ1;
    typ2 = Cpp::IsPointerType(typ2) ? Cpp::GetPointeeType(typ2) : typ2;
    return Cpp::IsSameType(typ1, typ2);
}

std::string Cppyy::GetDoxygenComment(TCppScope_t scope, bool strip_markers)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetDoxygenComment(scope, strip_markers);
}

bool Cppyy::IsConstMethod(TCppMethod_t method)
{
    if (!method)
        return false;
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::IsConstMethod(method);
}

void Cppyy::GetTemplatedMethods(TCppScope_t scope, std::vector<Cppyy::TCppMethod_t> &methods)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    Cpp::GetFunctionTemplatedDecls(scope, methods);
}

Cppyy::TCppIndex_t Cppyy::GetNumTemplatedMethods(TCppScope_t scope, bool /*accept_namespace*/)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    std::vector<Cppyy::TCppMethod_t> mc;
    Cpp::GetFunctionTemplatedDecls(scope, mc);
    return mc.size();
}

std::string Cppyy::GetTemplatedMethodName(TCppScope_t scope, TCppIndex_t imeth)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    std::vector<Cppyy::TCppMethod_t> mc;
    Cpp::GetFunctionTemplatedDecls(scope, mc);

    if (imeth < mc.size()) return Cpp::GetName(TCppScope_t(mc[imeth].data));

    return "";
}

bool Cppyy::ExistsMethodTemplate(TCppScope_t scope, const std::string& name)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::ExistsFunctionTemplate(name, scope);
}

bool Cppyy::IsTemplatedMethod(TCppMethod_t method)
{
    return Cpp::IsTemplatedFunction(method);
}

bool Cppyy::IsStaticTemplate(TCppScope_t scope, const std::string& name)
{
    std::vector<TCppMethod_t> candidate_methods;
    Cpp::GetClassTemplatedMethods(name, scope, candidate_methods);
    bool is_static = true;
    for (auto i: candidate_methods) {
        if (!Cpp::IsStaticMethod(i)) {
            is_static = false;
            break;
        }
    }
    return is_static;
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
    
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);

    std::vector<Cppyy::TCppMethod_t> unresolved_candidate_methods;
    Cpp::GetClassTemplatedMethods(pureName, scope, unresolved_candidate_methods);
    if (unresolved_candidate_methods.empty() && name.find("operator") == 0) {
        // try operators
        Cppyy::GetClassOperators(scope, pureName, unresolved_candidate_methods);
    }

    // CPyCppyy assumes that we attempt instantiation here
    std::vector<Cpp::TemplateArgInfo> arg_types;
    std::vector<Cpp::TemplateArgInfo> templ_params;
    Cppyy::AppendTypesSlow(proto, arg_types, scope);
    Cppyy::AppendTypesSlow(explicit_params, templ_params, scope);
    Cppyy::TCppMethod_t cppmeth = nullptr;
    cppmeth = Cpp::BestOverloadFunctionMatch(
        unresolved_candidate_methods, templ_params, arg_types);

    // If overload resolution failed but explicit template arguments were
    // supplied, fall back to direct template-argument substitution: ask Sema
    // to instantiate each candidate with the explicit args. Sema's SFINAE
    // rejects overloads whose substitution fails (e.g. the initializer_list
    // form of std::make_any with non-init-list explicit args), so iterating
    // gives back exactly the viable specialisation. The wrapper-side argument
    // conversion then handles e.g. taking the address of an instance when the
    // substituted parameter is a pointer.
    if (!cppmeth && !templ_params.empty()) {
        for (const auto& cand : unresolved_candidate_methods) {
            if (Cpp::DeclRef spec = Cpp::InstantiateTemplate(
                    TCppScope_t(cand.data), templ_params.data(),
                    templ_params.size(), /*instantiate_body=*/false)) {
                cppmeth = spec.data;
                break;
            }
        }
    }

    return TCppMethod_t(cppmeth.data);
    // if it fails, use Sema to propogate info about why it failed (DeductionInfo)
}

static inline bool is_basic_string_of(const std::string& n, const char* charT) {
    // Match "std::basic_string<charT" whether or not the default template
    // arguments (char_traits, allocator) are spelled out; the character type
    // must be followed by ',' or '>' so that e.g. "char" does not match
    // "char16_t".
    std::string prefix = "std::basic_string<";
    prefix += charT;
    if (n.compare(0, prefix.size(), prefix) != 0)
        return false;
    return n.size() > prefix.size() && (n[prefix.size()] == ',' || n[prefix.size()] == '>');
}

static inline std::string type_remap(const std::string& n1,
                                     const std::string& n2) {
    // Operator lookups of (C++ string, Python str) should succeed for the
    // combos of string/str, wstring/str, string/unicode and wstring/unicode;
    // since C++ does not have a operator+(std::string, std::wstring), we'll
    // have to look up the same type and rely on the converters in
    // CPyCppyy/_cppyy.
    if (n1 == "str" || n1 == "unicode" || is_basic_string_of(n1, "char")) {
        if (is_basic_string_of(n2, "wchar_t"))
            return "std::basic_string<wchar_t>&";                      // match like for like
        return "std::basic_string<char>&"; // probably best bet
    } else if (is_basic_string_of(n1, "wchar_t")) {
        return "std::basic_string<wchar_t>&";
    } else if (n1 == "complex") {
        return "std::complex<double>";
    }
    return n1;
}

void Cppyy::GetClassOperators(Cppyy::TCppScope_t klass,
                              const std::string& opname,
                              std::vector<TCppMethod_t>& operators) {
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    std::string op = opname.substr(8);
    Cpp::GetOperator(klass, Cpp::GetOperatorFromSpelling(op), operators,
                     /*kind=*/Cpp::OperatorArity::kBoth);
}

Cppyy::TCppMethod_t Cppyy::GetGlobalOperator(
    TCppScope_t scope, const std::string& lc, const std::string& rc, const std::string& opname)
{
    std::string rc_type = type_remap(rc, lc);
    std::string lc_type = type_remap(lc, rc);

    std::vector<TCppMethod_t> overloads;
    Cpp::GetOperator(scope, Cpp::GetOperatorFromSpelling(opname), overloads,
                     /*kind=*/Cpp::OperatorArity::kBoth);

    // Avoid pushing nullptr into arg_types which would crash
    // BestOverloadFunctionMatch when it dereferences each entry's QualType.
    auto resolve_arg_type = [](const std::string& name) -> Cppyy::TCppType_t {
        if (auto s = Cppyy::GetScope(name))
            if (auto t = Cppyy::GetTypeFromScope(s))
                return Cppyy::GetReferencedType(t);
        return Cppyy::GetType(name, /*enable_slow_lookup=*/true);
    };
    
    std::vector<Cpp::TemplateArgInfo> arg_types;
    if (auto l = resolve_arg_type(lc_type))
        arg_types.emplace_back(l.data);
    else
        return nullptr;

    if (!rc_type.empty()) {
        if (auto r = resolve_arg_type(rc_type))
            arg_types.emplace_back(r.data);
        else
            return nullptr;
    }
    Cppyy::TCppMethod_t cppmeth = Cpp::BestOverloadFunctionMatch(
        overloads, {}, arg_types);
    if (cppmeth)
        return cppmeth;
    return nullptr;
}

// method properties ---------------------------------------------------------
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

// data member reflection information ----------------------------------------
void Cppyy::GetDatamembers(TCppScope_t scope, std::vector<TCppScope_t>& datamembers)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    Cpp::GetDatamembers(scope, datamembers);
    Cpp::GetStaticDatamembers(scope, datamembers);
    Cpp::GetEnumConstantDatamembers(scope, datamembers, false);
}

bool Cppyy::CheckDatamember(TCppScope_t scope, const std::string& name) {
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return (bool) Cpp::LookupDatamember(name, scope);
}

bool Cppyy::IsLambdaClass(TCppType_t type) {
    return Cpp::IsLambdaClass(type);
}

Cppyy::TCppScope_t Cppyy::WrapLambdaFromVariable(TCppScope_t var) {
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
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

Cppyy::TCppMethod_t Cppyy::AdaptFunctionForLambdaReturn(Cppyy::TCppMethod_t fn) {
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);

    std::string fn_name = Cpp::GetQualifiedCompleteName(TCppScope_t(fn.data));
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
      if (res) return TCppMethod_t(res.data);
    }
    return fn;
}

Cppyy::TCppType_t Cppyy::GetDatamemberType(TCppScope_t var)
{
  std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
  return Cpp::GetVariableType(Cpp::GetUnderlyingScope(var));
}

std::string Cppyy::GetDatamemberTypeAsString(TCppScope_t scope)
{
  std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
  return Cpp::GetTypeAsString(
      Cpp::GetVariableType(Cpp::GetUnderlyingScope(scope)));
}

std::string Cppyy::GetTypeAsString(TCppType_t type)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetTypeAsString(type);
}

intptr_t Cppyy::GetDatamemberOffset(TCppScope_t var, TCppScope_t klass)
{
  std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
  return Cpp::GetVariableOffset(Cpp::GetUnderlyingScope(var), klass);
}

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
  return Cpp::IsStaticVariable(Cppyy::GetUnderlyingScope(var));
}

bool Cppyy::IsConstVar(TCppScope_t var)
{
    return Cpp::IsConstVariable(var);
}

Cppyy::TCppMethod_t Cppyy::ReduceReturnType(TCppMethod_t fn, TCppType_t reduce) {
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);

    std::string fn_name = Cpp::GetQualifiedCompleteName(TCppScope_t(fn.data));
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
      if (res) return TCppMethod_t(res.data);
    }
    return fn;
}

std::vector<long int>  Cppyy::GetDimensions(TCppType_t type)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetDimensions(type);
}

// enum properties -----------------------------------------------------------
std::vector<Cppyy::TCppScope_t> Cppyy::GetEnumConstants(TCppScope_t scope)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetEnumConstants(scope);
}

Cppyy::TCppType_t Cppyy::GetEnumConstantType(TCppScope_t scope)
{
  std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
  return Cpp::GetEnumConstantType(Cpp::GetUnderlyingScope(scope));
}

Cppyy::TCppIndex_t Cppyy::GetEnumDataValue(TCppScope_t scope)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::GetEnumConstantValue(scope);
}

Cppyy::TCppScope_t Cppyy::InstantiateTemplate(
             TCppScope_t tmpl, Cpp::TemplateArgInfo* args, size_t args_size)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    return Cpp::InstantiateTemplate(tmpl, args, args_size,
                                    /*instantiate_body=*/false);
}

void Cppyy::DumpScope(TCppScope_t scope)
{
    std::lock_guard<RInterOpMutex> Lock(InterOpMutex);
    Cpp::DumpScope(scope);
}
