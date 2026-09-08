#ifndef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
// silence warnings about getenv, strncpy, etc.
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "precommondefs.h" // This defines several system feature macros and should be included before any system header.

// Bindings
#include "cppjit_interop.h"

using namespace cppjit;
#include "callcontext.h"

typedef cppjit::cpyrt::Parameter Parameter;

static inline size_t CALL_NARGS(size_t nargs) { return nargs & ~DIRECT_CALL; }

#ifndef _WIN32
#include <dlfcn.h>
#endif

// Standard
#include <algorithm> // for std::count, std::remove
#include <cassert>
#include <csignal>
#include <cstdlib> // for getenv
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <new>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <typeinfo>
#include <vector>

std::recursive_mutex InterOpMutex;

// builtin types
static std::set<std::string> g_builtins = {"bool",
                                           "char",
                                           "signed char",
                                           "unsigned char",
                                           "wchar_t",
                                           "short",
                                           "unsigned short",
                                           "int",
                                           "unsigned int",
                                           "long",
                                           "unsigned long",
                                           "long long",
                                           "unsigned long long",
                                           "float",
                                           "double",
                                           "long double",
                                           "void"};

// configuration
static bool gEnableFastPath = true;

// global initialization -----------------------------------------------------
namespace {

static inline bool is_integral(std::string& s) {
  if (s == "false") {
    s = "0";
    return true;
  } else if (s == "true") {
    s = "1";
    return true;
  }
  return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

struct InterOpPaths {
  std::string Library;
  std::vector<std::string> IncludeDirs;
  std::string ClangIncludeDir; // empty when no usable resource dir is known
};

// One set of coordinates, two anchors: prefer CppInterOp next to our own
// load location so wheels relocate; fall back to the build-time install
// prefix. An absolute coordinate (an external CppInterOp) replaces the
// anchor in the join ([fs.path.append]).
static InterOpPaths cppinterop_paths() {
  std::filesystem::path anchor = CPPINTEROP_INSTALL_PREFIX;
#ifndef _WIN32
  Dl_info info;
  if (dladdr((void*)&cppinterop_paths, &info) && info.dli_fname) {
    const std::filesystem::path here =
        std::filesystem::path(info.dli_fname).parent_path();
    std::error_code ec;
    if (std::filesystem::exists(here / CPPINTEROP_LIBRARY, ec))
      anchor = here;
  }
#endif
  InterOpPaths Paths;
  Paths.Library = (anchor / CPPINTEROP_LIBRARY).string();
  // The include coordinate may carry several ':'-separated directories (an
  // external CppInterOp build tree splits source and generated headers).
  std::istringstream includeSpec{CPPINTEROP_INCLUDE_DIR};
  for (std::string dir; std::getline(includeSpec, dir, ':');)
    if (!dir.empty())
      Paths.IncludeDirs.push_back((anchor / dir).string());
  // A bundled install ships the build clang's builtin headers (see the
  // CMake install rule); an empty coordinate or a missing directory falls
  // back to resource-dir detection.
  const std::string clangSpec = CPPJIT_CLANG_INCLUDE_DIR;
  if (!clangSpec.empty()) {
    const std::filesystem::path bundled = anchor / clangSpec;
    std::error_code ec;
    if (std::filesystem::exists(bundled / "include", ec))
      Paths.ClangIncludeDir = bundled.string();
  }
  return Paths;
}

// The one place libclangCppInterOp is dlopen'd.
static bool loadDispatchAPI(const InterOpPaths& Paths) {
  if (!Cpp::LoadDispatchAPI(Paths.Library.c_str())) {
    std::cerr << "[cppjit] Failed to load CppInterOp" << std::endl;
    return false;
  }
  return true;
}

// CppInterOp itself appends CPPINTEROP_EXTRA_INTERPRETER_ARGS inside
// CreateInterpreter, so nothing needs to be forwarded from here.
static interop::TInterp_t
acquireOrCreateInterpreter(const InterOpPaths& Paths) {
  if (auto existingInterp = Cpp::GetInterpreter())
    return existingInterp;

  std::vector<const char*> args = {"-std=c++17"};
#if !(defined(__arm64__) && defined(__APPLE__))
  // apple silicon clang rejects -march=native
  args.push_back("-march=native");
#endif
  // Without clang's builtin headers the interpreter fails at its first
  // #include. Prefer the bundled copy: it matches the build clang and
  // needs no LLVM on the host. DetectResourceDir refuses version
  // mismatches, and CppInterOp itself probes only bare `clang`.
  std::string resourceDir = Paths.ClangIncludeDir;
  if (resourceDir.empty())
    resourceDir = Cpp::DetectResourceDir("clang-" CPPJIT_CLANG_MAJOR);
  if (!resourceDir.empty()) {
    args.push_back("-resource-dir");
    args.push_back(resourceDir.c_str());
  }
  return Cpp::CreateInterpreter(args, /*GpuArgs=*/{});
}

static void configureInterpreter(const InterOpPaths& Paths) {
  std::set<std::string> bi{g_builtins};
  for (const auto& name : bi) {
    for (const char* a : {"*", "&", "*&", "[]", "*[]"})
      g_builtins.insert(name + a);
  }

  if (getenv("CPPJIT_DISABLE_FASTPATH"))
    gEnableFastPath = false;

  // set opt level (default to 2 if not given; Cling itself defaults to 0)
  int optLevel = 2;

  if (getenv("CPPJIT_OPT_LEVEL"))
    optLevel = atoi(getenv("CPPJIT_OPT_LEVEL"));

  if (optLevel != 0) {
    std::ostringstream s;
    s << "#pragma cling optimize " << optLevel;
    Cpp::Process(s.str().c_str());
  }

  for (const std::string& dir : Paths.IncludeDirs)
    Cpp::AddIncludePath(dir.c_str());
  Cpp::LoadLibrary("libstdc++", /* lookup= */ true);
}

static void preloadHeaders() {
  const char* code = "#include <algorithm>\n"
                     "#include <numeric>\n"
                     "#include <complex>\n"
                     "#include <iostream>\n"
                     "#include <string.h>\n" // for strcpy
                     "#include <string>\n"
                     "#include <vector>\n"
                     "#include <utility>\n"
                     "#include <memory>\n"
                     "#include <functional>\n" // for the dispatcher code to
                                               // use std::function
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
}

static void defineRuntimeHelpers() {
  Cpp::Declare("namespace __cppjit_internal { template<class C1, class C2>"
               " bool is_equal(const C1& c1, const C2& c2) { return "
               "(bool)(c1 == c2); } }",
               /*silent=*/false);
  Cpp::Declare("namespace __cppjit_internal { template<class C1, class C2>"
               " bool is_not_equal(const C1& c1, const C2& c2) { return "
               "(bool)(c1 != c2); } }",
               /*silent=*/false);

  // helper for multiple inheritance
  Cpp::Declare("namespace __cppjit_internal { struct Sep; }",
               /*silent=*/false);
}

} // unnamed namespace

// Load CppInterOp and set up the interpreter. A dlopen during static
// initialization is unsafe, so _cpython_cppjit.py calls this explicitly
// before the first libcppjit use. Thread-safe and idempotent; returns 1 on
// success.
extern "C" {
RPY_EXPORTED int LoadCppInterOp();
}

extern "C" int LoadCppInterOp() {
  static std::once_flag Once;
  static int Loaded = 0;
  std::call_once(Once, [] {
    std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
    const InterOpPaths Paths = cppinterop_paths();
    if (!loadDispatchAPI(Paths))
      return;

    acquireOrCreateInterpreter(Paths);
    configureInterpreter(Paths);
    preloadHeaders();
    defineRuntimeHelpers();

    Loaded = 1;
  });
  return Loaded;
}

// local helpers -------------------------------------------------------------
static inline char* cppstring_to_cstring(const std::string& cppstr) {
  char* cstr = (char*)malloc(cppstr.size() + 1);
  memcpy(cstr, cppstr.c_str(), cppstr.size() + 1);
  return cstr;
}

// direct interpreter access -------------------------------------------------
// Returns false on failure and true on success
bool interop::Compile(const std::string& code, bool silent) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  // Declare returns an enum which equals 0 on success
  return !Cpp::Declare(code.c_str(), silent);
}

std::string interop::ToString(TCppScope_t klass, TCppObject_t obj) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  if (klass && obj && !Cpp::IsNamespace(klass))
    return Cpp::ObjToString(Cpp::GetQualifiedCompleteName(klass).c_str(),
                            obj.data);
  return "";
}

// // name to opaque C++ scope representation
// -----------------------------------
std::string interop::ResolveName(const std::string& name) {
  if (!name.empty()) {
    if (interop::TCppType_t type =
            interop::GetType(name, /*enable_slow_lookup=*/true))
      return interop::GetTypeAsString(interop::ResolveType(type));
    return name;
  }
  return "";
}

interop::TCppType_t interop::ResolveEnumReferenceType(TCppType_t type) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  if (Cpp::GetValueKind(type) != Cpp::ValueKind::LValue)
    return type;

  TCppType_t nonReferenceType = Cpp::GetNonReferenceType(type);
  if (Cpp::IsEnumType(nonReferenceType)) {
    TCppType_t underlying_type =
        Cpp::GetIntegerTypeFromEnumType(nonReferenceType);
    return Cpp::GetReferencedType(underlying_type, /*rvalue=*/false);
  }
  return type;
}

interop::TCppType_t interop::ResolveEnumPointerType(TCppType_t type) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  if (!Cpp::IsPointerType(type))
    return type;

  TCppType_t PointeeType = Cpp::GetPointeeType(type);
  if (Cpp::IsEnumType(PointeeType)) {
    TCppType_t underlying_type = Cpp::GetIntegerTypeFromEnumType(PointeeType);
    return Cpp::GetPointerType(underlying_type);
  }
  return type;
}

interop::TCppType_t int_like_type(interop::TCppType_t type) {
  interop::TCppType_t check_int_typedefs = type;
  if (Cpp::IsPointerType(check_int_typedefs))
    check_int_typedefs = Cpp::GetPointeeType(check_int_typedefs);
  if (Cpp::IsReferenceType(check_int_typedefs))
    check_int_typedefs =
        Cpp::GetReferencedType(check_int_typedefs, /*rvalue=*/false);

  if (Cpp::GetTypeAsString(check_int_typedefs) == "int8_t" ||
      Cpp::GetTypeAsString(check_int_typedefs) == "uint8_t")
    return check_int_typedefs;
  return nullptr;
}

interop::TCppType_t interop::ResolveType(TCppType_t type) {
  if (!type)
    return type;

  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);

  TCppType_t check_int_typedefs = int_like_type(type);
  if (check_int_typedefs)
    return type;

  interop::TCppType_t canonType = Cpp::GetCanonicalType(type);

  if (Cpp::IsEnumType(canonType)) {
    if (Cpp::GetTypeAsString(type) != "std::byte")
      return Cpp::GetIntegerTypeFromEnumType(canonType);
  }
  if (Cpp::HasTypeQualifier(canonType, Cpp::QualKind::Restrict)) {
    return Cpp::RemoveTypeQualifier(canonType, Cpp::QualKind::Restrict);
  }

  return canonType;
}

interop::TCppType_t interop::GetRealType(TCppType_t type) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  TCppType_t check_int_typedefs = int_like_type(type);
  if (check_int_typedefs)
    return check_int_typedefs;
  return Cpp::GetUnderlyingType(type);
}

interop::TCppType_t interop::GetPointerType(TCppType_t type) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetPointerType(type);
}

interop::TCppType_t interop::GetReferencedType(TCppType_t type, bool rvalue) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetReferencedType(type, rvalue);
}

bool interop::IsRValueReferenceType(TCppType_t type) {
  return Cpp::GetValueKind(type) == Cpp::ValueKind::RValue;
}

bool interop::IsLValueReferenceType(TCppType_t type) {
  return Cpp::GetValueKind(type) == Cpp::ValueKind::LValue;
}

bool interop::IsClassType(TCppType_t type) { return Cpp::IsRecordType(type); }

bool interop::IsIntegerType(TCppType_t type, bool* is_signed /*= nullptr*/) {
  if (is_signed) {
    Cpp::Signedness sign;
    bool res = Cpp::IsIntegerType(type, &sign);
    *is_signed = (sign == Cpp::Signedness::kSigned);
    return res;
  }
  return Cpp::IsIntegerType(type, nullptr);
}

bool interop::IsPointerType(TCppType_t type) {
  return Cpp::IsPointerType(type);
}

bool interop::IsFunctionPointerType(TCppType_t type) {
  return Cpp::IsFunctionPointerType(type);
}

std::string trim(const std::string& line) {
  if (line.empty())
    return "";
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
        if (end_pos > start_pos)
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

interop::TCppScope_t GetEnumFromCompleteName(const std::string& name) {
  std::string delim = "::";
  size_t start = 0;
  size_t end = name.find(delim);
  interop::TCppScope_t curr_scope;
  while (end != std::string::npos) {
    curr_scope = Cpp::GetNamed(name.substr(start, end - start), curr_scope);
    start = end + delim.length();
    end = name.find(delim, start);
  }
  return Cpp::GetNamed(name.substr(start, end), curr_scope);
}
static bool is_identifier(std::string_view s) {
  if (s.empty())
    return false;
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
bool interop::AppendTypesSlow(const std::string& name,
                              std::vector<Cpp::TemplateArgInfo>& types,
                              interop::TCppScope_t parent) {

  // Add no new type if string is empty
  if (name.empty())
    return true;

  // The ast printer gave us garbage.
  if (name == "<unnamed>")
    return true;

  auto replace_all = [](std::string& str, const std::string& from,
                        const std::string& to) {
    if (from.empty())
      return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
      str.replace(start_pos, from.length(), to);
      start_pos += to.length();
    }
  };

  std::string resolved_name = name;
  replace_all(resolved_name, "std::initializer_list<",
              "std::vector<"); // replace initializer_list with vector

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
    return true;
  }

  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);

  // We might have an entire expression such as int, double.
  static unsigned long long struct_count = 0;
  std::string code =
      "template<typename ...T> struct __cppjit_interop_AppendTypesSlow {};\n";
  if (!struct_count)
    Cpp::Declare(code.c_str(), /*silent=*/true); // initialize the trampoline

  // The trampoline declares its variable in the global scope, so a name
  // written relative to a parent (e.g. "vector<int>" looked up in std)
  // won't resolve. Try the name as given, then qualified by the parent.
  std::vector<std::string> candidates = {resolved_name};
  if (parent && parent != Cpp::GetGlobalScope() &&
      (interop::IsNamespace(parent) || interop::IsClass(parent)))
    candidates.push_back(Cpp::GetQualifiedCompleteName(parent) +
                         "::" + resolved_name);

  for (const std::string& candidate : candidates) {
    std::string var = "__cppjit_interop_s" + std::to_string(struct_count++);
    // nodebug: with -g the variable's debug info would carry the full DIE
    // tree of every template argument (all member declarations included) --
    // a large, uncacheable per-lookup cost on heavyweight types.
    if (!Cpp::Declare(("__cppjit_interop_AppendTypesSlow<" + candidate +
                       "> __attribute__((nodebug)) " + var + ";\n")
                          .c_str(),
                      /*silent=*/true)) {
      TCppType_t varN =
          Cpp::GetVariableType(Cpp::GetNamed(var.c_str(), /*parent=*/nullptr));
      TCppScope_t instance_class = Cpp::GetScopeFromType(varN);
      size_t oldSize = types.size();
      Cpp::GetClassTemplateInstantiationArgs(instance_class, types);
      return oldSize == types.size();
    }
  }

  // We split each individual types based on , and resolve it
  // FIXME: see discussion on should we support template instantiation with
  // string:
  //   https://github.com/compiler-research/cppyy-backend/pull/137#discussion_r2079357491
  //   We should consider eliminating the `split_comma_saparated_types` and
  //   `is_integral` string parsing.
  std::vector<std::string> individual_types;
  if (!split_comma_saparated_types(resolved_name, individual_types))
    return true;

  for (std::string& i : individual_types) {
    // Try going via interop::GetType first.
    const char* integral_value = nullptr;
    interop::TCppType_t type = nullptr;

    type = GetType(i, /*enable_slow_lookup=*/true);
    if (!type && parent &&
        (interop::IsNamespace(parent) || interop::IsClass(parent))) {
      type =
          interop::GetTypeFromScope(interop::GetNamed(resolved_name, parent));
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

interop::TCppType_t interop::GetType(const std::string& name,
                                     bool enable_slow_lookup /* = false */) {
  // The ast printer gave us garbage.
  if (name == "<unnamed>")
    return nullptr;
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);

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
      throw std::runtime_error(
          "Calling interop::GetType with qualified name '" + name + "'\n");
    return nullptr;
  }

  // Here we might need to deal with integral types such as 3.14.

  static unsigned long long var_count = 0;
  std::string id = "__cppjit_interop_GetType_" + std::to_string(var_count++);
  std::string using_clause = "using " + id + " = __typeof__(" + name + ");\n";

  if (!Cpp::Declare(using_clause.c_str(), /*silent=*/true)) {
    TCppScope_t lookup = Cpp::GetNamed(id);
    TCppType_t lookup_ty = Cpp::GetTypeFromScope(lookup);
    return Cpp::GetCanonicalType(lookup_ty);
  }
  return nullptr;
}

interop::TCppType_t interop::GetComplexType(const std::string& name) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetComplexType(Cpp::GetType(name));
}

std::string interop::ResolveEnum(TCppScope_t handle) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  std::string type =
      Cpp::GetTypeAsString(Cpp::GetIntegerTypeFromEnumScope(handle));
  if (type == "signed char")
    return "char";
  return type;
}

interop::TCppScope_t interop::GetUnderlyingScope(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetUnderlyingScope(scope);
}

interop::TCppScope_t interop::GetScope(const std::string& name,
                                       TCppScope_t parent_scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  if (interop::TCppScope_t scope = Cpp::GetScope(name, parent_scope))
    return scope;
  if (!parent_scope || parent_scope == Cpp::GetGlobalScope())
    if (interop::TCppScope_t scope = Cpp::GetScopeFromCompleteName(name))
      return scope;

  // FIXME: avoid string parsing here
  if (name.find('<') != std::string::npos) {
    // Templated type; may need instantiation. Resolve the whole type
    // expression (e.g. "std::array<float, 3>") and read back its scope.
    // Splitting off the argument list and resolving it directly cannot
    // represent non-type arguments such as the `3` in std::array<float, 3>.
    std::vector<Cpp::TemplateArgInfo> types;
    InterOpMutex.unlock(); // unlock to allow AppendTypesSlow
    bool added_new_type =
        !interop::AppendTypesSlow(name, types, /*parent=*/parent_scope);
    std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
    if (added_new_type && types.size() == 1) {
      // A pointer or reference spelling (e.g. "std::chrono::nanoseconds *",
      // the return type of std::array<nanoseconds, N>::begin()) does not
      // name a scope; GetScopeFromType would silently strip the pointer and
      // return the pointee's scope, misclassifying the name.
      if (Cpp::IsPointerType(types[0].m_Type) ||
          Cpp::IsReferenceType(types[0].m_Type))
        return nullptr;
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

interop::TCppScope_t interop::GetFullScope(const std::string& name) {
  return interop::GetScope(name);
}

interop::TCppScope_t interop::GetTypeScope(TCppScope_t var) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetScopeFromType(Cpp::GetVariableType(var));
}

interop::TCppScope_t interop::GetNamed(const std::string& name,
                                       TCppScope_t parent_scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetNamed(name, parent_scope);
}

interop::TCppScope_t interop::GetParentScope(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetParentScope(scope);
}

interop::TCppScope_t interop::GetScopeFromType(TCppType_t type) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetScopeFromType(type);
}

interop::TCppType_t interop::GetTypeFromScope(TCppScope_t klass) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetTypeFromScope(klass);
}

interop::TCppScope_t interop::GetGlobalScope() {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetGlobalScope();
}

bool interop::IsTemplate(TCppScope_t handle) { return Cpp::IsTemplate(handle); }

bool interop::IsTemplateInstantiation(TCppScope_t handle) {
  return Cpp::IsTemplateSpecialization(handle);
}

bool interop::IsTypedefed(TCppScope_t handle) {
  return Cpp::IsTypedefed(handle);
}

namespace {
class AutoCastRTTI {
public:
  virtual ~AutoCastRTTI() {}
};
} // namespace

interop::TCppScope_t interop::GetActualClass(TCppScope_t klass,
                                             TCppObject_t obj) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);

  if (!Cpp::IsClassPolymorphic(klass))
    return klass;

  const std::type_info* typ = &typeid(*(AutoCastRTTI*)obj.data);

  std::string mangled_name = typ->name();
  std::string demangled_name = Cpp::Demangle(mangled_name);

  if (TCppScope_t scope = interop::GetScope(demangled_name))
    return scope;

  return klass;
}

size_t interop::SizeOf(TCppScope_t klass) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::SizeOf(klass);
}

size_t interop::SizeOfType(TCppType_t klass) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetSizeOfType(klass);
}

bool interop::IsBuiltin(const std::string& type_name) {
  static std::set<std::string> s_builtins = {"bool",
                                             "char",
                                             "signed char",
                                             "unsigned char",
                                             "wchar_t",
                                             "short",
                                             "unsigned short",
                                             "int",
                                             "unsigned int",
                                             "long",
                                             "unsigned long",
                                             "long long",
                                             "unsigned long long",
                                             "float",
                                             "double",
                                             "long double",
                                             "void"};
  if (s_builtins.find(trim(type_name)) != s_builtins.end())
    return true;

  if (strstr(type_name.c_str(), "std::complex"))
    return true;

  return false;
}

bool interop::IsBuiltin(TCppType_t type) { return Cpp::IsBuiltin(type); }

bool interop::IsComplete(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::IsComplete(scope);
}

// // memory management
// ---------------------------------------------------------
interop::TCppObject_t interop::Allocate(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::Allocate(scope, /*count=*/1);
}

void interop::Deallocate(TCppScope_t scope, TCppObject_t instance) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  Cpp::Deallocate(scope, instance, /*count=*/1);
}

interop::TCppObject_t interop::Construct(TCppScope_t scope,
                                         void* arena /*=nullptr*/) {
  std::lock_guard<std::recursive_mutex> Lock(
      InterOpMutex); // TODO: this shouldn't locks the JIT call
  return Cpp::Construct(scope, arena, /*count=*/1);
}

void interop::Destruct(TCppScope_t scope, TCppObject_t instance) {
  std::lock_guard<std::recursive_mutex> Lock(
      InterOpMutex); // TODO: this shouldn't locks the JIT call
  Cpp::Destruct(instance, scope, true, /*count=*/0);
}

static inline bool copy_args(Parameter* args, size_t nargs, void** vargs) {
  bool runRelease = false;
  for (size_t i = 0; i < nargs; ++i) {
    switch (args[i].fTypeCode) {
    case 'X': /* (void*)type& with free */
      runRelease = true;
    case 'V': /* (void*)type& */
      vargs[i] = args[i].fValue.fVoidp;
      break;
    case 'r': /* const type& */
      vargs[i] = args[i].fRef;
      break;
    default: /* all other types in union */
      vargs[i] = (void*)&args[i].fValue.fVoidp;
      break;
    }
  }
  return runRelease;
}

static inline void release_args(Parameter* args, size_t nargs) {
  for (size_t i = 0; i < nargs; ++i) {
    if (args[i].fTypeCode == 'X')
      free(args[i].fValue.fVoidp);
  }
}

static inline bool WrapperCall(interop::TCppMethod_t method, size_t nargs,
                               void* args_, void* self, void* result) {
  Parameter* args = (Parameter*)args_;
  // bool is_direct = nargs & DIRECT_CALL;
  nargs = CALL_NARGS(nargs);

  // if (!is_ready(wrap, is_direct))
  //     return false;        // happens with compilation error
  InterOpMutex.lock();
  if (Cpp::JitCall JC = Cpp::MakeFunctionCallable(method)) {
    InterOpMutex.unlock();
    bool runRelease = false;
    // const auto& fgen = /* is_direct ? faceptr.fDirect : */ faceptr;
    if (nargs <= cpyrt::SMALL_ARGS_N) {
      void* smallbuf[cpyrt::SMALL_ARGS_N];
      if (nargs)
        runRelease = copy_args(args, nargs, smallbuf);
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
    if (runRelease)
      release_args(args, nargs);
    return true;
  }
  InterOpMutex.unlock();
  return false;
}

template <typename T>
static inline T CallT(interop::TCppMethod_t method, interop::TCppObject_t self,
                      size_t nargs, void* args) {
  T t{};
  if (WrapperCall(method, nargs, args, self.data, &t))
    return t;
  throw std::runtime_error("failed to resolve function");
  return (T)-1;
}

#ifdef PRINT_DEBUG
#define _IMP_CALL_PRINT_STMT(type) printf("IMP CALL with type: %s\n", #type);
#else
#define _IMP_CALL_PRINT_STMT(type)
#endif

#define CPPJIT_IMP_CALL(typecode, rtype)                                       \
  rtype interop::Call##typecode(TCppMethod_t method, TCppObject_t self,        \
                                size_t nargs, void* args) {                    \
    _IMP_CALL_PRINT_STMT(rtype)                                                \
    return CallT<rtype>(method, self, nargs, args);                            \
  }

void interop::CallV(TCppMethod_t method, TCppObject_t self, size_t nargs,
                    void* args) {
  if (!WrapperCall(method, nargs, args, self.data, nullptr))
    return /* TODO ... report error */;
}

// clang-format off
CPPJIT_IMP_CALL(B,  unsigned char)
CPPJIT_IMP_CALL(C,  char         )
CPPJIT_IMP_CALL(H,  short        )
CPPJIT_IMP_CALL(I,  int          )
CPPJIT_IMP_CALL(L,  long         )
CPPJIT_IMP_CALL(LL, long long    )
CPPJIT_IMP_CALL(F,  float        )
CPPJIT_IMP_CALL(D,  double       )
CPPJIT_IMP_CALL(LD, long double  )
// clang-format on

void* interop::CallR(TCppMethod_t method, TCppObject_t self, size_t nargs,
                     void* args) {
  void* r = nullptr;
  if (WrapperCall(method, nargs, args, self.data, &r))
    return r;
  return nullptr;
}

char* interop::CallS(TCppMethod_t method, TCppObject_t self, size_t nargs,
                     void* args, size_t* length) {
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

interop::TCppObject_t interop::CallConstructor(TCppMethod_t method,
                                               TCppScope_t /*klass*/,
                                               size_t nargs, void* args) {
  void* obj = nullptr;
  WrapperCall(method, nargs, args, nullptr, &obj);
  return (TCppObject_t)obj;
}

void interop::CallDestructor(TCppScope_t scope, TCppObject_t self) {
  std::lock_guard<std::recursive_mutex> Lock(
      InterOpMutex); // TODO: this shouldn't locks the JIT call
  Cpp::Destruct(self, scope, /*withFree=*/false, /*count=*/0);
}

interop::TCppObject_t interop::CallO(TCppMethod_t method, TCppObject_t self,
                                     size_t nargs, void* args,
                                     TCppType_t result_type) {
  void* obj = ::operator new(interop::SizeOfType(result_type));
  if (WrapperCall(method, nargs, args, self.data, obj))
    return (TCppObject_t)obj;
  ::operator delete(obj);
  return TCppObject_t{};
}

interop::TCppFuncAddr_t interop::GetFunctionAddress(TCppMethod_t method,
                                                    bool /*check_enabled*/) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetFunctionAddress(method);
}

// handling of function argument buffer --------------------------------------
void* interop::AllocateFunctionArgs(size_t nargs) {
  return new Parameter[nargs];
}

void interop::DeallocateFunctionArgs(void* args) { delete[] (Parameter*)args; }

size_t interop::GetFunctionArgSizeof() { return sizeof(Parameter); }

size_t interop::GetFunctionArgTypeoffset() {
  return offsetof(Parameter, fTypeCode);
}

// scope reflection information ----------------------------------------------
bool interop::IsNamespace(TCppScope_t scope) {
  if (!scope)
    return false;

  // Test if this scope represents a namespace.
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::IsNamespace(scope) || Cpp::GetGlobalScope() == scope;
}

bool interop::IsClass(TCppScope_t scope) {
  // Test if this scope represents a namespace.
  return Cpp::IsClass(scope);
}
//
bool interop::IsAbstract(TCppScope_t scope) {
  // Test if this type may not be instantiated.
  return Cpp::IsAbstract(scope);
}

bool interop::IsEnumScope(TCppScope_t scope) { return Cpp::IsEnumScope(scope); }

bool interop::IsEnumConstant(TCppScope_t scope) {
  return Cpp::IsEnumConstant(interop::GetUnderlyingScope(scope));
}

bool interop::IsEnumType(TCppType_t type) { return Cpp::IsEnumType(type); }

bool interop::IsAggregate(TCppScope_t type) {
  // Test if this type is a "plain old data" type
  return Cpp::IsAggregate(type);
}

bool interop::IsDefaultConstructable(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  // Test if this type has a default constructor or is a "plain old data" type
  return Cpp::HasDefaultConstructor(scope);
}

bool interop::IsVariable(TCppScope_t scope) { return Cpp::IsVariable(scope); }

void interop::GetAllCppNames(TCppScope_t scope,
                             std::set<std::string>& cppnames) {
  // Collect all known names of C++ entities under scope. This is useful for
  // IDEs employing tab-completion, for example. Note that functions names need
  // not be unique as they can be overloaded.
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  Cpp::GetAllCppNames(scope, cppnames);
}

// class reflection information ----------------------------------------------
std::vector<interop::TCppScope_t>
interop::GetUsingNamespaces(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetUsingNamespaces(scope);
}

// class reflection information ----------------------------------------------
std::string interop::GetFinalName(TCppScope_t klass) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetCompleteName(Cpp::GetUnderlyingScope(klass));
}

std::string interop::GetScopedFinalName(TCppScope_t klass) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetQualifiedCompleteName(klass);
}

bool interop::HasVirtualDestructor(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  TCppMethod_t func = Cpp::GetDestructor(scope);
  return Cpp::IsVirtualMethod(func);
}

interop::TCppIndex_t interop::GetNumBases(TCppScope_t klass) {
  // Get the total number of base classes that this class has.
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetNumBases(klass);
}

////////////////////////////////////////////////////////////////////////////////
/// \fn interop::TCppIndex_t interop::GetNumBasesLongestBranch(TCppScope_t
/// klass) \brief Retrieve number of base classes in the longest branch of the
///        inheritance tree of the input class.
/// \param[in] klass The class to start the retrieval process from.
///
/// This is a helper function for interop::GetNumBasesLongestBranch.
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
interop::TCppIndex_t interop::GetNumBasesLongestBranch(TCppScope_t klass) {
  std::vector<size_t> num;
  for (TCppIndex_t ibase = 0; ibase < GetNumBases(klass); ++ibase)
    num.push_back(
        GetNumBasesLongestBranch(interop::GetBaseScope(klass, ibase)));
  if (num.empty())
    return 0;
  return *std::max_element(num.begin(), num.end()) + 1;
}

std::string interop::GetBaseName(TCppScope_t klass, TCppIndex_t ibase) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetName(Cpp::GetBaseClass(klass, ibase));
}

interop::TCppScope_t interop::GetBaseScope(TCppScope_t klass,
                                           TCppIndex_t ibase) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetBaseClass(klass, ibase);
}

bool interop::IsSubclass(TCppScope_t derived, TCppScope_t base) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::IsSubclass(derived, base);
}

static std::set<std::string> gSmartPtrTypes = {
    "std::auto_ptr", "std::shared_ptr", "std::unique_ptr", "std::weak_ptr"};

bool interop::IsSmartPtr(TCppScope_t klass) {
  const std::string& rn = interop::GetScopedFinalName(klass);
  if (gSmartPtrTypes.find(rn.substr(0, rn.find("<"))) != gSmartPtrTypes.end())
    return true;
  return false;
}

bool interop::GetSmartPtrInfo(const std::string& tname, TCppScope_t* raw,
                              TCppMethod_t* deref) {
  // TODO: We can directly accept scope instead of name
  const std::string& rn = ResolveName(tname);
  if (gSmartPtrTypes.find(rn.substr(0, rn.find("<"))) == gSmartPtrTypes.end())
    return false;

  if (!raw && !deref)
    return true;

  TCppScope_t scope = interop::GetScope(rn);
  if (!scope)
    return false;

  std::vector<TCppMethod_t> ops;
  {
    std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
    Cpp::GetOperator(scope, Cpp::Operator::OP_Arrow, ops,
                     /*kind=*/Cpp::OperatorArity::kBoth);
  }
  if (ops.size() != 1)
    return false;

  if (deref)
    *deref = ops[0];
  if (raw)
    *raw = interop::GetScopeFromType(interop::GetMethodReturnType(ops[0]));
  return (!deref || *deref) && (!raw || *raw);
}

// type offsets --------------------------------------------------------------
ptrdiff_t interop::GetBaseOffset(TCppScope_t derived, TCppScope_t base,
                                 TCppObject_t /*address*/, int direction,
                                 bool rerror) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  intptr_t offset = Cpp::GetBaseClassOffset(derived, base);

  if (offset == -1) // Cling error, treat silently
    return rerror ? (ptrdiff_t)offset : 0;

  return (ptrdiff_t)(direction < 0 ? -offset : offset);
}

// method/function reflection information ------------------------------------
// A deleted overload is not callable, and leaving it in the set adds a
// spurious conversion error to every failed-call report.
static void
remove_deleted_methods(std::vector<interop::TCppMethod_t>& methods) {
  methods.erase(std::remove_if(methods.begin(), methods.end(),
                               [](interop::TCppMethod_t m) {
                                 return Cpp::IsFunctionDeleted(m);
                               }),
                methods.end());
}

void interop::GetClassMethods(TCppScope_t scope,
                              std::vector<interop::TCppMethod_t>& methods) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  Cpp::GetClassMethods(scope, methods);
  remove_deleted_methods(methods);
}

std::vector<interop::TCppMethod_t>
interop::GetMethodsFromName(TCppScope_t scope, const std::string& name) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  std::vector<interop::TCppMethod_t> methods =
      Cpp::GetFunctionsUsingName(scope, name);
  remove_deleted_methods(methods);
  return methods;
}

std::string interop::GetName(TCppScope_t method) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetName(method);
}

std::string interop::GetFullName(TCppScope_t method) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetCompleteName(method);
}

interop::TCppType_t interop::GetMethodReturnType(TCppMethod_t method) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetFunctionReturnType(method);
}

std::string interop::GetMethodReturnTypeAsString(TCppMethod_t method) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetTypeAsString(
      Cpp::GetCanonicalType(Cpp::GetFunctionReturnType(method)));
}

interop::TCppIndex_t interop::GetMethodNumArgs(TCppMethod_t method) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetFunctionNumArgs(method);
}

interop::TCppIndex_t interop::GetMethodReqArgs(TCppMethod_t method) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetFunctionRequiredArgs(method);
}

std::string interop::GetMethodArgName(TCppMethod_t method, TCppIndex_t iarg) {
  if (!method)
    return "<unknown>";

  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetFunctionArgName(method, iarg);
}

interop::TCppType_t interop::GetMethodArgType(TCppMethod_t method,
                                              TCppIndex_t iarg) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetFunctionArgType(method, iarg);
}

std::string interop::GetMethodArgTypeAsString(TCppMethod_t method,
                                              TCppIndex_t iarg) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetTypeAsString(Cpp::RemoveTypeQualifier(
      Cpp::GetFunctionArgType(method, iarg), Cpp::QualKind::Const));
}

std::string interop::GetMethodArgCanonTypeAsString(TCppMethod_t method,
                                                   TCppIndex_t iarg) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetTypeAsString(
      Cpp::GetCanonicalType(Cpp::GetFunctionArgType(method, iarg)));
}

std::string interop::GetMethodArgDefault(TCppMethod_t method,
                                         TCppIndex_t iarg) {
  if (!method)
    return "";

  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetFunctionArgDefault(method, iarg);
}

interop::TCppIndex_t
interop::CompareMethodArgType(TCppMethod_t /*method*/, TCppIndex_t /*iarg*/,
                              const std::string& /*req_type*/) {
  // if (method) {
  //     TFunction* f = m2f(method);
  //     TMethodArg* arg = (TMethodArg
  //     *)f->GetListOfMethodArgs()->At((int)iarg); void *argqtp =
  //     gInterpreter->TypeInfo_QualTypePtr(arg->GetTypeInfo());

  //     TypeInfo_t *reqti = gInterpreter->TypeInfo_Factory(req_type.c_str());
  //     void *reqqtp = gInterpreter->TypeInfo_QualTypePtr(reqti);

  //     if (ArgSimilarityScore(argqtp, reqqtp) < 10) {
  //         return ArgSimilarityScore(argqtp, reqqtp);
  //     }
  //     else { // Match using underlying types
  //         if(gInterpreter->IsPointerType(argqtp))
  //             argqtp =
  //             gInterpreter->TypeInfo_QualTypePtr(gInterpreter->GetPointerType(argqtp));

  //         // Handles reference types and strips qualifiers
  //         TypeInfo_t *arg_ul = gInterpreter->GetNonReferenceType(argqtp);
  //         TypeInfo_t *req_ul = gInterpreter->GetNonReferenceType(reqqtp);
  //         argqtp =
  //         gInterpreter->TypeInfo_QualTypePtr(gInterpreter->GetUnqualifiedType(gInterpreter->TypeInfo_QualTypePtr(arg_ul)));
  //         reqqtp =
  //         gInterpreter->TypeInfo_QualTypePtr(gInterpreter->GetUnqualifiedType(gInterpreter->TypeInfo_QualTypePtr(req_ul)));

  //         return ArgSimilarityScore(argqtp, reqqtp);
  //     }
  // }
  return 0; // Method is not valid
}

std::string interop::GetMethodSignature(TCppMethod_t method,
                                        bool show_formal_args,
                                        TCppIndex_t max_args) {
  std::ostringstream sig;
  sig << "(";
  int nArgs = GetMethodNumArgs(method);
  if (max_args != (TCppIndex_t)-1)
    nArgs = std::min(nArgs, (int)max_args);
  for (int iarg = 0; iarg < nArgs; ++iarg) {
    sig << interop::GetMethodArgTypeAsString(method, iarg);
    if (show_formal_args) {
      std::string argname = interop::GetMethodArgName(method, iarg);
      if (!argname.empty())
        sig << " " << argname;
      std::string defvalue = interop::GetMethodArgDefault(method, iarg);
      if (!defvalue.empty())
        sig << " = " << defvalue;
    }
    if (iarg != nArgs - 1)
      sig << ", ";
  }
  sig << ")";
  return sig.str();
}

interop::TCppType_t interop::GetFnTypeFromStdFn(TCppType_t fn_type) {
  fn_type = Cpp::IsReferenceType(fn_type) ? Cpp::GetNonReferenceType(fn_type)
                                          : fn_type;
  fn_type =
      Cpp::IsPointerType(fn_type) ? Cpp::GetPointeeType(fn_type) : fn_type;
  TCppScope_t scope = Cpp::GetScopeFromType(fn_type);
  std::vector<Cpp::TemplateArgInfo> args;
  Cpp::GetClassTemplateArgs(scope, args);
  assert(args.size() == 1);
  if (args.size() == 1)
    return args[0].m_Type;
  return nullptr;
}

void interop::GetFnTypeSig(TCppType_t fn_type,
                           std::vector<TCppType_t>& arg_types) {
  fn_type = Cpp::IsReferenceType(fn_type) ? Cpp::GetNonReferenceType(fn_type)
                                          : fn_type;
  fn_type =
      Cpp::IsPointerType(fn_type) ? Cpp::GetPointeeType(fn_type) : fn_type;
  Cpp::GetFnTypeSignature(fn_type, arg_types);
}

bool interop::IsSameType(TCppType_t typ1, TCppType_t typ2) {
  return Cpp::IsSameType(typ1, typ2);
}

bool interop::IsFunctionType(TCppType_t typ) {
  typ = Cpp::IsReferenceType(typ) ? Cpp::GetNonReferenceType(typ) : typ;
  typ = Cpp::IsPointerType(typ) ? Cpp::GetPointeeType(typ) : typ;
  return Cpp::IsFunctionProtoType(typ);
}

bool interop::IsSimilarFnTypes(TCppType_t typ1, TCppType_t typ2) {
  typ1 = Cpp::IsReferenceType(typ1) ? Cpp::GetNonReferenceType(typ1) : typ1;
  typ2 = Cpp::IsReferenceType(typ2) ? Cpp::GetNonReferenceType(typ2) : typ2;
  typ1 = Cpp::IsPointerType(typ1) ? Cpp::GetPointeeType(typ1) : typ1;
  typ2 = Cpp::IsPointerType(typ2) ? Cpp::GetPointeeType(typ2) : typ2;
  return Cpp::IsSameType(typ1, typ2);
}

std::string interop::GetDoxygenComment(TCppScope_t scope, bool strip_markers) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetDoxygenComment(scope, strip_markers);
}

bool interop::IsConstMethod(TCppMethod_t method) {
  if (!method)
    return false;
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::IsConstMethod(method);
}

void interop::GetTemplatedMethods(TCppScope_t scope,
                                  std::vector<interop::TCppMethod_t>& methods) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  Cpp::GetFunctionTemplatedDecls(scope, methods);
}

interop::TCppIndex_t
interop::GetNumTemplatedMethods(TCppScope_t scope, bool /*accept_namespace*/) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  std::vector<interop::TCppMethod_t> mc;
  Cpp::GetFunctionTemplatedDecls(scope, mc);
  return mc.size();
}

std::string interop::GetTemplatedMethodName(TCppScope_t scope,
                                            TCppIndex_t imeth) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  std::vector<interop::TCppMethod_t> mc;
  Cpp::GetFunctionTemplatedDecls(scope, mc);

  if (imeth < mc.size())
    return Cpp::GetName(TCppScope_t(mc[imeth].data));

  return "";
}

bool interop::ExistsMethodTemplate(TCppScope_t scope, const std::string& name) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::ExistsFunctionTemplate(name, scope);
}

bool interop::IsTemplatedMethod(TCppMethod_t method) {
  return Cpp::IsTemplatedFunction(method);
}

bool interop::IsStaticTemplate(TCppScope_t scope, const std::string& name) {
  std::vector<TCppMethod_t> candidate_methods;
  Cpp::GetClassTemplatedMethods(name, scope, candidate_methods);
  bool is_static = true;
  for (auto i : candidate_methods) {
    if (!Cpp::IsStaticMethod(i)) {
      is_static = false;
      break;
    }
  }
  return is_static;
}

interop::TCppMethod_t interop::GetMethodTemplate(TCppScope_t scope,
                                                 const std::string& name,
                                                 const std::string& proto) {
  std::string pureName;
  std::string explicit_params;

  if ((name.find("operator<") != 0) && (name.find('<') != std::string::npos)) {
    pureName = name.substr(0, name.find('<'));
    size_t start = name.find('<');
    size_t end = name.rfind('>');
    explicit_params = name.substr(start + 1, end - start - 1);
  } else {
    pureName = name;
  }

  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);

  std::vector<interop::TCppMethod_t> unresolved_candidate_methods;
  Cpp::GetClassTemplatedMethods(pureName, scope, unresolved_candidate_methods);
  if (unresolved_candidate_methods.empty() && name.find("operator") == 0) {
    // try operators
    interop::GetClassOperators(scope, pureName, unresolved_candidate_methods);
  }

  // cpyrt assumes that we attempt instantiation here
  std::vector<Cpp::TemplateArgInfo> arg_types;
  std::vector<Cpp::TemplateArgInfo> templ_params;
  interop::AppendTypesSlow(proto, arg_types, scope);
  interop::AppendTypesSlow(explicit_params, templ_params, scope);
  interop::TCppMethod_t cppmeth = nullptr;
  cppmeth = Cpp::BestOverloadFunctionMatch(unresolved_candidate_methods,
                                           templ_params, arg_types);

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
              TCppScope_t(cand.data), templ_params.data(), templ_params.size(),
              /*instantiate_body=*/false)) {
        cppmeth = spec.data;
        break;
      }
    }
  }

  return TCppMethod_t(cppmeth.data);
  // if it fails, use Sema to propogate info about why it failed (DeductionInfo)
}

static inline std::string type_remap(const std::string& n1,
                                     const std::string& n2) {
  // Operator lookups of (C++ string, Python str) should succeed for the
  // combos of string/str, wstring/str, string/unicode and wstring/unicode;
  // since C++ does not have a operator+(std::string, std::wstring), we'll
  // have to look up the same type and rely on the converters in
  // cpyrt/_cppjit.
  if (n1 == "str" || n1 == "unicode" || n1 == "std::basic_string<char>") {
    if (n2 == "std::basic_string<wchar_t>")
      return "std::basic_string<wchar_t>&"; // match like for like
    return "std::basic_string<char>&";      // probably best bet
  } else if (n1 == "std::basic_string<wchar_t>") {
    return "std::basic_string<wchar_t>&";
  } else if (n1 == "complex") {
    return "std::complex<double>";
  }
  return n1;
}

void interop::GetClassOperators(interop::TCppScope_t klass,
                                const std::string& opname,
                                std::vector<TCppMethod_t>& operators) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  std::string op = opname.substr(8);
  Cpp::GetOperator(klass, Cpp::GetOperatorFromSpelling(op), operators,
                   /*kind=*/Cpp::OperatorArity::kBoth);
}

interop::TCppMethod_t interop::GetGlobalOperator(TCppScope_t scope,
                                                 const std::string& lc,
                                                 const std::string& rc,
                                                 const std::string& opname) {
  std::string rc_type = type_remap(rc, lc);
  std::string lc_type = type_remap(lc, rc);

  std::vector<TCppMethod_t> overloads;
  Cpp::GetOperator(scope, Cpp::GetOperatorFromSpelling(opname), overloads,
                   /*kind=*/Cpp::OperatorArity::kBoth);

  // Avoid pushing nullptr into arg_types which would crash
  // BestOverloadFunctionMatch when it dereferences each entry's QualType.
  auto resolve_arg_type = [](const std::string& name) -> interop::TCppType_t {
    if (auto s = interop::GetScope(name))
      if (auto t = interop::GetTypeFromScope(s))
        return interop::GetReferencedType(t);
    return interop::GetType(name, /*enable_slow_lookup=*/true);
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
  interop::TCppMethod_t cppmeth =
      Cpp::BestOverloadFunctionMatch(overloads, {}, arg_types);
  if (cppmeth)
    return cppmeth;
  return nullptr;
}

// method properties ---------------------------------------------------------
bool interop::IsDeletedMethod(TCppMethod_t method) {
  return Cpp::IsFunctionDeleted(method);
}

bool interop::IsPublicMethod(TCppMethod_t method) {
  return Cpp::IsPublicMethod(method);
}

bool interop::IsProtectedMethod(TCppMethod_t method) {
  return Cpp::IsProtectedMethod(method);
}

bool interop::IsPrivateMethod(TCppMethod_t method) {
  return Cpp::IsPrivateMethod(method);
}

bool interop::IsConstructor(TCppMethod_t method) {
  return Cpp::IsConstructor(method);
}

bool interop::IsDestructor(TCppMethod_t method) {
  return Cpp::IsDestructor(method);
}

bool interop::IsStaticMethod(TCppMethod_t method) {
  return Cpp::IsStaticMethod(method);
}

bool interop::IsExplicit(TCppMethod_t method) {
  return Cpp::IsExplicit(method);
}

// data member reflection information ----------------------------------------
void interop::GetDatamembers(TCppScope_t scope,
                             std::vector<TCppScope_t>& datamembers) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  Cpp::GetDatamembers(scope, datamembers);
  Cpp::GetStaticDatamembers(scope, datamembers);
  Cpp::GetEnumConstantDatamembers(scope, datamembers, false);
}

bool interop::CheckDatamember(TCppScope_t scope, const std::string& name) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return (bool)Cpp::LookupDatamember(name, scope);
}

bool interop::IsLambdaClass(TCppType_t type) {
  return Cpp::IsLambdaClass(type);
}

interop::TCppScope_t interop::WrapLambdaFromVariable(TCppScope_t var) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  std::ostringstream code;
  std::string name = interop::GetFinalName(var);
  code << "namespace __cppjit_internal_wrap_g {\n"
       << "  " << "std::function " << name
       << " = ::" << Cpp::GetQualifiedName(var) << ";\n"
       << "}\n";

  if (interop::Compile(code.str().c_str())) {
    TCppScope_t res = Cpp::GetNamed(
        name, Cpp::GetScope("__cppjit_internal_wrap_g", /*parent=*/nullptr));
    if (res)
      return res;
  }
  return var;
}

interop::TCppMethod_t
interop::AdaptFunctionForLambdaReturn(interop::TCppMethod_t fn) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);

  std::string fn_name = Cpp::GetQualifiedCompleteName(TCppScope_t(fn.data));
  std::string signature = interop::GetMethodSignature(fn, true);

  std::ostringstream call;
  call << "(";
  for (size_t i = 0, n = interop::GetMethodNumArgs(fn); i < n; i++) {
    call << interop::GetMethodArgName(fn, i);
    if (i != n - 1)
      call << ", ";
  }
  call << ")";

  std::ostringstream code;
  static int i = 0;
  std::string name = "lambda_return_convert_" + std::to_string(++i);
  code << "namespace __cppjit_internal_wrap_g {\n"
       << "auto " << name << signature << "{" << "return std::function("
       << fn_name << call.str() << "); }\n"
       << "}\n";
  if (interop::Compile(code.str().c_str())) {
    TCppScope_t res = Cpp::GetNamed(
        name, Cpp::GetScope("__cppjit_internal_wrap_g", /*parent=*/nullptr));
    if (res)
      return TCppMethod_t(res.data);
  }
  return fn;
}

interop::TCppType_t interop::GetDatamemberType(TCppScope_t var) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetVariableType(Cpp::GetUnderlyingScope(var));
}

std::string interop::GetDatamemberTypeAsString(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetTypeAsString(
      Cpp::GetVariableType(Cpp::GetUnderlyingScope(scope)));
}

std::string interop::GetTypeAsString(TCppType_t type) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetTypeAsString(type);
}

intptr_t interop::GetDatamemberOffset(TCppScope_t var, TCppScope_t klass) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetVariableOffset(Cpp::GetUnderlyingScope(var), klass);
}

// data member properties ----------------------------------------------------
bool interop::IsPublicData(TCppScope_t datamem) {
  return Cpp::IsPublicVariable(datamem);
}

bool interop::IsProtectedData(TCppScope_t datamem) {
  return Cpp::IsProtectedVariable(datamem);
}

bool interop::IsPrivateData(TCppScope_t datamem) {
  return Cpp::IsPrivateVariable(datamem);
}

bool interop::IsStaticDatamember(TCppScope_t var) {
  return Cpp::IsStaticVariable(interop::GetUnderlyingScope(var));
}

bool interop::IsConstVar(TCppScope_t var) { return Cpp::IsConstVariable(var); }

interop::TCppMethod_t interop::ReduceReturnType(TCppMethod_t fn,
                                                TCppType_t reduce) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);

  std::string fn_name = Cpp::GetQualifiedCompleteName(TCppScope_t(fn.data));
  std::string signature = interop::GetMethodSignature(fn, true);
  std::string result_type = interop::GetTypeAsString(reduce);

  std::ostringstream call;
  call << "(";
  for (size_t i = 0, n = interop::GetMethodNumArgs(fn); i < n; i++) {
    call << interop::GetMethodArgName(fn, i);
    if (i != n - 1)
      call << ", ";
  }
  call << ")";

  std::ostringstream code;
  static int i = 0;
  std::string name = "reduced_function_" + std::to_string(++i);
  code << "namespace __cppjit_internal_wrap_g {\n"
       << result_type << " " << name << signature << "{" << "return ("
       << result_type << ")::" << fn_name << call.str() << "; }\n"
       << "}\n";
  if (interop::Compile(code.str().c_str())) {
    TCppScope_t res = Cpp::GetNamed(
        name, Cpp::GetScope("__cppjit_internal_wrap_g", /*parent=*/nullptr));
    if (res)
      return TCppMethod_t(res.data);
  }
  return fn;
}

std::vector<long int> interop::GetDimensions(TCppType_t type) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetDimensions(type);
}

// enum properties -----------------------------------------------------------
std::vector<interop::TCppScope_t> interop::GetEnumConstants(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetEnumConstants(scope);
}

interop::TCppType_t interop::GetEnumConstantType(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetEnumConstantType(Cpp::GetUnderlyingScope(scope));
}

interop::TCppIndex_t interop::GetEnumDataValue(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::GetEnumConstantValue(scope);
}

interop::TCppScope_t interop::InstantiateTemplate(TCppScope_t tmpl,
                                                  Cpp::TemplateArgInfo* args,
                                                  size_t args_size) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  return Cpp::InstantiateTemplate(tmpl, args, args_size,
                                  /*instantiate_body=*/false);
}

void interop::DumpScope(TCppScope_t scope) {
  std::lock_guard<std::recursive_mutex> Lock(InterOpMutex);
  Cpp::DumpScope(scope);
}
