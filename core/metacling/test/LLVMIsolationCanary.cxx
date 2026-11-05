/// \file
/// Defines LLVM and Clang functions that libCling calls, with default
/// visibility. When this library is preloaded, or loaded with RTLD_GLOBAL, it
/// is a candidate for every LLVM symbol that is resolved through the global
/// scope instead of inside libCling. Any call reaching it means that libCling's
/// LLVM is not isolated, so it aborts loudly.

#include "LLVMIsolationCanary.h"

#include <cstdio>
#include <cstdlib>

[[noreturn]] static void Leak(const char *symbol)
{
   std::fprintf(stderr, "LLVM isolation canary: %s resolved outside of libCling\n", symbol);
   std::abort();
}

#define LLVM_ISOLATION_CANARY(name, mangled)                                                \
   extern "C" __attribute__((visibility("default"))) void canary_##name() __asm__(mangled); \
   extern "C" void canary_##name()                                                          \
   {                                                                                        \
      Leak(mangled);                                                                        \
   }
LLVM_ISOLATION_DEFINED_SYMBOLS(LLVM_ISOLATION_CANARY)

// Weak references to LLVM symbols that the canary does not define. The dynamic
// linker binds them when the library is loaded; if libCling exports any of them,
// they bind to libCling, which the bindings tests detect. Otherwise they stay null.
#define LLVM_ISOLATION_PROBE(name, mangled) extern "C" __attribute__((weak)) const char probe_##name[] __asm__(mangled);
LLVM_ISOLATION_PROBED_SYMBOLS(LLVM_ISOLATION_PROBE)

#define LLVM_ISOLATION_PROBE_ADDRESS(name, mangled) probe_##name,
__attribute__((used)) static const void *const gLLVMIsolationProbes[] = {
   LLVM_ISOLATION_PROBED_SYMBOLS(LLVM_ISOLATION_PROBE_ADDRESS)};
