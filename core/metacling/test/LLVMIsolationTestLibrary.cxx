/// \file
/// The libraries of the LLVM isolation self-tests, one per role:
///   CALLER: calls llvm::errs() when loaded, without defining it, as a library
///   that
///           relies on the LLVM of its host. The reference is weak, so that it
///           links.

#ifdef _WIN32
#define LLVM_ISOLATION_EXPORT __declspec(dllexport)
#else
#define LLVM_ISOLATION_EXPORT __attribute__((visibility("default")))
#endif

namespace llvm {
#if defined(CALLER)
class raw_fd_ostream;
__attribute__((weak)) raw_fd_ostream &errs();
#elif defined(PROVIDER)
LLVM_ISOLATION_EXPORT int LLVMIsolationTestFunction()
{
   return 42;
}
#elif defined(LEAKY)
int LLVMIsolationTestFunction();
LLVM_ISOLATION_EXPORT int LLVMIsolationTestExport()
{
   return LLVMIsolationTestFunction();
}
#ifndef _WIN32
__attribute__((weak, visibility("default"))) int LLVMIsolationTestWeak = 0;
#endif
#endif
} // namespace llvm

#if defined(CALLER)
__attribute__((constructor)) static void LLVMIsolationCallErrs()
{
   if (&llvm::errs)
      llvm::errs();
}
#endif
