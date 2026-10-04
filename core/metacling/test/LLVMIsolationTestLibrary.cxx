/// \file
/// The libraries of the LLVM isolation self-tests, one per role:
///   CALLER: calls llvm::errs() when loaded, without defining it, as a library
///   that
///           relies on the LLVM of its host. The reference is weak, so that it
///           links.

namespace llvm {
#if defined(CALLER)
class raw_fd_ostream;
__attribute__((weak)) raw_fd_ostream &errs();
#elif defined(PROVIDER)
__attribute__((visibility("default"))) int LLVMIsolationTestFunction()
{
   return 42;
}
#elif defined(LEAKY)
int LLVMIsolationTestFunction();
__attribute__((visibility("default"))) int LLVMIsolationTestExport()
{
   return LLVMIsolationTestFunction();
}
__attribute__((weak, visibility("default"))) int LLVMIsolationTestWeak = 0;
#endif
} // namespace llvm

#if defined(CALLER)
__attribute__((constructor)) static void LLVMIsolationCallErrs()
{
   if (&llvm::errs)
      llvm::errs();
}
#endif
