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
#endif
} // namespace llvm

#if defined(CALLER)
__attribute__((constructor)) static void LLVMIsolationCallErrs()
{
   if (&llvm::errs)
      llvm::errs();
}
#endif
