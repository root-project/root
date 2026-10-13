/// \file
/// LLVM and Clang symbols used by the LLVM isolation tests, as X(identifier,
/// mangled name). test-metacling-llvm-isolation-canary-names checks that the
/// LLVM ROOT was built with defines all of them, so that they cannot silently
/// become stale when LLVM changes.

/// Functions libCling calls; the canary defines them to abort.
#define LLVM_ISOLATION_DEFINED_SYMBOLS(X)                                                            \
   X(errs, "_ZN4llvm4errsEv")                                         /* llvm::errs() */             \
   X(outs, "_ZN4llvm4outsEv")                                         /* llvm::outs() */             \
   X(nulls, "_ZN4llvm5nullsEv")                                       /* llvm::nulls() */            \
   X(options, "_ZN4llvm2cl20getRegisteredOptionsERNS0_10SubCommandE") /* cl::getRegisteredOptions */ \
   X(is_absolute, "_ZN4llvm3sys4path11is_absoluteERKNS_5TwineENS1_5StyleE")                          \
   X(module_c1, "_ZN4llvm6ModuleC1ENS_9StringRefERNS_11LLVMContextE") /* Module::Module */           \
   X(module_c2, "_ZN4llvm6ModuleC2ENS_9StringRefERNS_11LLVMContextE")                                \
   X(type_info, "_ZNK5clang10ASTContext15getTypeInfoImplEPKNS_4TypeE")

/// Symbols the canary references weakly without defining them.
#define LLVM_ISOLATION_PROBED_SYMBOLS(X)                                                   \
   X(anchor, "_ZN4llvm2cl6Option6anchorEv")                       /* cl::Option::anchor */ \
   X(write, "_ZN4llvm11raw_ostream5writeEPKcm")                   /* raw_ostream::write */ \
   X(vtable, "_ZTVN4llvm11raw_ostreamE")                          /* raw_ostream vtable */ \
   X(twine, "_ZNK4llvm5Twine8toVectorERNS_15SmallVectorImplIcEE") /* Twine::toVector */
