//===- PassManager.cpp - Infrastructure for managing & running IR passes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassManagerImpl.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/FileSystem.h"
#include <optional>

using namespace llvm;

namespace llvm {
// Explicit template instantiations and specialization defininitions for core
// template typedefs.
template class AllAnalysesOn<Module>;
template class AllAnalysesOn<Function>;
template class PassManager<Module>;
template class PassManager<Function>;
template class AnalysisManager<Module>;
template class AnalysisManager<Function>;
template class InnerAnalysisManagerProxy<FunctionAnalysisManager, Module>;
template class OuterAnalysisManagerProxy<ModuleAnalysisManager, Function>;

template <>
bool FunctionAnalysisManagerModuleProxy::Result::invalidate(
    Module &M, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &Inv) {
  // If literally everything is preserved, we're done.
  if (PA.areAllPreserved())
    return false; // This is still a valid proxy.

  // If this proxy isn't marked as preserved, then even if the result remains
  // valid, the key itself may no longer be valid, so we clear everything.
  //
  // Note that in order to preserve this proxy, a module pass must ensure that
  // the FAM has been completely updated to handle the deletion of functions.
  // Specifically, any FAM-cached results for those functions need to have been
  // forcibly cleared. When preserved, this proxy will only invalidate results
  // cached on functions *still in the module* at the end of the module pass.
  auto PAC = PA.getChecker<FunctionAnalysisManagerModuleProxy>();
  if (!PAC.preserved() && !PAC.preservedSet<AllAnalysesOn<Module>>()) {
    InnerAM->clear();
    return true;
  }

  // Directly check if the relevant set is preserved.
  bool AreFunctionAnalysesPreserved =
      PA.allAnalysesInSetPreserved<AllAnalysesOn<Function>>();

  // Now walk all the functions to see if any inner analysis invalidation is
  // necessary.
  for (Function &F : M) {
    std::optional<PreservedAnalyses> FunctionPA;

    // Check to see whether the preserved set needs to be pruned based on
    // module-level analysis invalidation that triggers deferred invalidation
    // registered with the outer analysis manager proxy for this function.
    if (auto *OuterProxy =
            InnerAM->getCachedResult<ModuleAnalysisManagerFunctionProxy>(F))
      for (const auto &OuterInvalidationPair :
           OuterProxy->getOuterInvalidations()) {
        AnalysisKey *OuterAnalysisID = OuterInvalidationPair.first;
        const auto &InnerAnalysisIDs = OuterInvalidationPair.second;
        if (Inv.invalidate(OuterAnalysisID, M, PA)) {
          if (!FunctionPA)
            FunctionPA = PA;
          for (AnalysisKey *InnerAnalysisID : InnerAnalysisIDs)
            FunctionPA->abandon(InnerAnalysisID);
        }
      }

    // Check if we needed a custom PA set, and if so we'll need to run the
    // inner invalidation.
    if (FunctionPA) {
      InnerAM->invalidate(F, *FunctionPA);
      continue;
    }

    // Otherwise we only need to do invalidation if the original PA set didn't
    // preserve all function analyses.
    if (!AreFunctionAnalysesPreserved)
      InnerAM->invalidate(F, PA);
  }

  // Return false to indicate that this result is still a valid proxy.
  return false;
}

static thread_local int DumpModuleIdent = 0;
static thread_local std::vector<int> DumpModuleIdents;

static void __attribute__((noinline))
dumpPassTrap() {
    printf( "  %s\n", __FUNCTION__);
    return;
}

static bool
isDumpPassTrap( const char *str0) {
    bool r = false;
    char *str = strdup( str0);
    char *p = str;
    char *q = 0;
    std::vector<int> idents;

    while ( p ) {
        int ident = strtol( p, &q, 10);
        if ( p < q ) {
            p = q;
            idents.push_back( ident);
        } else {
            p = 0;
        }
    }

    free( str);
    if ( idents.size() == DumpModuleIdents.size() + 1 ) {
        if ( idents[0] == DumpModuleIdent ) {
            r = true;
            for ( int i = 1; i < (int)idents.size(); ++i ) {
                if ( idents[i] != DumpModuleIdents[i - 1] ) {
                    r = false;
                    break;
                }
            }
        }
    }

    return (r);
}

static std::string
dumpPassName( const StringRef &PN, const char *irtype, bool isbefore) {
    std::string r;
    static bool need_init = true;
    static bool need_dump = false;

    if ( need_init ) {
        const char *envval = getenv( "LLVM_PASS_DUMP");

        need_dump = envval && atoi( envval);
    }

    if ( need_dump ) {
        char buff[1024];
        char name[128];
        int bn = 0;
        const char *pass_trap = getenv( "LLVM_PASS_TRAP");

        snprintf( name, 1024, "%s", PN.str().c_str());
        for ( int k = 0; name[k]; ++k ) {
            if ( isalnum( name[k]) || (name[k] == '_') ) {
            } else {
                name[k] = 0;
                break;
            }
        }

        if ( pass_trap && isDumpPassTrap( pass_trap) ) {
            dumpPassTrap();
        }

        bn += snprintf( buff + bn, 1024 - bn, "pass.%03d.", DumpModuleIdent);
        for ( int k = 0; k < (int)DumpModuleIdents.size() - !isbefore; ++k ) {
            bn += snprintf( buff + bn, 1024 - bn, "pass.%03d.",
                            DumpModuleIdents[k]);
        }
        bn += snprintf( buff + bn, 1024 - bn, "%s.%s.%s.txt",
                        isbefore ? "abeg" : "xend", irtype, name);

        r = buff;

        if ( isbefore ) {
            DumpModuleIdents.push_back( 0);
        } else {
            if ( !DumpModuleIdents.empty() ) {
                DumpModuleIdents.resize( DumpModuleIdents.size() - 1);
            }

            if ( DumpModuleIdents.empty() ) {
                DumpModuleIdent++;
            } else {
                DumpModuleIdents.back()++;
            }
        }
    }

    return (r);
}

void detail::PassDumper<Module>::dumpPass( const StringRef &PN, Module &IR, bool isbefore) {
    std::string fname = dumpPassName( PN, "module", isbefore);

    if ( !fname.empty() ) {
        std::error_code ec;
        FILE *f = 0;

        f = fopen( fname.c_str(), "w");
        fclose( f);

        raw_fd_ostream *fs = new raw_fd_ostream( fname, ec, sys::fs::OpenFlags::OF_None);
        IR.print( *fs, 0);
        delete fs;
        dbgs() << "dump module pass : " << fname << "\n";
    }
}

void detail::PassDumper<Function>::dumpPass( const StringRef &PN, Function &IR, bool isbefore) {
    std::string fname = dumpPassName( PN, "function", isbefore);

    if ( !fname.empty() ) {
        std::error_code ec;
        FILE *f = 0;

        f = fopen( fname.c_str(), "w");
        fclose( f);

        raw_fd_ostream *fs = new raw_fd_ostream( fname, ec, sys::fs::OpenFlags::OF_None);
        IR.print( *fs, 0);
        delete fs;
        dbgs() << "dump function pass : " << fname << "\n";
    }
}

void detail::PassDumper<MachineFunction>::dumpPass( const StringRef &PN, MachineFunction &IR, bool isbefore) {
    std::string fname = dumpPassName( PN, "machinefunction", isbefore);

    if ( !fname.empty() ) {
        std::error_code ec;
        FILE *f = 0;

        f = fopen( fname.c_str(), "w");
        fclose( f);

        raw_fd_ostream *fs = new raw_fd_ostream( fname, ec, sys::fs::OpenFlags::OF_None);
#ifdef NDEBUG
        IR.print( *fs, 0);
#endif /* NDEBUG */
        delete fs;
        dbgs() << "dump function pass : " << fname << "\n";
    }
}

} // namespace llvm

void ModuleToFunctionPassAdaptor::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  OS << "function";
  if (EagerlyInvalidate)
    OS << "<eager-inv>";
  OS << '(';
  Pass->printPipeline(OS, MapClassName2PassName);
  OS << ')';
}

PreservedAnalyses ModuleToFunctionPassAdaptor::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  // Request PassInstrumentation from analysis manager, will use it to run
  // instrumenting callbacks for the passes later.
  PassInstrumentation PI = AM.getResult<PassInstrumentationAnalysis>(M);

  PreservedAnalyses PA = PreservedAnalyses::all();
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // Check the PassInstrumentation's BeforePass callbacks before running the
    // pass, skip its execution completely if asked to (callback returns
    // false).
    if (!PI.runBeforePass<Function>(*Pass, F))
      continue;

    PreservedAnalyses PassPA;
    {
      detail::PassDumper<Module>::dumpPass( Pass->name(), M, true);
      PassPA = Pass->run(F, FAM);
      detail::PassDumper<Module>::dumpPass( Pass->name(), M, false);
    }

    // We know that the function pass couldn't have invalidated any other
    // function's analyses (that's the contract of a function pass), so
    // directly handle the function analysis manager's invalidation here.
    FAM.invalidate(F, EagerlyInvalidate ? PreservedAnalyses::none() : PassPA);

    PI.runAfterPass(*Pass, F, PassPA);

    // Then intersect the preserved set so that invalidation of module
    // analyses will eventually occur when the module pass completes.
    PA.intersect(std::move(PassPA));
  }

  // The FunctionAnalysisManagerModuleProxy is preserved because (we assume)
  // the function passes we ran didn't add or remove any functions.
  //
  // We also preserve all analyses on Functions, because we did all the
  // invalidation we needed to do above.
  PA.preserveSet<AllAnalysesOn<Function>>();
  PA.preserve<FunctionAnalysisManagerModuleProxy>();
  return PA;
}

AnalysisSetKey CFGAnalyses::SetKey;

AnalysisSetKey PreservedAnalyses::AllAnalysesKey;
