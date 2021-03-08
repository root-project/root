#include "RooBatchCompute.h"

#include "TEnv.h"
#include "TSystem.h"

#include <iostream>
#include <string>
#include <exception>


/**
 * The dispatch pointer points to the instance of the compute library in use, provided it has been loaded. 
 * The pointer is of type RooBatchComputeInterface*, so that calling functions through it are always virtual calls.
 * \see RooBatchComputeInterface, RooBatchComputeClass, RF_ARCH
 */
RooBatchCompute::RooBatchComputeInterface* RooBatchCompute::dispatch=nullptr;

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dynamically load a library and throw exception in case of failure
void loadWithErrorChecking(const std::string& libName)
{
  const auto returnValue = gSystem->Load(libName.c_str());
  if (returnValue == -1 || returnValue == -2) {
    throw std::runtime_error("RooFit was unable to load its computation library " + libName);
  } else if (returnValue == 1) {
    // Library should not have been loaded before we tried to do it.
    throw std::logic_error("RooFit computation library " + libName + " was loaded before RooFit initialisation began.");
  } 
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Inspect hardware capabilities, and load the optimal library for RooFit computations.
void loadComputeLibrary() {

#ifdef R__RF_ARCHITECTURE_SPECIFIC_LIBS

  // Check if user has requested a specific library architecture in .rootrc
  const std::string userChoice = gEnv->GetValue("RooFit.ComputationLibraryArch","auto");
  if (userChoice!="auto" && userChoice!="avx512" && userChoice!="avx2" && userChoice!="avx" && userChoice!="sse4.1" && userChoice!="generic")
    throw std::invalid_argument("Supported options for `RooFit.ComputationLibraryArch` are `auto`, `avx512`, `avx2`, `avx`, `sse4.1`, `generic`.");
  
  __builtin_cpu_init();
#if __GNUC__ > 5 || defined(__clang__)
  if (__builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq")
  && (userChoice=="avx512" || userChoice=="auto") )  {
    loadWithErrorChecking("libRooBatchCompute_AVX512");
    return;
  } else
#endif
  if (__builtin_cpu_supports("avx2") && (userChoice=="avx2" || userChoice=="auto")) {
    loadWithErrorChecking("libRooBatchCompute_AVX2");
    return;
  } else if (__builtin_cpu_supports("avx") && (userChoice=="avx" || userChoice=="auto")) {
    loadWithErrorChecking("libRooBatchCompute_AVX");
    return;
  } else if (__builtin_cpu_supports("sse4.1") && (userChoice=="sse4.1" || userChoice=="auto")) {
    loadWithErrorChecking("libRooBatchCompute_SSE4.1");
    return;
  }
  
#endif //R__RF_ARCHITECTURE_SPECIFIC_LIBS

  if (gDebug>0) {
    std::cout << "In " << __func__ << "(), " << __FILE__ << ":" << __LINE__ << ": Vector instruction sets not supported, using generic implementation." << std::endl;
  }
  loadWithErrorChecking("libRooBatchCompute_GENERIC");

}

} //end anonymous namespace

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// A RAII that performs RooFit's static initialisation.
static struct RooBatchComputeInitialiser {
  RooBatchComputeInitialiser() {
    loadComputeLibrary();
  }
} __RooBatchComputeInitialiser;

