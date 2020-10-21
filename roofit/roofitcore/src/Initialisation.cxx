#include "RConfigure.h"
#include "TEnv.h"
#include "TSystem.h"

#include <iostream>
#include <vector>
#include <string>
#include <exception>

/**
\file Initialisation.cxx
\ingroup Roofitcore

Run static initialisers on first load of RooFitCore.
**/

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Print RooFit banner.
void doBanner() {
#ifndef __ROOFIT_NOBANNER
  if (gEnv->GetValue("RooFit.Banner", 1) == 0)
    return;

  /// RooFit version tag.
  constexpr char VTAG[] = "3.60";

  std::cout << '\n'
      << "\033[1mRooFit v" << VTAG << " -- Developed by Wouter Verkerke and David Kirkby\033[0m " << '\n'
      << "                Copyright (C) 2000-2013 NIKHEF, University of California & Stanford University" << '\n'
      << "                All rights reserved, please read http://roofit.sourceforge.net/license.txt" << '\n'
      << std::endl;
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Inspect cpu capabilities, and load architecture-specific libraries for RooFitCore/RooFit computations.
void loadComputeLibrary() {

  std::vector<std::string> libNames;
  libNames.push_back("libRooFitCompute");

  std::string libSuffix;

#ifdef R__RF_ARCHITECTURE_SPECIFIC_LIBS

  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2")) {
    libSuffix = "_AVX2";
  } else if (__builtin_cpu_supports("avx")) {
    libSuffix = "_AVX";
  } else if (__builtin_cpu_supports("sse4.1")) {
    libSuffix = "_SSE4.1";
  }

#if __GNUC__ > 5 || defined(__clang__)
  //skylake-avx512 support
  if (__builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq"))  {
    libSuffix = "_AVX512";
  }
#endif

  if (gEnv->GetValue("RooFit.LoadOptimisedComputationLibrary", 1) == 0) {
    libSuffix = "";
    if (gDebug>0) {
      std::cout << "In roofitcore/Initialisation.cxx:loadComputeLibrary(): RooFit.LoadOptimisedComputationLibrary is set to 0, using generic RooFitCompute library." << std::endl;
    }
  }

#else //R__RF_ARCHITECTURE_SPECIFIC_LIBS not defined

  if (gDebug>0) {
    std::cout << "In roofitcore/Initialisation.cxx:loadComputeLibrary(): Architecture specifics libraries not supported." << std::endl;
  }

#endif //R__RF_ARCHITECTURE_SPECIFIC_LIBS

  for (auto&& libName : libNames) {
    libName += libSuffix;
    const auto returnValue = gSystem->Load(libName.c_str());

    if (returnValue == -1 || returnValue == -2) {
      throw std::runtime_error("RooFit was unable to load its computation library " + libName);
    } else if (returnValue == 1) {
      // Library should not have been loaded before we tried to do it.
      throw std::logic_error("RooFit computation library " + libName + " was loaded before RooFit initialisation began.");
    } else if (gDebug>0) {
      std::cout << "In roofitcore/Initialisation.cxx:loadComputeLibrary(): Library " + libName + " was loaded successfully" << std::endl;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// A RAII that performs RooFit's static initialisation.
static struct RooFitInitialiser {
  RooFitInitialiser() {
    doBanner();
    loadComputeLibrary();
  }
} __rooFitInitialiser;

}
