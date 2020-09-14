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
//  libNames.push_back("libRooFitCoreCompute");
//  libNames.push_back("libRooFitCompute");
#ifdef R__HAS_MATHMORE
//  libNames.push_back("libRooFitMoreCompute");
#endif
  {
    // Try to load HistFactory compute library as well. Need to check first if it exists.
    TString libName("libHistFactory");
//    if (gSystem->FindDynamicLibrary(libName, true) != nullptr)
//      libNames.push_back("libHistFactoryCompute");
  }

  std::string libSuffix;
#if defined(R__RF_ARCHITECTURE_SPECIFIC_LIBS) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx2")) {
    libSuffix = "_AVX2";
  } else if (__builtin_cpu_supports("avx")) {
    libSuffix = "_AVX";
  } else if (__builtin_cpu_supports("sse4.1")) {
    libSuffix = "_SSE4.1";
  }

#if __GNUC__ > 5 || defined(__clang__)
  if (__builtin_cpu_supports("avx512f")) {
    libSuffix = "_AVX512f";
  }
#endif

#endif

  for (auto&& libName : libNames) {
    libName += libSuffix;
    const auto returnValue = gSystem->Load(libName.c_str());

    if (returnValue == -1 || returnValue == -2) {
      throw std::runtime_error("RooFit was unable to load its computation library " + libName);
    }
    // Library should not have been loaded before we tried to do it.
    if (returnValue == 1) {
      throw std::logic_error("RooFit computation library " + libName + " was loaded before RooFit initialisation began.");
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
