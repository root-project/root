/*
 * Project: RooFit
 * Authors:
 *   Emmanouil Michalainas, CERN, December 2018
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooBatchCompute.h"

#include "TEnv.h"
#include "TSystem.h"
#include "TError.h"

#include <string>
#include <exception>

// First initialisation of the pointers. When implementations of the batch compute library
// are loaded, they will overwrite the pointers.
RooBatchCompute::RooBatchComputeInterface *RooBatchCompute::dispatchCPU = nullptr;
RooBatchCompute::RooBatchComputeInterface *RooBatchCompute::dispatchCUDA = nullptr;

namespace {

/// Dynamically load a library and throw exception in case of failure
void loadWithErrorChecking(const std::string &libName)
{
   const auto returnValue = gSystem->Load(libName.c_str());
   if (returnValue == -1 || returnValue == -2)
      throw std::runtime_error("RooFit was unable to load its computation library " + libName);
   else if (returnValue == 1) // Library should not have been loaded before we tried to do it.
      throw std::logic_error("RooFit computation library " + libName +
                             " was loaded before RooFit initialisation began.");
}

} // end anonymous namespace

namespace RooBatchCompute {

/// Inspect hardware capabilities, and load the optimal library for RooFit computations.
void init()
{
   // Check if the library was not initialised already
   static bool isInitialised = false;
   if (isInitialised)
      return;
   isInitialised = true;

   const std::string userChoice = gEnv->GetValue("RooFit.BatchCompute", "auto");
#ifdef R__RF_ARCHITECTURE_SPECIFIC_LIBS
#ifdef R__HAS_CUDA
   if(gSystem->Load("libRooBatchCompute_CUDA") != 0) {
      RooBatchCompute::dispatchCUDA = nullptr;
   }
#endif // R__HAS_CUDA

   __builtin_cpu_init();
#if __GNUC__ > 5 || defined(__clang__)
   bool supported_avx512 = __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512vl") &&
                           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq");
#else
   bool supported_avx512 = false;
#endif

   if (userChoice == "auto") {
      if (supported_avx512)
         loadWithErrorChecking("libRooBatchCompute_AVX512");
      else if (__builtin_cpu_supports("avx2"))
         loadWithErrorChecking("libRooBatchCompute_AVX2");
      else if (__builtin_cpu_supports("avx"))
         loadWithErrorChecking("libRooBatchCompute_AVX");
      else if (__builtin_cpu_supports("sse4.1"))
         loadWithErrorChecking("libRooBatchCompute_SSE4.1");
   } else if (userChoice == "avx512")
      loadWithErrorChecking("libRooBatchCompute_AVX512");
   else if (userChoice == "avx2")
      loadWithErrorChecking("libRooBatchCompute_AVX2");
   else if (userChoice == "avx")
      loadWithErrorChecking("libRooBatchCompute_AVX");
   else if (userChoice == "sse")
      loadWithErrorChecking("libRooBatchCompute_SSE4.1");
   else if (userChoice != "generic")
      throw std::invalid_argument(
         "Supported options for `RooFit.BatchCompute` are `auto`, `avx512`, `avx2`, `avx`, `sse`, `generic`.");
#endif // R__RF_ARCHITECTURE_SPECIFIC_LIBS

   if (RooBatchCompute::dispatchCPU == nullptr)
      loadWithErrorChecking("libRooBatchCompute_GENERIC");
}

} // namespace RooBatchCompute
