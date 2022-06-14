/*
 * Project: RooFit
 * Authors:
 *   Emmanouil Michalainas, CERN, September 2020
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
\file RooBatchCompute.cxx
\class RbcClass
\ingroup Roobatchcompute

This file contains the code for cpu computations using the RooBatchCompute library.
**/

#include "RooBatchCompute.h"
#include "RooVDTHeaders.h"
#include "Batches.h"

#include "ROOT/RConfig.hxx"
#include "ROOT/TExecutor.hxx"

#include <algorithm>

#ifndef RF_ARCH
#error "RF_ARCH should always be defined"
#endif

namespace RooBatchCompute {
namespace RF_ARCH {

std::vector<void (*)(BatchesHandle)> getFunctions();

/// This class overrides some RooBatchComputeInterface functions, for the
/// purpose of providing a CPU specific implementation of the library.
class RooBatchComputeClass : public RooBatchComputeInterface {
private:
   const std::vector<void (*)(BatchesHandle)> _computeFunctions;

public:
   RooBatchComputeClass() : _computeFunctions(getFunctions())
   {
      // Set the dispatch pointer to this instance of the library upon loading
      dispatchCPU = this;
   }

   Architecture architecture() const override { return Architecture::RF_ARCH; };
   std::string architectureName() const override
   {
      // transform to lower case to match the original architecture name passed to the compiler
#ifdef _QUOTEVAL_ // to quote the value of the preprocessor macro instead of the name
#error "It's unexpected that _QUOTEVAL_ is defined at this point!"
#endif
#define _QUOTEVAL_(x) _QUOTE_(x)
      std::string out = _QUOTEVAL_(RF_ARCH);
#undef _QUOTEVAL_
      std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
      ;
      return out;
   };

   /** Compute multiple values using optimized functions.
   This method creates a Batches object and passes it to the correct compute function.
   In case Implicit Multithreading is enabled, the events to be processed are equally
   divided among the tasks to be generated and computed in parallel.
   \param computer An enum specifying the compute function to be used.
   \param output The array where the computation results are stored.
   \param nEvents The number of events to be processed.
   \param vars A std::vector containing pointers to the variables involved in the computation.
   \param extraArgs An optional std::vector containing extra double values that may participate in the computation. **/
   void compute(cudaStream_t *, Computer computer, RestrictArr output, size_t nEvents, const VarVector &vars,
                const ArgVector &extraArgs) override
   {
      static std::vector<double> buffer;
      buffer.resize(vars.size() * bufferSize);

      if (ROOT::IsImplicitMTEnabled()) {
         ROOT::Internal::TExecutor ex;
         unsigned int nThreads = ex.GetPoolSize();

         auto task = [&](std::size_t idx) -> int {
            // Fill a std::vector<Batches> with the same object and with ~nEvents/nThreads
            // Then advance every object but the first to split the work between threads
            Batches batches(output, nEvents / nThreads + (nEvents % nThreads > 0), vars, extraArgs, buffer.data());
            batches.advance(batches.getNEvents() * idx);

            // Set the number of events of the last Batches object as the remaining events
            if (idx == nThreads - 1) {
               batches.setNEvents(nEvents - idx * batches.getNEvents());
            }

            int events = batches.getNEvents();
            batches.setNEvents(bufferSize);
            while (events > bufferSize) {
               _computeFunctions[computer](batches);
               batches.advance(bufferSize);
               events -= bufferSize;
            }
            batches.setNEvents(events);
            _computeFunctions[computer](batches);
            return 0;
         };

         std::vector<std::size_t> indices(nThreads);
         for (unsigned int i = 1; i < nThreads; i++) {
            indices[i] = i;
         }
         ex.Map(task, indices);
      } else {
         // Fill a std::vector<Batches> with the same object and with ~nEvents/nThreads
         // Then advance every object but the first to split the work between threads
         Batches batches(output, nEvents, vars, extraArgs, buffer.data());

         int events = batches.getNEvents();
         batches.setNEvents(bufferSize);
         while (events > bufferSize) {
            _computeFunctions[computer](batches);
            batches.advance(bufferSize);
            events -= bufferSize;
         }
         batches.setNEvents(events);
         _computeFunctions[computer](batches);
      }
   }
   /// Return the sum of an input array
   double sumReduce(cudaStream_t *, InputArr input, size_t n) override
   {
      long double sum = 0.0;
      for (size_t i = 0; i < n; i++)
         sum += input[i];
      return sum;
   }
}; // End class RooBatchComputeClass

/// Static object to trigger the constructor which overwrites the dispatch pointer.
static RooBatchComputeClass computeObj;

/** Construct a Batches object
\param output The array where the computation results are stored.
\param nEvents The number of events to be processed.
\param vars A std::vector containing pointers to the variables involved in the computation.
\param extraArgs An optional std::vector containing extra double values that may participate in the computation.
\param buffer A 2D array that is used as a buffer for scalar variables.
For every scalar parameter a buffer (one row of the buffer) is filled with copies of the scalar
value, so that it behaves as a batch and facilitates auto-vectorization. The Batches object can be
passed by value to a compute function to perform efficient computations. **/
Batches::Batches(RestrictArr output, size_t nEvents, const VarVector &vars, const ArgVector &extraArgs, double *buffer)
   : _nEvents(nEvents), _nBatches(vars.size()), _nExtraArgs(extraArgs.size()), _output(output)
{
   _arrays.resize(vars.size());
   for (size_t i = 0; i < vars.size(); i++) {
      const RooSpan<const double> &span = vars[i];
      if (span.size() > 1)
         _arrays[i].set(span.data()[0], span.data(), true);
      else {
         std::fill_n(&buffer[i * bufferSize], bufferSize, span.data()[0]);
         _arrays[i].set(span.data()[0], &buffer[i * bufferSize], false);
      }
   }
   _extraArgs = extraArgs;
}

} // End namespace RF_ARCH
} // End namespace RooBatchCompute
