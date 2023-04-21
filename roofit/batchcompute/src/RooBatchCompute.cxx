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
#include "RooNaNPacker.h"
#include "RooVDTHeaders.h"
#include "Batches.h"

#include <ROOT/RConfig.hxx>
#include <ROOT/TExecutor.hxx>

#include <Math/Util.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

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
                ArgVector &extraArgs) override
   {
      static std::vector<double> buffer;
      buffer.resize(vars.size() * bufferSize);

      if (ROOT::IsImplicitMTEnabled()) {
         ROOT::Internal::TExecutor ex;
         std::size_t nThreads = ex.GetPoolSize();

         std::size_t nEventsPerThread = nEvents / nThreads + (nEvents % nThreads > 0);

         // Reset the number of threads to the number we actually need given nEventsPerThread
         nThreads = nEvents / nEventsPerThread + (nEvents % nEventsPerThread > 0);

         auto task = [&](std::size_t idx) -> int {
            // Fill a std::vector<Batches> with the same object and with ~nEvents/nThreads
            // Then advance every object but the first to split the work between threads
            Batches batches(output, nEventsPerThread, vars, extraArgs, buffer.data());
            batches.advance(batches.getNEvents() * idx);

            // Set the number of events of the last Batches object as the remaining events
            if (idx == nThreads - 1) {
               batches.setNEvents(nEvents - idx * batches.getNEvents());
            }

            std::size_t events = batches.getNEvents();
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

         std::size_t events = batches.getNEvents();
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
   double reduceSum(cudaStream_t *, InputArr input, size_t n) override;
   ReduceNLLOutput reduceNLL(cudaStream_t *, RooSpan<const double> probas, RooSpan<const double> weightSpan,
                             RooSpan<const double> weights, double weightSum,
                             RooSpan<const double> binVolumes) override;
}; // End class RooBatchComputeClass

namespace {

inline std::pair<double, double> getLog(double prob, ReduceNLLOutput &out)
{
   if (std::abs(prob) > 1e6) {
      out.nLargeValues++;
   }

   if (prob <= 0.0) {
      out.nNonPositiveValues++;
      return {std::log(prob), -prob};
   }

   if (std::isnan(prob)) {
      out.nNaNValues++;
      return {prob, RooNaNPacker::unpackNaN(prob)};
   }

   return {std::log(prob), 0.0};
}

} // namespace

double RooBatchComputeClass::reduceSum(cudaStream_t *, InputArr input, size_t n)
{
   return ROOT::Math::KahanSum<double, 4u>::Accumulate(input, input + n).Sum();
}

ReduceNLLOutput RooBatchComputeClass::reduceNLL(cudaStream_t *, RooSpan<const double> probas,
                                                RooSpan<const double> weightSpan, RooSpan<const double> weights,
                                                double weightSum, RooSpan<const double> binVolumes)
{
   ReduceNLLOutput out;

   double badness = 0.0;

   for (std::size_t i = 0; i < probas.size(); ++i) {

      const double eventWeight = weightSpan.size() > 1 ? weightSpan[i] : weightSpan[0];

      if (0. == eventWeight)
         continue;

      std::pair<double, double> logOut = getLog(probas[i], out);
      double term = logOut.first;
      badness += logOut.second;

      if (!binVolumes.empty()) {
         term -= std::log(weights[i]) - std::log(binVolumes[i]) - std::log(weightSum);
      }

      term *= -eventWeight;

      out.nllSum.Add(term);
   }

   if (badness != 0.) {
      // Some events with evaluation errors. Return "badness" of errors.
      out.nllSum = ROOT::Math::KahanSum<double>(RooNaNPacker::packFloatIntoNaN(badness));
   }

   return out;
}

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
Batches::Batches(RestrictArr output, size_t nEvents, const VarVector &vars, ArgVector &extraArgs, double *buffer)
   : _extraArgs{extraArgs.data()},
     _nEvents(nEvents),
     _nBatches(vars.size()),
     _nExtraArgs(extraArgs.size()),
     _output(output)
{
   _arrays.resize(vars.size());
   for (size_t i = 0; i < vars.size(); i++) {
      const RooSpan<const double> &span = vars[i];
      if (span.empty()) {
         std::stringstream ss;
         ss << "The span number " << i << " passed to Batches::Batches() is empty!";
         throw std::runtime_error(ss.str());
      } else if (span.size() > 1)
         _arrays[i].set(span.data()[0], span.data(), true);
      else {
         std::fill_n(&buffer[i * bufferSize], bufferSize, span.data()[0]);
         _arrays[i].set(span.data()[0], &buffer[i * bufferSize], false);
      }
   }
}

} // End namespace RF_ARCH
} // End namespace RooBatchCompute
