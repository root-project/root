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
\file RooBatchCompute.cu
\class RbcClass
\ingroup Roobatchcompute

This file contains the code for cuda computations using the RooBatchCompute library.
**/

#include "RooBatchCompute.h"
#include "Batches.h"

#include "ROOT/RConfig.h"
#include "TError.h"

#include <algorithm>
#include <thrust/reduce.h>

#ifdef __CUDACC__
#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      Fatal((func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s", cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}
#endif

#ifndef RF_ARCH
#error "RF_ARCH should always be defined"
#endif

namespace RooBatchCompute {
namespace RF_ARCH {

std::vector<void (*)(Batches)> getFunctions();

/// This class overrides some RooBatchComputeInterface functions, for the
/// purpose of providing a cuda specific implementation of the library.
class RooBatchComputeClass : public RooBatchComputeInterface {
private:
   const std::vector<void (*)(Batches)> _computeFunctions;

public:
   RooBatchComputeClass() : _computeFunctions(getFunctions())
   {
      dispatchCUDA = this; // Set the dispatch pointer to this instance of the library upon loading
   }

   Architecture architecture() const override { return Architecture::RF_ARCH; };
   std::string architectureName() const override
   {
      // transform to lower case to match the original architecture name passed to the compiler
      std::string out = _QUOTE_(RF_ARCH);
      std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
      ;
      return out;
   };

   /** Compute multiple values using cuda kernels.
   This method creates a Batches object and passes it to the correct compute function.
   The compute function is launched as a cuda kernel.
   \param computer An enum specifying the compute function to be used.
   \param output The array where the computation results are stored.
   \param nEvents The number of events to be processed.
   \param varData A std::map containing the values of the variables involved in the computation.
   \param vars A std::vector containing pointers to the variables involved in the computation.
   \param extraArgs An optional std::vector containing extra double values that may participate in the computation. **/
   void compute(cudaStream_t *stream, Computer computer, RestrictArr output, size_t nEvents, const DataMap &varData,
                const VarVector &vars, const ArgVector &extraArgs) override
   {
      Batches batches(output, nEvents, varData, vars, extraArgs);
      _computeFunctions[computer]<<<128, 512, 0, *stream>>>(batches);
   }
   /// Return the sum of an input array
   double sumReduce(cudaStream_t *stream, InputArr input, size_t n) override
   {
      return thrust::reduce(thrust::cuda::par.on(*stream), input, input + n, 0.0);
   }

   // cuda functions
   virtual void *cudaMalloc(size_t nBytes)
   {
      void *ret;
      ERRCHECK(::cudaMalloc(&ret, nBytes));
      return ret;
   }
   virtual void cudaFree(void *ptr) { ERRCHECK(::cudaFree(ptr)); }
   virtual void *cudaMallocHost(size_t nBytes)
   {
      void *ret;
      ERRCHECK(::cudaMallocHost(&ret, nBytes));
      return ret;
   }
   virtual void cudaFreeHost(void *ptr) { ERRCHECK(::cudaFreeHost(ptr)); }
   virtual cudaEvent_t *newCudaEvent(bool forTiming)
   {
      auto ret = new cudaEvent_t;
      ERRCHECK(cudaEventCreateWithFlags(ret, forTiming ? 0 : cudaEventDisableTiming));
      return ret;
   }
   virtual void deleteCudaEvent(cudaEvent_t *event)
   {
      ERRCHECK(cudaEventDestroy(*event));
      delete event;
   }
   virtual void cudaEventRecord(cudaEvent_t *event, cudaStream_t *stream)
   {
      ERRCHECK(::cudaEventRecord(*event, *stream));
   }
   virtual cudaStream_t *newCudaStream()
   {
      auto ret = new cudaStream_t;
      ERRCHECK(cudaStreamCreate(ret));
      return ret;
   }
   virtual void deleteCudaStream(cudaStream_t *stream)
   {
      ERRCHECK(cudaStreamDestroy(*stream));
      delete stream;
   }
   virtual bool streamIsActive(cudaStream_t *stream)
   {
      cudaError_t err = cudaStreamQuery(*stream);
      if (err == cudaErrorNotReady)
         return true;
      else if (err == cudaSuccess)
         return false;
      ERRCHECK(err);
      return false;
   }
   virtual void cudaStreamWaitEvent(cudaStream_t *stream, cudaEvent_t *event)
   {
      ERRCHECK(::cudaStreamWaitEvent(*stream, *event));
   }
   virtual float cudaEventElapsedTime(cudaEvent_t *begin, cudaEvent_t *end)
   {
      float ret;
      ERRCHECK(::cudaEventElapsedTime(&ret, *begin, *end));
      return ret;
   }
   void memcpyToCUDA(void *dest, const void *src, size_t nBytes, cudaStream_t *stream) override
   {
      if (stream)
         ERRCHECK(cudaMemcpyAsync(dest, src, nBytes, cudaMemcpyHostToDevice, *stream));
      else
         ERRCHECK(cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice));
   }
   void memcpyToCPU(void *dest, const void *src, size_t nBytes, cudaStream_t *stream) override
   {
      if (stream)
         ERRCHECK(cudaMemcpyAsync(dest, src, nBytes, cudaMemcpyDeviceToHost, *stream));
      else
         ERRCHECK(cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost));
   }
}; // End class RooBatchComputeClass

/// Static object to trigger the constructor which overwrites the dispatch pointer.
static RooBatchComputeClass computeObj;

/** Construct a Batches object
\param output The array where the computation results are stored.
\param nEvents The number of events to be processed.
\param varData A std::map containing the values of the variables involved in the computation.
\param vars A std::vector containing pointers to the variables involved in the computation.
\param extraArgs An optional std::vector containing extra double values that may participate in the computation.
For every scalar parameter a `Batch` object inside the `Batches` object is set accordingly;
a data member of type double gets assigned the scalar value. This way, when the cuda kernel
is launched this scalar value gets copied automatically and thus no call to cudaMemcpy is needed **/
Batches::Batches(RestrictArr output, size_t nEvents, const DataMap &varData, const VarVector &vars,
                 const ArgVector &extraArgs, double[maxParams][bufferSize])
   : _nEvents(nEvents), _nBatches(vars.size()), _nExtraArgs(extraArgs.size()), _output(output)
{
   for (int i = 0; i < vars.size(); i++) {
      const RooSpan<const double> &span = varData.at(vars[i]);
      size_t size = span.size();
      if (size == 1)
         _arrays[i].set(span[0], nullptr, false);
      else
         _arrays[i].set(0.0, span.data(), true);
   }
   std::copy(extraArgs.cbegin(), extraArgs.cend(), _extraArgs);
}

} // End namespace RF_ARCH
} // End namespace RooBatchCompute
