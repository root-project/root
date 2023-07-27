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

#include <ROOT/RConfig.hxx>
#include <TError.h>

#include <algorithm>
#include <iostream>

#ifndef RF_ARCH
#error "RF_ARCH should always be defined"
#endif

namespace CudaInterface = RooFit::Detail::CudaInterface;

namespace RooBatchCompute {
namespace RF_ARCH {

constexpr int blockSize = 512;

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
      return out;
   };

   /** Compute multiple values using cuda kernels.
   This method creates a Batches object and passes it to the correct compute function.
   The compute function is launched as a cuda kernel.
   \param computer An enum specifying the compute function to be used.
   \param output The array where the computation results are stored.
   \param nEvents The number of events to be processed.
   \param vars A std::vector containing pointers to the variables involved in the computation.
   \param extraArgs An optional std::vector containing extra double values that may participate in the computation. **/
   void compute(RooBatchCompute::Config const &cfg, Computer computer, RestrictArr output, size_t nEvents,
                const VarVector &vars, ArgVector &extraArgs) override
   {
      Batches batches(output, nEvents, vars, extraArgs);
      const int gridSize = std::ceil(double(nEvents) / blockSize);
      _computeFunctions[computer]<<<gridSize, blockSize, 0, *cfg.cudaStream()>>>(batches);
   }
   /// Return the sum of an input array
   double reduceSum(RooBatchCompute::Config const &cfg, InputArr input, size_t n) override;
   ReduceNLLOutput reduceNLL(RooBatchCompute::Config const &cfg, RooSpan<const double> probas,
                             RooSpan<const double> weightSpan, RooSpan<const double> weights, double weightSum,
                             RooSpan<const double> binVolumes) override;
}; // End class RooBatchComputeClass

// Guarantees unique access to this shared memory
inline __device__ double atomicAddDouble(double *address, double val)
{
   unsigned long long int *address_as_ull = (unsigned long long int *)address;
   unsigned long long int old = *address_as_ull, assumed;
   do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
   } while (assumed != old);
   return __longlong_as_double(old);
}

// This is the same implementation of the ROOT::Math::KahanSum::operator+=(KahanSum) but in GPU
inline __device__ void KahanSumAlgorithm(double *shared, size_t n, double *__restrict__ result, int carry_index)
{
   double sum, y, t, a;

   // Stride in first iteration = half of the block dim. Then the half of the half...
   for (int i = blockDim.x / 2; i > 0; i >>= 1) {
      if (threadIdx.x < i && (threadIdx.x + i) < n) {
         sum = shared[threadIdx.x];
         a = shared[threadIdx.x + i];

         // c is zero the first time around. Then is done a summation as the c variable is NEGATIVE
         y = a - (shared[carry_index] + shared[carry_index + i]);
         t = sum + y; // Alas, sum is big, y small, so low-order digits of y are lost.

         // (t - sum) cancels the high-order part of y; subtracting y recovers NEGATIVE (low part of y)
         shared[carry_index] = (t - sum) - y;

         // Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
         shared[threadIdx.x] = t;
      }
      __syncthreads();
   } // Next time around, the lost low part will be added to y in a fresh attempt.
     // Wait until all threads of the block have finished its work

   if (threadIdx.x == 0 && shared[0] != 0.0) {
      atomicAddDouble(result, shared[0]);
   }
}

__global__ void kahanSum(const double *__restrict__ input, size_t n, double *__restrict__ result, bool nll)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   int carry_index = threadIdx.x + blockDim.x;

   // The first half of the shared memory is for storing the summation and the second half for the carry or compensation
   extern __shared__ double shared[];

   if (gthIdx == 0)
      *result = 0.0;

   if (gthIdx < n) {
      // In shared memory only indexes from 0-blockDim.x are available
      if (nll == 1)
         shared[thIdx] = nll == 1 ? -std::log(input[gthIdx]) : 0;
      else
         shared[thIdx] = input[gthIdx];
      shared[carry_index] = 0.0f; // A running compensation for lost low-order bits.
   } else {
      shared[thIdx] = 0.0f;
      shared[carry_index] = 0.0f;
   }

   // Wait until all threads in each block have loaded their elements
   __syncthreads();

   KahanSumAlgorithm(shared, n, result, carry_index);
}

__global__ void kahanSumWeighted(const double *__restrict__ input, const double *__restrict__ weights, size_t n,
                                 double *__restrict__ result)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   int carry_index = threadIdx.x + blockDim.x;

   // The first half of the shared memory is for storing the summation and the second half for the carry or compensation
   extern __shared__ double shared[];

   if (gthIdx == 0)
      *result = 0.0;

   if (gthIdx < n) {
      // In shared memory only indexes from 0-blockDim.x are available
      shared[thIdx] = -std::log(input[gthIdx]) * weights[gthIdx];
      shared[carry_index] = 0.0f; // A running compensation for lost low-order bits.
   } else {
      shared[thIdx] = 0.0f;
      shared[carry_index] = 0.0f;
   }

   // Wait until all threads in each block have loaded their elements
   __syncthreads();

   KahanSumAlgorithm(shared, n, result, carry_index);
}

double RooBatchComputeClass::reduceSum(RooBatchCompute::Config const &cfg, InputArr input, size_t n)
{
   const int gridSize = std::ceil(double(n) / blockSize);
   cudaStream_t stream = *cfg.cudaStream();
   CudaInterface::DeviceArray<double> devOut{1};
   double tmp = 0.0;
   kahanSum<<<gridSize, blockSize, 2 * blockSize * sizeof(double), stream>>>(input, n, devOut.data(), 0);
   CudaInterface::copyDeviceToHost(devOut.data(), &tmp, 1);
   return tmp;
}

ReduceNLLOutput RooBatchComputeClass::reduceNLL(RooBatchCompute::Config const &cfg, RooSpan<const double> probas,
                                                RooSpan<const double> weightSpan, RooSpan<const double> weights,
                                                double weightSum, RooSpan<const double> binVolumes)
{
   ReduceNLLOutput out;
   const int gridSize = std::ceil(double(probas.size()) / blockSize);
   CudaInterface::DeviceArray<double> devOut(gridSize);
   double tmp = 0.0;
   cudaStream_t stream = *cfg.cudaStream();

   if (weightSpan.size() == 1) {
      kahanSum<<<gridSize, blockSize, 2 * blockSize * sizeof(double), stream>>>(probas.data(), probas.size(),
                                                                                devOut.data(), 1);
      CudaInterface::copyDeviceToHost(devOut.data(), &tmp, 1);
      tmp *= weightSpan[0];
   } else {
      kahanSumWeighted<<<gridSize, blockSize, 2 * blockSize * sizeof(double), stream>>>(
         probas.data(), weightSpan.data(), probas.size(), devOut.data());
      CudaInterface::copyDeviceToHost(devOut.data(), &tmp, 1);
   }

   out.nllSum.Add(tmp);
   return out;
}

/// Static object to trigger the constructor which overwrites the dispatch pointer.
static RooBatchComputeClass computeObj;

/** Construct a Batches object
\param output The array where the computation results are stored.
\param nEvents The number of events to be processed.
\param vars A std::vector containing pointers to the variables involved in the computation.
\param extraArgs An optional std::vector containing extra double values that may participate in the computation.
For every scalar parameter a `Batch` object inside the `Batches` object is set accordingly;
a data member of type double gets assigned the scalar value. This way, when the cuda kernel
is launched this scalar value gets copied automatically and thus no call to cudaMemcpy is needed **/
Batches::Batches(RestrictArr output, size_t nEvents, const VarVector &vars, ArgVector &extraArgs, double *)
   : _nEvents(nEvents), _nBatches(vars.size()), _nExtraArgs(extraArgs.size()), _output(output)
{
   if (vars.size() > maxParams) {
      throw std::runtime_error(std::string("Size of vars is ") + std::to_string(vars.size()) +
                               ", which is larger than maxParams = " + std::to_string(maxParams) + "!");
   }
   if (extraArgs.size() > maxExtraArgs) {
      throw std::runtime_error(std::string("Size of extraArgs is ") + std::to_string(extraArgs.size()) +
                               ", which is larger than maxExtraArgs = " + std::to_string(maxExtraArgs) + "!");
   }

   for (int i = 0; i < vars.size(); i++) {
      const RooSpan<const double> &span = vars[i];
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
