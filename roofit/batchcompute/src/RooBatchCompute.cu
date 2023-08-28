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

namespace {

void fillBatches(Batches &batches, RestrictArr output, size_t nEvents, std::size_t nBatches, std::size_t nExtraArgs)
{
   batches._nEvents = nEvents;
   batches._nBatches = nBatches;
   batches._nExtraArgs = nExtraArgs;
   batches._output = output;
}

void fillArrays(Batch *arrays, const VarVector &vars)
{
   for (int i = 0; i < vars.size(); i++) {
      const std::span<const double> &span = vars[i];
      size_t size = span.size();
      if (size == 1)
         arrays[i].set(span[0], nullptr, false);
      else
         arrays[i].set(0.0, span.data(), true);
   }
}

} // namespace

std::vector<void (*)(BatchesHandle)> getFunctions();

/// This class overrides some RooBatchComputeInterface functions, for the
/// purpose of providing a cuda specific implementation of the library.
class RooBatchComputeClass : public RooBatchComputeInterface {
private:
   const std::vector<void (*)(BatchesHandle)> _computeFunctions;

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
      using namespace RooFit::Detail::CudaInterface;

      const std::size_t memSize = sizeof(Batches) + vars.size() * sizeof(Batch) + extraArgs.size() * sizeof(double);

      std::vector<char> hostMem(memSize);
      auto batches = reinterpret_cast<Batches *>(hostMem.data());
      auto arrays = reinterpret_cast<Batch *>(batches + 1);
      auto extraArgsHost = reinterpret_cast<double *>(arrays + vars.size());

      DeviceArray<char> deviceMem(memSize);
      auto batchesDevice = reinterpret_cast<Batches *>(deviceMem.data());
      auto arraysDevice = reinterpret_cast<Batch *>(batchesDevice + 1);
      auto extraArgsDevice = reinterpret_cast<double *>(arraysDevice + vars.size());

      fillBatches(*batches, output, nEvents, vars.size(), extraArgs.size());
      fillArrays(arrays, vars);
      batches->_arrays = arraysDevice;

      if (!extraArgs.empty()) {
         std::copy(std::cbegin(extraArgs), std::cend(extraArgs), extraArgsHost);
         batches->_extraArgs = extraArgsDevice;
      }

      copyHostToDevice(hostMem.data(), deviceMem.data(), hostMem.size(), cfg.cudaStream());

      const int gridSize = std::ceil(double(nEvents) / blockSize);
      _computeFunctions[computer]<<<gridSize, blockSize, 0, *cfg.cudaStream()>>>(*batchesDevice);

      // The compute might have modified the mutable extra args, so we need to
      // copy them back. This can be optimized if necessary in the future by
      // flagging if the extra args were actually changed.
      if (!extraArgs.empty()) {
         copyDeviceToHost(extraArgsDevice, extraArgs.data(), extraArgs.size(), cfg.cudaStream());
      }
   }
   /// Return the sum of an input array
   double reduceSum(RooBatchCompute::Config const &cfg, InputArr input, size_t n) override;
   ReduceNLLOutput reduceNLL(RooBatchCompute::Config const &cfg, std::span<const double> probas,
                             std::span<const double> weightSpan, std::span<const double> weights, double weightSum,
                             std::span<const double> binVolumes) override;
}; // End class RooBatchComputeClass

// This is the same implementation of the ROOT::Math::KahanSum::operator+=(KahanSum) but in GPU
inline __device__ void KahanSumAlgorithm(double *shared, size_t n, double *__restrict__ result, int carry_index)
{
   // Stride in first iteration = half of the block dim. Then the half of the half...
   for (int i = blockDim.x / 2; i > 0; i >>= 1) {
      if (threadIdx.x < i && (threadIdx.x + i) < n) {
         const double sum = shared[threadIdx.x];
         const double a = shared[threadIdx.x + i];

         // c is zero the first time around. Then is done a summation as the c variable is NEGATIVE
         const double y = a - (shared[carry_index] + shared[carry_index + i]);
         const double t = sum + y; // Alas, sum is big, y small, so low-order digits of y are lost.

         // (t - sum) cancels the high-order part of y; subtracting y recovers NEGATIVE (low part of y)
         shared[carry_index] = (t - sum) - y;

         // Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
         shared[threadIdx.x] = t;
      }
      __syncthreads();
   } // Next time around, the lost low part will be added to y in a fresh attempt.
     // Wait until all threads of the block have finished its work

   if (threadIdx.x == 0) {
      result[blockIdx.x] = shared[0];
      result[blockIdx.x + gridDim.x] = shared[carry_index];
   }
}

__global__ void kahanSum(const double *__restrict__ input, const double *__restrict__ carries, size_t n,
                         double *__restrict__ result, bool nll)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   int carry_index = threadIdx.x + blockDim.x;

   // The first half of the shared memory is for storing the summation and the second half for the carry or compensation
   extern __shared__ double shared[];

   if (gthIdx < n) {
      // In shared memory only indexes from 0-blockDim.x are available
      shared[thIdx] = nll == 1 ? -std::log(input[gthIdx]) : input[gthIdx];
      shared[carry_index] = carries ? carries[gthIdx] : 0.0; // A running compensation for lost low-order bits.
   } else {
      shared[thIdx] = 0.0;
      shared[carry_index] = 0.0;
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

   if (gthIdx < n) {
      // In shared memory only indexes from 0-blockDim.x are available
      shared[thIdx] = -std::log(input[gthIdx]) * weights[gthIdx];
   } else {
      shared[thIdx] = 0.0;
   }
   shared[carry_index] = 0.0; // A running compensation for lost low-order bits.

   // Wait until all threads in each block have loaded their elements
   __syncthreads();

   KahanSumAlgorithm(shared, n, result, carry_index);
}

double RooBatchComputeClass::reduceSum(RooBatchCompute::Config const &cfg, InputArr input, size_t n)
{
   const int gridSize = std::ceil(double(n) / blockSize);
   cudaStream_t stream = *cfg.cudaStream();
   CudaInterface::DeviceArray<double> devOut(2 * gridSize);
   const int shMemSize = 2 * blockSize * sizeof(double);
   kahanSum<<<gridSize, blockSize, shMemSize, stream>>>(input, nullptr, n, devOut.data(), 0);
   kahanSum<<<1, blockSize, shMemSize, stream>>>(devOut.data(), devOut.data() + gridSize, gridSize, devOut.data(), 0);
   double tmp = 0.0;
   CudaInterface::copyDeviceToHost(devOut.data(), &tmp, 1);
   return tmp;
}

ReduceNLLOutput RooBatchComputeClass::reduceNLL(RooBatchCompute::Config const &cfg, std::span<const double> probas,
                                                std::span<const double> weightSpan, std::span<const double> weights,
                                                double weightSum, std::span<const double> binVolumes)
{
   ReduceNLLOutput out;
   const int gridSize = std::ceil(double(probas.size()) / blockSize);
   CudaInterface::DeviceArray<double> devOut(2 * gridSize);
   cudaStream_t stream = *cfg.cudaStream();
   const int shMemSize = 2 * blockSize * sizeof(double);

   if (weightSpan.size() == 1) {
      kahanSum<<<gridSize, blockSize, shMemSize, stream>>>(probas.data(), nullptr, probas.size(), devOut.data(), 1);
   } else {
      kahanSumWeighted<<<gridSize, blockSize, shMemSize, stream>>>(probas.data(), weightSpan.data(), probas.size(),
                                                                   devOut.data());
   }

   kahanSum<<<1, blockSize, shMemSize, stream>>>(devOut.data(), devOut.data() + gridSize, gridSize, devOut.data(), 0);

   double tmpSum = 0.0;
   double tmpCarry = 0.0;
   CudaInterface::copyDeviceToHost(devOut.data(), &tmpSum, 1);
   CudaInterface::copyDeviceToHost(devOut.data() + 1, &tmpCarry, 1);

   if (weightSpan.size() == 1) {
      tmpSum *= weightSpan[0];
      tmpCarry *= weightSpan[0];
   }

   out.nllSum = ROOT::Math::KahanSum<double>{tmpSum, tmpCarry};
   return out;
}

/// Static object to trigger the constructor which overwrites the dispatch pointer.
static RooBatchComputeClass computeObj;

} // End namespace RF_ARCH
} // End namespace RooBatchCompute
