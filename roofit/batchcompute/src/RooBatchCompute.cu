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
#include <vector>

#ifndef RF_ARCH
#error "RF_ARCH should always be defined"
#endif

namespace CudaInterface = RooFit::Detail::CudaInterface;

namespace RooBatchCompute {
namespace RF_ARCH {

constexpr int blockSize = 512;

namespace {

void fillBatches(Batches &batches, double *output, size_t nEvents, std::size_t nBatches, std::size_t nExtraArgs)
{
   batches.nEvents = nEvents;
   batches.nBatches = nBatches;
   batches.nExtra = nExtraArgs;
   batches.output = output;
}

void fillArrays(Batch *arrays, VarSpan vars, double *buffer, double *bufferDevice, std::size_t nEvents)
{
   for (int i = 0; i < vars.size(); i++) {
      const std::span<const double> &span = vars[i];
      arrays[i]._isVector = span.empty() || span.size() >= nEvents;
      if (!arrays[i]._isVector) {
         // In the scalar case, the value is not on the GPU yet, so we have to
         // copy the value to the GPU buffer.
         buffer[i] = span[0];
         arrays[i]._array = bufferDevice + i;
      } else {
         // In the vector input cases, they are already on the GPU, so we can
         // fill be buffer with some dummy value and set the input span
         // directly.
         buffer[i] = 0.0;
         arrays[i]._array = span.data();
      }
   }
}

int getGridSize(std::size_t n)
{
   // The grid size should be not larger than the order of number of streaming
   // multiprocessors (SMs) in an Nvidia GPU. The number 84 was chosen because
   // the developers were using an Nvidia RTX A4500, which has 46 SMs. This was
   // multiplied by a factor of 1.5, as recommended by stackoverflow.
   //
   // But when there are not enough elements to load the GPU, the number should
   // be lower: that's why there is the std::ceil().
   //
   // Note: for grid sizes larger than 512, the Kahan summation kernels give
   // wrong results. This problem is not understood, but also not really worth
   // investigating further, as that number is unreasonably large anyway.
   constexpr int maxGridSize = 84;
   return std::min(int(std::ceil(double(n) / blockSize)), maxGridSize);
}

} // namespace

std::vector<void (*)(Batches &)> getFunctions();

/// This class overrides some RooBatchComputeInterface functions, for the
/// purpose of providing a cuda specific implementation of the library.
class RooBatchComputeClass : public RooBatchComputeInterface {
private:
   const std::vector<void (*)(Batches &)> _computeFunctions;

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
   \param vars A std::span containing pointers to the variables involved in the computation.
   \param extraArgs An optional std::span containing extra double values that may participate in the computation. **/
   void compute(RooBatchCompute::Config const &cfg, Computer computer, std::span<double> output, VarSpan vars,
                ArgSpan extraArgs) override
   {
      using namespace RooFit::Detail::CudaInterface;

      std::size_t nEvents = output.size();

      const std::size_t memSize = sizeof(Batches) + vars.size() * sizeof(Batch) + vars.size() * sizeof(double) +
                                  extraArgs.size() * sizeof(double);

      std::vector<char> hostMem(memSize);
      auto batches = reinterpret_cast<Batches *>(hostMem.data());
      auto arrays = reinterpret_cast<Batch *>(batches + 1);
      auto scalarBuffer = reinterpret_cast<double *>(arrays + vars.size());
      auto extraArgsHost = reinterpret_cast<double *>(scalarBuffer + vars.size());

      DeviceArray<char> deviceMem(memSize);
      auto batchesDevice = reinterpret_cast<Batches *>(deviceMem.data());
      auto arraysDevice = reinterpret_cast<Batch *>(batchesDevice + 1);
      auto scalarBufferDevice = reinterpret_cast<double *>(arraysDevice + vars.size());
      auto extraArgsDevice = reinterpret_cast<double *>(scalarBufferDevice + vars.size());

      fillBatches(*batches, output.data(), nEvents, vars.size(), extraArgs.size());
      fillArrays(arrays, vars, scalarBuffer, scalarBufferDevice, nEvents);
      batches->args = arraysDevice;

      if (!extraArgs.empty()) {
         std::copy(std::cbegin(extraArgs), std::cend(extraArgs), extraArgsHost);
         batches->extra = extraArgsDevice;
      }

      copyHostToDevice(hostMem.data(), deviceMem.data(), hostMem.size(), cfg.cudaStream());

      const int gridSize = getGridSize(nEvents);
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
                             std::span<const double> weights, std::span<const double> offsetProbas) override;
}; // End class RooBatchComputeClass

inline __device__ void kahanSumUpdate(double &sum, double &carry, double a, double otherCarry)
{
   // c is zero the first time around. Then is done a summation as the c variable is NEGATIVE
   const double y = a - (carry + otherCarry);
   const double t = sum + y; // Alas, sum is big, y small, so low-order digits of y are lost.

   // (t - sum) cancels the high-order part of y; subtracting y recovers NEGATIVE (low part of y)
   carry = (t - sum) - y;

   // Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
   sum = t;
}

// This is the same implementation of the ROOT::Math::KahanSum::operator+=(KahanSum) but in GPU
inline __device__ void kahanSumReduction(double *shared, size_t n, double *__restrict__ result, int carry_index)
{
   // Stride in first iteration = half of the block dim. Then the half of the half...
   for (int i = blockDim.x / 2; i > 0; i >>= 1) {
      if (threadIdx.x < i && (threadIdx.x + i) < n) {
         kahanSumUpdate(shared[threadIdx.x], shared[carry_index], shared[threadIdx.x + i], shared[carry_index + i]);
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
   const int nThreadsTotal = blockSize * gridDim.x;

   // The first half of the shared memory is for storing the summation and the second half for the carry or compensation
   extern __shared__ double shared[];

   double sum = 0.0;
   double carry = 0.0;

   for (int i = gthIdx; i < n; i += nThreadsTotal) {
      // Note: it does not make sense to use the nll option and provide at the
      // same time external carries.
      double val = nll == 1 ? -std::log(input[i]) : input[i];
      kahanSumUpdate(sum, carry, val, carries ? carries[i] : 0.0);
   }

   shared[thIdx] = sum;
   shared[carry_index] = carry;

   // Wait until all threads in each block have loaded their elements
   __syncthreads();

   kahanSumReduction(shared, n, result, carry_index);
}

__global__ void nllSumKernel(const double *__restrict__ probas, const double *__restrict__ weights,
                             const double *__restrict__ offsetProbas, size_t n, double *__restrict__ result)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   int carry_index = threadIdx.x + blockDim.x;
   const int nThreadsTotal = blockSize * gridDim.x;

   // The first half of the shared memory is for storing the summation and the second half for the carry or compensation
   extern __shared__ double shared[];

   double sum = 0.0;
   double carry = 0.0;

   for (int i = gthIdx; i < n; i += nThreadsTotal) {
      // Note: it does not make sense to use the nll option and provide at the
      // same time external carries.
      double val = -std::log(probas[i]);
      if (offsetProbas)
         val += std::log(offsetProbas[i]);
      if (weights)
         val = weights[i] * val;
      kahanSumUpdate(sum, carry, val, 0.0);
   }

   shared[thIdx] = sum;
   shared[carry_index] = carry;

   // Wait until all threads in each block have loaded their elements
   __syncthreads();

   kahanSumReduction(shared, n, result, carry_index);
}

double RooBatchComputeClass::reduceSum(RooBatchCompute::Config const &cfg, InputArr input, size_t n)
{
   if (n == 0)
      return 0.0;
   const int gridSize = getGridSize(n);
   cudaStream_t stream = *cfg.cudaStream();
   CudaInterface::DeviceArray<double> devOut(2 * gridSize);
   constexpr int shMemSize = 2 * blockSize * sizeof(double);
   kahanSum<<<gridSize, blockSize, shMemSize, stream>>>(input, nullptr, n, devOut.data(), 0);
   kahanSum<<<1, blockSize, shMemSize, stream>>>(devOut.data(), devOut.data() + gridSize, gridSize, devOut.data(), 0);
   double tmp = 0.0;
   CudaInterface::copyDeviceToHost(devOut.data(), &tmp, 1, cfg.cudaStream());
   return tmp;
}

ReduceNLLOutput RooBatchComputeClass::reduceNLL(RooBatchCompute::Config const &cfg, std::span<const double> probas,
                                                std::span<const double> weights, std::span<const double> offsetProbas)
{
   ReduceNLLOutput out;
   if (probas.empty()) {
      return out;
   }
   const int gridSize = getGridSize(probas.size());
   CudaInterface::DeviceArray<double> devOut(2 * gridSize);
   cudaStream_t stream = *cfg.cudaStream();
   constexpr int shMemSize = 2 * blockSize * sizeof(double);

   nllSumKernel<<<gridSize, blockSize, shMemSize, stream>>>(
      probas.data(), weights.size() == 1 ? nullptr : weights.data(),
      offsetProbas.empty() ? nullptr : offsetProbas.data(), probas.size(), devOut.data());

   kahanSum<<<1, blockSize, shMemSize, stream>>>(devOut.data(), devOut.data() + gridSize, gridSize, devOut.data(), 0);

   double tmpSum = 0.0;
   double tmpCarry = 0.0;
   CudaInterface::copyDeviceToHost(devOut.data(), &tmpSum, 1, cfg.cudaStream());
   CudaInterface::copyDeviceToHost(devOut.data() + 1, &tmpCarry, 1, cfg.cudaStream());

   if (weights.size() == 1) {
      tmpSum *= weights[0];
      tmpCarry *= weights[0];
   }

   out.nllSum = tmpSum;
   out.nllSumCarry = tmpCarry;
   return out;
}

/// Static object to trigger the constructor which overwrites the dispatch pointer.
static RooBatchComputeClass computeObj;

} // End namespace RF_ARCH
} // End namespace RooBatchCompute
