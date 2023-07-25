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

#include "ROOT/RConfig.hxx"
#include "TError.h"

#include <algorithm>

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

constexpr int gridSize = 128;
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
      ;
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
   void compute(cudaStream_t *stream, Computer computer, RestrictArr output, size_t nEvents, const VarVector &vars,
                ArgVector &extraArgs) override
   {
      Batches batches(output, nEvents, vars, extraArgs);
      _computeFunctions[computer]<<<gridSize, blockSize, 0, *stream>>>(batches);
   }
   /// Return the sum of an input array
   double reduceSum(cudaStream_t *stream, InputArr input, size_t n) override;
   ReduceNLLOutput reduceNLL(cudaStream_t *, RooSpan<const double> probas, RooSpan<const double> weightSpan,
                             RooSpan<const double> weights, double weightSum,
                             RooSpan<const double> binVolumes) override;

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
      ERRCHECK(::cudaStreamWaitEvent(*stream, *event, 0));
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

template <class T>
class DeviceArray {
public:
   DeviceArray(std::size_t n) : _size{n} { cudaMalloc(reinterpret_cast<void **>(&_deviceArray), n * sizeof(T)); }
   DeviceArray(T const *hostArray, std::size_t n) : _size{n}
   {
      cudaMalloc((void **)&_deviceArray, n * sizeof(T));
      cudaMemcpy(_deviceArray, hostArray, n * sizeof(T), cudaMemcpyHostToDevice);
   }
   DeviceArray(DeviceArray const &other) = delete;
   DeviceArray &operator=(DeviceArray const &other) = delete;
   ~DeviceArray() { cudaFree(_deviceArray); }

   std::size_t size() const { return _size; }
   T *data() { return _deviceArray; }
   T const *data() const { return _deviceArray; }

   void copyBack(T *hostArray, std::size_t n)
   {
      cudaMemcpy(hostArray, _deviceArray, sizeof(T) * n, cudaMemcpyDeviceToHost);
   }

private:
   T *_deviceArray = nullptr;
   std::size_t _size = 0;
};

template <class T, class U>
__global__ void sumMultiBlock(const T *__restrict__ gArr, int arraySize, U *__restrict__ gOut)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   const int gridSize = blockSize * gridDim.x;
   U sum = 0;
   for (int i = gthIdx; i < arraySize; i += gridSize)
      sum += gArr[i];
   __shared__ U shArr[blockSize];
   shArr[thIdx] = sum;
   __syncthreads();
   for (int size = blockSize / 2; size > 0; size /= 2) { // uniform
      if (thIdx < size)
         shArr[thIdx] += shArr[thIdx + size];
      __syncthreads();
   }
   if (thIdx == 0)
      gOut[blockIdx.x] = shArr[0];
}

__global__ void nllSumMultiBlock(const double *__restrict__ probas, int probasSize, double *__restrict__ out)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   const int gridSize = blockSize * gridDim.x;
   double sum = 0.0;
   for (int i = gthIdx; i < probasSize; i += gridSize) {
      sum -= std::log(probas[i]);
   }
   __shared__ double shArr[blockSize];
   shArr[thIdx] = sum;
   __syncthreads();
   for (int size = blockSize / 2; size > 0; size /= 2) { // uniform
      if (thIdx < size)
         shArr[thIdx] += shArr[thIdx + size];
      __syncthreads();
   }
   if (thIdx == 0)
      out[blockIdx.x] = shArr[0];
}

__global__ void nllSumWeightedMultiBlock(const double *__restrict__ probas, const double *__restrict__ weights,
                                         int probasSize, double *__restrict__ out)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   const int gridSize = blockSize * gridDim.x;
   double sum = 0.0;
   for (int i = gthIdx; i < probasSize; i += gridSize) {
      if (weights[i] != 0.0) {
         sum -= weights[i] * std::log(probas[i]);
      }
   }
   __shared__ double shArr[blockSize];
   shArr[thIdx] = sum;
   __syncthreads();
   for (int size = blockSize / 2; size > 0; size /= 2) { // uniform
      if (thIdx < size)
         shArr[thIdx] += shArr[thIdx + size];
      __syncthreads();
   }
   if (thIdx == 0)
      out[blockIdx.x] = shArr[0];
}

double RooBatchComputeClass::reduceSum(cudaStream_t *stream, InputArr input, size_t n)
{
   DeviceArray<double> devOut{gridSize};
   double tmp = 0.0;
   sumMultiBlock<<<gridSize, blockSize, 0, *stream>>>(input, n, devOut.data());
   sumMultiBlock<<<1, blockSize, 0, *stream>>>(devOut.data(), gridSize, devOut.data());
   devOut.copyBack(&tmp, 1);
   return tmp;
}

ReduceNLLOutput RooBatchComputeClass::reduceNLL(cudaStream_t *stream, RooSpan<const double> probas,
                                                RooSpan<const double> weightSpan, RooSpan<const double> weights,
                                                double weightSum, RooSpan<const double> binVolumes)
{
   ReduceNLLOutput out;
   DeviceArray<double> devOut{gridSize};
   double tmp = 0.0;

   if (weightSpan.size() == 1) {
      nllSumMultiBlock<<<gridSize, blockSize, 0, *stream>>>(probas.data(), probas.size(), devOut.data());
      sumMultiBlock<<<1, blockSize, 0, *stream>>>(devOut.data(), gridSize, devOut.data());
      devOut.copyBack(&tmp, 1);
      tmp *= weightSpan[0];
   } else {
      nllSumWeightedMultiBlock<<<gridSize, blockSize, 0, *stream>>>(probas.data(), weightSpan.data(), probas.size(),
                                                                    devOut.data());
      sumMultiBlock<<<1, blockSize, 0, *stream>>>(devOut.data(), gridSize, devOut.data());
      devOut.copyBack(&tmp, 1);
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
