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
\ingroup roofit_dev_docs_batchcompute

This file contains the code for cuda computations using the RooBatchCompute library.
**/

#include "RooBatchCompute.h"
#include "Batches.h"
#include "CudaInterface.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <map>
#include <queue>
#include <unordered_map>
#include <vector>

namespace RooBatchCompute {
namespace CUDA {

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

/// Scratch memory attached to a CUDA stream, used for staging small
/// per-kernel-launch data like the Batches descriptor and reduction results.
///
/// The slots form a ring: acquire() returns the next slot, waiting for the
/// completion of the work that was previously enqueued from that slot if it
/// is still in flight (which is rare, given the depth of the ring). Each slot
/// pairs a pinned host buffer with a device buffer of the same capacity, so
/// staging copies are truly asynchronous and no cudaMalloc()/cudaFree() calls
/// happen in the evaluation hot loop.
///
/// Like the rest of the RooBatchCompute library, this class is not
/// thread-safe: RooFit evaluates on a single thread per process.
class StreamScratch {
public:
   struct Slot {
      char *host = nullptr; // pinned host memory
      char *device = nullptr;
      std::size_t capacity = 0;
      cudaEvent_t event = nullptr; // recorded after the last enqueued use
      bool inFlight = false;
   };

   StreamScratch() = default;
   StreamScratch(StreamScratch const &) = delete;
   StreamScratch &operator=(StreamScratch const &) = delete;

   Slot &acquire(std::size_t n)
   {
      Slot &slot = _slots[_next];
      _next = (_next + 1) % _slots.size();
      if (slot.inFlight) {
         ERRCHECK(cudaEventSynchronize(slot.event));
         slot.inFlight = false;
      }
      if (slot.capacity < n) {
         if (slot.host)
            ERRCHECK(cudaFreeHost(slot.host));
         if (slot.device)
            ERRCHECK(cudaFree(slot.device));
         const std::size_t newCapacity = std::max<std::size_t>(n, 1024);
         ERRCHECK(cudaMallocHost(reinterpret_cast<void **>(&slot.host), newCapacity));
         ERRCHECK(cudaMalloc(reinterpret_cast<void **>(&slot.device), newCapacity));
         slot.capacity = newCapacity;
      }
      if (slot.event == nullptr) {
         ERRCHECK(cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming));
      }
      return slot;
   }

   /// Mark the last enqueued use of the slot on the stream. The slot will not
   /// be handed out again before that work has completed.
   void release(Slot &slot, cudaStream_t stream)
   {
      ERRCHECK(cudaEventRecord(slot.event, stream));
      slot.inFlight = true;
   }

   ~StreamScratch()
   {
      // Don't use ERRCHECK here: throwing from a destructor would terminate.
      for (Slot &slot : _slots) {
         if (slot.inFlight)
            cudaEventSynchronize(slot.event);
         if (slot.event)
            cudaEventDestroy(slot.event);
         if (slot.host)
            cudaFreeHost(slot.host);
         if (slot.device)
            cudaFree(slot.device);
      }
   }

private:
   std::array<Slot, 64> _slots;
   std::size_t _next = 0;
};

} // namespace

std::vector<void (*)(Batches &)> getFunctions();

/// This class overrides some RooBatchComputeInterface functions, for the
/// purpose of providing a cuda specific implementation of the library.
class RooBatchComputeClass : public RooBatchComputeInterface {

public:
   RooBatchComputeClass() : _computeFunctions(getFunctions())
   {
      dispatchCUDA = this; // Set the dispatch pointer to this instance of the library upon loading
   }

   Architecture architecture() const override { return Architecture::CUDA; }
   std::string architectureName() const override { return "cuda"; }

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
      using namespace CudaInterface;

      std::size_t nEvents = output.size();

      const std::size_t memSize = sizeof(Batches) + vars.size() * sizeof(Batch) + vars.size() * sizeof(double) +
                                  extraArgs.size() * sizeof(double);

      cudaStream_t stream = *cfg.cudaStream();
      StreamScratch::Slot &slot = scratch(cfg.cudaStream()).acquire(memSize);

      // The staging area has the same layout in the pinned host buffer and in
      // the device buffer, so it can be uploaded with a single copy.
      auto batches = reinterpret_cast<Batches *>(slot.host);
      auto arrays = reinterpret_cast<Batch *>(batches + 1);
      auto scalarBuffer = reinterpret_cast<double *>(arrays + vars.size());
      auto extraArgsHost = reinterpret_cast<double *>(scalarBuffer + vars.size());

      auto batchesDevice = reinterpret_cast<Batches *>(slot.device);
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

      copyHostToDevice(slot.host, slot.device, memSize, cfg.cudaStream());

      const int gridSize = getGridSize(nEvents);
      _computeFunctions[computer]<<<gridSize, blockSize, 0, stream>>>(*batchesDevice);

      // Only the NormalizedPdf computer mutates its extra args: it uses them
      // as output parameters for the evaluation error counts. Only then the
      // extra args need to be copied back, and the stream needs to be
      // synchronized because the caller inspects the counters right after
      // this function returns.
      if (computer == NormalizedPdf && !extraArgs.empty()) {
         copyDeviceToHost(extraArgsDevice, extraArgs.data(), extraArgs.size(), cfg.cudaStream());
         ERRCHECK(cudaStreamSynchronize(stream));
      }

      scratch(cfg.cudaStream()).release(slot, stream);
   }
   /// Return the sum of an input array
   double reduceSum(RooBatchCompute::Config const &cfg, InputArr input, size_t n) override;
   ReduceNLLOutput reduceNLL(RooBatchCompute::Config const &cfg, std::span<const double> probas,
                             std::span<const double> weights, std::span<const double> offsetProbas) override;

   std::unique_ptr<AbsBufferManager> createBufferManager() const override;

   CudaInterface::CudaStream *newCudaStream() const override { return new CudaInterface::CudaStream{}; }
   void deleteCudaStream(CudaInterface::CudaStream *stream) const override
   {
      _scratchMap.erase(stream);
      delete stream;
   }
   void synchronizeCudaStream(CudaInterface::CudaStream *stream) const override
   {
      ERRCHECK(::cudaStreamSynchronize(*stream));
   }

private:
   StreamScratch &scratch(CudaInterface::CudaStream *stream) { return _scratchMap[stream]; }

   const std::vector<void (*)(Batches &)> _computeFunctions;
   mutable std::unordered_map<CudaInterface::CudaStream *, StreamScratch> _scratchMap;

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
                             const double *__restrict__ offsetProbas, size_t nProbas, double scalarProba,
                             size_t nWeights, double *__restrict__ result)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   int carry_index = threadIdx.x + blockDim.x;
   const int nThreadsTotal = blockSize * gridDim.x;

   // The first half of the shared memory is for storing the summation and the second half for the carry or compensation
   extern __shared__ double shared[];

   double sum = 0.0;
   double carry = 0.0;

   for (int i = gthIdx; i < nWeights; i += nThreadsTotal) {
      // Note: it does not make sense to use the nll option and provide at the
      // same time external carries.
      double val = -std::log(nProbas == 1 ? scalarProba : probas[i]);
      if (offsetProbas)
         val += std::log(offsetProbas[i]);
      val = weights[i] * val;
      kahanSumUpdate(sum, carry, val, 0.0);
   }

   shared[thIdx] = sum;
   shared[carry_index] = carry;

   // Wait until all threads in each block have loaded their elements
   __syncthreads();

   kahanSumReduction(shared, nWeights, result, carry_index);
}

double RooBatchComputeClass::reduceSum(RooBatchCompute::Config const &cfg, InputArr input, size_t n)
{
   if (n == 0)
      return 0.0;
   const int gridSize = getGridSize(n);
   cudaStream_t stream = *cfg.cudaStream();
   StreamScratch::Slot &slot = scratch(cfg.cudaStream()).acquire(2 * gridSize * sizeof(double));
   auto devOut = reinterpret_cast<double *>(slot.device);
   auto hostOut = reinterpret_cast<double *>(slot.host);
   constexpr int shMemSize = 2 * blockSize * sizeof(double);
   kahanSum<<<gridSize, blockSize, shMemSize, stream>>>(input, nullptr, n, devOut, 0);
   kahanSum<<<1, blockSize, shMemSize, stream>>>(devOut, devOut + gridSize, gridSize, devOut, 0);
   CudaInterface::copyDeviceToHost(devOut, hostOut, 1, cfg.cudaStream());
   ERRCHECK(cudaStreamSynchronize(stream));
   const double result = hostOut[0];
   scratch(cfg.cudaStream()).release(slot, stream);
   return result;
}

ReduceNLLOutput RooBatchComputeClass::reduceNLL(RooBatchCompute::Config const &cfg, std::span<const double> probas,
                                                std::span<const double> weights, std::span<const double> offsetProbas)
{
   ReduceNLLOutput out;
   if (probas.empty()) {
      return out;
   }
   const int gridSize = getGridSize(weights.size());
   cudaStream_t stream = *cfg.cudaStream();
   StreamScratch::Slot &slot = scratch(cfg.cudaStream()).acquire(2 * gridSize * sizeof(double));
   auto devOut = reinterpret_cast<double *>(slot.device);
   auto hostOut = reinterpret_cast<double *>(slot.host);
   constexpr int shMemSize = 2 * blockSize * sizeof(double);

#ifndef NDEBUG
   for (auto span : {probas, weights, offsetProbas}) {
      // Scalar spans can point to host memory (e.g. the scalar buffer of an
      // observable-independent pdf), so only spans with more than one element
      // are required to be on the device.
      cudaPointerAttributes attr;
      assert(span.size() <= 1 || span.data() == nullptr ||
             (cudaPointerGetAttributes(&attr, span.data()) == cudaSuccess && attr.type == cudaMemoryTypeDevice));
   }
#endif

   nllSumKernel<<<gridSize, blockSize, shMemSize, stream>>>(
      probas.data(), weights.data(), offsetProbas.empty() ? nullptr : offsetProbas.data(), probas.size(),
      probas.size() == 1 ? probas[0] : 0.0, weights.size(), devOut);

   kahanSum<<<1, blockSize, shMemSize, stream>>>(devOut, devOut + gridSize, gridSize, devOut, 0);

   // The sum and its Kahan carry are adjacent in the output buffer, so they
   // can be read back with a single copy.
   CudaInterface::copyDeviceToHost(devOut, hostOut, 2, cfg.cudaStream());
   ERRCHECK(cudaStreamSynchronize(stream));

   out.nllSum = hostOut[0];
   out.nllSumCarry = hostOut[1];
   scratch(cfg.cudaStream()).release(slot, stream);
   return out;
}

namespace {

class ScalarBufferContainer {
public:
   ScalarBufferContainer() {}
   ScalarBufferContainer(std::size_t size)
   {
      if (size != 1)
         throw std::runtime_error("ScalarBufferContainer can only be of size 1");
   }

   double const *hostReadPtr() const { return &_val; }
   double const *deviceReadPtr() const { return &_val; }

   double *hostWritePtr() { return &_val; }
   double *deviceWritePtr() { return &_val; }

   void assignFromHost(std::span<const double> input) { _val = input[0]; }
   void assignFromDevice(std::span<const double> input)
   {
      CudaInterface::copyDeviceToHost(input.data(), &_val, input.size(), nullptr);
   }

private:
   double _val;
};

class CPUBufferContainer {
public:
   CPUBufferContainer(std::size_t size) : _vec(size) {}

   double const *hostReadPtr() const { return _vec.data(); }
   double const *deviceReadPtr() const
   {
      throw std::bad_function_call();
      return nullptr;
   }

   double *hostWritePtr() { return _vec.data(); }
   double *deviceWritePtr()
   {
      throw std::bad_function_call();
      return nullptr;
   }

   void assignFromHost(std::span<const double> input) { _vec.assign(input.begin(), input.end()); }
   void assignFromDevice(std::span<const double> input)
   {
      CudaInterface::copyDeviceToHost(input.data(), _vec.data(), input.size(), nullptr);
   }

private:
   std::vector<double> _vec;
};

class GPUBufferContainer {
public:
   GPUBufferContainer(std::size_t size) : _arr(size) {}

   double const *hostReadPtr() const
   {
      throw std::bad_function_call();
      return nullptr;
   }
   double const *deviceReadPtr() const { return _arr.data(); }

   double *hostWritePtr() const
   {
      throw std::bad_function_call();
      return nullptr;
   }
   double *deviceWritePtr() const { return const_cast<double *>(_arr.data()); }

   void assignFromHost(std::span<const double> input)
   {
      CudaInterface::copyHostToDevice(input.data(), deviceWritePtr(), input.size(), nullptr);
   }
   void assignFromDevice(std::span<const double> input)
   {
      CudaInterface::copyDeviceToDevice(input.data(), deviceWritePtr(), input.size(), nullptr);
   }

private:
   CudaInterface::DeviceArray<double> _arr;
};

class PinnedBufferContainer {
public:
   PinnedBufferContainer(std::size_t size) : _arr{size}, _gpuBuffer{size} {}
   std::size_t size() const { return _arr.size(); }

   void setCudaStream(CudaInterface::CudaStream *stream) { _cudaStream = stream; }

   double const *hostReadPtr() const
   {

      if (_lastAccess == LastAccessType::GPU_WRITE) {
         CudaInterface::copyDeviceToHost(_gpuBuffer.deviceReadPtr(), const_cast<double *>(_arr.data()), size(),
                                         _cudaStream);
         // The copy is asynchronous, and the caller reads the host memory
         // right away, so the stream needs to be synchronized here.
         if (_cudaStream) {
            ERRCHECK(cudaStreamSynchronize(*_cudaStream));
         }
      }

      _lastAccess = LastAccessType::CPU_READ;
      return const_cast<double *>(_arr.data());
   }
   double const *deviceReadPtr() const
   {

      if (_lastAccess == LastAccessType::CPU_WRITE) {
         CudaInterface::copyHostToDevice(_arr.data(), _gpuBuffer.deviceWritePtr(), size(), _cudaStream);
      }

      _lastAccess = LastAccessType::GPU_READ;
      return _gpuBuffer.deviceReadPtr();
   }

   double *hostWritePtr()
   {
      _lastAccess = LastAccessType::CPU_WRITE;
      return _arr.data();
   }
   double *deviceWritePtr()
   {
      _lastAccess = LastAccessType::GPU_WRITE;
      return _gpuBuffer.deviceWritePtr();
   }

   void assignFromHost(std::span<const double> input) { std::copy(input.begin(), input.end(), hostWritePtr()); }
   void assignFromDevice(std::span<const double> input)
   {
      CudaInterface::copyDeviceToDevice(input.data(), deviceWritePtr(), input.size(), _cudaStream);
   }

private:
   enum class LastAccessType {
      CPU_READ,
      GPU_READ,
      CPU_WRITE,
      GPU_WRITE
   };

   CudaInterface::PinnedHostArray<double> _arr;
   GPUBufferContainer _gpuBuffer;
   CudaInterface::CudaStream *_cudaStream = nullptr;
   mutable LastAccessType _lastAccess = LastAccessType::CPU_READ;
};

template <class Container>
class BufferImpl : public AbsBuffer {
public:
   using Queue = std::queue<std::unique_ptr<Container>>;

   BufferImpl(std::size_t size, Queue &queue) : _queue{queue}
   {
      if (_queue.empty()) {
         _vec = std::make_unique<Container>(size);
      } else {
         _vec = std::move(_queue.front());
         _queue.pop();
      }
   }

   ~BufferImpl() override { _queue.emplace(std::move(_vec)); }

   double const *hostReadPtr() const override { return _vec->hostReadPtr(); }
   double const *deviceReadPtr() const override { return _vec->deviceReadPtr(); }

   double *hostWritePtr() override { return _vec->hostWritePtr(); }
   double *deviceWritePtr() override { return _vec->deviceWritePtr(); }

   void assignFromHost(std::span<const double> input) override { _vec->assignFromHost(input); }
   void assignFromDevice(std::span<const double> input) override { _vec->assignFromDevice(input); }

   Container &vec() { return *_vec; }

private:
   std::unique_ptr<Container> _vec;
   Queue &_queue;
};

using ScalarBuffer = BufferImpl<ScalarBufferContainer>;
using CPUBuffer = BufferImpl<CPUBufferContainer>;
using GPUBuffer = BufferImpl<GPUBufferContainer>;
using PinnedBuffer = BufferImpl<PinnedBufferContainer>;

struct BufferQueuesMaps {
   std::map<std::size_t, ScalarBuffer::Queue> scalarBufferQueuesMap;
   std::map<std::size_t, CPUBuffer::Queue> cpuBufferQueuesMap;
   std::map<std::size_t, GPUBuffer::Queue> gpuBufferQueuesMap;
   std::map<std::size_t, PinnedBuffer::Queue> pinnedBufferQueuesMap;
};

class BufferManager : public AbsBufferManager {

public:
   BufferManager() : _queuesMaps{std::make_unique<BufferQueuesMaps>()} {}

   std::unique_ptr<AbsBuffer> makeScalarBuffer() override
   {
      return std::make_unique<ScalarBuffer>(1, _queuesMaps->scalarBufferQueuesMap[1]);
   }
   std::unique_ptr<AbsBuffer> makeCpuBuffer(std::size_t size) override
   {
      return std::make_unique<CPUBuffer>(size, _queuesMaps->cpuBufferQueuesMap[size]);
   }
   std::unique_ptr<AbsBuffer> makeGpuBuffer(std::size_t size) override
   {
      return std::make_unique<GPUBuffer>(size, _queuesMaps->gpuBufferQueuesMap[size]);
   }
   std::unique_ptr<AbsBuffer> makePinnedBuffer(std::size_t size, CudaInterface::CudaStream *stream = nullptr) override
   {
      auto out = std::make_unique<PinnedBuffer>(size, _queuesMaps->pinnedBufferQueuesMap[size]);
      out->vec().setCudaStream(stream);
      return out;
   }

private:
   std::unique_ptr<BufferQueuesMaps> _queuesMaps;
};

} // namespace

std::unique_ptr<AbsBufferManager> RooBatchComputeClass::createBufferManager() const
{
   return std::make_unique<BufferManager>();
}

/// Static object to trigger the constructor which overwrites the dispatch pointer.
static RooBatchComputeClass computeObj;

} // End namespace CUDA
} // End namespace RooBatchCompute
