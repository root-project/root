/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN  11/2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit/Detail/Buffers.h"

#include <stdexcept>
#include <functional>
#include <queue>
#include <map>

#ifdef ROOFIT_CUDA
namespace CudaInterface = RooFit::Detail::CudaInterface;
using CudaInterface::CudaStream;
#endif

namespace RooFit {
namespace Detail {

class ScalarBufferContainer {
public:
   ScalarBufferContainer() {}
   ScalarBufferContainer(std::size_t size)
   {
      if (size != 1)
         throw std::runtime_error("ScalarBufferContainer can only be of size 1");
   }
   std::size_t size() const { return 1; }

   double const *cpuReadPtr() const { return &_val; }
   double const *gpuReadPtr() const { return &_val; }

   double *cpuWritePtr() { return &_val; }
   double *gpuWritePtr() { return &_val; }

private:
   double _val;
};

class CPUBufferContainer {
public:
   CPUBufferContainer(std::size_t size) : _vec(size) {}
   std::size_t size() const { return _vec.size(); }

   double const *cpuReadPtr() const { return _vec.data(); }
   double const *gpuReadPtr() const
   {
      throw std::bad_function_call();
      return nullptr;
   }

   double *cpuWritePtr() { return _vec.data(); }
   double *gpuWritePtr()
   {
      throw std::bad_function_call();
      return nullptr;
   }

private:
   std::vector<double> _vec;
};

#ifdef ROOFIT_CUDA
class GPUBufferContainer {
public:
   GPUBufferContainer(std::size_t size) : _arr(size) {}
   std::size_t size() const { return _arr.size(); }

   double const *cpuReadPtr() const
   {
      throw std::bad_function_call();
      return nullptr;
   }
   double const *gpuReadPtr() const { return _arr.data(); }

   double *cpuWritePtr() const
   {
      throw std::bad_function_call();
      return nullptr;
   }
   double *gpuWritePtr() const { return const_cast<double *>(_arr.data()); }

private:
   CudaInterface::DeviceArray<double> _arr;
};

class PinnedBufferContainer {
public:
   PinnedBufferContainer(std::size_t size) : _arr{size}, _gpuBuffer{size} {}
   std::size_t size() const { return _arr.size(); }

   void setCudaStream(CudaStream *stream) { _cudaStream = stream; }

   double const *cpuReadPtr() const
   {

      if (_lastAccess == LastAccessType::GPU_WRITE) {
         CudaInterface::copyDeviceToHost(_gpuBuffer.gpuReadPtr(), const_cast<double *>(_arr.data()), size(),
                                         _cudaStream);
      }

      _lastAccess = LastAccessType::CPU_READ;
      return const_cast<double *>(_arr.data());
   }
   double const *gpuReadPtr() const
   {

      if (_lastAccess == LastAccessType::CPU_WRITE) {
         CudaInterface::copyHostToDevice(_arr.data(), _gpuBuffer.gpuWritePtr(), size(), _cudaStream);
      }

      _lastAccess = LastAccessType::GPU_READ;
      return _gpuBuffer.gpuReadPtr();
   }

   double *cpuWritePtr()
   {
      _lastAccess = LastAccessType::CPU_WRITE;
      return _arr.data();
   }
   double *gpuWritePtr()
   {
      _lastAccess = LastAccessType::GPU_WRITE;
      return _gpuBuffer.gpuWritePtr();
   }

private:
   enum class LastAccessType { CPU_READ, GPU_READ, CPU_WRITE, GPU_WRITE };

   CudaInterface::PinnedHostArray<double> _arr;
   GPUBufferContainer _gpuBuffer;
   CudaStream *_cudaStream = nullptr;
   mutable LastAccessType _lastAccess = LastAccessType::CPU_READ;
};
#endif // ROOFIT_CUDA

template <class Container>
class BufferImpl : public AbsBuffer {
public:
   using Queue = std::queue<std::unique_ptr<Container>>;
   using QueuesMap = std::map<std::size_t, Queue>;

   BufferImpl(std::size_t size, QueuesMap &queuesMap) : _queue{queuesMap[size]}
   {
      if (_queue.empty()) {
         _vec = std::make_unique<Container>(size);
      } else {
         _vec = std::move(_queue.front());
         _queue.pop();
      }
   }

   ~BufferImpl() override { _queue.emplace(std::move(_vec)); }

   double const *cpuReadPtr() const override { return _vec->cpuReadPtr(); }
   double const *gpuReadPtr() const override { return _vec->gpuReadPtr(); }

   double *cpuWritePtr() override { return _vec->cpuWritePtr(); }
   double *gpuWritePtr() override { return _vec->gpuWritePtr(); }

   Container &vec() { return *_vec; }

private:
   std::unique_ptr<Container> _vec;
   Queue &_queue;
};

using ScalarBuffer = BufferImpl<ScalarBufferContainer>;
using CPUBuffer = BufferImpl<CPUBufferContainer>;
#ifdef ROOFIT_CUDA
using GPUBuffer = BufferImpl<GPUBufferContainer>;
using PinnedBuffer = BufferImpl<PinnedBufferContainer>;
#endif

struct BufferQueuesMaps {
   ScalarBuffer::QueuesMap scalarBufferQueuesMap;
   CPUBuffer::QueuesMap cpuBufferQueuesMap;
#ifdef ROOFIT_CUDA
   GPUBuffer::QueuesMap gpuBufferQueuesMap;
   PinnedBuffer::QueuesMap pinnedBufferQueuesMap;
#endif
};

BufferManager::BufferManager()
{
   _queuesMaps = std::make_unique<BufferQueuesMaps>();
}

BufferManager::~BufferManager() = default;

std::unique_ptr<AbsBuffer> BufferManager::makeScalarBuffer()
{
   return std::make_unique<ScalarBuffer>(1, _queuesMaps->scalarBufferQueuesMap);
}
std::unique_ptr<AbsBuffer> BufferManager::makeCpuBuffer(std::size_t size)
{
   return std::make_unique<CPUBuffer>(size, _queuesMaps->cpuBufferQueuesMap);
}
#ifdef ROOFIT_CUDA
std::unique_ptr<AbsBuffer> BufferManager::makeGpuBuffer(std::size_t size)
{
   return std::make_unique<GPUBuffer>(size, _queuesMaps->gpuBufferQueuesMap);
}
std::unique_ptr<AbsBuffer> BufferManager::makePinnedBuffer(std::size_t size, CudaStream *stream)
{
   auto out = std::make_unique<PinnedBuffer>(size, _queuesMaps->pinnedBufferQueuesMap);
   out->vec().setCudaStream(stream);
   return out;
}
#endif // ROOFIT_CUDA

} // end namespace Detail
} // end namespace RooFit
