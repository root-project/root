/*
 * Project: RooFit
 * Author:
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "CudaInterface.h"

#include <stdexcept>
#include <sstream>
#include <string>

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      std::stringstream errMsg;
      errMsg << func << "(), " << file + ":" << std::to_string(line) << " : " << cudaGetErrorString(error);
      throw std::runtime_error(errMsg.str());
   }
}

namespace RooBatchCompute {
namespace CudaInterface {

DeviceMemory::DeviceMemory(std::size_t n, std::size_t typeSize) : _size{n}
{
   void *ret;
   ERRCHECK(::cudaMalloc(&ret, n * typeSize));
   _data.reset(ret);
}
PinnedHostMemory::PinnedHostMemory(std::size_t n, std::size_t typeSize) : _size{n}
{
   void *ret;
   ERRCHECK(::cudaMallocHost(&ret, n * typeSize));
   _data.reset(ret);
}

template <>
void Deleter<DeviceMemory>::operator()(void *ptr)
{
   ERRCHECK(::cudaFree(ptr));
   ptr = nullptr;
}
template <>
void Deleter<PinnedHostMemory>::operator()(void *ptr)
{
   ERRCHECK(::cudaFreeHost(ptr));
   ptr = nullptr;
}

/**
 * Creates a new CUDA event.
 *
 * @param[in] forTiming       Set to true if the event is intended for timing purposes.
 *                            If `false`, the `cudaEventDisableTiming` is passed to CUDA.
 * @return                    CudaEvent object representing the new event.
 */
CudaEvent::CudaEvent(bool forTiming)
{
   auto event = new cudaEvent_t;
   ERRCHECK(cudaEventCreateWithFlags(event, forTiming ? 0 : cudaEventDisableTiming));
   _ptr.reset(event);
}

template <>
void Deleter<CudaEvent>::operator()(void *ptr)
{
   auto event = reinterpret_cast<cudaEvent_t *>(ptr);
   ERRCHECK(cudaEventDestroy(*event));
   delete event;
   ptr = nullptr;
}

template <>
void Deleter<CudaStream>::operator()(void *ptr)
{
   auto stream = reinterpret_cast<cudaStream_t *>(ptr);
   ERRCHECK(cudaStreamDestroy(*stream));
   delete stream;
   ptr = nullptr;
}

/**
 * Records a CUDA event.
 *
 * @param[in] event           CudaEvent object representing the event to be recorded.
 * @param[in] stream          CudaStream in which to record the event.
 */
void cudaEventRecord(CudaEvent &event, CudaStream &stream)
{
   ERRCHECK(::cudaEventRecord(event, stream));
}

/**
 * Creates a new CUDA stream.
 *
 * @return                    CudaStream object representing the new stream.
 */
CudaStream::CudaStream()
{
   auto stream = new cudaStream_t;
   ERRCHECK(cudaStreamCreate(stream));
   _ptr.reset(stream);
}

/**
 * Checks if a CUDA stream is currently active.
 *
 * @return                    True if the stream is active, false otherwise.
 */
bool CudaStream::isActive()
{
   cudaError_t err = cudaStreamQuery(*this);
   if (err == cudaErrorNotReady)
      return true;
   else if (err == cudaSuccess)
      return false;
   ERRCHECK(err);
   return false;
}

/**
 * Makes a CUDA stream wait for a CUDA event.
 *
 * @param[in] event           CudaEvent object representing the event to wait for.
 */
void CudaStream::waitForEvent(CudaEvent &event)
{
   ERRCHECK(::cudaStreamWaitEvent(*this, event, 0));
}

/**
 * Calculates the elapsed time between two CUDA events.
 *
 * @param[in] begin           CudaEvent representing the start event.
 * @param[in] end             CudaEvent representing the end event.
 * @return                    Elapsed time in milliseconds.
 */
float cudaEventElapsedTime(CudaEvent &begin, CudaEvent &end)
{
   float ret;
   ERRCHECK(::cudaEventElapsedTime(&ret, begin, end));
   return ret;
}

/// \cond ROOFIT_INTERNAL

void copyHostToDeviceImpl(const void *src, void *dest, size_t nBytes, CudaStream *stream)
{
   if (stream)
      ERRCHECK(cudaMemcpyAsync(dest, src, nBytes, cudaMemcpyHostToDevice, *stream));
   else
      ERRCHECK(cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice));
}

void copyDeviceToHostImpl(const void *src, void *dest, size_t nBytes, CudaStream *stream)
{
   if (stream)
      ERRCHECK(cudaMemcpyAsync(dest, src, nBytes, cudaMemcpyDeviceToHost, *stream));
   else
      ERRCHECK(cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost));
}

void copyDeviceToDeviceImpl(const void *src, void *dest, size_t nBytes, CudaStream *stream)
{
   if (stream)
      ERRCHECK(cudaMemcpyAsync(dest, src, nBytes, cudaMemcpyDeviceToDevice, *stream));
   else
      ERRCHECK(cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToDevice));
}

/// \endcond

} // namespace CudaInterface
} // namespace RooBatchCompute
