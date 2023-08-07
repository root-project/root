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

#include <RooFit/Detail/CudaInterface.h>

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

namespace RooFit {
namespace Detail {
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
}
template <>
void Deleter<PinnedHostMemory>::operator()(void *ptr)
{
   ERRCHECK(::cudaFreeHost(ptr));
}

/**
 * Creates a new CUDA event.
 *
 * @param[in] forTiming       Set to true if the event is intended for timing purposes.
 *                            If `false`, the `cudaEventDisableTiming` is passed to CUDA.
 * @return                    CudaEvent object representing the new event.
 */
CudaEvent newCudaEvent(bool forTiming)
{
   CudaEvent ret;
   ret.reset(new cudaEvent_t);
   ERRCHECK(cudaEventCreateWithFlags(ret.get(), forTiming ? 0 : cudaEventDisableTiming));
   return ret;
}

/**
 * Destroys a CUDA event.
 *
 * @param[in] event           CudaEvent object representing the event to be destroyed.
 */
void deleteCudaEvent(CudaEvent event)
{
   ERRCHECK(cudaEventDestroy(*event.get()));
   delete event.get();
   event.reset(nullptr);
}

/**
 * Records a CUDA event.
 *
 * @param[in] event           CudaEvent object representing the event to be recorded.
 * @param[in] stream          CudaStream in which to record the event.
 */
void cudaEventRecord(CudaEvent event, CudaStream stream)
{
   ERRCHECK(::cudaEventRecord(*event.get(), *stream.get()));
}

/**
 * Creates a new CUDA stream.
 *
 * @return                    CudaStream object representing the new stream.
 */
CudaStream newCudaStream()
{
   CudaStream ret;
   ret.reset(new cudaStream_t);
   ERRCHECK(cudaStreamCreate(ret.get()));
   return ret;
}

/**
 * Destroys a CUDA stream.
 *
 * @param[in] stream          CudaStream object representing the stream to be destroyed.
 */
void deleteCudaStream(CudaStream stream)
{
   ERRCHECK(cudaStreamDestroy(*stream.get()));
   delete stream.get();
   stream.reset(nullptr);
}

/**
 * Checks if a CUDA stream is currently active.
 *
 * @param[in] stream          CudaStream object representing the stream to be checked.
 * @return                    True if the stream is active, false otherwise.
 */
bool streamIsActive(CudaStream stream)
{
   cudaError_t err = cudaStreamQuery(*stream.get());
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
 * @param[in] stream          CudaStream object representing the stream to wait on.
 * @param[in] event           CudaEvent object representing the event to wait for.
 */
void cudaStreamWaitEvent(CudaStream stream, CudaEvent event)
{
   ERRCHECK(::cudaStreamWaitEvent(*stream.get(), *event.get(), 0));
}

/**
 * Calculates the elapsed time between two CUDA events.
 *
 * @param[in] begin           CudaEvent representing the start event.
 * @param[in] end             CudaEvent representing the end event.
 * @return                    Elapsed time in milliseconds.
 */
float cudaEventElapsedTime(CudaEvent begin, CudaEvent end)
{
   float ret;
   ERRCHECK(::cudaEventElapsedTime(&ret, *begin.get(), *end.get()));
   return ret;
}

/// \internal
void copyHostToDeviceImpl(const void *src, void *dest, size_t nBytes, CudaStream stream)
{
   if (stream.get())
      ERRCHECK(cudaMemcpyAsync(dest, src, nBytes, cudaMemcpyHostToDevice, *stream.get()));
   else
      ERRCHECK(cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice));
}

/// \internal
void copyDeviceToHostImpl(const void *src, void *dest, size_t nBytes, CudaStream stream)
{
   if (stream.get())
      ERRCHECK(cudaMemcpyAsync(dest, src, nBytes, cudaMemcpyDeviceToHost, *stream.get()));
   else
      ERRCHECK(cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost));
}

} // namespace CudaInterface
} // namespace Detail
} // namespace RooFit
