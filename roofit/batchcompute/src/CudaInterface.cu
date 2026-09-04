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

template <>
void Deleter<CudaStream>::operator()(void *ptr)
{
   auto stream = reinterpret_cast<cudaStream_t *>(ptr);
   ERRCHECK(cudaStreamDestroy(*stream));
   delete stream;
   ptr = nullptr;
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
