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

#ifndef RooFit_Detail_CudaInterface_h
#define RooFit_Detail_CudaInterface_h

#include <cstddef>

namespace RooFit {
namespace Detail {
namespace CudaInterface {

/// Wrapper around cudaEvent_t.
class CudaEvent {
public:
   operator bool() const { return _ptr; }
// When compiling with NVCC, we allow setting and getting the actual CUDA objects from the wrapper.
#ifdef __CUDACC__
   inline cudaEvent_t *&get() { return reinterpret_cast<cudaEvent_t *&>(_ptr); }
#endif
   void *_ptr = nullptr;
};

/// Wrapper around cudaStream_t.
class CudaStream {
public:
   operator bool() const { return _ptr; }

// When compiling with NVCC, we allow setting and getting the actual CUDA objects from the wrapper.
#ifdef __CUDACC__
   inline cudaStream_t *&get() { return reinterpret_cast<cudaStream_t *&>(_ptr); }
#endif
private:
   void *_ptr = nullptr;
};

void *cudaMallocImpl(std::size_t);
void cudaFree(void *);
void *cudaMallocHostImpl(std::size_t);
void cudaFreeHost(void *);
CudaEvent newCudaEvent(bool /*forTiming*/);
void deleteCudaEvent(CudaEvent);
CudaStream newCudaStream();
void deleteCudaStream(CudaStream);
bool streamIsActive(CudaStream);
void cudaEventRecord(CudaEvent, CudaStream);
void cudaStreamWaitEvent(CudaStream, CudaEvent);
float cudaEventElapsedTime(CudaEvent, CudaEvent);
void memcpyToCUDA(void *, const void *, std::size_t, CudaStream = {});
void memcpyToCPU(void *, const void *, std::size_t, CudaStream = {});

/**
 * Allocates device memory on the CUDA GPU.
 *
 * @tparam T                  Element type of the allocated array.
 * @param[in] nBytes          Size in bytes of the memory to allocate.
 * @return                    Pointer to the allocated device memory.
 */
template <class T>
T *cudaMalloc(size_t n)
{
   return static_cast<T *>(cudaMallocImpl(n * sizeof(T)));
}

/**
 * Allocates memory on the host.
 *
 * @tparam T                  Element type of the allocated array.
 * @param[in] nBytes          Size in bytes of the memory to allocate.
 * @return                    Pointer to the allocated host memory.
 */
template <class T>
T *cudaMallocHost(size_t n)
{
   return static_cast<T *>(cudaMallocHostImpl(n * sizeof(T)));
}

} // namespace CudaInterface
} // namespace Detail
} // namespace RooFit

#endif
