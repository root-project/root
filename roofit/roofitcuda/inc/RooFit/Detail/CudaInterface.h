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
#include <memory>

#include <vector>

namespace RooFit {
namespace Detail {
namespace CudaInterface {

template <class T>
struct Deleter {
   void operator()(void *ptr);
};

/// Wrapper around cudaEvent_t.
class CudaEvent {
public:
   operator bool() const { return _ptr; }
// When compiling with NVCC, we allow setting and getting the actual CUDA objects from the wrapper.
#ifdef __CUDACC__
   inline cudaEvent_t *get() { return reinterpret_cast<cudaEvent_t *&>(_ptr); }
   void reset(void *ptr) { _ptr = ptr; }
#endif
private:
   void *_ptr = nullptr;
};

/// Wrapper around cudaStream_t.
class CudaStream {
public:
   operator bool() const { return _ptr; }

// When compiling with NVCC, we allow setting and getting the actual CUDA objects from the wrapper.
#ifdef __CUDACC__
   inline cudaStream_t *get() { return reinterpret_cast<cudaStream_t *&>(_ptr); }
   void reset(void *ptr) { _ptr = ptr; }
#endif
private:
   void *_ptr = nullptr;
};

CudaEvent newCudaEvent(bool /*forTiming*/);
void deleteCudaEvent(CudaEvent);
CudaStream newCudaStream();
void deleteCudaStream(CudaStream);
bool streamIsActive(CudaStream);
void cudaEventRecord(CudaEvent, CudaStream);
void cudaStreamWaitEvent(CudaStream, CudaEvent);
float cudaEventElapsedTime(CudaEvent, CudaEvent);
void copyHostToDeviceImpl(const void *src, void *dest, std::size_t n, CudaStream = {});
void copyDeviceToHostImpl(const void *src, void *dest, std::size_t n, CudaStream = {});

/**
 * Copies data from the host to the CUDA device.
 *
 * @param[in] src             Pointer to the source memory on the host.
 * @param[in] dest            Pointer to the destination memory on the device.
 * @param[in] nBytes          Number of bytes to copy.
 * @param[in] stream          CudaStream for asynchronous memory transfer (optional).
 */
template <class T>
void copyHostToDevice(const T *src, T *dest, std::size_t n, CudaStream = {})
{
   copyHostToDeviceImpl(src, dest, sizeof(T) * n);
}

/**
 * Copies data from the CUDA device to the host.
 *
 * @param[in] src             Pointer to the source memory on the device.
 * @param[in] dest            Pointer to the destination memory on the host.
 * @param[in] nBytes          Number of bytes to copy.
 * @param[in] stream          CudaStream for asynchronous memory transfer (optional).
 */
template <class T>
void copyDeviceToHost(const T *src, T *dest, std::size_t n, CudaStream = {})
{
   copyDeviceToHostImpl(src, dest, sizeof(T) * n);
}

class DeviceMemory {
public:
   DeviceMemory(std::size_t n, std::size_t typeSize);

   std::size_t size() const { return _size; }
   void *data() { return _data.get(); }
   void const *data() const { return _data.get(); }

private:
   std::unique_ptr<void, Deleter<DeviceMemory>> _data;
   std::size_t _size = 0;
};

class PinnedHostMemory {
public:
   PinnedHostMemory(std::size_t n, std::size_t typeSize);

   std::size_t size() const { return _size; }
   void *data() { return _data.get(); }
   void const *data() const { return _data.get(); }

private:
   std::unique_ptr<void, Deleter<PinnedHostMemory>> _data;
   std::size_t _size = 0;
};

template <class Data_t, class Memory_t>
class Array : public Memory_t {
public:
   Array(std::size_t n) : Memory_t{n, sizeof(Data_t)} {}
   inline Data_t *data() { return static_cast<Data_t *>(Memory_t::data()); }
   inline Data_t const *data() const { return static_cast<Data_t const *>(Memory_t::data()); }
};

template <class T>
using DeviceArray = Array<T, DeviceMemory>;

template <class T>
using PinnedHostArray = Array<T, PinnedHostMemory>;

} // namespace CudaInterface
} // namespace Detail
} // namespace RooFit

#endif
