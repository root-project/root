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

#ifndef CudaInterface_h
#define CudaInterface_h

#include <cstddef>
#include <memory>

namespace RooBatchCompute {

/*
 * C++ interface around CUDA functionality.
 *
 * Generally, if the call to the underlying CUDA function does not return
 * `cudaSuccess`, a `std::runtime_error` is thrown.
 *
 * \ingroup RooFitCuda
 */
namespace CudaInterface {

/// \cond ROOFIT_INTERNAL

template <class T>
struct Deleter {
   void operator()(void *ptr);
};

/// \endcond

/*
 * Wrapper around cudaEvent_t.
 */
class CudaEvent {
public:
   CudaEvent(bool forTiming);

// When compiling with NVCC, we allow setting and getting the actual CUDA objects from the wrapper.
#ifdef __CUDACC__
   inline operator cudaEvent_t() { return *reinterpret_cast<cudaEvent_t *>(_ptr.get()); }
#endif
private:
   std::unique_ptr<void, Deleter<CudaEvent>> _ptr;
};

/*
 * Wrapper around cudaStream_t.
 */
class CudaStream {
public:
   CudaStream();

   bool isActive();
   void waitForEvent(CudaEvent &);

// When compiling with NVCC, we allow setting and getting the actual CUDA objects from the wrapper.
#ifdef __CUDACC__
   inline cudaStream_t *get() { return reinterpret_cast<cudaStream_t *&>(_ptr); }
   inline operator cudaStream_t() { return *reinterpret_cast<cudaStream_t *>(_ptr.get()); }
#endif
private:
   std::unique_ptr<void, Deleter<CudaStream>> _ptr;
};

void cudaEventRecord(CudaEvent &, CudaStream &);
float cudaEventElapsedTime(CudaEvent &, CudaEvent &);

/// \cond ROOFIT_INTERNAL
void copyHostToDeviceImpl(const void *src, void *dest, std::size_t n, CudaStream * = nullptr);
void copyDeviceToHostImpl(const void *src, void *dest, std::size_t n, CudaStream * = nullptr);
void copyDeviceToDeviceImpl(const void *src, void *dest, std::size_t n, CudaStream * = nullptr);
/// \endcond

/**
 * Copies data from the host to the CUDA device.
 *
 * @param[in] src             Pointer to the source memory on the host.
 * @param[in] dest            Pointer to the destination memory on the device.
 * @param[in] n               Number of bytes to copy.
 * @param[in] stream          CudaStream for asynchronous memory transfer (optional).
 */
template <class T>
void copyHostToDevice(const T *src, T *dest, std::size_t n, CudaStream * = nullptr)
{
   copyHostToDeviceImpl(src, dest, sizeof(T) * n);
}

/**
 * Copies data from the CUDA device to the host.
 *
 * @param[in] src             Pointer to the source memory on the device.
 * @param[in] dest            Pointer to the destination memory on the host.
 * @param[in] n               Number of bytes to copy.
 * @param[in] stream          CudaStream for asynchronous memory transfer (optional).
 */
template <class T>
void copyDeviceToHost(const T *src, T *dest, std::size_t n, CudaStream * = nullptr)
{
   copyDeviceToHostImpl(src, dest, sizeof(T) * n);
}

/**
 * Copies data from the CUDA device to the CUDA device.
 *
 * @param[in] src             Pointer to the source memory on the device.
 * @param[in] dest            Pointer to the destination memory on the device.
 * @param[in] n               Number of bytes to copy.
 * @param[in] stream          CudaStream for asynchronous memory transfer (optional).
 */
template <class T>
void copyDeviceToDevice(const T *src, T *dest, std::size_t n, CudaStream * = nullptr)
{
   copyDeviceToDeviceImpl(src, dest, sizeof(T) * n);
}

/// \cond ROOFIT_INTERNAL

// The user should not use these "Memory" classes directly, but instead the typed
// "Array" classes. That's why we tell doxygen that this is internal.

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
/// \endcond

/**
 * @class Array
 * @brief A templated class for managing an array of data using a specified memory type.
 *
 * The Array class provides a convenient interface for managing an array of
 * data using different memory types (e.g., memory on the host or device).
 * The memory is automatically freed at the end of the lifetime.
 *
 * @tparam Data_t The type of data elements to be stored in the array.
 * @tparam Memory_t The type of memory that provides storage for the array.
 */
template <class Data_t, class Memory_t>
class Array : public Memory_t {
public:
   /**
    * @brief Constructor to create an Array object with a specified size.
    * @param n The size of the array (number of elements).
    */
   Array(std::size_t n) : Memory_t{n, sizeof(Data_t)} {}

   // Needs to be declared explicitly for doxygen to mention it.
   /**
    * @brief Get the size of the array.
    * @return The size of the array (number of elements).
    *
    * This function returns the number of elements in the array.
    */
   inline std::size_t size() const { return Memory_t::size(); }

   /**
    * @brief Get a pointer to the start of the array.
    * @return A pointer to the start of the array.
    *
    * This function returns a pointer to the underlying memory.
    * It allows direct manipulation of array elements.
    */
   inline Data_t *data() { return static_cast<Data_t *>(Memory_t::data()); }

   /**
    * @brief Get a const pointer to the start of the array.
    * @return A const pointer to the start of the array.
    *
    * This function returns a const pointer to the underlying memory.
    * It allows read-only access to array elements.
    */
   inline Data_t const *data() const { return static_cast<Data_t const *>(Memory_t::data()); }
};

/**
 * An array of specific type that is allocated on the device with `cudaMalloc` and freed with `cudaFree`.
 */
template <class T>
using DeviceArray = Array<T, DeviceMemory>;

/**
 * A pinned array of specific type that allocated on the host with `cudaMallocHost` and freed with `cudaFreeHost`.
 * The memory is "pinned", i.e. page-locked and accessible to the device for fast copying.
 * \see The documentation of `cudaMallocHost` on <a
 * href="https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html">developer.download.nvidia.com</a>.
 */
template <class T>
using PinnedHostArray = Array<T, PinnedHostMemory>;

} // namespace CudaInterface
} // namespace RooBatchCompute

#endif
