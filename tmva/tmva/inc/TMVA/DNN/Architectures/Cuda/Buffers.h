// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 07/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////
// Device and host buffer for CUDA architectures. //
////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CUDA_BUFFERS
#define TMVA_DNN_ARCHITECTURES_CUDA_BUFFERS

#include "cuda.h"
#include "cuda_runtime.h"
#include <memory>
#include "Types.h"

namespace TMVA {
namespace DNN  {

class TCudaDeviceBuffer;

/** Wrapper class for pinned memory double-precision buffers on the host.
 * Uses std::shared_pointer with custom destructor to ensure consistent
 * memory management and allow for easy copying/moving. Copying from device
 * will set the corresponding data stream which can then be used to synchronize
 * on the results.
 */
class TCudaHostBuffer
{
private:

   size_t                          fOffset;
   mutable cudaStream_t            fComputeStream;
   std::shared_ptr<CudaDouble_t *> fDevicePointer;

   struct TDestructor
   {
       TDestructor()                     = default;
       TDestructor(const TDestructor  &) = default;
       TDestructor(      TDestructor &&) = default;
       TDestructor & operator=(const TDestructor  &) = default;
       TDestructor & operator=(      TDestructor &&) = default;
       void operator()(CudaDouble_t ** devicePointer);
       friend TCudaDeviceBuffer;
   } fDestructor;

   friend TCudaDeviceBuffer;

public:

   TCudaHostBuffer(size_t size);
   TCudaHostBuffer(CudaDouble_t *);
   TCudaHostBuffer() = default;
   TCudaHostBuffer(const TCudaHostBuffer  &) = default;
   TCudaHostBuffer(      TCudaHostBuffer &&) = default;
   TCudaHostBuffer & operator=(const TCudaHostBuffer  &) = default;
   TCudaHostBuffer & operator=(      TCudaHostBuffer &&) = default;

   /** Return sub-buffer of the current buffer. */
   TCudaHostBuffer GetSubBuffer(size_t offset, size_t size);

   /** Convert to raw device data pointer.*/
   operator CudaDouble_t * () const;
   CudaDouble_t & operator[](size_t index)
   {
      return (*fDevicePointer + fOffset)[index];
   }
   CudaDouble_t   operator[](size_t index)   const
   {
      return (*fDevicePointer + fOffset)[index];
   }

};

/** Wrapper class for on-device memory double-precision buffers. Uses
 *  std::shared_pointer with custom destructor to ensure consistent
 *  memory management and allow for easy copying/moving. A device
 *  buffer has an associated CUDA compute stream , which is used for
 *  implicit synchronization of data transfers.
 */
class TCudaDeviceBuffer
{
private:

   size_t                          fOffset;
   size_t                          fSize;
   cudaStream_t                    fComputeStream;
   std::shared_ptr<CudaDouble_t *> fDevicePointer;

   struct TDestructor
   {
       TDestructor()                     = default;
       TDestructor(const TDestructor  &) = default;
       TDestructor(      TDestructor &&) = default;
       TDestructor & operator=(const TDestructor  &) = default;
       TDestructor & operator=(      TDestructor &&) = default;
       void operator()(CudaDouble_t ** devicePointer);
       friend TCudaDeviceBuffer;
   } fDestructor;

public:

   TCudaDeviceBuffer(size_t size);
   TCudaDeviceBuffer(size_t size,    cudaStream_t stream);
   TCudaDeviceBuffer(CudaDouble_t *, size_t size, cudaStream_t stream);
   TCudaDeviceBuffer() = default;
   TCudaDeviceBuffer(const TCudaDeviceBuffer  &) = default;
   TCudaDeviceBuffer(      TCudaDeviceBuffer &&) = default;
   TCudaDeviceBuffer & operator=(const TCudaDeviceBuffer  &) = default;
   TCudaDeviceBuffer & operator=(      TCudaDeviceBuffer &&) = default;

   /** Return sub-buffer of the current buffer. */
   TCudaDeviceBuffer GetSubBuffer(size_t offset, size_t size);
   /** Convert to raw device data pointer.*/
   operator CudaDouble_t * () const;

   void CopyFrom(const TCudaHostBuffer &) const;
   void CopyTo(const TCudaHostBuffer &)   const;

   cudaStream_t GetComputeStream() const {return fComputeStream;}
   void SetComputeStream(cudaStream_t stream) {fComputeStream = stream;}

};


} // namespace DNN
} // namespace TMVA
#endif
