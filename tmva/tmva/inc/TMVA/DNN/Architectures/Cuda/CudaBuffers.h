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

#ifndef TMVA_DNN_ARCHITECTURES_CUDA_CUDABUFFERS
#define TMVA_DNN_ARCHITECTURES_CUDA_CUDABUFFERS

#include "cuda.h"
#include "cuda_runtime.h"

#include <memory>

namespace TMVA {
namespace DNN  {

template<typename AFloat>
class TCudaDeviceBuffer;

/** TCudaHostBuffer
 *
 * Wrapper class for pinned memory buffers on the host.  Uses
 * std::shared_pointer with custom destructor to ensure consistent
 * memory management and allow for easy copying/moving of the
 * buffers. Copying is asynchronous and will set the cudaStream of the
 * device buffer so that subsequent computations on the device buffer
 * can be performed on the same stream.
 *
 * \tparam AFloat The floating point type to be stored in the buffers.
 */
template<typename AFloat>
class TCudaHostBuffer
{
private:

   size_t                    fOffset;        ///< Offset for sub-buffers
   size_t                    fSize;
   mutable cudaStream_t      fComputeStream; ///< cudaStream for data transfer
   std::shared_ptr<AFloat *> fHostPointer;   ///< Pointer to the buffer data

   // Custom destructor required to free pinned host memory using cudaFree.
   struct TDestructor
   {
       TDestructor()                     = default;
       TDestructor(const TDestructor  &) = default;
       TDestructor(      TDestructor &&) = default;
       TDestructor & operator=(const TDestructor  &) = default;
       TDestructor & operator=(      TDestructor &&) = default;
       void operator()(AFloat ** devicePointer);
   } fDestructor;

   friend TCudaDeviceBuffer<AFloat>;

public:

   TCudaHostBuffer(size_t size);
   TCudaHostBuffer(AFloat *);
   TCudaHostBuffer() = default;
   TCudaHostBuffer(const TCudaHostBuffer  &) = default;
   TCudaHostBuffer(      TCudaHostBuffer &&) = default;
   TCudaHostBuffer & operator=(const TCudaHostBuffer  &) = default;
   TCudaHostBuffer & operator=(      TCudaHostBuffer &&) = default;

   /** Return sub-buffer of the current buffer. */
   TCudaHostBuffer GetSubBuffer(size_t offset, size_t size);
   /** Sets the entire buffer to a constant value */
   void            SetConstVal(const AFloat constVal);

   operator AFloat * () const;

   inline AFloat & operator[](size_t index);
   inline AFloat   operator[](size_t index) const;

   size_t GetSize() const {return fSize;}

};

/** TCudaDeviceBuffer
 *
 *  Service class for on-device memory buffers. Uses
 *  std::shared_pointer with custom destructor to ensure consistent
 *  memory management and allow for easy copying/moving. A device
 *  buffer has an associated CUDA compute stream , which is used for
 *  implicit synchronization of data transfers.
 *
 * \tparam AFloat The floating point type to be stored in the buffers.
 */
template<typename AFloat>
class TCudaDeviceBuffer
{
private:

   size_t                    fOffset;        ///< Offset for sub-buffers
   size_t                    fSize;
   cudaStream_t              fComputeStream; ///< cudaStream for data transfer
   std::shared_ptr<AFloat *> fDevicePointer; ///< Pointer to the buffer data

   // Custom destructor required to free pinned host memory using cudaFree.
   struct TDestructor
   {
       TDestructor()                     = default;
       TDestructor(const TDestructor  &) = default;
       TDestructor(      TDestructor &&) = default;
       TDestructor & operator=(const TDestructor  &) = default;
       TDestructor & operator=(      TDestructor &&) = default;
       void operator()(AFloat ** devicePointer);
       friend TCudaDeviceBuffer;
   } fDestructor;

public:

   TCudaDeviceBuffer(size_t size);
   TCudaDeviceBuffer(size_t size,    cudaStream_t stream);
   TCudaDeviceBuffer(AFloat *, size_t size, cudaStream_t stream);
   TCudaDeviceBuffer() = default;
   TCudaDeviceBuffer(const TCudaDeviceBuffer  &) = default;
   TCudaDeviceBuffer(      TCudaDeviceBuffer &&) = default;
   TCudaDeviceBuffer & operator=(const TCudaDeviceBuffer  &) = default;
   TCudaDeviceBuffer & operator=(      TCudaDeviceBuffer &&) = default;

   /** Return sub-buffer of the current buffer. */
   TCudaDeviceBuffer GetSubBuffer(size_t offset, size_t size);
   /** Convert to raw device data pointer.*/
   operator AFloat * () const;

   void CopyFrom(const TCudaHostBuffer<AFloat> &) const;
   void CopyTo(const TCudaHostBuffer<AFloat> &)   const;

   size_t GetSize() const {return fSize;}
   cudaStream_t GetComputeStream() const {return fComputeStream;}
   void SetComputeStream(cudaStream_t stream) {fComputeStream = stream;}

};

//
// Inline Functions.
//______________________________________________________________________________

template<typename AFloat>
AFloat & TCudaHostBuffer<AFloat>::operator[](size_t index)
{
   return (*fHostPointer + fOffset)[index];
}

template<typename AFloat>
AFloat   TCudaHostBuffer<AFloat>::operator[](size_t index)   const
{
   return (*fHostPointer + fOffset)[index];
}


} // namespace DNN
} // namespace TMVA
#endif
