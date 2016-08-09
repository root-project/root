// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
// Contains the TCudaMatrix class for the representation of matrices //
// on CUDA devices as well as the TCudaDeviceReference class which   //
// is a helper class to emulate lvalue references to floating point //
// values on the device.                                            //
//////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CUDA_CUDAMATRIX
#define TMVA_DNN_ARCHITECTURES_CUDA_CUDAMATRIX

#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand_kernel.h"

#include "TMatrixT.h"

#include "Types.h"
#include "Buffers.h"

#define CUDACHECK(ans) {cudaError((ans), __FILE__, __LINE__); }

namespace TMVA {
namespace DNN {

/** Function to check cuda return code. Taken from
 * http://stackoverflow.com/questions/14038589/
 */
inline void cudaError(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//____________________________________________________________________________
//
// Cuda Device Reference
//____________________________________________________________________________

/** TCudaDeviceReference
 *
 * Helper class emulating lvalue references for CudaDouble_t values that are
 * physically on the device. Allows for example to assign to matrix elements.
 * Not that device access through CudaDeviceReferences enforces synchronization
 * with all streams and thus qualifies as performance killer. Only used for
 * testing.
 */
class TCudaDeviceReference
{
private:

    CudaDouble_t * fDevicePointer;

public:

TCudaDeviceReference(CudaDouble_t * devicePointer)
  : fDevicePointer(devicePointer)
   {
      // Nothing to do here.
   }

   operator CudaDouble_t()
   {
      CudaDouble_t buffer;
      cudaMemcpy(& buffer, fDevicePointer, sizeof(CudaDouble_t),
                 cudaMemcpyDeviceToHost);
      return buffer;
    }

    void operator=(const TCudaDeviceReference &other)
    {
       cudaMemcpy(fDevicePointer, other.fDevicePointer, sizeof(CudaDouble_t),
                  cudaMemcpyDeviceToDevice);
    }

    void operator=(CudaDouble_t value)
    {
       CudaDouble_t buffer = value;
          cudaMemcpy(fDevicePointer, & buffer, sizeof(CudaDouble_t),
                     cudaMemcpyHostToDevice);
    }

    void operator+=(CudaDouble_t value)
    {
       CudaDouble_t buffer;
       cudaMemcpy(& buffer, fDevicePointer, sizeof(CudaDouble_t),
                  cudaMemcpyDeviceToHost);
       buffer += value;
       cudaMemcpy(fDevicePointer, & buffer, sizeof(CudaDouble_t),
                  cudaMemcpyHostToDevice);
    }

    void operator-=(CudaDouble_t value)
    {
       CudaDouble_t buffer;
       cudaMemcpy(& buffer, fDevicePointer, sizeof(CudaDouble_t),
                  cudaMemcpyDeviceToHost);
       buffer -= value;
       cudaMemcpy(fDevicePointer, & buffer, sizeof(CudaDouble_t),
                  cudaMemcpyHostToDevice);
    }
};

//____________________________________________________________________________
//
// Cuda Matrix
//____________________________________________________________________________

/** TCudaMatrix Class
 *
 * The TCudaMatrix class represents matrices on a CUDA device. The class takes
 * care of allocating and freeing device memory. It also holds static variables
 * to certain device resources such as the Cublas handle as the computation stream.
 *
 * Computations on all matrix instances are performend in a single,
 * non-standard stream. That means that they are guaranteed to be
 * executed in order, but may not wait for asynchronous data transfer
 * in other streams.
 *
 * Each matrix also has an associated data stream, which defaults to the
 * default (0) stream if not explicitly provided.
 */
class TCudaMatrix
{
public:

private:

   static size_t          fInstances;    ///< Number of existing TCudaMatrix instances.
   static cublasHandle_t  fCublasHandle;
   static CudaDouble_t  * fDeviceReturn; ///< Buffer for kernel return values.
   static curandState_t * fCurandStates;
   static size_t          fNCurandStates;

   size_t            fNRows;
   size_t            fNCols;
   TCudaDeviceBuffer fElementBuffer;

public:

   TCudaMatrix();
   TCudaMatrix(size_t i, size_t j);
   TCudaMatrix(const TMatrixT<CudaDouble_t> &);
   TCudaMatrix(TCudaDeviceBuffer buffer, size_t m, size_t n);

   TCudaMatrix(const TCudaMatrix  &) = default;
   TCudaMatrix(      TCudaMatrix &&) = default;
   TCudaMatrix & operator=(const TCudaMatrix  &) = default;
   TCudaMatrix & operator=(      TCudaMatrix &&) = default;
   ~TCudaMatrix() = default;

   /** Return the compute stream associated with this matrix */
   inline cudaStream_t GetComputeStream() const;
   inline void         SetComputeStream(cudaStream_t stream);
   /** Blocking synchronization with the associated compute stream, if it's
    * not the default stream. */
   inline void Synchronize(const TCudaMatrix &) const;
   /** Set the return buffer on the device to the specified value. This is
    * required for example for reductions in order to initialize the
    * accumulator. */
   inline static void ResetDeviceReturn(CudaDouble_t value = 0.0);
   /** Transfer the value in the device return buffer to the host. */
   inline static CudaDouble_t GetDeviceReturn();

   /** Return device pointer to the device return buffer */
   inline static CudaDouble_t *  GetDeviceReturnPointer() {return fDeviceReturn;}
   inline static curandState_t * GetCurandStatesPointer() {return fCurandStates;}

   /** Convert cuda matrix to Root TMatrix. */
   operator TMatrixT<CudaDouble_t>() const;

   void CopyToDevice(CudaDouble_t *source);
   void CopyFromDevice(CudaDouble_t *dest);

   size_t GetNrows() const {return fNRows;}
   size_t GetNcols() const {return fNCols;}
   size_t GetNoElements() const {return fNRows * fNCols;}
   const CudaDouble_t * GetDataPointer() const {return fElementBuffer;}
   CudaDouble_t *       GetDataPointer()       {return fElementBuffer;}
   const cublasHandle_t & GetCublasHandle() const    {return fCublasHandle;}

   /** Access to elements of device matrices provided through TCudaDeviceReference
    *  class. Note that access is synchronous end enforces device synchronization
    *  on all streams. Only used for testing. */
   TCudaDeviceReference operator()(size_t i, size_t j) const
   {
      CudaDouble_t * elementPointer = fElementBuffer;
      elementPointer += j * fNRows + i;
      return TCudaDeviceReference(elementPointer);
   }

private:

   void InitializeCuda();
   void InitializeCurandStates();

};

//
// Inline Functions.
//______________________________________________________________________________
inline cudaStream_t TCudaMatrix::GetComputeStream() const
{
   return fElementBuffer.GetComputeStream();
}

//______________________________________________________________________________
inline void TCudaMatrix::SetComputeStream(cudaStream_t stream)
{
   return fElementBuffer.SetComputeStream(stream);
}

//______________________________________________________________________________
inline void TCudaMatrix::Synchronize(const TCudaMatrix &A) const {
   cudaEvent_t event;
   cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
   cudaEventRecord(event, A.GetComputeStream());
   cudaStreamWaitEvent(fElementBuffer.GetComputeStream(), event, 0);
   cudaEventDestroy(event);
}

//______________________________________________________________________________
inline void TCudaMatrix::ResetDeviceReturn(CudaDouble_t value)
{
   CudaDouble_t buffer = value;
   cudaMemcpy(fDeviceReturn, & buffer, sizeof(CudaDouble_t),
              cudaMemcpyHostToDevice);
}

//______________________________________________________________________________
inline CudaDouble_t TCudaMatrix::GetDeviceReturn()
{
   CudaDouble_t buffer;
   cudaMemcpy(& buffer, fDeviceReturn, sizeof(CudaDouble_t),
              cudaMemcpyDeviceToHost);
   return buffer;
}

} // namespace DNN
} // namespace TMVA

#endif
