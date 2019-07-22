// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////
// Contains the TCudaMatrix class for the representation of matrices //
// on CUDA devices as well as the TCudaDeviceReference class which   //
// is a helper class to emulate lvalue references to floating point  //
// values on the device.                                             //
///////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CUDA_CUDATENSOR
#define TMVA_DNN_ARCHITECTURES_CUDA_CUDATENSOR

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand_kernel.h"

//#include "TMatrixT.h"
#include "CudaBuffers.h"
#include "CudaMatrix.h"
//#include "TMVA/RTensor.hxx"


namespace TMVA {
namespace DNN {




//____________________________________________________________________________
//
// Cuda Tensor
//____________________________________________________________________________

/** TCudaTensor Class
 *
 * The TCudaTGensor class extends the TCudaMatrix class for dimensions > 2. 
 *
 */
template<typename AFloat>
class TCudaTensor
{
public:

private:

   static size_t          fInstances;    ///< Current number of matrix instances.
   static cublasHandle_t  fCublasHandle;
   static AFloat        * fDeviceReturn; ///< Buffer for kernel return values.
   //?? static AFloat        * fOnes;         ///< Vector used for summations of columns.
   //static size_t          fNOnes;        ///< Current length of the one vector.
   static curandState_t * fCurandStates;
   static size_t          fNCurandStates;

   size_t *                  fShape;
   size_t                    fNDim; 
   size_t                    fSize;

   TCudaDeviceBuffer<AFloat> fElementBuffer;

public:

   //static AFloat * GetOnes() {return fOnes;}

   TCudaTensor();
   TCudaTensor(size_t size, size_t ndim, const size_t *  shape);
   TCudaTensor(size_t size, const AFloat * data, size_t ndim, const size_t * shape );
   TCudaTensor(TCudaDeviceBuffer<AFloat> buffer, size_t ndim, const size_t * shape);

   TCudaTensor(const TCudaTensor  &) = default;
   TCudaTensor(      TCudaTensor &&) = default;
   TCudaTensor & operator=(const TCudaTensor  &) = default;
   TCudaTensor & operator=(      TCudaTensor &&) = default;
   ~TCudaTensor() = default;

   /** Convert cuda matrix to Root TMatrix. Performs synchronous data transfer. */
   //operator Experimental::RTensor<AFloat>() const;

   inline cudaStream_t GetComputeStream() const;
   inline void         SetComputeStream(cudaStream_t stream);
   /** Set the return buffer on the device to the specified value. This is
    * required for example for reductions in order to initialize the
    * accumulator. */
   //inline static void ResetDeviceReturn(AFloat value = 0.0);
   /** Transfer the value in the device return buffer to the host. This
    *  tranfer is synchronous */
   //inline static AFloat GetDeviceReturn();
   /** Return device pointer to the device return buffer */
   inline static AFloat *        GetDeviceReturnPointer() {return fDeviceReturn;}
   inline static curandState_t * GetCurandStatesPointer() {return fCurandStates;}

   /** Blocking synchronization with the associated compute stream, if it's
    * not the default stream. */
   inline void Synchronize(const TCudaTensor &) const;

   const size_t * GetShape() const {return fShape;}
   size_t GetDimAt(size_t i) const {return fShape[i];}
   size_t GetNDim() const {return fNDim;}
   size_t GetSize() const {return fSize;}
    
   const AFloat * GetDataPointer() const {return fElementBuffer;}
   AFloat *       GetDataPointer()       {return fElementBuffer;}
   const cublasHandle_t & GetCublasHandle() const    {return fCublasHandle;}

   /** Access to elements of device matrices provided through TCudaDeviceReference
    *  class. Note that access is synchronous end enforces device synchronization
    *  on all streams. Only used for testing. */
   //TCudaDeviceReference<AFloat> operator()(size_t i, size_t j) const;

   void Print() const { 
      //TMatrixT<AFloat> mat(*this); 
      //mat.Print(); 
   }

   void Zero() {
      cudaMemset(GetDataPointer(), 0, sizeof(AFloat) * GetSize());
   }


private:

   /** Initializes all shared devices resource and makes sure that a sufficient
    *  number of curand states are allocated on the device and initialized as
    *  well as that the one-vector for the summation over columns has the right
    *  size. */
   void InitializeCuda();
   void InitializeCurandStates();

};

//
// Inline Functions.
//______________________________________________________________________________
#if 0
inline void cudaError(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif


//______________________________________________________________________________
template<typename AFloat>
inline cudaStream_t TCudaTensor<AFloat>::GetComputeStream() const
{
   return fElementBuffer.GetComputeStream();
}

//______________________________________________________________________________
template<typename AFloat>
inline void TCudaTensor<AFloat>::SetComputeStream(cudaStream_t stream)
{
   return fElementBuffer.SetComputeStream(stream);
}

//______________________________________________________________________________
template<typename AFloat>
inline void TCudaTensor<AFloat>::Synchronize(const TCudaTensor &A) const
{
   cudaEvent_t event;
   cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
   cudaEventRecord(event, A.GetComputeStream());
   cudaStreamWaitEvent(fElementBuffer.GetComputeStream(), event, 0);
   cudaEventDestroy(event);
}

//______________________________________________________________________________
// template<typename AFloat>
// inline void TCudaTensor<AFloat>::ResetDeviceReturn(AFloat value)
// {
//    AFloat buffer = value;
//    cudaMemcpy(fDeviceReturn, & buffer, sizeof(AFloat), cudaMemcpyHostToDevice);
// }

// //______________________________________________________________________________
// template<typename AFloat>
// inline AFloat TCudaTensor<AFloat>::GetDeviceReturn()
// {
//    AFloat buffer;
//    cudaMemcpy(& buffer, fDeviceReturn, sizeof(AFloat), cudaMemcpyDeviceToHost);
//    return buffer;
// }

//______________________________________________________________________________
// template<typename AFloat>
// TCudaDeviceReference<AFloat> TCudaTensor<AFloat>::operator()(size_t i, size_t j) const
// {
//     AFloat * elementPointer = fElementBuffer;
//     elementPointer += j * fNRows + i;
//     return TCudaDeviceReference<AFloat>(elementPointer);
// }

} // namespace DNN
} // namespace TMVA

#endif
