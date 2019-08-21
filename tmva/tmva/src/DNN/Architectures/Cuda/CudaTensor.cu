// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////
// Implementation of the TCudaTensor class. //
/////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda/CudaTensor.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"

#include <algorithm>
#include <cassert>
#include <iostream>

namespace TMVA {
namespace DNN  {


// Static members.
//____________________________________________________________________________
/*template<typename AFloat>
size_t                   TCudaTensor<AFloat>::fInstances        = 0;*/
/*template<typename AFloat>
cublasHandle_t           TCudaTensor<AFloat>::fCublasHandle     = nullptr;*/
/*template<typename AFloat>
cudnnHandle_t            TCudaTensor<AFloat>::fCudnnHandle      = nullptr;*/
template<typename AFloat>
std::vector<cudnnHandle_t> TCudaTensor<AFloat>::fCudnnHandle(1);
/*template<typename AFloat>
cudnnTensorDescriptor_t  TCudaTensor<AFloat>::fTensorDescriptor = nullptr;*/
template<typename AFloat>
cudnnDataType_t          TCudaTensor<AFloat>::fDataType         = CUDNN_DATA_FLOAT;
/*template<typename AFloat>
AFloat                   * TCudaTensor<AFloat>::fDeviceReturn   = nullptr;*/
/*template<typename AFloat>
AFloat                   * TCudaTensor<AFloat>::fOnes           = nullptr;*/
/*template<typename AFloat>
curandState_t            * TCudaTensor<AFloat>::fCurandStates   = nullptr;*/
/*template<typename AFloat>
size_t                   TCudaTensor<AFloat>::fNCurandStates    = 0;*/
/*template<typename AFloat>
size_t                   TCudaTensor<AFloat>::fNOnes            = 0;*/
/*template<typename AFloat>
std::vector<std::vector<int> >         TCudaTensor<AFloat>::fStreamIndxs(std:vector<int>(), std::vector<int>());*/
template<typename AFloat>
std::vector<int>         TCudaTensor<AFloat>::fInstances(1,0);

/// This information is needed for the multi-dimensional indexing. See here:
/// https://en.wikipedia.org/wiki/Row-_and_column-major_order
/// https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html
template<typename AFloat>
std::vector<std::size_t> TCudaTensor<AFloat>::ComputeStridesFromShape(const std::vector<std::size_t> &shape, 
   bool rowmajorLayout)
{
   const auto size = shape.size();
   std::vector<std::size_t> strides(size);
   if (rowmajorLayout)  {
      for (std::size_t i = 0; i < size; i++) {
         if (i == 0) {
            strides[size - 1 - i] = 1;
         } else {
            strides[size - 1 - i] = strides[size - 1 - i + 1] * shape[size - 1 - i + 1];
         }
      }
   } else  {
      for (std::size_t i = 0; i < size; i++) {
         if (i == 0) {
            strides[i] = 1;
         } else {
            strides[i] = strides[i - 1] * shape[i - 1];
         }
      }
   }
   return strides;
}

// Constructors.
//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor()
    : fShape(), fStrides(), fNDim(0), fSize(0), fElementBuffer(), fStreamIndx(0), fTensorDescriptor(nullptr)
{
   InitializeCuda();
}

//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(std::vector<TMatrixT<Double_t> >& inputTensor, 
                                 const std::vector<size_t> & shape,
                                 TCudaTensor::MemoryLayout layout,
                                 int device, int streamIndx)
    : fShape(shape), fStrides(shape.size()), fNDim(shape.size()),  
      fElementBuffer(inputTensor.size()*inputTensor[0].GetNoElements(), 0),
      fDevice(device), fStreamIndx(streamIndx), fTensorDescriptor(nullptr),
      fMemoryLayout(layout)
{
   // Need a shape array with at least 4 entries for cuDNN tensors
   if (fNDim < 4) {
       std::puts("No matching cuDNN tensor description for given input dimension(s). "
                 "Inputs should be given as: batch size, no. channels, image dimensions. "
                 "Unused dimensions should be set to one.");
       exit(EXIT_FAILURE);
   }
   
   size_t inputDepth  = inputTensor.size();
   size_t inputHeight = inputTensor[0].GetNcols();
   size_t inputWidth  = inputTensor[0].GetNrows();
   
   fSize = inputDepth * inputHeight * inputWidth;
   
   fStrides = ComputeStridesFromShape(fShape, layout==MemoryLayout::RowMajor);
   
   InitializeCuda();
      
   //fElementBuffer = TCudaDeviceBuffer<AFloat>(fSize);
   TCudaHostBuffer<AFloat> hostBuffer (fElementBuffer.GetSize());
   //AFloat * hostBuffer = new AFloat[inputDepth * inputHeight * inputWidth];
   for (size_t i = 0; i < inputDepth; i++) {
      for (size_t j = 0; j < inputHeight; j++) {
         for (size_t k = 0; k < inputWidth; k++) {
            size_t bufferIndex = i * inputHeight * inputWidth + j * inputWidth + k;
            hostBuffer[bufferIndex] = static_cast<AFloat>(inputTensor[i](k, j));
         }
      }
   }

   fElementBuffer.CopyFrom(hostBuffer);
}

//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(const std::vector<size_t> & shape,
                                 TCudaTensor::MemoryLayout layout,
                                 int device, int streamIndx)
    : fShape(shape), fStrides(shape.size()), fNDim(shape.size()), fDevice(device), fStreamIndx(streamIndx),
      fTensorDescriptor(nullptr), fMemoryLayout(layout)
{
   fStrides = ComputeStridesFromShape(fShape, layout==MemoryLayout::RowMajor);
   
   fSize = (layout==MemoryLayout::RowMajor) ? fStrides.front()*fShape.front() : 
                                              fStrides.back()*fShape.back(); 

   fElementBuffer = TCudaDeviceBuffer<AFloat>(fSize, 0);
   
   InitializeCuda();
}

//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(const AFloat * host_data, const std::vector<size_t> & shape,
                                 TCudaTensor::MemoryLayout layout,
                                 int device, int streamIndx)
   : TCudaTensor(shape, layout, device, streamIndx)
{
   // do I need to allocate this buffer ???? 
   // is not a mem leak
   // AFloat * buffer = new AFloat[fSize];
   // size_t index = 0;
   // for (size_t j = 0; j < fSize; ++j) {
   //       buffer[j] = static_cast<AFloat>(host_data[j]);
   //    }
   // }

   cudaMemcpy(fElementBuffer, host_data, fSize * sizeof(AFloat),
              cudaMemcpyHostToDevice);
}

//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(TCudaDeviceBuffer<AFloat> buffer,  
                                 const std::vector<size_t> & shape,
                                 TMVA::Experimental::MemoryLayout layout,
                                 int device, int streamIndx)
   : fNDim(shape.size()), fElementBuffer(buffer), fShape(shape), fStrides( shape.size()), fDevice(device), 
     fStreamIndx(streamIndx), fTensorDescriptor(nullptr), fMemoryLayout(layout)
{
   fStrides = ComputeStridesFromShape(fShape, layout==MemoryLayout::RowMajor);
   
   fSize = (layout==MemoryLayout::RowMajor) ? fStrides.front()*fShape.front() : 
                                              fStrides.back()*fShape.back();  
   InitializeCuda();  
}

//____________________________________________________________________________
//FIXME: Go to shared_ptr implementation of instance tracking
template <typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(const TCudaTensor<AFloat>& oldTensor, size_t /*dim*/) :
   TCudaTensor(oldTensor.fShape, oldTensor.fMemoryLayout, oldTensor.fDevice, oldTensor.fStreamIndx)
{
   // No deep copy
   fStrides       = oldTensor.fStrides;
   fElementBuffer = oldTensor.fElementBuffer;   
        
   InitializeCuda();
}

//____________________________________________________________________________
template <typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(const TCudaMatrix<AFloat>& matrix, size_t dim) :
   TCudaTensor( matrix.GetDeviceBuffer(), {matrix.GetNrows(), matrix.GetNcols()}, MemoryLayout::ColumnMajor)
{
   // No deep copy
   if (dim > 2) {
      // change shape from (nrows,ncols) to (nrows,ncols,1,1)
      // this works onlt for coolum major layout since this is same of TCudaMatrix
      fShape.insert(fShape.end(), dim-2, 1);
      fStrides.insert(fStrides.end(),dim-2,fSize);
      fNDim = dim; 
   }

   InitializeCuda();
}

//____________________________________________________________________________
template <typename AFloat>
TCudaTensor<AFloat>::~TCudaTensor() 
{
//#if USE_CUDNN
   CUDNNCHECK(cudnnDestroyTensorDescriptor(fTensorDescriptor));

   // When all tensors in a streamIndx are destroyed, release cudnn resources 
   //if (--fInstances[fStreamIndx] <= 0) CUDNNCHECK(cudnnDestroy(fCudnnHandle[fStreamIndx]));
//#endif
}

//____________________________________________________________________________
template <typename AFloat>
inline void TCudaTensor<AFloat>::InitializeCuda()
{
   // If fNDim >= 4, a cudnn tensor is required
   if (fNDim >= 4) {
      // Also check whether a new streamIndx has been opened
      if (fInstances.size() - 1 < fStreamIndx) {
         // If need to resize once, need probably to resize more often
         fInstances.resize(2*fStreamIndx + 1, 0);
         fCudnnHandle.resize(2*fStreamIndx + 1, nullptr);
      }
      if (fInstances[fStreamIndx] == 0) {
        CUDNNCHECK(cudnnCreate(&fCudnnHandle[fStreamIndx]));
        //CUDNNCHECK(cudnnSetStream(fCudnnHandle[fStreamIndx], fElementBuffer.GetComputeStream()));
        
        //cublasCreate(&fCublasHandle);
        //CUDACHECK(cudaMalloc(& fDeviceReturn, sizeof(AFloat)));
        //CUDACHECK(cudaMalloc(& fCurandStates, TDevice::NThreads(*this)));
      }
      // if (TDevice::NThreads(*this) > (int) fNCurandStates) {
      //     fNCurandStates = TDevice::NThreads(*this);
      //     if (fCurandStates) {
      //         cudaFree(fCurandStates);
      //     }
      //     cudaMalloc(&fCurandStates, TDevice::NThreads(*this) * sizeof(curandState_t));
      //     InitializeCurandStates();
      // }

      CUDNNCHECK(cudnnCreateTensorDescriptor(&fTensorDescriptor));
      fInstances[fStreamIndx]++;
      
      // Prevent template specialization of entire class
      if      (std::is_same<AFloat, double>::value) {fDataType = CUDNN_DATA_DOUBLE;}
      else if (std::is_same<AFloat, float>::value)  {fDataType = CUDNN_DATA_FLOAT;}

      // cuDNN NdTensor format has a minsize of 4 tensor dimensions
      // 4D tensor is more performant on lower dimensions and supports all folowing operations
      if (fNDim == 4) {
         CUDNNCHECK(cudnnSetTensor4dDescriptor(fTensorDescriptor,
                                               CUDNN_TENSOR_NCHW,// Layout of the tensor in memory
                                               fDataType,
                                               (int)fShape[0],   // batch size
                                               (int)fShape[1],   // no. channels
                                               (int)fShape[2],   // image height
                                               (int)fShape[3])); // image width
      }
      // Some operations in cudnn may not work with this tensor description
      else {
        std::cout << "Dim = "<< fNDim 
                  <<". Currently only 4D tensors are supported for the TMVA cuDNN backend."
                  << std::endl;
        /*CUDNNCHECK(cudnnSetTensorNdDescriptor(fTensorDescriptor,
                                              fDataType,
                                              (int)fNDim,
                                              (int *)fShape.data(),
                                              (int *)fStrides.data()));*/
      }
   
      size_t tensorSize;
      CUDNNCHECK(cudnnGetTensorSizeInBytes(fTensorDescriptor, &tensorSize));
      assert(fSize == tensorSize/sizeof(AFloat));
   }

}

//____________________________________________________________________________
template<typename AFloat>
void TCudaTensor<AFloat>::InitializeCurandStates()
{
   // dim3 blockDims = TDevice::BlockDims2D();
   // dim3 gridDims  = TDevice::GridDims2D(*this);
   // CurandInitializationKernel<<<gridDims, blockDims>>>(time(nullptr), fCurandStates);
}

#if 0
// Conversion to RTensor
//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::operator Experimental::RTensor<AFloat>() const
{
   std::vector<size_t> shape(fNDims, fNDims + fDim)
   
   Experimental::RTensor<AFloat> hostTensor( shape)

   AFloat * buffer = new AFloat[fSize];
   cudaMemcpy(buffer, fElementBuffer, fSize * sizeof(AFloat),
              cudaMemcpyDeviceToHost);

   int index = 0;
   for (int j = 0; j < fSize; j++) {
         hostTensor.GetData()[j] = static_cast<AFloat>(buffer[j]);
      }
   }

   delete[] buffer;
   return hostTensor;
}
#endif
// Explicit Instantiations.

template class TCudaTensor<float>;
template class TCudaTensor<double>;

} // namespace DNN
} // namespace TMVA
