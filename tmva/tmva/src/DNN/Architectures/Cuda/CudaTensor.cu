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
    : fShape(shape), fStrides( shape.size()), fNDim(shape.size()),  
      fElementBuffer(inputTensor.size()*inputTensor[0].GetNoElements(), 0),
      fDevice(device), fStreamIndx(streamIndx), fTensorDescriptor(nullptr)
{
   //assert(fNDim == fShape.size());
   // Need a shape array with at least 4 entries for cuDNN tensors
   std::cout << "Dimensions :\t" << fNDim << std::endl;
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
   
   std::cout << "depth\t" << inputDepth << std::endl;
   std::cout << "height\t" << inputHeight << std::endl;
   std::cout << "width\t" << inputWidth << std::endl;
   std::cout << "size\t" << fSize << std::endl;
   
   // Reduce shape size afterwards for loop and direct array access
   //fStrides = new size_t[fNDim];
   for (int i = 0; i < fNDim - 1; ++i) {
       fStrides[i] = shape[i+1];
       for (int j = 0; j < i; j++) {
          fStrides[j] *= shape[i+1];
       }
   }
   // Last stride should be one for cudnn
   fStrides[fNDim - 1] = 1;
   
   std::cout << "Shape:" << std::endl;
   for (int i = 0; i < fNDim; ++i) {
      std::cout << fShape[i] << std::endl;
   }
   std::cout << "Strides:" << std::endl;
   for (int i = 0; i < fNDim; ++i) {
      std::cout << fStrides[i] << std::endl;
   }
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
    : fShape(shape), fStrides( shape.size()), fNDim( shape.size()), fDevice(device), fStreamIndx(streamIndx),
      fTensorDescriptor(nullptr)
{
   assert(fNDim == fShape.size());
   // Need a shape array with at least 4 entries for cuDNN tensors
//    if (fNDim < 4) {
//        std::puts("No matching cuDNN tensor description for given input dimension(s). "
//                  "Inputs should be given as: batch size, no. channels, image dimensions. "
//                  "Unused dimensions should be set to one.");
// //       exit(EXIT_FAILURE);
//   }
   
   // Reduce shape size afterwards for loop and direct array access
   //fStrides = new size_t[fNDim];
   for (int i = 0; i < fNDim - 1; ++i) {
       fStrides[i] = shape[i+1];
       for (int j = 0; j < i; j++) {
          fStrides[j] *= shape[i+1];
       }
   }
   // Last stride should be one for cudnn
   fStrides[fNDim - 1] = 1;
   
   fSize = fStrides[0]*shape[0];

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
   assert(fNDim == fShape.size());
   // Need a shape array with at least 4 entries for cuDNN tensors
   // if (shape.size() < 4) {
   //     std::puts("No matching cuDNN tensor description for given input dimension(s). "
   //               "Inputs should be given as: batch size, no. channels, image dimensions. "
   //               "Unused dimensions should be set to one.");
   //     //     exit(EXIT_FAILURE);
   // }
   
   // Reduce shape size afterwards for loop and direct array access
   for (int i = 0; i < fNDim - 1; ++i) {
       fStrides[i] = shape[i+1];
       for (int j = 0; j < i; j++) {
          fStrides[j] *= shape[i+1];
       }
   }
   // Last stride should be one for cudnn
   fStrides[fNDim - 1] = 1;
   
   fSize = fStrides[0]*shape[0];
   
   InitializeCuda();  
}

//____________________________________________________________________________
//FIXME: Go to shared_ptr implementation
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
   TCudaTensor( matrix.GetDeviceBuffer(), {matrix.GetNrows(), matrix.GetNcols()})
{
   // No deep copy
   fMemoryLayout = MemoryLayout::ColumnMajor;
   fStrides       = { 1 , matrix.GetNrows() };  //CM layout
   fNDim = dim; 

   if (dim > 2) {
      // change shape from (nrows,ncols) to (nrows,ncols,1,1)
      fShape.insert( fShape.end(), dim-2, 1);
      fStrides.insert(fStrides.begin(),dim-2,1);
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
//#if USE_CUDNN
   
   // Also check whether a new streamIndx has been opened
   if (fInstances.size() - 1 < fStreamIndx) {
      // If need to resize once, need probably to resize more often
      fInstances.resize(2*fStreamIndx + 1, 0);
      fCudnnHandle.resize(2*fStreamIndx + 1, nullptr);
   }
   if (fInstances[fStreamIndx] == 0) {
     CUDNNCHECK(cudnnCreate(&fCudnnHandle[fStreamIndx]));
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
   //fInstances++;
      
   if      (std::is_same<AFloat, double>::value) { fDataType = CUDNN_DATA_DOUBLE;}
   else if (std::is_same<AFloat, float>::value)  { fDataType = CUDNN_DATA_FLOAT;}
   
   // No tensor can be set by unparametrized constructor
   if (fShape.size() == 0) {
      return;
   }
   // cuDNN NdTensor format has a minsize of 4 tensor dimensions
   // 4D tensor is more performant on lower dimensions and supports all folowing operations
   else if (fNDim == 4) {
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
     CUDNNCHECK(cudnnSetTensorNdDescriptor(fTensorDescriptor,
                                           fDataType,
                                           (int)fNDim,
                                           (int *)fShape.data(),
                                           (int *)fStrides.data()));
   }
   
   size_t tensorSize;
   CUDNNCHECK(cudnnGetTensorSizeInBytes(fTensorDescriptor, &tensorSize));
   assert(fSize == tensorSize/sizeof(AFloat));
//#endif
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
