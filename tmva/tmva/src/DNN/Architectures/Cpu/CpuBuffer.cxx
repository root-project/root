// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 12/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////
// CPU Buffer interface class for the generic data loader. //
/////////////////////////////////////////////////////////////

#include <vector>
#include <memory>
#include "TMVA/DataSetInfo.h"
#include "TMVA/DNN/DataLoader.h"
#include "TMVA/DNN/TensorDataLoader.h"
#include "TMVA/DNN/Architectures/Cpu.h"
#include "Rtypes.h"
#include <iostream>

namespace TMVA {
namespace DNN {

//______________________________________________________________________________
template <typename AReal>
void TCpuBuffer<AReal>::TDestructor::operator()(AReal **pointer)
{
   delete[] * pointer;
   delete[] pointer;
}

//______________________________________________________________________________
template <typename AReal>
TCpuBuffer<AReal>::TCpuBuffer(size_t size) : fSize(size), fOffset(0)
{
   AReal **pointer = new AReal *[1];
   *pointer = new AReal[size];
   fBuffer = std::shared_ptr<AReal *>(pointer, fDestructor);
}

//______________________________________________________________________________
template <typename AReal>
TCpuBuffer<AReal> TCpuBuffer<AReal>::GetSubBuffer(size_t offset, size_t size) const
{
   TCpuBuffer buffer = *this;
   buffer.fOffset = offset;
   buffer.fSize = size;
   return buffer;
}

//______________________________________________________________________________
template <typename AReal>
void TCpuBuffer<AReal>::CopyFrom(const TCpuBuffer &other)
{
   //std::copy*this->fBuffer, *other.fBuffer);
   std::copy( *other.fBuffer, *other.fBuffer+fSize, *this->fBuffer);
}

//______________________________________________________________________________
template <typename AReal>
void TCpuBuffer<AReal>::CopyTo(TCpuBuffer &other) const
{
   std::copy( *this->fBuffer, *this->fBuffer+fSize, *other.fBuffer);
   //std::swap(*this->fBuffer, *other.fBuffer);
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TCpu<Float_t>>::CopyInput(TCpuBuffer<Float_t> &buffer, IndexIterator_t sampleIterator,
                                                         size_t batchSize)
{
   const TMatrixT<Float_t> &inputMatrix = std::get<0>(fData);
   size_t n = inputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = static_cast<Float_t>(inputMatrix(sampleIndex, j));
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TCpu<Float_t>>::CopyOutput(TCpuBuffer<Float_t> &buffer, IndexIterator_t sampleIterator,
                                                          size_t batchSize)
{
   const TMatrixT<Float_t> &outputMatrix = std::get<1>(fData);
   size_t n = outputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = static_cast<Float_t>(outputMatrix(sampleIndex, j));
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TCpu<Float_t>>::CopyWeights(TCpuBuffer<Float_t> &buffer, IndexIterator_t sampleIterator,
                                                           size_t batchSize)
{
   const TMatrixT<Float_t> &outputMatrix = std::get<2>(fData);

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      buffer[i] = static_cast<Float_t>(outputMatrix(sampleIndex, 0));
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TCpu<Double_t>>::CopyInput(TCpuBuffer<Double_t> &buffer, IndexIterator_t sampleIterator,
                                                           size_t batchSize)
{
   const TMatrixT<Double_t> &inputMatrix = std::get<0>(fData);
   size_t n = inputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = inputMatrix(sampleIndex, j);
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TCpu<Double_t>>::CopyOutput(TCpuBuffer<Double_t> &buffer,
                                                            IndexIterator_t sampleIterator, size_t batchSize)
{
   const TMatrixT<Double_t> &outputMatrix = std::get<1>(fData);
   size_t n = outputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = outputMatrix(sampleIndex, j);
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TCpu<Double_t>>::CopyWeights(TCpuBuffer<Double_t> &buffer,
                                                             IndexIterator_t sampleIterator, size_t batchSize)
{
   const TMatrixT<Double_t> &outputMatrix = std::get<2>(fData);

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      buffer[i] = static_cast<Double_t>(outputMatrix(sampleIndex, 0));
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<TMVAInput_t, TCpu<Double_t>>::CopyInput(TCpuBuffer<Double_t> &buffer, IndexIterator_t sampleIterator,
                                                         size_t batchSize)
{
   Event *event = std::get<0>(fData)[0];
   size_t n  = event->GetNVariables();
   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = * sampleIterator++;
      event = std::get<0>(fData)[sampleIndex];
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = event->GetValue(j);
      }
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<TMVAInput_t, TCpu<Double_t>>::CopyOutput(TCpuBuffer<Double_t> &buffer, IndexIterator_t sampleIterator,
                                                          size_t batchSize)
{
  const DataSetInfo &info = std::get<1>(fData);
  size_t n = buffer.GetSize() / batchSize;

  // Copy target(s).

  for (size_t i = 0; i < batchSize; i++) {
    size_t sampleIndex = *sampleIterator++;
    Event *event = std::get<0>(fData)[sampleIndex];
    for (size_t j = 0; j < n; j++) {
      // Copy output matrices.
      size_t bufferIndex = j * batchSize + i;
      // Classification
      if (event->GetNTargets() == 0) {
        if (n == 1) {
          // Binary.
          buffer[bufferIndex] = (info.IsSignal(event)) ? 1.0 : 0.0;
        } else {
          // Multiclass.
          buffer[bufferIndex] = 0.0;
          if (j == event->GetClass()) {
            buffer[bufferIndex] = 1.0;
          }
        }
      } else {
        buffer[bufferIndex] = static_cast<Float_t>(event->GetTarget(j));
      }
    }
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<TMVAInput_t, TCpu<Double_t>>::CopyWeights(TCpuBuffer<Double_t> &buffer, IndexIterator_t sampleIterator,
                                                           size_t batchSize)
{
   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator++;
      Event *event = std::get<0>(fData)[sampleIndex];
      buffer[i] = event->GetWeight();
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<TMVAInput_t, TCpu<Float_t>>::CopyInput(TCpuBuffer<Float_t> &buffer, IndexIterator_t sampleIterator,
                                                       size_t batchSize)
{
   Event *event = std::get<0>(fData)[0];
   size_t n  = event->GetNVariables();
   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = * sampleIterator++;
      event = std::get<0>(fData)[sampleIndex];
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = static_cast<Float_t>(event->GetValue(j));
      }
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<TMVAInput_t, TCpu<Float_t>>::CopyOutput(TCpuBuffer<Float_t> &buffer, IndexIterator_t sampleIterator,
                                                        size_t batchSize)
{
  const DataSetInfo &info = std::get<1>(fData);
  size_t n = buffer.GetSize() / batchSize;

  // Copy target(s).

  for (size_t i = 0; i < batchSize; i++) {
    size_t sampleIndex = *sampleIterator++;
    Event *event = std::get<0>(fData)[sampleIndex];
    for (size_t j = 0; j < n; j++) {
      // Copy output matrices.
      size_t bufferIndex = j * batchSize + i;
      // Classification
      if (event->GetNTargets() == 0) {
        if (n == 1) {
          // Binary.
          buffer[bufferIndex] = (info.IsSignal(event)) ? 1.0 : 0.0;
        } else {
          // Multiclass.
          buffer[bufferIndex] = 0.0;
          if (j == event->GetClass()) {
            buffer[bufferIndex] = 1.0;
          }
        }
      } else {
        buffer[bufferIndex] = static_cast<Float_t>(event->GetTarget(j));
      }
    }
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<TMVAInput_t, TCpu<Float_t>>::CopyWeights(TCpuBuffer<Float_t> &buffer, IndexIterator_t sampleIterator,
                                                         size_t batchSize)
{
   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator++;
      Event *event = std::get<0>(fData)[sampleIndex];
      buffer[i] = static_cast<Float_t>(event->GetWeight());
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TCpu<Float_t>>::CopyTensorInput(TCpuBuffer<Float_t> &buffer,
                                                                   IndexIterator_t sampleIterator)
{
   const std::vector<TMatrixT<Double_t>> &inputTensor = std::get<0>(fData);

   if (fBatchDepth == 1) {
      for (size_t i = 0; i < fBatchHeight; i++) {
         size_t sampleIndex = *sampleIterator;
         for (size_t j = 0; j < fBatchWidth; j++) {
            size_t bufferIndex = j * fBatchHeight + i;
            buffer[bufferIndex] = static_cast<Float_t>(inputTensor[0](sampleIndex, j));
         }
         sampleIterator++;
      }
   } else {
      for (size_t i = 0; i < fBatchDepth; i++) {
         size_t sampleIndex = *sampleIterator;
         for (size_t j = 0; j < fBatchHeight; j++) {
            for (size_t k = 0; k < fBatchWidth; k++) {
               size_t bufferIndex = i * fBatchHeight * fBatchWidth + k * fBatchHeight + j;
               buffer[bufferIndex] = static_cast<Float_t>(inputTensor[sampleIndex](j, k));
            }
         }
         sampleIterator++;
      }
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TCpu<Float_t>>::CopyTensorOutput(TCpuBuffer<Float_t> &buffer,
                                                                    IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &outputMatrix = std::get<1>(fData);
   size_t n = outputMatrix.GetNcols();

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * fBatchSize + i;
         buffer[bufferIndex] = static_cast<Float_t>(outputMatrix(sampleIndex, j));
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TCpu<Float_t>>::CopyTensorWeights(TCpuBuffer<Float_t> &buffer,
                                                                     IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &outputMatrix = std::get<2>(fData);

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      buffer[i] = static_cast<Float_t>(outputMatrix(sampleIndex, 0));
      sampleIterator++;
   }
}

#if 0
//______________________________________________________________________________
template <>
TTensorBatch<TCpu<Float_t> > TTensorDataLoader<TensorInput, TCpu<Float_t> >::GetTensorBatch()
{
   // After copying the data to the device, wrap the device buffer in the respective
   // architectures matrix type
   DeviceBufferTuple DeviceBuffers = CopyTensorBatches();

   Tensor_t inputTensor( std::get<0>(DeviceBuffers), { fBatchHeight, fBatchWidth, fBatchSize } );
   // size_t jump = fBatchHeight * fBatchWidth;
   // for (size_t i = 0; i < fBatchSize; i++) {
   //    DeviceBuffer_t subInputDeviceBuffer = std::get<0>(DeviceBuffers).GetSubBuffer(i * jump, jump);
   //    inputTensor.emplace_back(subInputDeviceBuffer, fBatchHeight, fBatchWidth);
   // }
   Matrix_t outputMatrix(std::get<1>(DeviceBuffers), fBatchSize, fNOutputFeatures);
   Matrix_t weightMatrix(std::get<2>(DeviceBuffers), fBatchSize, fNOutputFeatures);

   fBatchIndex++;
   return TTensorBatch<TCpu<Float_t> >(inputTensor, outputMatrix, weightMatrix);
}
#endif

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TCpu<Double_t>>::CopyTensorInput(TCpuBuffer<Double_t> &buffer,
                                                                     IndexIterator_t sampleIterator)
{
   const std::vector<TMatrixT<Double_t>> &inputTensor = std::get<0>(fData);

   if (fBatchDepth == 1) {
      for (size_t i = 0; i < fBatchHeight; i++) {
         size_t sampleIndex = *sampleIterator;
         for (size_t j = 0; j < fBatchWidth; j++) {
            size_t bufferIndex = j * fBatchHeight + i;
            buffer[bufferIndex] = inputTensor[0](sampleIndex, j);
         }
         sampleIterator++;
      }
   } else {
      for (size_t i = 0; i < fBatchDepth; i++) {
         size_t sampleIndex = *sampleIterator;
         for (size_t j = 0; j < fBatchHeight; j++) {
            for (size_t k = 0; k < fBatchWidth; k++) {
               size_t bufferIndex = i * fBatchHeight * fBatchWidth + k * fBatchHeight + j;
               buffer[bufferIndex] = inputTensor[sampleIndex](j, k);
            }
         }
         sampleIterator++;
      }
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TCpu<Double_t>>::CopyTensorOutput(TCpuBuffer<Double_t> &buffer,
                                                                      IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &outputMatrix = std::get<1>(fData);
   size_t n = outputMatrix.GetNcols();

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * fBatchSize + i;
         buffer[bufferIndex] = outputMatrix(sampleIndex, j);
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TCpu<Double_t>>::CopyTensorWeights(TCpuBuffer<Double_t> &buffer,
                                                                       IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &outputMatrix = std::get<2>(fData);

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      buffer[i] = static_cast<Double_t>(outputMatrix(sampleIndex, 0));
      sampleIterator++;
   }
}
#if 0
//______________________________________________________________________________
template <>
TTensorBatch<TCpu<Double_t> > TTensorDataLoader<TensorInput, TCpu<Double_t> >::GetTensorBatch()
{
   // After copying the data to the device, wrap the device buffer in the respective
   // architectures matrix type
   DeviceBufferTuple DeviceBuffers = CopyTensorBatches();

   Tensor_t inputTensor( std::get<0>(DeviceBuffers), { fBatchHeight, fBatchWidth, fBatchSize } );
   // size_t jump = fBatchHeight * fBatchWidth;
   // for (size_t i = 0; i < fBatchSize; i++) {
   //    DeviceBuffer_t subInputDeviceBuffer = std::get<0>(DeviceBuffers).GetSubBuffer(i * jump, jump);
   //    inputTensor.emplace_back(subInputDeviceBuffer, fBatchHeight, fBatchWidth);
   // }

   Matrix_t outputMatrix(std::get<1>(DeviceBuffers), fBatchSize, fNOutputFeatures);
   Matrix_t weightMatrix(std::get<2>(DeviceBuffers), fBatchSize, fNOutputFeatures);

   fBatchIndex++;
   return TTensorBatch<TCpu<Double_t> >(inputTensor, outputMatrix, weightMatrix);
}
#endif


///- re-implement specialization for Double_t
//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TCpu<Double_t>>::CopyTensorInput(TCpuBuffer<Double_t> &buffer,
                                                                     IndexIterator_t sampleIterator)
{
   // one event, one  example in the batch

   if (fBatchDepth == 1 && fBatchHeight == fBatchSize) {
      for (size_t i = 0; i < fBatchHeight; i++) {
         size_t sampleIndex = *sampleIterator;
         Event * event = std::get<0>(fData)[sampleIndex];
         for (size_t j = 0; j < fBatchWidth; j++) {
            size_t bufferIndex = j * fBatchHeight + i;
            buffer[bufferIndex] = event->GetValue(j);
         }
         sampleIterator++;
      }
   } else if (fBatchDepth == fBatchSize) {
      // batchDepth is batch size
      for (size_t i = 0; i < fBatchDepth; i++) {
         size_t sampleIndex = *sampleIterator;
         Event * event = std::get<0>(fData)[sampleIndex];
         for (size_t j = 0; j < fBatchHeight; j++) {
            for (size_t k = 0; k < fBatchWidth; k++) {
               // because of the ordering of tensor in memory is NHWC
               size_t bufferIndex = i * fBatchHeight * fBatchWidth + k * fBatchHeight + j;
               buffer[bufferIndex] = event->GetValue(j * fBatchWidth + k);
            }
         }
         sampleIterator++;
      }
   }
   else {
      Error("TTensorDataLoader","Inconsistency between batch depth and batch size");
      R__ASSERT(0);   // one event, one  example in the batch
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TCpu<Double_t>>::CopyTensorOutput(TCpuBuffer<Double_t> &buffer,
                                                                      IndexIterator_t sampleIterator)
{
   const DataSetInfo &info = std::get<1>(fData);
   size_t n = buffer.GetSize() / fBatchSize;

   // Copy target(s).

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator++;
      Event *event = std::get<0>(fData)[sampleIndex];
      for (size_t j = 0; j < n; j++) {
         // Copy output matrices.
         size_t bufferIndex = j * fBatchSize + i;
         // Classification
         if (event->GetNTargets() == 0) {
            if (n == 1) {
               // Binary.
               buffer[bufferIndex] = (info.IsSignal(event)) ? 1.0 : 0.0;
            } else {
               // Multiclass.
               buffer[bufferIndex] = 0.0;
               if (j == event->GetClass()) {
                  buffer[bufferIndex] = 1.0;
               }
            }
         } else {
            buffer[bufferIndex] = static_cast<Float_t>(event->GetTarget(j));
         }
      }
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TCpu<Double_t>>::CopyTensorWeights(TCpuBuffer<Double_t> &buffer,
                                                                       IndexIterator_t sampleIterator)
{
   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator++;
      Event *event = std::get<0>(fData)[sampleIndex];
       buffer[i] = event->GetWeight();
   }
}

#if 0
//______________________________________________________________________________
template <>
TTensorBatch<TCpu<Double_t> > TTensorDataLoader<TMVAInput_t, TCpu<Double_t> >::GetTensorBatch()
{
   // After copying the data to the device, wrap the device buffer in the respective
   // architectures matrix type
   DeviceBufferTuple DeviceBuffers = CopyTensorBatches();


   Tensor_t inputTensor( std::get<0>(DeviceBuffers), { fBatchHeight, fBatchWidth, fBatchSize } );
   // size_t jump = fBatchHeight * fBatchWidth;
   // for (size_t i = 0; i < fBatchSize; i++) {
   //    DeviceBuffer_t subInputDeviceBuffer = std::get<0>(DeviceBuffers).GetSubBuffer(i * jump, jump);
   //    inputTensor.emplace_back(subInputDeviceBuffer, fBatchHeight, fBatchWidth);
   // }
   Matrix_t outputMatrix(std::get<1>(DeviceBuffers), fBatchSize, fNOutputFeatures);
   Matrix_t weightMatrix(std::get<2>(DeviceBuffers), fBatchSize, fNOutputFeatures);

   fBatchIndex++;
   return TTensorBatch<TCpu<Double_t> >(inputTensor, outputMatrix, weightMatrix);
}
#endif

///- re-implement specialization for Float_t
//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TCpu<Float_t>>::CopyTensorInput(TCpuBuffer<Float_t> &buffer,
                                                                   IndexIterator_t sampleIterator)
{
   // one event, one  example in the batch

   if (fBatchDepth == 1 && fBatchHeight == fBatchSize) {
      for (size_t i = 0; i < fBatchHeight; i++) {
         size_t sampleIndex = *sampleIterator;
         Event * event = std::get<0>(fData)[sampleIndex];
         for (size_t j = 0; j < fBatchWidth; j++) {
            size_t bufferIndex = j * fBatchHeight + i;
            buffer[bufferIndex] = event->GetValue(j);
         }
         sampleIterator++;
      }
   } else if (fBatchDepth == fBatchSize) {
      // batchDepth is batch size
      for (size_t i = 0; i < fBatchDepth; i++) {
         size_t sampleIndex = *sampleIterator;
         Event * event = std::get<0>(fData)[sampleIndex];
         for (size_t j = 0; j < fBatchHeight; j++) {
            for (size_t k = 0; k < fBatchWidth; k++) {
               // because of the column-major ordering
               size_t bufferIndex = i * fBatchHeight * fBatchWidth + k * fBatchHeight + j;
               buffer[bufferIndex] = event->GetValue(j * fBatchWidth + k);
            }
         }
         sampleIterator++;
      }
   }
   else {
      Error("TTensorDataLoader","Inconsistency between batch depth and batch size");
      R__ASSERT(0);
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TCpu<Float_t>>::CopyTensorOutput(TCpuBuffer<Float_t> &buffer,
                                                                      IndexIterator_t sampleIterator)
{
   const DataSetInfo &info = std::get<1>(fData);
   size_t n = buffer.GetSize() / fBatchSize;

   // Copy target(s).

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator++;
      Event *event = std::get<0>(fData)[sampleIndex];
      for (size_t j = 0; j < n; j++) {
         // Copy output matrices.
         size_t bufferIndex = j * fBatchSize + i;
         // Classification
         if (event->GetNTargets() == 0) {
            if (n == 1) {
               // Binary.
               buffer[bufferIndex] = (info.IsSignal(event)) ? 1.0 : 0.0;
            } else {
               // Multiclass.
               buffer[bufferIndex] = 0.0;
               if (j == event->GetClass()) {
                  buffer[bufferIndex] = 1.0;
               }
            }
         } else {
            buffer[bufferIndex] = static_cast<Float_t>(event->GetTarget(j));
         }
      }
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TCpu<Float_t>>::CopyTensorWeights(TCpuBuffer<Float_t> &buffer,
                                                                       IndexIterator_t sampleIterator)
{
   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator++;
      Event *event = std::get<0>(fData)[sampleIndex];
      buffer[i] = event->GetWeight();
   }
}

#if 0
//______________________________________________________________________________
template <>
TTensorBatch<TCpu<Float_t> > TTensorDataLoader<TMVAInput_t, TCpu<Float_t> >::GetTensorBatch()
{
   // After copying the data to the device, wrap the device buffer in the respective
   // architectures matrix type
   DeviceBufferTuple DeviceBuffers = CopyTensorBatches();

   Tensor_t inputTensor( std::get<0>(DeviceBuffers), { fBatchHeight, fBatchWidth, fBatchSize } );
   // std::vector<Matrix_t> inputTensor;
   // size_t jump = fBatchHeight * fBatchWidth;
   // for (size_t i = 0; i < fBatchSize; i++) {
   //    DeviceBuffer_t subInputDeviceBuffer = std::get<0>(DeviceBuffers).GetSubBuffer(i * jump, jump);
   //    inputTensor.emplace_back(subInputDeviceBuffer, fBatchHeight, fBatchWidth);
   // }
   Matrix_t outputMatrix(std::get<1>(DeviceBuffers), fBatchSize, fNOutputFeatures);
   Matrix_t weightMatrix(std::get<2>(DeviceBuffers), fBatchSize, fNOutputFeatures);

   fBatchIndex++;
   return TTensorBatch<TCpu<Float_t> >(inputTensor, outputMatrix, weightMatrix);
}
#endif

//______________________________________________________________________________
// Explicit instantiations.
template class TCpuBuffer<Double_t>;
template class TCpuBuffer<Float_t>;

template class TTensorDataLoader<TensorInput, TCpu<Float_t>>;
template class TTensorDataLoader<TMVAInput_t, TCpu<Float_t>>;
template class TTensorDataLoader<TensorInput, TCpu<Double_t>>;
template class TTensorDataLoader<TMVAInput_t, TCpu<Double_t>>;

} // namespace DNN
} // namespace TMVA
