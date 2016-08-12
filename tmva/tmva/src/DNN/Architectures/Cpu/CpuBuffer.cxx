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
#include "TMVA/DNN/DataLoader.h"
#include "TMVA/DNN/Architectures/Cpu.h"
#include "Rtypes.h"
#include <iostream>

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename AReal>
void TCpuBuffer<AReal>::TDestructor::operator()(AReal ** pointer)
{
   delete[] * pointer;
   delete[] pointer;
}

//______________________________________________________________________________
template<typename AReal>
TCpuBuffer<AReal>::TCpuBuffer(size_t size)
    : fSize(size), fOffset(0)
{
   AReal ** pointer = new AReal * [1];
   * pointer        = new AReal[size];
   fBuffer          = std::shared_ptr<AReal *>(pointer, fDestructor);
}

//______________________________________________________________________________
template<typename AReal>
TCpuBuffer<AReal> TCpuBuffer<AReal>::GetSubBuffer(size_t offset, size_t size)
{
   TCpuBuffer buffer = *this;
   buffer.fOffset = offset;
   buffer.fSize   = size;
   return buffer;
}

//______________________________________________________________________________
template<typename AReal>
void TCpuBuffer<AReal>::CopyFrom(TCpuBuffer & other)
{
   std::swap(*this->fBuffer, *other.fBuffer);
}

//______________________________________________________________________________
template<typename AReal>
void TCpuBuffer<AReal>::CopyTo(TCpuBuffer & other)
{
   //other = *this;
}

//______________________________________________________________________________
template<>
void TDataLoader<MatrixInput_t, TCpu<Double_t, false>>::CopyInput(
    TCpuBuffer<Double_t> & buffer,
    IndexIterator_t sampleIterator,
    size_t batchSize)
{
   const TMatrixT<Double_t> &inputMatrix  = std::get<0>(fData);
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
template<>
void TDataLoader<MatrixInput_t, TCpu<Double_t, false>>::CopyOutput(
    TCpuBuffer<Double_t> & buffer,
    IndexIterator_t sampleIterator,
    size_t batchSize)
{
   const TMatrixT<Double_t> &outputMatrix  = std::get<1>(fData);
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

template class TCpuBuffer<Double_t>;

} // namespace DNN
} // namespace TMVA


