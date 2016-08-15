// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 08/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////
// Generic data loader for neural network input data. //
////////////////////////////////////////////////////////

#ifndef TMVA_DNN_DATALOADER
#define TMVA_DNN_DATALOADER

#include "TMatrix.h"
#include <vector>

#include "TMVA/Event.h"

namespace TMVA {
namespace DNN  {

//
// Input Data Types
//______________________________________________________________________________
using MatrixInput_t    = std::pair<const TMatrixT<Double_t> &,
                                   const TMatrixT<Double_t> &>;
using TMVAInput_t      = std::vector<Event*>;

//
// TBatch Class.
//______________________________________________________________________________
template <typename Architecture_t>
class TBatch
{
private:

   using Matrix_t       = typename Architecture_t::Matrix_t;

   Matrix_t fInputMatrix;
   Matrix_t fOutputMatrix;

public:

   TBatch(Matrix_t &&, Matrix_t &&);
   TBatch(const TBatch  &) = default;
   TBatch(      TBatch &&) = default;
   TBatch & operator=(const TBatch  &) = default;
   TBatch & operator=(      TBatch &&) = default;

   Matrix_t & GetInput()  {return fInputMatrix;}
   Matrix_t & GetOutput() {return fOutputMatrix;}
};

//
// TBatchIterator Class.
//______________________________________________________________________________

template<typename Data_t, typename Architecture_t> class TDataLoader;

template<typename Data_t, typename Architecture_t>
class TBatchIterator
{
private:

   TDataLoader<Data_t, Architecture_t> & fDataLoader;
   size_t fBatchIndex;

public:

TBatchIterator(TDataLoader<Data_t, Architecture_t> & dataLoader, size_t index = 0)
: fDataLoader(dataLoader), fBatchIndex(index)
{
   // Nothing to do here.
}

   TBatch<Architecture_t> operator*() {return fDataLoader.GetBatch();}
   TBatchIterator operator++() {fBatchIndex++; return *this;}
   bool operator!=(const TBatchIterator & other) {
      return fBatchIndex != other.fBatchIndex;
   }
};

//
// TDataLoader Class.
//______________________________________________________________________________
template<typename Data_t, typename Architecture_t>
class TDataLoader
{
private:

   using HostBuffer_t    = typename Architecture_t::HostBuffer_t;
   using DeviceBuffer_t  = typename Architecture_t::DeviceBuffer_t;
   using IndexIterator_t = typename std::vector<size_t>::iterator;
   using Matrix_t        = typename Architecture_t::Matrix_t;
   using BatchIterator_t = TBatchIterator<Data_t, Architecture_t>;

   const Data_t  & fData;

   size_t fNSamples;
   size_t fBatchSize;
   size_t fNInputFeatures;
   size_t fNOutputFeatures;
   size_t fBatchIndex;

   size_t fNStreams;
   std::vector<DeviceBuffer_t> fDeviceBuffers;
   std::vector<HostBuffer_t>  fHostBuffers;

   std::vector<size_t> fSampleIndices;

public:

   TDataLoader(const Data_t & data, size_t nSamples, size_t batchSize,
               size_t nInputFeatures, size_t nOutputFeatures, size_t nStreams = 4);
   TDataLoader(const TDataLoader  &) = default;
   TDataLoader(      TDataLoader &&) = default;
   TDataLoader & operator=(const TDataLoader  &) = default;
   TDataLoader & operator=(      TDataLoader &&) = default;

   /** Copy input matrix into the given host buffer. Function to be specialized by
    *  the architecture-specific backend. */
   void  CopyInput(HostBuffer_t &buffer, IndexIterator_t begin, size_t batchSize);
   /** Copy output matrix into the given host buffer. Function to be specialized
    * by the architecture-spcific backend. */
   void CopyOutput(HostBuffer_t &buffer, IndexIterator_t begin, size_t batchSize);

   BatchIterator_t begin() {return TBatchIterator<Data_t, Architecture_t>(*this);}
   BatchIterator_t end()
   {
      return TBatchIterator<Data_t, Architecture_t>(*this,(fNSamples / fBatchSize)+1);
   }

   void Shuffle();
   TBatch<Architecture_t> GetBatch();

};

//
// TBatch Class.
//______________________________________________________________________________
template<typename Architecture_t>
TBatch<Architecture_t>::TBatch(Matrix_t && inputMatrix, Matrix_t && outputMatrix)
    : fInputMatrix(inputMatrix), fOutputMatrix(outputMatrix)
{
    // Nothing to do here.
}

//
// TDataLoader Class.
//______________________________________________________________________________
template<typename Data_t, typename Architecture_t>
TDataLoader<Data_t, Architecture_t>::TDataLoader(
    const Data_t & data, size_t nSamples, size_t batchSize,
    size_t nInputFeatures, size_t nOutputFeatures, size_t nStreams)
    : fData(data), fNSamples(nSamples), fBatchSize(batchSize),
      fNInputFeatures(nInputFeatures), fNOutputFeatures(nOutputFeatures),
      fBatchIndex(0), fNStreams(nStreams), fDeviceBuffers(), fHostBuffers(),
      fSampleIndices()
{
   size_t inputMatrixSize  = fBatchSize * fNInputFeatures;
   size_t outputMatrixSize = fBatchSize * fNOutputFeatures;

   for (size_t i = 0; i < fNStreams; i++)
   {
      fHostBuffers.push_back(HostBuffer_t(inputMatrixSize + outputMatrixSize));
      fDeviceBuffers.push_back(DeviceBuffer_t(inputMatrixSize + outputMatrixSize));
   }

   fSampleIndices.reserve(fNSamples);
   for (size_t i = 0; i < fNSamples; i++) {
      fSampleIndices.push_back(i);
   }
}

template<typename Data_t, typename Architecture_t>
TBatch<Architecture_t> TDataLoader<Data_t, Architecture_t>::GetBatch()
{
   fBatchIndex %= (fNSamples / fBatchSize); // Cycle through samples.


   size_t inputMatrixSize  = fBatchSize * fNInputFeatures;
   size_t outputMatrixSize = fBatchSize * fNOutputFeatures;

   size_t streamIndex = fBatchIndex % fNStreams;
   HostBuffer_t   & hostBuffer   = fHostBuffers[streamIndex];
   DeviceBuffer_t & deviceBuffer = fDeviceBuffers[streamIndex];

   HostBuffer_t inputHostBuffer  = hostBuffer.GetSubBuffer(0, inputMatrixSize);
   HostBuffer_t outputHostBuffer = hostBuffer.GetSubBuffer(inputMatrixSize,
                                                           outputMatrixSize);

   DeviceBuffer_t inputDeviceBuffer  = deviceBuffer.GetSubBuffer(0, inputMatrixSize);
   DeviceBuffer_t outputDeviceBuffer = deviceBuffer.GetSubBuffer(inputMatrixSize,
                                                                 outputMatrixSize);
   size_t sampleIndex = fBatchIndex * fBatchSize;
   IndexIterator_t sampleIndexIterator = fSampleIndices.begin() + sampleIndex;

   CopyInput(inputHostBuffer,   sampleIndexIterator, fBatchSize);
   CopyOutput(outputHostBuffer, sampleIndexIterator, fBatchSize);

   deviceBuffer.CopyFrom(hostBuffer);
   Matrix_t  inputMatrix(inputDeviceBuffer,  fBatchSize, fNInputFeatures);
   Matrix_t outputMatrix(outputDeviceBuffer, fBatchSize, fNOutputFeatures);

   fBatchIndex++;
   return TBatch<Architecture_t>(std::move(inputMatrix), std::move(outputMatrix));
}

template<typename Data_t, typename Architecture_t>
void TDataLoader<Data_t, Architecture_t>::Shuffle()
{
   std::random_shuffle(fSampleIndices.begin(), fSampleIndices.end());
}

}
}
#endif
