// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 12/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// This file contains the declaration of the Batch, BatchIterator //
// and DataLoader classes for the reference implementation.       //
////////////////////////////////////////////////////////////////////

#include <vector>
#include "TMatrix.h"
#include "TMVA/Event.h"

#ifndef TMVA_DNN_ARCHITECTURES_REFERENCE_DATALOADER
#define TMVA_DNN_ARCHITECTURES_REFERENCE_DATALOADER

namespace TMVA
{
namespace DNN
{

// Input Data Types
using MatrixInput_t = std::pair<const TMatrixT<Double_t> &,
                                const TMatrixT<Double_t> &>;
using TMVAInput_t   = std::vector<TMVA::Event*>;

/** Reference Batch Class
 *
 * The Batch class for the reference implementation. Provides the required
 * GetInput() and GetOutput() method which returns the input and output
 * corresponding to this batch in matrix form.
 *
 * \tparam Real_t The floating point type to represent floating point numbers.
 */
template<typename Real_t>
class TReferenceBatch
{
   using Matrix_t = TMatrixT<Double_t>;

   const Matrix_t & fInput;
   const Matrix_t & fOutput;

public:

   /** Create a batch containing the provided input and output in matrix
    *  form.
    *
    * \param input Reference to the input data in matrix form.
    * \param output Reference to the expected output (truth) in matrix form.
    */
   TReferenceBatch(Matrix_t & input, Matrix_t & output)
   : fInput(input), fOutput(output)
   {
      // Nothing to do here.
   }

   Matrix_t GetInput()  {return fInput;}
   Matrix_t GetOutput() {return fOutput;}

};

/** Reference DataLoader Class
 *
 * DataLoader class for the reference implementation. The
 * TReferenceDataLoader class takes a reference to the input data and
 * prepares the batches for the training and evaluation of the neural
 * network.
 *
 * \tparam Data_t Defines the input data type, i.e. the class that
 *  provides access to the training and test data. To instantiate this
 *  class with a given input data type, the CopyBatch function
 *  template must be specialized for this type.
 *
 *  \tparam Real_t The floating point type used to represent scalars.
 */
template<typename Data_t, typename Real_t>
class TReferenceDataLoader
{
   using Scalar_t = Real_t;
   using Matrix_t = TMatrixT<Real_t>;
   using BatchIterator_t = typename std::vector<TReferenceBatch<Real_t>>::iterator;
   using IndexIterator_t = typename std::vector<size_t>::iterator;

   const Data_t & fInput;

   size_t fNSamples;
   size_t fBatchSize;
   size_t fNInputFeatures;
   size_t fNOutputFeatures;
   size_t fNBatches;

   std::vector<TMatrixT<Double_t>> fInputMatrices;
   std::vector<TMatrixT<Double_t>> fOutputMatrices;
   std::vector<TReferenceBatch<Real_t>> fBatches;
   std::vector<size_t>        fSampleIndices;

public:

   /** Create data loader from the given input data \p input. The data set
    *  shoule consists of \p nsamples samples. Will create a vector of
    *  batches that are sample randomly from the input data.
    *
    *  \param nsamples Number of samples in the training set.
    *  \param batchSize The batch size.
    *  \param ninputFeatures Number of input features
    *  \param noutputFeatures Number of output features.
   */
   TReferenceDataLoader(const Data_t &input,
                       size_t nsamples,
                       size_t batchSize,
                       size_t ninputFeatures,
                       size_t noutputFeatures);

   /** This function copies a batch from the input data into the the matrices
    *  used as input for the neural network.
    *
    * \param input  Reference to the matrix to write the input samples into.
    * \param output Reference to the matrix to write the output labels into.
    * \param begin  Index iterator containing the sample indices which to include
    *  in the current batch.
    * \param end    Iterator to the index of the last sample to include in thie
    *  batch.
    */
   void CopyBatch(Matrix_t &inputMatrix,
                  Matrix_t &outputMatrix,
                  const Data_t & input,
                  IndexIterator_t begin,
                  IndexIterator_t end);

   /** Return iterator over batches in one epoch the data set. Shuffles
    *  the training sample before returning the iterator. */
   BatchIterator_t begin();
   /** Iterator pointing to the end of the epoch, required for C++ range
    *  loops */
   BatchIterator_t end();

   size_t GetBatchSize() const       {return fBatchSize;}
   size_t GetNBatchesInEpoch() const {return fNBatches;}
};

template <>
void TReferenceDataLoader<MatrixInput_t, Double_t>::CopyBatch(Matrix_t &inputMatrix,
                                                             Matrix_t &outputMatrix,
                                                             const MatrixInput_t & input,
                                                             IndexIterator_t begin,
                                                             IndexIterator_t end);

template <>
void TReferenceDataLoader<TMVAInput_t, Double_t>::CopyBatch(Matrix_t &inputMatrix,
                                                           Matrix_t &outputMatrix,
                                                           const TMVAInput_t & input,
                                                           IndexIterator_t begin,
                                                           IndexIterator_t end);
} // namespace TMVA
} // namespace DNN

#endif
