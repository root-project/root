// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 21/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////
// Declaration of the CpuDataLoader, CpuBatchIterator and       //
// Cpu Batch for the multi-threaded CPU implementation of DNNs. //
//////////////////////////////////////////////////////////////////

#include <vector>
#include "TMatrix.h"
#include "CpuMatrix.h"
#include "TMVA/Event.h"

#ifndef TMVA_DNN_ARCHITECTURES_CPU_DATALOADER
#define TMVA_DNN_ARCHITECTURES_CPU_DATALOADER

namespace TMVA
{
namespace DNN
{

// Input Data Types
using MatrixInput_t = std::pair<const TMatrixT<Double_t> &,
                                const TMatrixT<Double_t> &>;
using TMVAInput_t   = std::vector<TMVA::Event*>;

// TCpuBatch
//______________________________________________________________________________
/** Cpu Batch Class
 *
 * Holds reference to the CpuMatrix representation of the input and output
 * data for the neural net.
 *
 * \tparam Real_t The floating point type to represent floating point numbers.
 */
template<typename Real_t>
class TCpuBatch
{
   using Matrix_t = TCpuMatrix<Real_t>;

         Matrix_t & fInput;
   const Matrix_t & fOutput;

public:

   /** Create a batch containing the provided input and output in matrix
    *  form.
    *
    * \param input Reference to the input data in matrix form.
    * \param output Reference to the expected output (truth) in matrix form.
    */
   TCpuBatch(Matrix_t & input, const Matrix_t & output)
   : fInput(input), fOutput(output)
   {
      // Nothing to do here.
   }

         Matrix_t & GetInput()  {return fInput;}
   const Matrix_t & GetOutput() {return fOutput;}
};

template<typename Data_t, typename Real_t>
class TCpuDataLoader;

// TCpuBatchIterator
//______________________________________________________________________________
/** Batch Iterator Class.
 *
 * Very simple iterator class that internally stores the current batch index
 * and accesses batches throught the GetBatch() member function provided by
 * the corresponding data loader object.
 */
template<typename Data_t, typename Real_t>
class TCpuBatchIterator
{
private:

   TCpuDataLoader<Data_t, Real_t> & fDataLoader;
   size_t fBatchIndex;

public:

   /** Construct a batch iterator iterating through the batches in an epoch from
    *  the given batch index on. If used explicitly use with \p batchIndex = 0
    *  otherwise it is not ensured that data loader has transferred the requested
    *  data.
    */
   TCpuBatchIterator(TCpuDataLoader<Data_t,Real_t> &dataLoader, size_t batchIndex);

   /** Return batch corresponding to the current (internal) batch index.
    *  Periodically triggers data reload from the data loader. */
   TCpuBatch<Real_t>   operator*();
   TCpuBatchIterator & operator++();

   bool operator!=(const TCpuBatchIterator &);
   bool operator==(const TCpuBatchIterator &);

   size_t GetBatchIndex() const {return fBatchIndex;}
};

// TCpuDataLoader
//______________________________________________________________________________
/** CpuDataLoader Class
 *
 * DataLoader class for the mult-threaded CPU implementation of DNNs. The
 * TCpuDataLoader class takes a reference to the input data and
 * prepares the batches for the training and evaluation of the neural
 * network.
 * The data loader has an internal buffer which holds a user-defined number
 * of batches. This is to avoid duplication of data sets due to preloading.
 * Filling of the buffers is autmatically triggered when iterating over
 * the batches in the current epoch.
 * Subsequent iterations through epochs of training data are in randomized
 * order.
 *
 * \tparam Data_t Defines the input data type, i.e. the class that
 *  provides access to the training and test data. To instantiate this
 *  class with a given input data type, the CopyBatch function
 *  template must be specialized for this type.
 *  \tparam Real_t The floating point type used to represent scalars.
 */
template<typename Data_t, typename Real_t>
class TCpuDataLoader
{
   using Scalar_t = Real_t;
   using Matrix_t = TCpuMatrix<Real_t>;
   using BatchIterator_t = TCpuBatchIterator<Data_t, Real_t>;
   using IndexIterator_t = typename std::vector<size_t>::iterator;

   const Data_t & fInput;

   size_t fNSamples;
   size_t fBatchSize;
   size_t fBufferSize;
   size_t fNInputFeatures;
   size_t fNOutputFeatures;
   size_t fNBatches;
   size_t fBatchIndex;

   std::vector<TCpuMatrix<Double_t>> fInputMatrices;
   std::vector<TCpuMatrix<Double_t>> fOutputMatrices;
   std::vector<size_t>               fSampleIndices;

public:

   /** Create data loader from the given input data \p input. The data set
    *  should consists of \p nsamples samples. Internally holds a buffer containing
    *  \p bufferSize batches, which are periodically filled with data during
    *  the iteration an epoch.
    *
    *  \param nsamples Number of samples in the training set.
    *  \param batchSize The batch size.
    *  \param ninputFeatures Number of input features
    *  \param noutputFeatures Number of output features.
    *  \param bufferSize Size of the internal data buffer.
   */
   TCpuDataLoader(const Data_t &input,
                  size_t nsamples,
                  size_t batchSize,
                  size_t ninputFeatures,
                  size_t noutputFeatures,
                  size_t bufferSize = 100);

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
    *  the sample order before returning the iterator. */
   BatchIterator_t begin();
   /** Iterator pointing to the end of the epoch, required for C++ range
    *  loops */
   BatchIterator_t end();
   /** Return batch corresponding to the batch Index \p batchIndex in the
    *  current epoch. Takes care of periodically reloading data into the
    *  internal buffer.*/
   TCpuBatch<Real_t> GetBatch(size_t batchIndex);

   size_t GetBatchSize() const       {return fBatchSize;}
   size_t GetNBatchesInEpoch() const {return fNBatches;}

private:

   inline void CopyData(size_t batchIndex);

};

template <>
void TCpuDataLoader<MatrixInput_t, Double_t>::CopyBatch(
    Matrix_t &inputMatrix,
    Matrix_t &outputMatrix,
    const MatrixInput_t & input,
    IndexIterator_t begin,
    IndexIterator_t end);

template <>
void TCpuDataLoader<TMVAInput_t, Double_t>::CopyBatch(
    Matrix_t &inputMatrix,
    Matrix_t &outputMatrix,
    const TMVAInput_t & input,
    IndexIterator_t begin,
    IndexIterator_t end);

} // namespace TMVA
} // namespace DNN

#endif
<<<<<<< HEAD

=======
>>>>>>> 055354f6262b9c10847d24ce0f683235cca9d892
