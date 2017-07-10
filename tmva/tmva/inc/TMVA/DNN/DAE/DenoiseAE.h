// @(#)root/tmva/tmva/dnn/dae:$Id$
// Author: Akshay Vashistha(ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd                                            *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////////
// Contains DAE class that represents the denoising autoencoder layer. //
//                                                                     //
/////////////////////////////////////////////////////////////////////////

#ifndef TMVA_DAE
#define TMVA_DAE

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"
#include "AE.h"

#include <cmath>
#include <iostream>
#include <cstdlib>

namespace TMVA
{
namespace DNN
{
namespace DAE
{

//______________________________________________________________________________
//
// The DAE layer
//______________________________________________________________________________

/** \class DAE
    Generic Denoising Layer class.
    This generic Denoising Layer class represents a layer to denoise the inputs.
    It inherits all of the properties of the generic virtual base class
    AELayer.
*/

template <typename Architecture_t> class DAE : public AELayer<Architecture_t> {

public:
  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;

  Matrix_t fWeights; ///< the weights associated

  Matrix_t fVBiases; ///< bias associated with visible layer

  Matrix_t fHBiases; ///< bias associated with hidden layer

  size_t fBatchSize; ///< Batch size used for training and evaluation.

  size_t fVisibleUnits; ///< number of visible units in one input set

  size_t fHiddenUnits; ///< number of hidden units in the hidden layer of
                       ///autoencoder

  DAE(size_t BatchSize, size_t VisibleUnits, size_t HiddenUnits);

  DAE(const DAE &);

  ~DAE();

  size_t GetBatchSize() const { return fBatchSize; }
  size_t GetVisibleUnits() const { return fVisibleUnits; }
  size_t GetHiddenUnits() const { return fHiddenUnits; }

  const Matrix_t &GetWeights() const { return fWeights; }
  Matrix_t &GetWeights() { return fWeights; }

  const Matrix_t &GetVBiases() const { return fVBiases; }
  Matrix_t &GetVBiases() { return fVBiases; }
  const Matrix_t &GetHBiases() const { return fHBiases; }
  Matrix_t &GetHBiases() { return fHBiases; }

  // This method corrupts the input. Currently it corrupts the random inputs
  // according to the corruption Level.
  void inline Corruption(Matrix_t &input, Matrix_t &corruptedInput,
                         Scalar_t corruptionLevel);

  // This encodes the input into a compressed form.
  void inline Encoding(Matrix_t &input, Matrix_t &compressedInput);

  // This reconstructs the input from the compressed units. The reconstructed Input
  // has same dimensions as that of the input.
  void inline Reconstruction(Matrix_t &compressedInput, Matrix_t &reconstructedInput);

  // this updates the parameters after passing it to the network.
  void TrainLayer(Matrix_t &input, Double_t learningRate, Double_t corruptionLevel);

}; // class DAE

//______________________________________________________________________________

template <typename Architecture_t>
DAE<Architecture_t>::DAE(size_t batchSize, size_t visibleUnits,
                         size_t hiddenUnits)
    : AELayer<Architecture_t>(batchSize, visibleUnits, hiddenUnits),
      fWeights(hiddenUnits, visibleUnits), fVBiases(visibleUnits, 1),
      fHBiases(hiddenUnits, 1)

{}

//______________________________________________________________________________

template <typename Architecture_t>
DAE<Architecture_t>::DAE(const DAE &dae)
    : AELayer<Architecture_t>(dae),
      fWeights(dae.GetHiddenUnits, dae.GetVisibleUnits),
      fVBiases(dae.GetHiddenUnits, 1), fHBiases(dae.GetVisibleUnits, 1)

{
  Architecture_t::Copy(fWeights, dae.GetWeights());
  Architecture_t::Copy(fHBiases, dae.GetHBiases());
  Architecture_t::Copy(fVBiases, dae.GetVBiases());

}
//______________________________________________________________________________

template <typename Architecture_t> DAE<Architecture_t>::~DAE() {}

//______________________________________________________________________________


//______________________________________________________________________________

template <typename Architecture_t>
auto DAE<Architecture_t>::Corruption(Matrix_t &input, Matrix_t &corruptedInput,
                                     Scalar_t corruptionLevel) -> void {
  Architecture_t::CorruptInput(input, corruptedInput, corruptionLevel);
}

//______________________________________________________________________________

template <typename Architecture_t>
auto DAE<Architecture_t>::Encoding(Matrix_t &input, Matrix_t &compressedInput)
    -> void {
  Architecture_t::EncodeInputs(input,compressedInput,fWeights);
  Architecture_t::AddBiases(compressedInput,fHBiases);
  Architecture_t::Sigmoid(compressedInput);

}
//______________________________________________________________________________
//______________________________________________________________________________


//using concept of tied weights, i.e. using same weights as associated with previous layer
//______________________________________________________________________________

template <typename Architecture_t>
auto DAE<Architecture_t>::Reconstruction(Matrix_t &compressedInput,
                                         Matrix_t &reconstructedInput) -> void {
  Architecture_t::ReconstructInput(compressedInput, reconstructedInput,
                                   fWeights);
  Architecture_t::AddBiases(reconstructedInput,fVBiases);
  Architecture_t::Sigmoid(reconstructedInput);
}
//______________________________________________________________________________

template <typename Architecture_t>
auto DAE<Architecture_t>::TrainLayer(Matrix_t &input, Double_t learningRate,
                                     Double_t corruptionLevel) -> void {
  Matrix_t corruptedInput(fVisibleUnits,1);
  Matrix_t compressedInput(fHiddenUnits,1);
  Matrix_t reconstructedInput(fVisibleUnits,1);
  Matrix_t VBiasError(fVisibleUnits,1);
  Matrix_t HBiasError(fHiddenUnits,1);
  Double_t p = 1 - corruptionLevel;

  Corruption(input,corruptedInput,p);
  Encoding(corruptedInput,compressedInput);
  Reconstruction(compressedInput,reconstructedInput);

  Architecture_t::UpdateParams(
      input, corruptedInput, compressedInput, reconstructedInput, fVBiases,
      fHBiases, fWeights, VBiasError, HBiasError, learningRate, fBatchSize);
}


}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_DAE */
