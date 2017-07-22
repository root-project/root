// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TDAE                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Denoising Layer for DeepAutoEncoders                                      *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Akshay Vashistha <akshayvashistha1995@gmail.com>  - CERN, Switzerland     *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/


#ifndef TMVA_DAE_DENOISELAYER
#define TMVA_DAE_DENOISELAYER

#include "TMatrix.h"

#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Functions.h"

#include <vector>

namespace TMVA {
namespace DNN {
namespace DAE {

/** \class TDAE
     Denoising Layer for Deep AutoEncoders. This Layer performs both forward
     and backward propagation. It is the unsupervised learning step for Deep
     AutoEncoders. It corrupts the input according to the given Corruption
     Level. Encodes the input to extract features. Reconstructs the output from
     features.
*/

template <typename Architecture_t>
class TDAE : public VGeneralLayer<Architecture_t> {
public:
  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;
  using Tensor_t = std::vector<Matrix_t>

  Matrix_t fWeights; ///< the weights associated

  Matrix_t fVBiases; ///< bias associated with visible layer

  Matrix_t fHBiases; ///< bias associated with hidden layer

  Tensor_t fCorruptedInput; ///< corrupted Input Units

  Tensor_t fCompressedInput; ///< compressed Input Units

  Tensor_t fReconstructedInput; ///<reconstructed Input Units

  size_t fVisibleUnits ///< total number of visible units

  size_t fHiddenUnits; ///< Number of compressed inputs

  Scalar_t fDropoutProbability; ///< Probability that an input is active.

  Matrix_t fVBiasError; ///< Errors associated with visible Units

  Matrix_t fHBiasError; ///< Errors associated with Hidden Units

  EActivationFunction fF; ///< Activation function of the layer.

  /*! Constructor. */
  TDAE(size_t BatchSize, size_t InputDepth,
       size_t InputHeight, size_t InputWidth,
       size_t HiddenUnits,Scalar_t DropoutProbability, EActivationFunction f);

  /*! Copy the denoise layer provided as a pointer */
  TDAE(TDAE<Architecture_t> *layer);

  /*! Copy constructor. */
  TDAE(const TDAE &);

   /*! Destructor. */
  ~TDAE();

  /*! Initializes the weights, biases associated with visible units and
   *  and biases associated with hidden units. These weights and biases are
   *  layer specific.
   */
  void Initialize(DNN::EInitialization m);

  /*! Corrupts the inputs matrix and stores it in the matrix 'corruptedInput'.
   *  The level of corruption is specified by 'corruptionLevel'.
   *  The input is in the form of Matrix.
   *  */
  void inline Corruption(Tensor_t &input, Scalar_t corruptionLevel);

  /*! Encodes the input matrix in compressed format. 'compressedInput' stores
   *  the compressed version. This compressed input is the feature generated
   *  from given inputs. */
  void inline Encoding(Tensor_t &input);

  /*! Reconstructs the input from the compressed version of inputs.
   *  'reconstructedInput' holds this reconstructed input in the form of matrix.
   *  The reconstructed input has same dimensions as original input.
   */
  void inline Reconstruction();


  void Forward(Tensor_t input, bool applyDropout = false);

  /*! This method is used by the deep network of autoencoders. This trains the
   *  inputs, updates the parameters. The updated parameters are passed on to
   *  Transform Layer in AutoEncoder's DeepNet.
  */
  void TrainLayer(Tensor_t &input,
                  Double_t learningRate,
                  Double_t corruptionLevel, bool applyDropout);

  void Print();

  /* Getters */
  size_t GetVisibleUnits() const { return fVisibleUnits; }
  size_t GetHiddenUnits() const { return fHiddenUnits; }

  const Matrix_t &GetWeights() const { return fWeights; }
  Matrix_t &GetWeights() { return fWeights; }

  const Matrix_t &GetVBiases() const { return fVBiases; }
  Matrix_t &GetVBiases() { return fVBiases; }

  const Matrix_t &GetHBiases() const { return fHBiases; }
  Matrix_t &GetHBiases() { return fHBiases; }

  const Matrix_t &GetVBiasError() const { return fVBiasError; }
  Matrix_t &GetVBiasError() { return fVBiasError; }

  const Matrix_t &GetHBiasError() const { return fHBiasError; }
  Matrix_t &GetHBiasError() { return fHBiasError; }

  const Tensor_t &GetCorruptedInput() const { return fCorruptedInput; }
  Tensor_t &GetCorruptedInput() { return fCorruptedInput; }

  Matrix_t &GetCorruptedInputAt(size_t i) { return fCorruptedInput[i]; }
  const Matrix_t &GetCorruptedInputAt(size_t i) const { return fCorruptedInput[i]; }

  const Tensor_t &GetCompressedInput() const { return fCompressedInput; }
  Tensor_t &GetCompressedInput() { return fCompressedInput; }

  Matrix_t &GetCompressedInputAt(size_t i) { return fCompressedInput[i]; }
  const Matrix_t &GetCompressedInputAt(size_t i) const { return fCompressedInput[i]; }

  const Tensor_t &GetReconstruedInput() const { return fReconstructedInput; }
  Tensor_t &GetReconstruedInputedInput() { return fReconstructedInput; }

  Matrix_t &GetReconstruedInputAt(size_t i) { return fReconstructedInput[i]; }
  const Matrix_t &GetReconstrucedInputAt(size_t i) const { return fReconstructedInput[i]; }


};

//
//
//  Denoise Layer Class - Implementation
//______________________________________________________________________________

template <typename Architecture_t>
TDAE<Architecture_t>::TDAE(size_t batchSize, size_t inputDepth,
                           size_t inputHeight, size_t inputWidth,
                           size_t hiddenUnits, Scalar_t dropoutProbability,
                           EActivationFunction f)
          : VGeneralLayer<Architecture_t>(batchSize),
            fVisibleUnits(inputDepth * inputHeight * inputWidth),
            fHiddenUnits(hiddenUnits)
            fWeights(hiddenUnits, inputDepth * inputHeight * inputWidth),
            fVBiases(inputDepth * inputHeight * inputWidth, 1),
            fHBiases(hiddenUnits, 1), fDropoutProbability(dropoutProbability),
            fF(f), fCorruptedInput(), fCompressedInput(), fReconstructedInput(),
            VBiasError(inputDepth * inputHeight * inputWidth, 1),
            HBiasError(hiddenUnits, 1)

{
  for (size_t i = 0; i < batchSize; i++)
  {
    fCorruptedInput.emplace_back(inputDepth * inputHeight * inputWidth, 1);
    fCompressedInput.emplace_back(hiddenUnits, 1);
    fReconstructedInput.emplace_back(inputDepth * inputHeight * inputWidth, 1);
  }
}

//______________________________________________________________________________
template <typename Architecture_t>
TDAE<Architecture_t>::TDAE(TDAE<Architecture_t> *layer)
                  : VGeneralLayer<Architecture_t>(layer),
                    fVisibleUnits(layer->GetVisibleUnits()),
                    fHiddenUnits(layer->GetHiddenUnits()),
                    fWeights(layer->GetHiddenUnits(), layer->GetVisibleUnits()),
                    fVBiases(layer->GetVisibleUnits(), 1),
                    fHBiases(layer->GetHiddenUnits(), 1),
                    fDropoutProbability(layer->GetDropoutProbability()),
                    fF(layer->GetActivationFunction()),
                    HBiasError(layer->GetHiddenUnits(), 1),
                    VBiasError(layer->GetVisibleUnits(), 1)

{
  size_t batchSize = layer->GetBatchSize();
  for (size_t i = 0; i < batchSize ; i++)
  {
    fCorruptedInput.emplace_back(layer->GetVisibleUnits(), 1);
    fCompressedInput.emplace_back(layer->GetHiddenUnits(), 1);
    fReconstructedInput.emplace_back(layer->GetVisibleUnits(), 1);
  }
}

//______________________________________________________________________________

template <typename Architecture_t>
TDAE<Architecture_t>::TDAE(const TDAE &dae)
                    : VGeneralLayer<Architecture_t>(dae),
                      fVisibleUnits(dae.GetVisibleUnits()),
                      fHiddenUnits(dae.GetHiddenUnits()),
                      fWeights(dae.GetHiddenUnits(), dae.GetVisibleUnits()),
                      fVBiases(dae.GetHiddenUnits(), 1),
                      fHBiases(dae.GetVisibleUnits(), 1),
                      fDropoutProbability(dae.fDropoutProbability),
                      fF(dae.fF),HBiasError(dae.GetHiddenUnits(), 1),
                      VBiasError(dae.GetVisibleUnits(),1)

{
  size_t batchSize = dae.GetBatchSize();
  for (size_t i = 0; i < batchSize ; i++)
  {
    fCorruptedInput.emplace_back(dae.GetVisibleUnits(), 1);
    fCompressedInput.emplace_back(dae.GetHiddenUnits(), 1);
    fReconstructedInput.emplace_back(dae.GetVisibleUnits(), 1);
  }

}
//______________________________________________________________________________

template <typename Architecture_t> TDAE<Architecture_t>::~TDAE() {}

//______________________________________________________________________________

template <typename Architecture_t>
auto TDAE<Architecture_t>::Initialize(DNN::EInitialization m)
-> void

{

  DNN::initialize<Architecture_t>(fWeights, m);
  DNN::initialize<Architecture_t>(fHBiases, DNN::EInitialization::kZero);
  DNN::initialize<Architecture_t>(fVBiases, DNN::EInitialization::kZero);

}

//______________________________________________________________________________

template <typename Architecture_t>
auto TDAE<Architecture_t>::Corruption(Tensor_t &input,
                                      Scalar_t corruptionLevel)
-> void
{
   for (size_t i = 0; i < this->GetBatchSize(); i++)
   {
     Architecture_t::CorruptInput(input[i], this->GetCorruptedInputAt(i), corruptionLevel);
   }
}

//______________________________________________________________________________

template <typename Architecture_t>
auto TDAE<Architecture_t>::Encoding(Tensor_t &input)
-> void
{
  for (size_t i = 0; i < this->GetBatchSize(); i++)
  {
    Architecture_t::EncodeInput(input[i], this->GetCompressedInputAt(i), this->GetWeights());
    Architecture_t::AddBiases(this->GetCompressedInputAt(i), this->GetHBiases());
    evaluate<Architecture_t>(this->GetCompressedInputAt(i), fF);
  }

}
//______________________________________________________________________________
//
// using concept of tied weights, i.e. using same weights as associated with
// previous layer for reconstruction.
//______________________________________________________________________________

template <typename Architecture_t>
auto TDAE<Architecture_t>::Reconstruction()
-> void
{
  for (size_t i = 0; i < this->GetBatchSize(); i++)
  {
    Architecture_t::ReconstructInput(this->GetCompressedInputAt(i),
                                   this->GetReconstructedInputAt(i),
                                   this->GetWeights());
    Architecture_t::AddBiases(this->GetReconstructedInputAt(i), this->GetVBiases());
    evaluate<Architecture_t>(this->GetReconstructedInputAt(i), fF);
  }
}

//______________________________________________________________________________

template <typename Architecture_t>
auto TConvLayer<Architecture_t>::Forward(Tensor_t input, bool applyDropout)
-> void
{
  for (size_t i = 0; i < this->GetBatchSize(); i++)
  {
    if (applyDropout && (this->GetDropoutProbability() != 1.0))
    {
      Architecture_t::Dropout(input[i], this->GetDropoutProbability());
    }
    Encoding(input,compressedInput);
    Reconstruction(compressedInput,reconstructedInput);

}
}
//______________________________________________________________________________

template <typename Architecture_t>
auto TDAE<Architecture_t>::TrainLayer(Tensor_t &input, Double_t learningRate,
                                      Double_t corruptionLevel, bool applyDropout)
-> void
{

  Double_t p = 1 - corruptionLevel;
  Corruption(input,corruptedInput,p);
  Forward(corruptedInput,applyDropout)

  for (size_t i = 0; i < this->GetBatchSize(); i++)
  {
    Architecture_t::UpdateParams(input, this->GetCorruptedInputAt(i),
                                this->GetCompressedInputAt(i),
                                this->GetReconstructedInputAt(i),
                                this->GetVBiases(),
                                this->GetHBiases(), this->GetWeights(),
                                this->GetVBiasError(), this->GetHBiasError(),
                                learningRate, this->GetBatchSize());
  }
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TDAE<Architecture_t>::Print() const
-> void
{
   std::cout << "Batch Size: " << this->GetBatchSize() << "\n"
             << "Input Units: " << this->GetVisibleUnits() << "\n"
             << "Hidden Units: " << this->GetHiddenUnits() << "\n";
}


}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_DAE_DENOISELAYER */
