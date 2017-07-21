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

template <typename Architecture_t>
class TDAE : public VGeneralLayer<Architecture_t> {
public:
  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;
  using Tensor_t = std::vector<Matrix_t>

  Matrix_t fWeights; ///< the weights associated

  Matrix_t fVBiases; ///< bias associated with visible layer

  Matrix_t fHBiases; ///< bias associated with hidden layer

  size_t fVisibleUnits ///< total number of visible units

  size_t fHiddenUnits; ///< Number of compressed inputs

  /*! Constructor. */
  TDAE(size_t BatchSize, size_t InputDepth,
       size_t InputHeight, size_t InputWidth,
       size_t HiddenUnits);

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
  void inline Corruption(Tensor_t &input,
                         Tensor_t &corruptedInput,
                         Scalar_t corruptionLevel);

  /*! Encodes the input matrix in compressed format. 'compressedInput' stores
   *  the compressed version. This compressed input is the feature generated
   *  from given inputs. */
  void inline Encoding(Tensor_t &input, Tensor_t &compressedInput);

  /*! Reconstructs the input from the compressed version of inputs.
   *  'reconstructedInput' holds this reconstructed input in the form of matrix.
   *  The reconstructed input has same dimensions as original input.
   */
  void inline Reconstruction(Tensor_t &compressedInput,
                             Tensor_t &reconstructedInput);

  /*! This method is used by the deep network of autoencoders. This trains the
   *  inputs, updates the parameters. The updated parameters are passed on to
   *  Transform Layer in AutoEncoder's DeepNet.
  */
  void TrainLayer(Tensor_t &input,
                  Double_t learningRate,
                  Double_t corruptionLevel);

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

};

//
//
//  Denoise Layer Class - Implementation
//______________________________________________________________________________

template <typename Architecture_t>
TDAE<Architecture_t>::TDAE(size_t batchSize, size_t inputDepth,
                           size_t inputHeight, size_t inputWidth,
                           size_t hiddenUnits)
          : VGeneralLayer<Architecture_t>(batchSize),
            fVisibleUnits(inputDepth * inputHeight * inputWidth),
            fHiddenUnits(hiddenUnits)
            fWeights(hiddenUnits, inputDepth * inputHeight * inputWidth),
            fVBiases(inputDepth * inputHeight * inputWidth, 1),
            fHBiases(hiddenUnits, 1)

{

}

//______________________________________________________________________________
template <typename Architecture_t>
TDAE<Architecture_t>::TDAE(TDAE<Architecture_t> *layer)
                  : VGeneralLayer<Architecture_t>(layer),
                    fVisibleUnits(layer->GetVisibleUnits()),
                    fHiddenUnits(layer->GetHiddenUnits()),
                    fWeights(layer->GetHiddenUnits(), layer->GetVisibleUnits()),
                    fVBiases(layer->GetVisibleUnits(), 1),
                    fHBiases(layer->GetHiddenUnits(), 1)

{
  Architecture_t::Copy(fWeights, layer.GetWeights());
  Architecture_t::Copy(fVBiases, layer.GetVBiases());
  Architecture_t::Copy(fHBiases, layer.GetHBiases());
}

//______________________________________________________________________________

template <typename Architecture_t>
TDAE<Architecture_t>::TDAE(const TDAE &dae)
                    : VGeneralLayer<Architecture_t>(dae),
                      fVisibleUnits(dae.GetVisibleUnits()),
                      fHiddenUnits(dae.GetHiddenUnits()),
                      fWeights(dae.GetHiddenUnits(), dae.GetVisibleUnits()),
                      fVBiases(dae.GetHiddenUnits(), 1),
                      fHBiases(dae.GetVisibleUnits(), 1)

{

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
                                      Tensor_t &corruptedInput,
                                      Scalar_t corruptionLevel)
-> void
{
   for (size_t i = 0; i < this->GetBatchSize(); i++)
   {
     Architecture_t::CorruptInput(input[i], corruptedInput[i], corruptionLevel);
   }
}

//______________________________________________________________________________

template <typename Architecture_t>
auto TDAE<Architecture_t>::Encoding(Tensor_t &input, Tensor_t &compressedInput)
-> void
{
  for (size_t i = 0; i < this->GetBatchSize(); i++)
  {
    Architecture_t::EncodeInput(input[i], compressedInput[i], this->GetWeights());
    Architecture_t::AddBiases(compressedInput[i], this->GetHBiases());
    Architecture_t::Sigmoid(compressedInput[i]);
  }

}
//______________________________________________________________________________
//
// using concept of tied weights, i.e. using same weights as associated with
// previous layer for reconstruction.
//______________________________________________________________________________

template <typename Architecture_t>
auto TDAE<Architecture_t>::Reconstruction(Tensor_t &compressedInput,
                                          Tensor_t &reconstructedInput)
-> void
{
  for (size_t i = 0; i < this->GetBatchSize(); i++)
  {
    Architecture_t::ReconstructInput(compressedInput[i],
                                   reconstructedInput[i],
                                   this->GetWeights());
    Architecture_t::AddBiases(reconstructedInput[i], this->GetVBiases());
    Architecture_t::Sigmoid(reconstructedInput[i]);
  }
}

//______________________________________________________________________________

template <typename Architecture_t>
auto TDAE<Architecture_t>::TrainLayer(Tensor_t &input, Double_t learningRate,
                                      Double_t corruptionLevel)
-> void
{

  Tensor_t corruptedInput;
  Tensor_t compressedInput;
  Tensor_t reconstructedInput;
  Matrix_t VBiasError;
  Matrix_t HBiasError;
  Double_t p = 1 - corruptionLevel;

  for (size_t i = 0; i < this->GetBatchSize(); i++)
  {
    corruptedInput.emplace_back(this->GetVisibleUnits(), 1);
    compressedInput.emplace_back(this->GetHiddenUnits(), 1)
    reconstructedInput.emplace_back(this->GetVisibleUnits(), 1);
  }


  Corruption(input,corruptedInput,p);
  Encoding(corruptedInput,compressedInput);
  Reconstruction(compressedInput,reconstructedInput);
  
  for (size_t i = 0; i < this->GetBatchSize(); i++)
  {
    Architecture_t::UpdateParams(input, corruptedInput, compressedInput,
                               reconstructedInput, this->GetVBiases(),
                               this->GetHBiases(), this->GetWeights(),
                               VBiasError, HBiasError, learningRate,
                               this->GetBatchSize());
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
#endif /* TMVA_TDAE */
