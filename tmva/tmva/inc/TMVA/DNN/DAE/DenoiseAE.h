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
// Contains TDAE class that represents the denoising autoencoder layer. //
//                                                                     //
/////////////////////////////////////////////////////////////////////////


#ifndef TMVA_TDAE
#define TMVA_TDAE

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
//The TDAE layer
//______________________________________________________________________________

/** \class TDAE
    Generic Denoising Layer class.
    This generic Denoising Layer class represents a layer to denoise the inputs.
    It inherits all of the properties of the generic virtual base class
    AELayer.
*/


template<typename Architecture_t>
   class TDAE : public AELayer<Architecture_t>
{

public:
  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;

  Matrix_t fWeights; ///< the weights associated

  Matrix_t fVBiases; ///< bias associated with visible layer

  Matrix_t fHBiases; ///< bias associated with hidden layer

//  size_t fBatchSize; ///< Batch size used for training and evaluation.

//  size_t fVisibleUnits; ///< number of visible units in one input set

//  size_t fHiddenUnits; ///< number of hidden units in the hidden layer of autoencoder

  TDAE(size_t BatchSize,
       size_t VisibleUnits,
       size_t HiddenUnits);

  TDAE(const TDAE &);

  ~TDAE();

  void Initialize(DNN::EInitialization m);

/*  size_t GetBatchSize()                     const {return fBatchSize;}
  size_t GetVisibleUnits()                  const {return fVisibleUnits;}
  size_t GetHiddenUnits()                   const {return fHiddenUnits;}*/

  const Matrix_t & GetWeights()             const {return fWeights;}
  Matrix_t & GetWeights()                         {return fWeights;}

  const Matrix_t & GetVBiases()             const {return fVBiases;}
  Matrix_t & GetVBiases()                         {return fVBiases;}
  const Matrix_t & GetHBiases()             const {return fHBiases;}
  Matrix_t & GetHBiases()                         {return fHBiases;}

  // This method corrupts the input. Currently it corrupts the random inputs
  // according to the corruption Level.
  void inline Corruption(Matrix_t & input, Matrix_t &corruptedInput, Scalar_t corruptionLevel);

  // This encodes the input into a compressed form.
  void inline Encoding(Matrix_t &input, Matrix_t &compressedInput);

  // This reconstructs the input from the compressed units. The reconstructed Input
  // has same dimensions as that of the input.
  void inline Reconstruction(Matrix_t &compressedInput, Matrix_t &reconstructedInput);

  // this updates the parameters after passing it to the network.
  void TrainLayer(Matrix_t &input, Double_t learningRate, Double_t corruptionLevel);





};//class TDAE


//______________________________________________________________________________

template<typename Architecture_t>
TDAE<Architecture_t>::TDAE(size_t batchSize,
                           size_t visibleUnits,
                           size_t hiddenUnits)
                   :AELayer<Architecture_t>(batchSize,visibleUnits,hiddenUnits),
                    fWeights(hiddenUnits,visibleUnits),fVBiases(visibleUnits,1),
                    fHBiases(hiddenUnits,1)

{
  std::cout<<"default constructor dae"<<std::endl;
}

//______________________________________________________________________________

template<typename Architecture_t>
TDAE<Architecture_t>::TDAE(const TDAE &dae)
                   : AELayer<Architecture_t>(dae),
                     fWeights(dae.GetHiddenUnits(),dae.GetVisibleUnits()),
                     fVBiases(dae.GetHiddenUnits(),1),
                     fHBiases(dae.GetVisibleUnits(),1)

{
  //check for weight and bias copy when called by stacked net
  //Architecture_t::Copy(fWeights, dae.GetWeights());
  //Architecture_t::Copy(fHBiases, dae.GetHBiases());
  //Architecture_t::Copy(fVBiases, dae.GetVBiases());


}
//______________________________________________________________________________
template<typename Architecture_t>
auto TDAE<Architecture_t>::Initialize(DNN::EInitialization m)
-> void
{
  DNN::initialize<Architecture_t>(fWeights,m);
  DNN::initialize<Architecture_t>(fHBiases, DNN::EInitialization::kZero);
  DNN::initialize<Architecture_t>(fVBiases, DNN::EInitialization::kZero);
  //std::cout<<"weights mat"<<std::endl;
  /*for(size_t i=0;i<(size_t)fWeights.GetNrows();i++)
  {
    for(size_t j=0;j<(size_t)fWeights.GetNcols();j++)
    {
      std::cout<<fWeights(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }*/
}
//______________________________________________________________________________

template<typename Architecture_t>
TDAE<Architecture_t>::~TDAE()
{

}


//______________________________________________________________________________


//______________________________________________________________________________

template<typename Architecture_t>
auto TDAE<Architecture_t>::Corruption(Matrix_t &input,
                                     Matrix_t &corruptedInput,
                                     Scalar_t corruptionLevel)
-> void
{
  Architecture_t::CorruptInput(input, corruptedInput, corruptionLevel);
}

//______________________________________________________________________________

template<typename Architecture_t>
auto TDAE<Architecture_t>::Encoding(Matrix_t &input, Matrix_t &compressedInput)
-> void
{
  Architecture_t::EncodeInput(input,compressedInput,this->GetWeights());
  Architecture_t::AddBiases(compressedInput,this->GetHBiases());
  Architecture_t::Sigmoid(compressedInput);

}
//______________________________________________________________________________
//______________________________________________________________________________


//using concept of tied weights, i.e. using same weights as associated with previous layer
//______________________________________________________________________________

template<typename Architecture_t>
auto TDAE<Architecture_t>::Reconstruction(Matrix_t &compressedInput,
                                         Matrix_t &reconstructedInput)
-> void
{
  Architecture_t::ReconstructInput(compressedInput,reconstructedInput,this->GetWeights());
  Architecture_t::AddBiases(reconstructedInput,this->GetVBiases());
  Architecture_t::Sigmoid(reconstructedInput);
}
//______________________________________________________________________________

template<typename Architecture_t>
auto TDAE<Architecture_t>::TrainLayer(Matrix_t &input, Double_t learningRate, Double_t corruptionLevel)
-> void
{
  Matrix_t corruptedInput(this->GetVisibleUnits(),1);
  Matrix_t compressedInput(this->GetHiddenUnits(),1);
  Matrix_t reconstructedInput(this->GetVisibleUnits(),1);
  Matrix_t VBiasError(this->GetVisibleUnits(),1);
  Matrix_t HBiasError(this->GetHiddenUnits(),1);
  Double_t p = 1 - corruptionLevel;

  Corruption(input,corruptedInput,p);
  Encoding(corruptedInput,compressedInput);
  Reconstruction(compressedInput,reconstructedInput);

  Architecture_t::UpdateParams(input,corruptedInput, compressedInput,
                               reconstructedInput, this->GetVBiases(),
                               this->GetHBiases(), this->GetWeights(),
                               VBiasError, HBiasError, learningRate,
                               this->GetBatchSize());

}


}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_TDAE */
