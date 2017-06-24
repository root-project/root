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


#ifndef TMVA_DAE
#define TMVA_DAE

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"
#include "ae.h"

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


  TDAE(size_t BatchSize,
       size_t VisibleUnits,
       size_t HiddenUnits);

  TDAE(const TDAE &);

  ~TDAE();


  // to corrupt input
  void inline Corruption(Matrix_t & input, Matrix_t &corruptedInput, size_t corruptionLevel);

  // to encode values
  void inline Encoding(Matrix_t &input, Matrix_t &compressedInput);

  //to reconstruct Input
  void inline Reconstruction(Matrix_t &compressedInput, Matrix_t &reconstructedInput);
  void TrainLayer(Matrix_t &input, Double_t learningRate, Double_t corruptionLevel);





};//class TDAE


//______________________________________________________________________________

template<typename Architecture_t>
TDAE<Architecture_t>::TDAE(size_t batchSize,
                           size_t visibleUnits,
                           size_t hiddenUnits)
: AELayer<Architecture_t>(batchSize,visibleUnits,hiddenUnits)

{

}

//______________________________________________________________________________

template<typename Architecture_t>
TDAE<Architecture_t>::TDAE(const TDAE &dae)
: AELayer<Architecture_t>(dae)

{
  Architecture_t::Copy(fWeights, dae.GetWeights());
  Architecture_t::Copy(fHBiases, dae.GetHBiases());
  Architecture_t::Copy(fVBiases, dae.GetVBiases());

}
//______________________________________________________________________________

template<typename Architecture_t>
TDAE<Architecture_t>::~TDAE()
{

}




//______________________________________________________________________________

template<typename Architecture_t>
TDAE<Architecture_t>::Corruption(Matrix_t &input, Matrix_t &corruptedInput, Double_t corruptionLevel)
-> void
{
      Architecture_t::CorruptInput(input, corruptedInput, corruptionLevel);
}

//______________________________________________________________________________

template<typename Architecture_t>
TDAE<Architecture_t>::Encoding(Matrix_t &input, Matrix_t &compressedInput)
-> void
{
  Architecture_t::EncodeInputs(input,compressedInput,fWeights);
  Architecture_t::AddBiases(compressedInput,fHBiases);
  Architecture_t::Sigmoid(compressedInput);

}
//______________________________________________________________________________
//______________________________________________________________________________


//using concept of tied weights, i.e. using same weights as associated with previous layer
//______________________________________________________________________________

template<typename Architecture_t>
TDAE<Architecture_t>::Reconstruction(Matrix_t &compressedInput,
                                       Matrix_t &reconstructedInput)
-> void
{
  Architecture_t::ReconstructInput(compressedInput,reconstructedInput,fWeights)
  Architecture_t::AddBiases(reconstructedInput,fVBiases);
  Architecture_t::Sigmoid(reconstructedInput);
}
//______________________________________________________________________________

template<typename Architecture_t>
TDAE<Architecture_t>::TrainLayer(Matrix_t &input, Double_t learningRate, Double_t corruptionLevel)
-> void
{
  Matrix_t corruptedInput(fVisibleUnits,1);
  Matrix_t compressedInput(fHiddenUnits,1);
  Matrix_t reconstructedInput(fVisibleUnits,1);
  Matrix_t VBiasError(fVisibleUnits,1);
  Matrix_t HBiasError(fHiddenUnits,1);
  Double_t p = 1 - corruptionLevel;

  Corruption(input,corruptedInput,p);
  Encoding(corruptedInput,compressedInput);
  Reconstruction(compressedInput,reconstructedInput);

  Architecture_t::UpdateParams(input, reconstructedInput, fVBiases, fHBiases, fWeights, learningRate,
                               corruptionLevel, fBatchSize);
}


}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_DAE */
