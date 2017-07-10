// @(#)root/tmva/tmva/dnn/dae:$Id$
// Author: Akshay Vashistha(ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd                                            *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


////////////////////////////////////////////////////////////////////////////////
// Contains AELayer class that represents the base class for denoising layer. //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#ifndef AELAYER_H_
#define AELAYER_H_

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"

#include <iostream>

namespace TMVA
{
namespace DNN
{
namespace DAE {

//______________________________________________________________________________
//
//  The AELayer Class
//______________________________________________________________________________

/** \class TLayer
    Generic layer class.
    This generic AE layer virtual class represents a base class of the layer.
    It contains its number of input units and the number of hidden units
    i.e. the compressed units.
    This allocates memory to the weight and bias matrices.
    The class also provides member functions for the initialization, and the encoding
    and decoding steps.
*/


template<typename Architecture_t>
class AELayer
{

public:
  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;

  size_t fBatchSize; ///< Batch size used for training and evaluation.

  size_t fVisibleUnits; ///< number of visible units in one input set

  size_t fHiddenUnits; ///< number of hidden units in the hidden layer of autoencoder

  Matrix_t fWeights; ///< the weights associated

  Matrix_t fVBiases; ///< bias associated with visible layer

  Matrix_t fHBiases; ///< bias associated with hidden layer

  Matrix_t fOutput; ///< outputs or activations


public:
  AELayer (size_t BatchSize,
           size_t VisibleUnits,
           size_t HiddenUnits);

  AELayer (const AELayer &);

  virtual ~AELayer();

  /*! Initialize fWeights according to the given initialization
    *  method. */
  void Initialize(EInitialization m);

  /* This takes the input and encodes it into the number of hidden units.
   * i.e the network is forced to learn a ”compressed” representation of the
   * input.  */
  virtual void Encoding(Matrix_t &input, Matrix_t &compressedInput) = 0;

  /* This reconstructs the compressed inputs to output with same dimension as
  *  that of output.
  */
  virtual void Reconstruction(Matrix_t &compressedInput,
                              Matrix_t &reconstructedInput) = 0;

  /* To print the weights and biases.
  */
  void Print() const;


  // Getter Functions

  size_t GetBatchSize()                     const {return fBatchSize;}
  size_t GetVisibleUnits()                  const {return fVisibleUnits;}
  size_t GetHiddenUnits()                   const {return fHiddenUnits;}

  const Matrix_t & GetWeights()             const {return fWeights;}
  Matrix_t & GetWeights()                         {return fWeights;}

  const Matrix_t & GetVBiases()             const {return fVBiases;}
  Matrix_t & GetVBiases()                         {return fVBiases;}

  const Matrix_t & GetHBiases()             const {return fHBiases;}
  Matrix_t & GetHBiases()                         {return fHBiases;}


};

//______________________________________________________________________________
// AELayer class implementation
//
//______________________________________________________________________________

template<typename Architecture_t>
AELayer<Architecture_t>::AELayer(size_t batchSize,
                                 size_t visibleUnits,
                                 size_t hiddenUnits)
: fBatchSize(batchSize), fVisibleUnits(visibleUnits),
  fHiddenUnits(hiddenUnits),fWeights(hiddenUnits,fVisibleUnits),
  fHBiases(hiddenUnits,1), fVBiases(visibleUnits,1)

{

}

//______________________________________________________________________________

template <typename Architecture_t>
AELayer<Architecture_t>::AELayer(const AELayer &ae)
    : fBatchSize(ae.fBatchSize), fVisibleUnits(ae.fVisibleUnits),
      fHiddenUnits(ae.fHiddenUnits),
      fWeights(ae.fHiddenUnits, ae.fVisibleUnits), fHBiases(ae.fHiddenUnits, 1),
      fVBiases(ae.fVisibleUnits, 1)

{
  Architecture_t::Copy(fWeights, ae.GetWeights());
  Architecture_t::Copy(fHBiases, ae.GetHBiases());
  Architecture_t::Copy(fVBiases, ae.GetVBiases());
}

//______________________________________________________________________________

template<typename Architecture_t>
AELayer<Architecture_t>::~AELayer()
{

}

//______________________________________________________________________________

template<typename Architecture_t>
auto AELayer<Architecture_t>::Initialize(DNN::EInitialization m)
-> void
{
  initialize<Architecture_t>(fWeights, m);
  initialize<Architecture_t>(fVBiases, EInitialization::kZero);
  initialize<Architecture_t>(fHBiases, EInitialization::kZero);

}
//______________________________________________________________________________



template<typename Architecture_t>
auto AELayer<Architecture_t>::Print() const
-> void
{
   std::cout << "\t\t\t Layer Weights: " << std::endl;
   for(size_t i = 0; i < fWeights.GetNrows(); i++)
   {
      for(size_t j = 0; j < fWeights.GetNcols(); j++)
      {
         std::cout<< fWeights(i, j) << "  ";
      }
      std::cout<<""<<std::endl;
   }

   std::cout << "Hidden Biases: " << std::endl;
   for(size_t i = 0; i < fHBiases.GetNrows(); i++)
   {
      for(size_t j = 0; j < fHBiases.GetNcols(); j++)
      {
        std::cout << fHBiases(i, j) << "  ";
      }
      std::cout<<""<<std::endl;
   }

   std::cout << "Visible Biases: " << std::endl;
   for(size_t i = 0; i < fVBiases.GetNrows(); i++)
   {
      for(size_t j = 0; j < fVBiases.GetNcols(); j++)
      {
        std::cout << fVBiases(i, j) << "  ";
      }
      std::cout<<""<<std::endl;
   }
}


}// namespace AE
}// namespace DNN
}// namespace TMVA
#endif /* AELAYER_H_ */
