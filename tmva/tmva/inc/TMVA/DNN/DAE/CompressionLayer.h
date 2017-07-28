// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TCompressionLayer                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Compressed Layer for DeepAutoEncoders                                     *
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


#ifndef TMVA_DAE_COMPRESSION_LAYER
#define TMVA_DAE_COMPRESSION_LAYER

#include "TMatrix.h"

#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Functions.h"

#include <iostream>
#include <vector>

namespace TMVA {
namespace DNN {
namespace DAE {

  /** \class TCompressionLayer
    *  Used to Compress the input values.
  */

template <typename Architecture_t>
class TCompressionLayer : public VGeneralLayer<Architecture_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;


   size_t fVisibleUnits; ///< total number of visible units

   size_t fHiddenUnits;

   Scalar_t fDropoutProbability; ///< Probability that an input is active.

   size_t fType; ///< Type of layer

   EActivationFunction fF; ///< Activation function of the layer.

   /*! Constructor. */
   TCompressionLayer(size_t BatchSize, size_t VisibleUnits, size_t HiddenUnits,
                     Scalar_t DropoutProbability, EActivationFunction f,
                     std::vector<Matrix_t> Weights, std::vector<Matrix_t> Biases);

   /*! Copy the denoise layer provided as a pointer */
   TCompressionLayer(TCompressionLayer<Architecture_t> *layer);

   /*! Copy constructor. */
   TCompressionLayer(const TCompressionLayer &);

   void Forward(std::vector<Matrix_t> input, bool applyDropout = false);

   void Backward(std::vector<Matrix_t> &gradients_backward,
                 const std::vector<Matrix_t> &activations_backward);

   void Print() const;

   size_t GetVisibleUnits() const { return fVisibleUnits; }
   size_t GetHiddenUnits() const {return fHiddenUnits;}
   size_t GetType() const {return fType;}
   Scalar_t GetDropoutProbability() const { return fDropoutProbability; }
   EActivationFunction GetActivationFunction() const { return fF; }
   //  const Matrix_t & GetWeights() const {return fWeights;}
   //  Matrix_t & GetWeights() {return fWeights;}

   //  const Matrix_t & GetBiases() const {return fBiases;}
   //  Matrix_t & GetBiases() {return fBiases;}

};

//______________________________________________________________________________
template <typename Architecture_t>
TCompressionLayer<Architecture_t>::TCompressionLayer(size_t batchSize, size_t visibleUnits,
                           size_t hiddenUnits, Scalar_t dropoutProbability, EActivationFunction f,
                           std::vector<Matrix_t> weights, std::vector<Matrix_t> biases)
   : VGeneralLayer<Architecture_t>(batchSize, 1, 1, 0, 0, 0, 0, 1, hiddenUnits, visibleUnits,1,hiddenUnits,
   1, batchSize, hiddenUnits, 1, EInitialization::kZero),
   fVisibleUnits(visibleUnits), fDropoutProbability(dropoutProbability),
   fType(2), fHiddenUnits(hiddenUnits), fF(f)//,
   //fWeights(hiddenUnits,visibleUnits),  fBiases(hiddenUnits,1)

 {
   for(size_t i=0; i<1; i++){

      Architecture_t::Copy(this->GetWeightsAt(i),weights[i]);
      Architecture_t::Copy(this->GetBiasesAt(i),biases[i]);
   }
 }
//______________________________________________________________________________
template <typename Architecture_t>
TCompressionLayer<Architecture_t>::TCompressionLayer(TCompressionLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer),
   fVisibleUnits(layer->GetVisibleUnits()),fType(2),
   fDropoutProbability(layer->GetDropoutProbability()),
   fHiddenUnits(layer->GetHiddenUnits()),fF(layer->GetActivationFunction())//,
   //  fWeights(layer->GetHiddenUnits(), layer->GetVisibleUnits()),
   //  fBiases(layer->GetHiddenUnits(),1)
{
   for(size_t i=0; i<1; i++)
   {
      Architecture_t::Copy(this->GetWeightsAt(i), layer->weights[i]);
      Architecture_t::Copy(this->GetBiasesAt(i), layer->biases[i]);
   }
   // Output Tensor will be created in General Layer
}
//______________________________________________________________________________
template <typename Architecture_t>
TCompressionLayer<Architecture_t>::TCompressionLayer(const TCompressionLayer &compress)
   : VGeneralLayer<Architecture_t>(compress),
   fVisibleUnits(compress.fVisibleUnits),fType(2),
   fDropoutProbability(compress.fDropoutProbability),
   fHiddenUnits(compress.fHiddenUnits),fF(compress.fF)//,
   //    fWeights(compress.fHiddenUnits,compress.fVisibleUnits),
   //    fBiases(compress.fHiddenUnits,1)


{
   for(size_t i=0; i<1; i++)
   {
      Architecture_t::Copy(this->GetWeightsAt(i), compress.weights[i]);
      Architecture_t::Copy(this->GetBiasesAt(i), compress.biases[i]);
   }
   // Output Tensor will be created in General Layer
}
//______________________________________________________________________________
template <typename Architecture_t>
auto TCompressionLayer<Architecture_t>::Forward(std::vector<Matrix_t> input, bool applyDropout) -> void {

   for (size_t i = 0; i < this->GetBatchSize(); i++) {

      Architecture_t::EncodeInput(input[i], this->GetOutputAt(i), this->GetWeightsAt(0));
      Architecture_t::AddBiases(this->GetOutputAt(i), this->GetBiasesAt(0));
      evaluate<Architecture_t>(this->GetOutputAt(i), fF);
  }
}
//______________________________________________________________________________
template <typename Architecture_t>
auto inline TCompressionLayer<Architecture_t>::Backward(std::vector<Matrix_t> &gradients_backward,
                                                     const std::vector<Matrix_t> &activations_backward) -> void

{
}
//______________________________________________________________________________
template<typename Architecture_t>
auto TCompressionLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "Batch Size: " << this->GetBatchSize() << "\n"
               << "Input Units: " << this->GetVisibleUnits() << "\n"
               << "Hidden units " << this->GetHiddenUnits() << "\n"
               << "Weights " << "\n";

      for(size_t j=0; j<this->GetWeightsAt(0).GetNrows(); j++)
      {
         for(size_t k=0; k<this->GetWeightsAt(0).GetNcols(); k++)
         {
            std::cout<<this->GetWeightsAt(0)(j,k)<<"\t";
         }
         std::cout<<std::endl;
      }

}
//______________________________________________________________________________

}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_DAE_Compression_LAYER*/
