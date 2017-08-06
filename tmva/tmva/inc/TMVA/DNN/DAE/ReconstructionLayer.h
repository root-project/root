// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TReconstructionLayer                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Reconstruction Layer for DeepAutoEncoders                                 *
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


#ifndef TMVA_DAE_RECONSTRUCTION_LAYER
#define TMVA_DAE_RECONSTRUCTION_LAYER

#include "TMatrix.h"

#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Functions.h"

#include <iostream>
#include <vector>

namespace TMVA {
namespace DNN {
namespace DAE {

/** \class TReconstructionLayer
  *  Reconstruction Layer for AutoEncoders.
  *  This reconstructs input based on the difference between Actual and Corrupted values.
  *  It takes weights and biases from previous layers. And used concept of tied weights
  *  to update parameters.
*/

template <typename Architecture_t>
class TReconstructionLayer : public VGeneralLayer<Architecture_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;


   size_t fVisibleUnits; ///< total number of visible units

   size_t fHiddenUnits; ///< Number of compressed inputs

   Matrix_t fVBiasError; ///< Errors associated with visible Units

   Matrix_t fHBiasError; ///< Errors associated with Hidden Units

   Scalar_t fLearningRate;

   size_t fType; ///< Type of layer

   EActivationFunction fF;

   Scalar_t  fCorruptionLevel;

   Scalar_t fDropoutProbability;



   /*! Constructor. */
   TReconstructionLayer(size_t BatchSize, size_t VisibleUnits,
                        size_t HiddenUnits, Scalar_t learningRate, EActivationFunction f,
                        std::vector<Matrix_t> weights, std::vector<Matrix_t> biases,
                        Scalar_t CorruptionLevel, Scalar_t dropoutProbability);

   /*! Copy the denoise layer provided as a pointer */
   TReconstructionLayer(TReconstructionLayer<Architecture_t> *layer);

   /*! Copy constructor. */
   TReconstructionLayer(const TReconstructionLayer &);

    /*! Destructor. */
   ~TReconstructionLayer();



   void Forward(std::vector<Matrix_t> input, bool applyDropout = false);

   void Backward(std::vector<Matrix_t> &compressedInput,
                 const std::vector<Matrix_t> &input,
                 std::vector<Matrix_t> &inp1,
                 std::vector<Matrix_t> &inp2);



   void Print() const;

   /* Getters */
   //size_t GetBatchSize() const { return fBatchSize;}
   size_t GetVisibleUnits() const { return fVisibleUnits; }
   size_t GetHiddenUnits() const { return fHiddenUnits; }
   size_t GetType() const {return fType;}
   Scalar_t GetCorruptionLevel() const {return fCorruptionLevel;}
   Scalar_t GetDropoutProbability() const { return fDropoutProbability; }
   Scalar_t GetLearningRate() const {return fLearningRate; }

   EActivationFunction GetActivationFunction() const { return fF; }

   const Matrix_t &GetVBiasError() const { return fVBiasError; }
   Matrix_t &GetVBiasError() { return fVBiasError; }

   const Matrix_t &GetHBiasError() const { return fHBiasError; }
   Matrix_t &GetHBiasError() { return fHBiasError; }

};

//
//
//  Denoise Layer Class - Implementation
//______________________________________________________________________________

template <typename Architecture_t>
TReconstructionLayer<Architecture_t>::TReconstructionLayer(size_t batchSize, size_t visibleUnits,
                           size_t hiddenUnits, Scalar_t learningRate, EActivationFunction f, std::vector<Matrix_t> weights,
                           std::vector<Matrix_t> biases,
                           Scalar_t corruptionLevel, Scalar_t dropoutProbability)
   : VGeneralLayer<Architecture_t>(batchSize, 1, 1, 0, 0, 0, 0, 1, {hiddenUnits},{visibleUnits},2, {hiddenUnits,visibleUnits},
   {1,1}, batchSize, visibleUnits, 1, EInitialization::kZero),
   fVisibleUnits(visibleUnits),
   fHiddenUnits(hiddenUnits),
   fVBiasError(visibleUnits, 1),
   fHBiasError(hiddenUnits, 1),
   fType(3), fF(f), fCorruptionLevel(corruptionLevel),
   fDropoutProbability(dropoutProbability),
   fLearningRate(learningRate)

{
   Architecture_t::Copy(this->GetWeightsAt(0),weights[0]);
   Architecture_t::Copy(this->GetBiasesAt(0),biases[0]);

}

//______________________________________________________________________________
template <typename Architecture_t>
TReconstructionLayer<Architecture_t>::TReconstructionLayer(TReconstructionLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer),
   fVisibleUnits(layer->GetVisibleUnits()),
   fHiddenUnits(layer->GetHiddenUnits()),
   fHBiasError(layer->GetHiddenUnits(), 1),
   fVBiasError(layer->GetVisibleUnits(), 1),fType(3),
   fF(layer->GetActivationFunction()),
   fCorruptionLevel(layer->GetCorruptionLevel()),
   fDropoutProbability(layer->GetDropoutProbability()),
   fLearningRate(layer->GetLearningRate())

{
   Architecture_t::Copy(this->GetWeightsAt(0),layer->weights[0]);
   Architecture_t::Copy(this->GetBiasesAt(0),layer->biases[0]);
}

//______________________________________________________________________________

template <typename Architecture_t>
TReconstructionLayer<Architecture_t>::TReconstructionLayer(const TReconstructionLayer &dae)
   : VGeneralLayer<Architecture_t>(dae),
   fVisibleUnits(dae.fVisibleUnits),
   fHiddenUnits(dae.fHiddenUnits),
   fHBiasError(dae.fHiddenUnits, 1),
   fVBiasError(dae.fVisibleUnits,1),fType(3),
   fF(dae.fActivationFunction),
   fCorruptionLevel(dae.fCorruptionLevel),
   fDropoutProbability(dae.fDropoutProbability),
   fLearningRate(dae.fLearningRate)

{
   Architecture_t::Copy(this->GetWeightsAt(0),dae.weights[0]);
   Architecture_t::Copy(this->GetBiasesAt(0),dae.biases[0]);

}

//______________________________________________________________________________

template <typename Architecture_t> TReconstructionLayer<Architecture_t>::~TReconstructionLayer() {}

//______________________________________________________________________________
//
// using concept of tied weights, i.e. using same weights as associated with
// previous layer for reconstruction.
//______________________________________________________________________________
// Input here should be compressedInput
// reconstruction step
template <typename Architecture_t>
auto TReconstructionLayer<Architecture_t>::Forward(std::vector<Matrix_t> input, bool applyDropout)
-> void
{
   std::cout<<"Reconstruction Forward starts "<<std::endl;
   for (size_t i = 0; i < this->GetBatchSize(); i++)
   {
      Architecture_t::ReconstructInput(input[i],this->GetOutputAt(i),this->GetWeightsAt(0));

      Architecture_t::AddBiases(this->GetOutputAt(i), this->GetBiasesAt(1));

      evaluate<Architecture_t>(this->GetOutputAt(i), fF);

   }
   std::cout<<"Reconstruction Forward ends "<<std::endl<<std::endl;

}



//______________________________________________________________________________
template <typename Architecture_t>
auto inline TReconstructionLayer<Architecture_t>::Backward(std::vector<Matrix_t> &compressedInput,
                                                     const std::vector<Matrix_t> &gradient,
                                                     std::vector<Matrix_t> &corruptedInput,
                                                     std::vector<Matrix_t> &input)
-> void
{
   for (size_t i = 0; i < this->GetBatchSize(); i++)
   {
      Architecture_t::UpdateParams(input[i], corruptedInput[i],
                                   compressedInput[i],
                                   this->GetOutputAt(i),
                                   this->GetBiasesAt(1),
                                   this->GetBiasesAt(0), this->GetWeightsAt(0),
                                   this->GetVBiasError(), this->GetHBiasError(),
                                   this->GetLearningRate(), this->GetBatchSize());
   }

}
//______________________________________________________________________________
template<typename Architecture_t>
auto TReconstructionLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "Batch Size: " << this->GetBatchSize() << "\n"
            << "Input Units: " << this->GetVisibleUnits() << "\n"
            << "Hidden Units: " << this->GetHiddenUnits() << "\n";
   std::cout<<"Reconstructed Input: "<<std::endl;
   for(size_t i=0; i<this->GetBatchSize(); i++)
   {
      for(size_t j=0; j<this->GetOutputAt(i).GetNrows(); j++)
      {
         for(size_t k=0; k<this->GetOutputAt(i).GetNcols(); k++)
            {
               std::cout<<this->GetOutputAt(i)(j,k)<<"\t";
	    }
            std::cout<<std::endl;
      }
      std::cout<<std::endl;
   }

}


}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_DAE_DENOISELAYER */
