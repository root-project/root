// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TReconstructionLayer                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Reconstruction Layer for DeepAutoEncoders                                      *
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
     Reconstruction Layer for AutoEncoders.
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

   std::vector<Matrix_t> fCorruptedInput; ///< corrupted Input Units

   std::vector<Matrix_t> fInput;

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

   /*! Reconstructs the input from the compressed version of inputs.
   *  'reconstructedInput' holds this reconstructed input in the form of matrix.
   *  The reconstructed input has same dimensions as original input.
   *  Should be called after Encoding.
   */
   void Corruption(std::vector<Matrix_t> input, bool applyDropout = false);

   void Forward(std::vector<Matrix_t> input, bool applyDropout = false);

   void Backward(std::vector<Matrix_t> &compressedInput,
                 const std::vector<Matrix_t> &input);



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

   const std::vector<Matrix_t> &GetCorruptedInput() const { return fCorruptedInput; }
   std::vector<Matrix_t> &GetCorruptedInput() { return fCorruptedInput; }

   Matrix_t &GetCorruptedInputAt(size_t i) { return fCorruptedInput[i]; }
   const Matrix_t &GetCorruptedInputAt(size_t i) const { return fCorruptedInput[i]; }

   const std::vector<Matrix_t> &GetInput() const { return fInput; }
   std::vector<Matrix_t> &GetInput() { return fInput; }

   Matrix_t &GetInputAt(size_t i) { return fInput[i]; }
   const Matrix_t &GetInputAt(size_t i) const { return fInput[i]; }
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
   fType(3), fF(f), fCorruptedInput(), fCorruptionLevel(corruptionLevel),
   fDropoutProbability(dropoutProbability), fInput(),
   fLearningRate(learningRate)

{
   Architecture_t::Copy(this->GetWeightsAt(0),weights[0]);
/*
   for(size_t i=0; i<2; i++)
   {
      Architecture_t::Copy(this->GetBiasesAt(i),biases[i]);
   }

   for(size_t i=0; i<hiddenUnits; i++)
   {
      this->GetHBiases()(i,0) = this->GetBiasesAt(0)(i,0);
   }

   Architecture_t::Copy(this->GetVBiases(), this->GetBiasesAt(1));
*/
   Architecture_t::Copy(this->GetBiasesAt(0),biases[0]);

   for (size_t i = 0; i < batchSize; i++)
   {
      fCorruptedInput.emplace_back(visibleUnits, 1);
      fInput.emplace_back(visibleUnits,1);
   }
   std::cout<<"Default Constructor Reconstruction: "<<std::endl;
   std::cout<<"visible units "<<visibleUnits<<std::endl;
   std::cout<<"hidden units "<<hiddenUnits<<std::endl;
   std::cout<<"weights rows: "<<this->GetWeightsAt(0).GetNrows()<<std::endl;
   std::cout<<"weights cols: "<<this->GetWeightsAt(0).GetNcols()<<std::endl;
   std::cout<<"Bias 0 rows "<<this->GetBiasesAt(0).GetNrows()<<std::endl;
   std::cout<<"Bias 1 rows "<<this->GetBiasesAt(1).GetNrows()<<std::endl<<std::endl;

}

//______________________________________________________________________________
template <typename Architecture_t>
TReconstructionLayer<Architecture_t>::TReconstructionLayer(TReconstructionLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer),
   fVisibleUnits(layer->GetVisibleUnits()),
   fHiddenUnits(layer->GetHiddenUnits()),
   fHBiasError(layer->GetHiddenUnits(), 1),
   fVBiasError(layer->GetVisibleUnits(), 1),fType(3),
   fF(layer->GetActivationFunction()), fCorruptedInput(),
   fCorruptionLevel(layer->GetCorruptionLevel()),
   fDropoutProbability(layer->GetDropoutProbability()),
   fInput(), fLearningRate(layer->GetLearningRate())

{
   Architecture_t::Copy(this->GetWeightsAt(0),layer->weights[0]);
/*
   for(size_t i=0; i<2; i++)
   {
     Architecture_t::Copy(this->GetBiasesAt(i),layer->biases[i]);
   }

   for(size_t i=0; i<layer->GetHiddenUnits(); i++)
   {
      this->GetHBiases()(i,0) = this->GetBiasesAt(0)(i,0);
   }

   Architecture_t::Copy(this->GetVBiases(), this->GetBiasesAt(1));
*/
   Architecture_t::Copy(this->GetBiasesAt(0),layer->biases[0]);

   size_t batchSize = layer->GetBatchSize();
   for (size_t i = 0; i < batchSize ; i++)
   {
      this->GetCorruptedInput().emplace_back(layer->GetVisibleUnits(), 1);
      this->GetInput().emplace_back(layer->GetVisibleUnits(),1);
   }
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
   fCorruptedInput(),
   fCorruptionLevel(dae.fCorruptionLevel),
   fDropoutProbability(dae.fDropoutProbability),
   fInput(),
   fLearningRate(dae.fLearningRate)

{
   Architecture_t::Copy(this->GetWeightsAt(0),dae.weights[0]);
/*
   for(size_t i=0; i<2; i++)
   {
      Architecture_t::Copy(this->GetBiasesAt(i),dae.biases[i]);
   }

   for(size_t i=0; i<dae.GetHiddenUnits(); i++)
   {
      this->GetHBiases()(i,0) = this->GetBiasesAt(0)(i,0);
   }

   Architecture_t::Copy(this->GetVBiases(), this->GetBiasesAt(1));
*/
   Architecture_t::Copy(this->GetBiasesAt(0),dae.biases[0]);

   size_t batchSize = dae.GetBatchSize();
   for (size_t i = 0; i < batchSize ; i++)
   {
      this->GetCorruptedInput().emplace_back(dae.GetVisibleUnits(), 1);
      this->GetInput().emplace_back(dae.GetVisibleUnits(),1);
   }
}

//______________________________________________________________________________

template <typename Architecture_t> TReconstructionLayer<Architecture_t>::~TReconstructionLayer() {}

//______________________________________________________________________________
template <typename Architecture_t>
auto TReconstructionLayer<Architecture_t>::Corruption(std::vector<Matrix_t> input, bool applyDropout)
-> void
{
   for (size_t i = 0; i < this->GetBatchSize(); i++)
   {
      if (applyDropout && (this->GetDropoutProbability() != 1.0))
      {
         Architecture_t::Dropout(input[i], this->GetDropoutProbability());
      }
      Architecture_t::CorruptInput(input[i], this->GetCorruptedInputAt(i), 1 - this->GetCorruptionLevel());
   }
}
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
   //Corruption(input, applyDropout);
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
                                                     const std::vector<Matrix_t> &input)
-> void
{
   std::cout<<"Reconstruction BackWard starts "<<std::endl;
   for(size_t i=0; i<this->GetBatchSize(); i++)
   {
      for(size_t j=0; j<input[i].GetNrows(); j++)
      {
         for(size_t k=0; k<input[i].GetNcols(); k++)
         {
	          this->GetInputAt(i)(j,k) = input[i](j,k);
	       }
      }
   }

   Corruption(this->GetInput(), false);
   // have to pass epochs
   // check for same corruption
   std::cout<<"batch size: "<<this->GetBatchSize()<<std::endl;

   std::cout<<"output size: "<<this->GetOutput().size()<<std::endl;
   std::cout<<"output rows: "<<this->GetOutputAt(0).GetNrows()<<std::endl;
   std::cout<<"output cols: "<<this->GetOutputAt(0).GetNcols()<<std::endl;
   std::cout<<"Weights cols "<<this->GetWeightsAt(0).GetNcols()<<std::endl;
   std::cout<<"Weights rows "<<this->GetWeightsAt(0).GetNrows()<<std::endl;

   for (size_t i = 0; i < this->GetBatchSize(); i++)
   {
      Architecture_t::UpdateParams(this->GetInputAt(i), this->GetCorruptedInputAt(i),
                                   compressedInput[i],
                                   this->GetOutputAt(i),
                                   this->GetBiasesAt(1),
                                   this->GetBiasesAt(0), this->GetWeightsAt(0),
                                   this->GetVBiasError(), this->GetHBiasError(),
                                   this->GetLearningRate(), this->GetBatchSize());
   }

   std::cout<<"Reconstruction Backward ends "<<std::endl<<std::endl;
}
//______________________________________________________________________________
template<typename Architecture_t>
auto TReconstructionLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "Batch Size: " << this->GetBatchSize() << "\n"
            << "Input Units: " << this->GetVisibleUnits() << "\n"
            << "Hidden Units: " << this->GetHiddenUnits() << "\n";
   std::cout<<"Reconstructed Input "<<std::endl;
   /*for(size_t i=0; i<this->GetBatchSize(); i++)
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
   std::cout<<this->GetBatchSize()<<std::endl;
   std::cout<<this->GetWeights().size()<<std::endl;
   std::cout<<this->GetOutput().size()<<std::endl;
   std::cout<<this->GetInput().size()<<std::endl;*/

}


}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_DAE_DENOISELAYER */
