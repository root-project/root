// @(#)root/tmva/tmva/dnn:$Id$
// Author: Ravi Kiran S

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TAdam                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Adam Optimizer Class                                                      *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Ravi Kiran S      <sravikiran0606@gmail.com>  - CERN, Switzerland         *
 *                                                                                *
 * Copyright (c) 2005-2018:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_DNN_ADAM
#define TMVA_DNN_ADAM

#include "TMatrix.h"
#include "TMVA/DNN/Optimizer.h"
#include "TMVA/DNN/Functions.h"

namespace TMVA {
namespace DNN {

/** \class TAdam
 *  Adam Optimizer class
 *
 *  This class represents the Adam Optimizer.
 */
template <typename Architecture_t, typename Layer_t = VGeneralLayer<Architecture_t>,
          typename DeepNet_t = TDeepNet<Architecture_t, Layer_t>>
class TAdam : public VOptimizer<Architecture_t, Layer_t, DeepNet_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

protected:
   Scalar_t fBeta1;   ///< The Beta1 constant used by the optimizer.
   Scalar_t fBeta2;   ///< The Beta2 constant used by the optimizer.
   Scalar_t fEpsilon; ///< The Smoothing term used to avoid division by zero.

   std::vector<std::vector<Matrix_t>> fFirstMomentWeights; ///< The decaying average of the first moment of the past
                                                           /// weight gradients associated with the deep net.
   std::vector<std::vector<Matrix_t>> fFirstMomentBiases; ///< The decaying average of the first moment of the past bias
                                                          /// gradients associated with the deep net.

   std::vector<std::vector<Matrix_t>> fSecondMomentWeights; ///< The decaying average of the second moment of the past
                                                            /// weight gradients associated with the deep net.
   std::vector<std::vector<Matrix_t>> fSecondMomentBiases;  ///< The decaying average of the second moment of the past
                                                            /// bias gradients associated with the deep net.

   /*! Update the weights, given the current weight gradients. */
   void UpdateWeights(size_t layerIndex, std::vector<Matrix_t> &weights, const std::vector<Matrix_t> &weightGradients);

   /*! Update the biases, given the current bias gradients. */
   void UpdateBiases(size_t layerIndex, std::vector<Matrix_t> &biases, const std::vector<Matrix_t> &biasGradients);

public:
   /*! Constructor. */
   TAdam(DeepNet_t &deepNet, Scalar_t learningRate = 0.001, Scalar_t beta1 = 0.9, Scalar_t beta2 = 0.999,
         Scalar_t epsilon = 1e-8);

   /*! Destructor. */
   ~TAdam() = default;

   /*! Getters */
   Scalar_t GetBeta1() const { return fBeta1; }
   Scalar_t GetBeta2() const { return fBeta2; }
   Scalar_t GetEpsilon() const { return fEpsilon; }

   std::vector<std::vector<Matrix_t>> &GetFirstMomentWeights() { return fFirstMomentWeights; }
   std::vector<Matrix_t> &GetFirstMomentWeightsAt(size_t i) { return fFirstMomentWeights[i]; }

   std::vector<std::vector<Matrix_t>> &GetFirstMomentBiases() { return fFirstMomentBiases; }
   std::vector<Matrix_t> &GetFirstMomentBiasesAt(size_t i) { return fFirstMomentBiases[i]; }

   std::vector<std::vector<Matrix_t>> &GetSecondMomentWeights() { return fSecondMomentWeights; }
   std::vector<Matrix_t> &GetSecondMomentWeightsAt(size_t i) { return fSecondMomentWeights[i]; }

   std::vector<std::vector<Matrix_t>> &GetSecondMomentBiases() { return fSecondMomentBiases; }
   std::vector<Matrix_t> &GetSecondMomentBiasesAt(size_t i) { return fSecondMomentBiases[i]; }
};

//
//
//  The Adam Optimizer Class - Implementation
//_________________________________________________________________________________________________
template <typename Architecture_t, typename Layer_t, typename DeepNet_t>
TAdam<Architecture_t, Layer_t, DeepNet_t>::TAdam(DeepNet_t &deepNet, Scalar_t learningRate, Scalar_t beta1,
                                                 Scalar_t beta2, Scalar_t epsilon)
   : VOptimizer<Architecture_t, Layer_t, DeepNet_t>(learningRate, deepNet), fBeta1(beta1), fBeta2(beta2),
     fEpsilon(epsilon)
{
   std::vector<Layer_t *> &layers = deepNet.GetLayers();
   const size_t layersNSlices = layers.size();
   fFirstMomentWeights.resize(layersNSlices);
   fFirstMomentBiases.resize(layersNSlices);
   fSecondMomentWeights.resize(layersNSlices);
   fSecondMomentBiases.resize(layersNSlices);

   for (size_t i = 0; i < layersNSlices; i++) {
      const size_t weightsNSlices = (layers[i]->GetWeights()).size();

      for (size_t j = 0; j < weightsNSlices; j++) {
         Matrix_t &currentWeights = layers[i]->GetWeightsAt(j);
         const size_t weightsNRows = currentWeights.GetNrows();
         const size_t weightsNCols = currentWeights.GetNcols();

         fFirstMomentWeights[i].emplace_back(weightsNRows, weightsNCols);
         fSecondMomentWeights[i].emplace_back(weightsNRows, weightsNCols);
         initialize<Architecture_t>(fFirstMomentWeights[i][j], EInitialization::kZero);
         initialize<Architecture_t>(fSecondMomentWeights[i][j], EInitialization::kZero);
      }

      const size_t biasesNSlices = (layers[i]->GetBiases()).size();

      for (size_t j = 0; j < biasesNSlices; j++) {
         Matrix_t &currentBiases = layers[i]->GetBiasesAt(j);
         const size_t biasesNRows = currentBiases.GetNrows();
         const size_t biasesNCols = currentBiases.GetNcols();

         fFirstMomentBiases[i].emplace_back(biasesNRows, biasesNCols);
         fSecondMomentBiases[i].emplace_back(biasesNRows, biasesNCols);
         initialize<Architecture_t>(fFirstMomentBiases[i][j], EInitialization::kZero);
         initialize<Architecture_t>(fSecondMomentBiases[i][j], EInitialization::kZero);
      }
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t, typename Layer_t, typename DeepNet_t>
auto TAdam<Architecture_t, Layer_t, DeepNet_t>::UpdateWeights(size_t layerIndex, std::vector<Matrix_t> &weights,
                                                              const std::vector<Matrix_t> &weightGradients) -> void
{
   std::vector<Matrix_t> &currentLayerFirstMomentWeights = this->GetFirstMomentWeightsAt(layerIndex);
   std::vector<Matrix_t> &currentLayerSecondMomentWeights = this->GetSecondMomentWeightsAt(layerIndex);

   for (size_t k = 0; k < currentLayerFirstMomentWeights.size(); k++) {

      // accumulation matrix used for temporary storing of the current accumulation
      Matrix_t accumulation(currentLayerFirstMomentWeights[k].GetNrows(), currentLayerFirstMomentWeights[k].GetNcols());

      // Mt = beta1 * Mt-1 + (1-beta1) * currentWeightGradients
      initialize<Architecture_t>(accumulation, EInitialization::kZero);
      Architecture_t::ScaleAdd(accumulation, currentLayerFirstMomentWeights[k], this->GetBeta1());
      Architecture_t::ScaleAdd(accumulation, weightGradients[k], 1 - (this->GetBeta1()));
      Architecture_t::Copy(currentLayerFirstMomentWeights[k], accumulation);

      // Vt = beta2 * Vt-1 + (1-beta2) * currentSquaredWeightGradients
      initialize<Architecture_t>(accumulation, EInitialization::kZero);
      Matrix_t currentSquaredWeightGradients(weightGradients[k].GetNrows(), weightGradients[k].GetNcols());
      Architecture_t::Copy(currentSquaredWeightGradients, weightGradients[k]);
      Architecture_t::SquareElementWise(currentSquaredWeightGradients);
      Architecture_t::ScaleAdd(accumulation, currentLayerSecondMomentWeights[k], this->GetBeta2());
      Architecture_t::ScaleAdd(accumulation, currentSquaredWeightGradients, 1 - (this->GetBeta2()));
      Architecture_t::Copy(currentLayerSecondMomentWeights[k], accumulation);
   }

   // alpha = learningRate * sqrt(1 - beta2^t) / (1-beta1^t)
   Scalar_t alpha = (this->GetLearningRate()) * (sqrt(1 - pow(this->GetBeta2(), this->GetGlobalStep()))) /
                    (1 - pow(this->GetBeta1(), this->GetGlobalStep()));

   // updating the weights.
   // theta = theta - alpha * Mt / (sqrt(Vt) + epsilon)
   for (size_t i = 0; i < weights.size(); i++) {
      Matrix_t currentWeightUpdates(weights[i].GetNrows(), weights[i].GetNcols());
      Architecture_t::Copy(currentWeightUpdates, currentLayerSecondMomentWeights[i]);
      Architecture_t::SqrtElementWise(currentWeightUpdates);
      Architecture_t::ConstAdd(currentWeightUpdates, this->GetEpsilon());
      Architecture_t::ReciprocalElementWise(currentWeightUpdates);
      Architecture_t::Hadamard(currentWeightUpdates, currentLayerFirstMomentWeights[i]);
      Architecture_t::ScaleAdd(weights[i], currentWeightUpdates, -alpha);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t, typename Layer_t, typename DeepNet_t>
auto TAdam<Architecture_t, Layer_t, DeepNet_t>::UpdateBiases(size_t layerIndex, std::vector<Matrix_t> &biases,
                                                             const std::vector<Matrix_t> &biasGradients) -> void
{
   std::vector<Matrix_t> &currentLayerFirstMomentBiases = this->GetFirstMomentBiasesAt(layerIndex);
   std::vector<Matrix_t> &currentLayerSecondMomentBiases = this->GetSecondMomentBiasesAt(layerIndex);

   for (size_t k = 0; k < currentLayerFirstMomentBiases.size(); k++) {

      // accumulation matrix used for temporary storing of the current accumulation
      Matrix_t accumulation(currentLayerFirstMomentBiases[k].GetNrows(), currentLayerFirstMomentBiases[k].GetNcols());

      // Mt = beta1 * Mt-1 + (1-beta1) * currentBiasGradients
      initialize<Architecture_t>(accumulation, EInitialization::kZero);
      Architecture_t::ScaleAdd(accumulation, currentLayerFirstMomentBiases[k], this->GetBeta1());
      Architecture_t::ScaleAdd(accumulation, biasGradients[k], 1 - (this->GetBeta1()));
      Architecture_t::Copy(currentLayerFirstMomentBiases[k], accumulation);

      // Vt = beta2 * Vt-1 + (1-beta2) * currentSquaredBiasGradients
      initialize<Architecture_t>(accumulation, EInitialization::kZero);
      Matrix_t currentSquaredBiasGradients(biasGradients[k].GetNrows(), biasGradients[k].GetNcols());
      Architecture_t::Copy(currentSquaredBiasGradients, biasGradients[k]);
      Architecture_t::SquareElementWise(currentSquaredBiasGradients);
      Architecture_t::ScaleAdd(accumulation, currentLayerSecondMomentBiases[k], this->GetBeta2());
      Architecture_t::ScaleAdd(accumulation, currentSquaredBiasGradients, 1 - (this->GetBeta2()));
      Architecture_t::Copy(currentLayerSecondMomentBiases[k], accumulation);
   }

   // alpha = learningRate * sqrt(1 - beta2^t) / (1-beta1^t)
   Scalar_t alpha = (this->GetLearningRate()) * (sqrt(1 - pow(this->GetBeta2(), this->GetGlobalStep()))) /
                    (1 - pow(this->GetBeta1(), this->GetGlobalStep()));

   // updating the biases.
   // theta = theta - alpha * Mt / (sqrt(Vt) + epsilon)
   for (size_t i = 0; i < biases.size(); i++) {
      Matrix_t currentBiasUpdates(biases[i].GetNrows(), biases[i].GetNcols());
      Architecture_t::Copy(currentBiasUpdates, currentLayerSecondMomentBiases[i]);
      Architecture_t::SqrtElementWise(currentBiasUpdates);
      Architecture_t::ConstAdd(currentBiasUpdates, this->GetEpsilon());
      Architecture_t::ReciprocalElementWise(currentBiasUpdates);
      Architecture_t::Hadamard(currentBiasUpdates, currentLayerFirstMomentBiases[i]);
      Architecture_t::ScaleAdd(biases[i], currentBiasUpdates, -alpha);
   }
}

} // namespace DNN
} // namespace TMVA

#endif