// @(#)root/tmva/tmva/dnn:$Id$
// Author: Ravi Kiran S

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TAdadelta                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Adadelta Optimizer Class                                                      *
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

#ifndef TMVA_DNN_ADADELTA
#define TMVA_DNN_ADADELTA

#include "TMatrix.h"
#include "TMVA/DNN/Optimizer.h"
#include "TMVA/DNN/Functions.h"

namespace TMVA {
namespace DNN {

/** \class TAdadelta
 *  Adadelta Optimizer class
 *
 *  This class represents the Adadelta Optimizer.
 */
template <typename Architecture_t, typename Layer_t = VGeneralLayer<Architecture_t>,
          typename DeepNet_t = TDeepNet<Architecture_t, Layer_t>>
class TAdadelta : public VOptimizer<Architecture_t, Layer_t, DeepNet_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

protected:
   Scalar_t fRho;     ///< The Rho constant used by the optimizer.
   Scalar_t fEpsilon; ///< The Smoothing term used to avoid division by zero.
   std::vector<std::vector<Matrix_t>> fPastSquaredWeightGradients; ///< The accumulation of the square of the past
                                                                   /// weight gradients associated with the deep net.
   std::vector<std::vector<Matrix_t>> fPastSquaredBiasGradients;   ///< The accumulation of the square of the past bias
                                                                   /// gradients associated with the deep net.

   std::vector<std::vector<Matrix_t>> fPastSquaredWeightUpdates; ///< The accumulation of the square of the past weight
                                                                 /// updates associated with the deep net.
   std::vector<std::vector<Matrix_t>> fPastSquaredBiasUpdates;   ///< The accumulation of the square of the past bias
                                                                 /// updates associated with the deep net.

   /*! Update the weights, given the current weight gradients. */
   void UpdateWeights(size_t layerIndex, std::vector<Matrix_t> &weights, const std::vector<Matrix_t> &weightGradients);

   /*! Update the biases, given the current bias gradients. */
   void UpdateBiases(size_t layerIndex, std::vector<Matrix_t> &biases, const std::vector<Matrix_t> &biasGradients);

public:
   /*! Constructor. */
   TAdadelta(DeepNet_t &deepNet, Scalar_t learningRate = 1.0, Scalar_t rho = 0.95, Scalar_t epsilon = 1e-8);

   /*! Destructor. */
   ~TAdadelta() = default;

   /*! Getters */
   Scalar_t GetRho() const { return fRho; }
   Scalar_t GetEpsilon() const { return fEpsilon; }

   std::vector<std::vector<Matrix_t>> &GetPastSquaredWeightGradients() { return fPastSquaredWeightGradients; }
   std::vector<Matrix_t> &GetPastSquaredWeightGradientsAt(size_t i) { return fPastSquaredWeightGradients[i]; }

   std::vector<std::vector<Matrix_t>> &GetPastSquaredBiasGradients() { return fPastSquaredBiasGradients; }
   std::vector<Matrix_t> &GetPastSquaredBiasGradientsAt(size_t i) { return fPastSquaredBiasGradients[i]; }

   std::vector<std::vector<Matrix_t>> &GetPastSquaredWeightUpdates() { return fPastSquaredWeightUpdates; }
   std::vector<Matrix_t> &GetPastSquaredWeightUpdatesAt(size_t i) { return fPastSquaredWeightUpdates[i]; }

   std::vector<std::vector<Matrix_t>> &GetPastSquaredBiasUpdates() { return fPastSquaredBiasUpdates; }
   std::vector<Matrix_t> &GetPastSquaredBiasUpdatesAt(size_t i) { return fPastSquaredBiasUpdates[i]; }
};

//
//
//  The Adadelta Optimizer Class - Implementation
//_________________________________________________________________________________________________
template <typename Architecture_t, typename Layer_t, typename DeepNet_t>
TAdadelta<Architecture_t, Layer_t, DeepNet_t>::TAdadelta(DeepNet_t &deepNet, Scalar_t learningRate, Scalar_t rho,
                                                         Scalar_t epsilon)
   : VOptimizer<Architecture_t, Layer_t, DeepNet_t>(learningRate, deepNet), fRho(rho), fEpsilon(epsilon)
{
   std::vector<Layer_t *> &layers = deepNet.GetLayers();
   const size_t layersNSlices = layers.size();
   fPastSquaredWeightGradients.resize(layersNSlices);
   fPastSquaredBiasGradients.resize(layersNSlices);
   fPastSquaredWeightUpdates.resize(layersNSlices);
   fPastSquaredBiasUpdates.resize(layersNSlices);

   for (size_t i = 0; i < layersNSlices; i++) {
      const size_t weightsNSlices = (layers[i]->GetWeights()).size();

      Architecture_t::CreateWeightTensors( fPastSquaredWeightGradients[i], layers[i]->GetWeights()); 
      Architecture_t::CreateWeightTensors( fPastSquaredWeightUpdates[i], layers[i]->GetWeights()); 

      for (size_t j = 0; j < weightsNSlices; j++) {
         initialize<Architecture_t>(fPastSquaredWeightGradients[i][j], EInitialization::kZero);
         initialize<Architecture_t>(fPastSquaredWeightUpdates[i][j], EInitialization::kZero);
      }

      const size_t biasesNSlices = (layers[i]->GetBiases()).size();

      Architecture_t::CreateWeightTensors( fPastSquaredBiasGradients[i], layers[i]->GetBiases()); 
      Architecture_t::CreateWeightTensors( fPastSquaredBiasUpdates[i], layers[i]->GetBiases()); 

      for (size_t j = 0; j < biasesNSlices; j++) {
         initialize<Architecture_t>(fPastSquaredBiasGradients[i][j], EInitialization::kZero);
         initialize<Architecture_t>(fPastSquaredBiasUpdates[i][j], EInitialization::kZero);
      }
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t, typename Layer_t, typename DeepNet_t>
auto TAdadelta<Architecture_t, Layer_t, DeepNet_t>::UpdateWeights(size_t layerIndex, std::vector<Matrix_t> &weights,
                                                                  const std::vector<Matrix_t> &weightGradients) -> void
{
   std::vector<Matrix_t> &currentLayerPastSquaredWeightGradients = this->GetPastSquaredWeightGradientsAt(layerIndex);
   std::vector<Matrix_t> &currentLayerPastSquaredWeightUpdates = this->GetPastSquaredWeightUpdatesAt(layerIndex);

   for (size_t k = 0; k < currentLayerPastSquaredWeightGradients.size(); k++) {

      // accumulation matrix used for temporary storing of the current accumulation
      Matrix_t accumulation(currentLayerPastSquaredWeightGradients[k].GetNrows(),
                            currentLayerPastSquaredWeightGradients[k].GetNcols());

      // Vt = rho * Vt-1 + (1-rho) * currentSquaredWeightGradients
      initialize<Architecture_t>(accumulation, EInitialization::kZero);
      Matrix_t currentSquaredWeightGradients(weightGradients[k].GetNrows(), weightGradients[k].GetNcols());
      Architecture_t::Copy(currentSquaredWeightGradients, weightGradients[k]);
      Architecture_t::SquareElementWise(currentSquaredWeightGradients);
      Architecture_t::ScaleAdd(accumulation, currentLayerPastSquaredWeightGradients[k], this->GetRho());
      Architecture_t::ScaleAdd(accumulation, currentSquaredWeightGradients, 1 - (this->GetRho()));
      Architecture_t::Copy(currentLayerPastSquaredWeightGradients[k], accumulation);
   }

   // updating the weights.
   for (size_t i = 0; i < weights.size(); i++) {

      // currentWeightUpdates = sqrt(Wt + epsilon) * currentGradients / sqrt(Vt + epsilon)

      // dummy1 = sqrt(Wt + epsilon)
      Matrix_t dummy1(currentLayerPastSquaredWeightUpdates[i].GetNrows(),
                      currentLayerPastSquaredWeightUpdates[i].GetNcols());
      Architecture_t::Copy(dummy1, currentLayerPastSquaredWeightUpdates[i]);
      Architecture_t::ConstAdd(dummy1, this->GetEpsilon());
      Architecture_t::SqrtElementWise(dummy1);

      Matrix_t currentWeightUpdates(currentLayerPastSquaredWeightGradients[i].GetNrows(),
                                    currentLayerPastSquaredWeightGradients[i].GetNcols());
      Architecture_t::Copy(currentWeightUpdates, currentLayerPastSquaredWeightGradients[i]);
      Architecture_t::ConstAdd(currentWeightUpdates, this->GetEpsilon());
      Architecture_t::SqrtElementWise(currentWeightUpdates);
      Architecture_t::ReciprocalElementWise(currentWeightUpdates);
      Architecture_t::Hadamard(currentWeightUpdates, weightGradients[i]);
      Architecture_t::Hadamard(currentWeightUpdates, dummy1);

      // theta = theta - learningRate * currentWeightUpdates
      Architecture_t::ScaleAdd(weights[i], currentWeightUpdates, -this->GetLearningRate());

      // accumulation matrix used for temporary storing of the current accumulation
      Matrix_t accumulation(currentLayerPastSquaredWeightUpdates[i].GetNrows(),
                            currentLayerPastSquaredWeightUpdates[i].GetNcols());

      // Wt = rho * Wt-1 + (1-rho) * currentSquaredWeightUpdates
      initialize<Architecture_t>(accumulation, EInitialization::kZero);
      Matrix_t currentSquaredWeightUpdates(currentWeightUpdates.GetNrows(), currentWeightUpdates.GetNcols());
      Architecture_t::Copy(currentSquaredWeightUpdates, currentWeightUpdates);
      Architecture_t::SquareElementWise(currentSquaredWeightUpdates);
      Architecture_t::ScaleAdd(accumulation, currentLayerPastSquaredWeightUpdates[i], this->GetRho());
      Architecture_t::ScaleAdd(accumulation, currentSquaredWeightUpdates, 1 - (this->GetRho()));
      Architecture_t::Copy(currentLayerPastSquaredWeightUpdates[i], accumulation);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t, typename Layer_t, typename DeepNet_t>
auto TAdadelta<Architecture_t, Layer_t, DeepNet_t>::UpdateBiases(size_t layerIndex, std::vector<Matrix_t> &biases,
                                                                 const std::vector<Matrix_t> &biasGradients) -> void
{
   std::vector<Matrix_t> &currentLayerPastSquaredBiasGradients = this->GetPastSquaredBiasGradientsAt(layerIndex);
   std::vector<Matrix_t> &currentLayerPastSquaredBiasUpdates = this->GetPastSquaredBiasUpdatesAt(layerIndex);

   for (size_t k = 0; k < currentLayerPastSquaredBiasGradients.size(); k++) {

      // accumulation matrix used for temporary storing of the current accumulation
      Matrix_t accumulation(currentLayerPastSquaredBiasGradients[k].GetNrows(),
                            currentLayerPastSquaredBiasGradients[k].GetNcols());

      // Vt = rho * Vt-1 + (1-rho) * currentSquaredBiasGradients
      initialize<Architecture_t>(accumulation, EInitialization::kZero);
      Matrix_t currentSquaredBiasGradients(biasGradients[k].GetNrows(), biasGradients[k].GetNcols());
      Architecture_t::Copy(currentSquaredBiasGradients, biasGradients[k]);
      Architecture_t::SquareElementWise(currentSquaredBiasGradients);
      Architecture_t::ScaleAdd(accumulation, currentLayerPastSquaredBiasGradients[k], this->GetRho());
      Architecture_t::ScaleAdd(accumulation, currentSquaredBiasGradients, 1 - (this->GetRho()));
      Architecture_t::Copy(currentLayerPastSquaredBiasGradients[k], accumulation);
   }

   // updating the biases.
   for (size_t i = 0; i < biases.size(); i++) {

      // currentBiasUpdates = sqrt(Wt + epsilon) * currentGradients / sqrt(Vt + epsilon)

      // dummy1 = sqrt(Wt + epsilon)
      Matrix_t dummy1(currentLayerPastSquaredBiasUpdates[i].GetNrows(),
                      currentLayerPastSquaredBiasUpdates[i].GetNcols());
      Architecture_t::Copy(dummy1, currentLayerPastSquaredBiasUpdates[i]);
      Architecture_t::ConstAdd(dummy1, this->GetEpsilon());
      Architecture_t::SqrtElementWise(dummy1);

      Matrix_t currentBiasUpdates(currentLayerPastSquaredBiasGradients[i].GetNrows(),
                                  currentLayerPastSquaredBiasGradients[i].GetNcols());
      Architecture_t::Copy(currentBiasUpdates, currentLayerPastSquaredBiasGradients[i]);
      Architecture_t::ConstAdd(currentBiasUpdates, this->GetEpsilon());
      Architecture_t::SqrtElementWise(currentBiasUpdates);
      Architecture_t::ReciprocalElementWise(currentBiasUpdates);
      Architecture_t::Hadamard(currentBiasUpdates, biasGradients[i]);
      Architecture_t::Hadamard(currentBiasUpdates, dummy1);

      // theta = theta - learningRate * currentBiasUpdates
      Architecture_t::ScaleAdd(biases[i], currentBiasUpdates, -this->GetLearningRate());

      // accumulation matrix used for temporary storing of the current accumulation
      Matrix_t accumulation(currentLayerPastSquaredBiasUpdates[i].GetNrows(),
                            currentLayerPastSquaredBiasUpdates[i].GetNcols());

      // Wt = rho * Wt-1 + (1-rho) * currentSquaredBiasUpdates
      initialize<Architecture_t>(accumulation, EInitialization::kZero);
      Matrix_t currentSquaredBiasUpdates(currentBiasUpdates.GetNrows(), currentBiasUpdates.GetNcols());
      Architecture_t::Copy(currentSquaredBiasUpdates, currentBiasUpdates);
      Architecture_t::SquareElementWise(currentSquaredBiasUpdates);
      Architecture_t::ScaleAdd(accumulation, currentLayerPastSquaredBiasUpdates[i], this->GetRho());
      Architecture_t::ScaleAdd(accumulation, currentSquaredBiasUpdates, 1 - (this->GetRho()));
      Architecture_t::Copy(currentLayerPastSquaredBiasUpdates[i], accumulation);
   }
}

} // namespace DNN
} // namespace TMVA

#endif