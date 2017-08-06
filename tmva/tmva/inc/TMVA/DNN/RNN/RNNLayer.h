// @(#)root/tmva/tmva/dnn/rnn:$Id$
// Author: Saurav Shekhar 19/07/17

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class : RNNLayer                                                               *
 *                                                                                *
 * Description:                                                                   *
 *       NeuralNetwork                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *       Saurav Shekhar    <sauravshekhar01@gmail.com> - ETH Zurich, Switzerland  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 * All rights reserved.                                                           *
 *       CERN, Switzerland                                                        *
 *                                                                                *
 * For the licensing terms see $ROOTSYS/LICENSE.                                  *
 * For the list of contributors see $ROOTSYS/README/CREDITS.                      *
 **********************************************************************************/

//#pragma once

//////////////////////////////////////////////////////////////////////
// <Description> //
//////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_RNN_LAYER
#define TMVA_DNN_RNN_LAYER

#include <cmath>
#include <iostream>
#include <vector>

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"

namespace TMVA
{
namespace DNN
{
namespace RNN
{

//______________________________________________________________________________
//
// Basic RNN Layer
//______________________________________________________________________________

/** \class BasicRNNLayer
      Generic implementation
*/
template<typename Architecture_t>
      class TBasicRNNLayer : public VGeneralLayer<Architecture_t>
{

public:

   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;
   using Tensor_t = std::vector<Matrix_t>;

private:

   /* from GeneralLayer:
    * fBatchSize
    * fInputDepth = 1
    * fInputHeight = 1
    * fInputWidth = inputSize
    * fOutputDepth = 1
    * fOutputHeight = 1
    * fOutputWidth = stateSize
    * fOutput = timeSteps x batchSize x stateSize */

   size_t fTimeSteps;              ///< Timesteps for RNN
   size_t fStateSize;              ///< Hidden state size of RNN
   bool   fRememberState;          ///< Remember state in next pass

   DNN::EActivationFunction fF;  ///< Activation function of the hidden state

   Matrix_t fState;                ///< Hidden State
   Matrix_t &fWeightsInput;         ///< Input weights, fWeights[0]
   Matrix_t &fWeightsState;         ///< Prev state weights, fWeights[1]
   Matrix_t &fBiases;               ///< Biases

   Matrix_t fDerivatives;          ///< First fDerivatives of the activations
   Matrix_t &fWeightInputGradients; ///< Gradients w.r.t. the input weights
   Matrix_t &fWeightStateGradients; ///< Gradients w.r.t. the recurring weights
   Matrix_t &fBiasGradients;        ///< Gradients w.r.t. the bias values

public:

   /** Constructor */
   TBasicRNNLayer(size_t batchSize, size_t stateSize, size_t inputSize,
                  size_t timeSteps, bool rememberState = false,
                  DNN::EActivationFunction f = DNN::EActivationFunction::kTanh,
                  bool training = true);

   /** Copy Constructor */
   TBasicRNNLayer(const TBasicRNNLayer &);

   /*! Initialize the weights according to the given initialization
    **  method. */
   void Initialize(DNN::EInitialization m);

   /*! Initialize the state
    **  method. */
   void InitState(DNN::EInitialization m = DNN::EInitialization::kZero);

   /*! Compute and return the next state with given input
   *  matrix */
   void Forward(Tensor_t input, bool isTraining = true);

   /*! Forward for a single cell (time unit) */
   void CellForward(Matrix_t &input);

   /*! Backpropagates the error. Must only be called directly at the corresponding
    *  call to Forward(...). */
   void Backward(Tensor_t &gradients_backward,
                 const Tensor_t &activations_backward,
                 std::vector<Matrix_t> &inp1,
                 std::vector<Matrix_t> &inp2);

   /* Updates weights and biases, given the learning rate */
   void Update(const Scalar_t learningRate);

   /*! Backward for a single time unit
    * a the corresponding call to Forward(...). */
   inline Matrix_t & CellBackward(Matrix_t & state_gradients_backward,
                              const Matrix_t & precStateActivations, const Matrix_t & currStateActivations,
                              const Matrix_t & input, Matrix_t & input_gradient);

   /*! Return a vector of all learnable weights */
   //std::vector<Matrix_t*> GetWeights() const;

   ///*! Return a vector of all learnable weights' gradients */
   //std::vector<Matrix_t*> GetWeightGradients() const;

   /*! Return a vector of all learnable biases */
   //std::vector<Matrix_t*> GetBiases();

   /*! Return a vector of all learnable bias' gradients */
   //std::vector<Matrix_t*> GetBiasGradients();

   /** Prints the info about the layer */
   void Print() const;

   /** Getters */
   const size_t GetTimeSteps()   const {return fTimeSteps;}
   const size_t GetStateSize()   const {return fStateSize;}
   const size_t GetInputSize()   const {return this->GetInputWidth();}
   inline bool IsRememberState()  const {return fRememberState;}
   inline DNN::EActivationFunction GetActivationFunction()  const {return fF;}
   Matrix_t        & GetState()            {return fState;}
   const Matrix_t & GetState()       const  {return fState;}
   Matrix_t        & GetWeightsInput()        {return fWeightsInput;}
   const Matrix_t & GetWeightsInput()   const {return fWeightsInput;}
   Matrix_t        & GetWeightsState()        {return fWeightsState;}
   const Matrix_t & GetWeightsState()   const {return fWeightsState;}
   Matrix_t        & GetBiases()              {return fBiases;}
   const Matrix_t & GetBiases()         const {return fBiases;}
   Matrix_t        & GetBiasGradients()            {return fBiasGradients;}
   const Matrix_t & GetBiasGradients() const {return fBiasGradients;}
   Matrix_t        & GetWeightInputGradients()         {return fWeightInputGradients;}
   const Matrix_t & GetWeightInputGradients()    const {return fWeightInputGradients;}
   Matrix_t        & GetWeightStateGradients()         {return fWeightStateGradients;}
   const Matrix_t & GetWeightStateGradients()    const {return fWeightStateGradients;}
};

//______________________________________________________________________________
//
// BasicRNNLayer Implementation
//______________________________________________________________________________

template<typename Architecture_t>
TBasicRNNLayer<Architecture_t>::TBasicRNNLayer(size_t batchSize, size_t stateSize, size_t inputSize,
                                              size_t timeSteps, bool rememberState,
                                              DNN::EActivationFunction f,
                                              bool training)
   : VGeneralLayer<Architecture_t>(batchSize, 1, 1, inputSize, 1, 1, stateSize, 2, {stateSize, stateSize}, {inputSize, stateSize},
   1, {stateSize}, {1}, timeSteps, batchSize, stateSize, DNN::EInitialization::kZero),
   fTimeSteps(timeSteps), fStateSize(stateSize), fRememberState(rememberState), fWeightsInput(this->GetWeightsAt(0)), fF(f),
   fState(batchSize, stateSize), fWeightsState(this->GetWeightsAt(1)), fBiases(this->GetBiasesAt(0)), fDerivatives(batchSize, stateSize),
   fWeightInputGradients(this->GetWeightGradientsAt(0)), fWeightStateGradients(this->GetWeightGradientsAt(1)), fBiasGradients(this->GetBiasGradientsAt(0))
{
   // Nothing
}

//______________________________________________________________________________
template <typename Architecture_t>
TBasicRNNLayer<Architecture_t>::TBasicRNNLayer(const TBasicRNNLayer &layer)
   : VGeneralLayer<Architecture_t>(layer), fTimeSteps(layer.fTimeSteps), fStateSize(layer.fStateSize),
   fRememberState(layer.fRememberState), fWeightsInput(this->GetWeightsAt(0)),
   fState(layer.GetBatchSize(), layer.GetStateSize()), fWeightsState(this->GetWeightsAt(1)),
   fBiases(this->GetBiasesAt(0)), fDerivatives(layer.GetBatchSize(), layer.GetStateSize()),
   fWeightInputGradients(this->GetWeightGradientsAt(0)), fF(layer.GetActivationFunction()),
   fWeightStateGradients(this->GetWeightGradientsAt(1)), fBiasGradients(this->GetBiasGradientsAt(0))
{
   // Gradient matrices not copied
   Architecture_t::Copy(fState, layer.GetState());
   Architecture_t::Copy(fDerivatives, layer.GetDerivatives());
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TBasicRNNLayer<Architecture_t>::Initialize(DNN::EInitialization m)
-> void
{
   DNN::initialize<Architecture_t>(fWeightsInput, m);
   DNN::initialize<Architecture_t>(fWeightsState, m);
   DNN::initialize<Architecture_t>(fBiases,  DNN::EInitialization::kZero);
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TBasicRNNLayer<Architecture_t>::InitState(DNN::EInitialization m)
-> void
{
   DNN::initialize<Architecture_t>(this->GetState(),  DNN::EInitialization::kZero);
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TBasicRNNLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "Batch Size: " << this->GetBatchSize() << "\n"
             << "Input Size: " << this->GetInputSize() << "\n"
             << "Hidden State Size: " << this->GetStateSize() << "\n";
}

////______________________________________________________________________________
//template<typename Architecture_t>
//auto TBasicRNNLayer<Architecture_t>::GetWeights() const
//->   std::vector<Matrix_t*>
//{
//  std::vector<Matrix_t*> weights;
//  weights.emplace_back(&fWeightsInput);
//  weights.emplace_back(&fWeightsState);
//  return weights;
//}
//
////______________________________________________________________________________
//template<typename Architecture_t>
//auto TBasicRNNLayer<Architecture_t>::GetWeightGradients() const
//->   std::vector<Matrix_t*>
//{
//  std::vector<Matrix_t*> weightGradients;
//  weightGradients.emplace_back(&fWeightInputGradients);
//  weightGradients.emplace_back(&fWeightStateGradients);
//  return weightGradients;
//}
//
//______________________________________________________________________________
//template<typename Architecture_t>
//auto TBasicRNNLayer<Architecture_t>::GetBiases() const
//->   std::vector<Matrix_t*>
//{
//  std::vector<Matrix_t*> biases;
//  biases.emplace_back(&fBiases);
//  return biases;
//}

//______________________________________________________________________________
//template<typename Architecture_t>
//auto TBasicRNNLayer<Architecture_t>::GetBiasGradients() const
//->   std::vector<Matrix_t*>
//{
//  std::vector<Matrix_t*> biasGradients;
//  biasGradients.emplace_back(&fBiasGradients);
//  return biasGradients;
//}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicRNNLayer<Architecture_t>::Forward(Tensor_t input, bool isTraining)
-> void
{
   if (!this->fRememberState) InitState(DNN::EInitialization::kZero);
   for (size_t t = 0; t < fTimeSteps; ++t) {
      CellForward(input[t]);
      Architecture_t::Copy(this->GetOutputAt(t), fState);
   }
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicRNNLayer<Architecture_t>::CellForward(Matrix_t &input)
-> void
{
   // State = act(W_input . input + W_state . state + bias)
   const DNN::EActivationFunction fF = this->GetActivationFunction();
   Matrix_t tmpState(fState.GetNrows(), fState.GetNcols());
   Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsState);
   Architecture_t::MultiplyTranspose(fState, input, fWeightsInput);
   Architecture_t::ScaleAdd(fState, tmpState);
   Architecture_t::AddRowWise(fState, fBiases);
   DNN::evaluate<Architecture_t>(fState, fF);
}

//____________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicRNNLayer<Architecture_t>::Backward(Tensor_t &gradients_backward,          // T x B x D
                                                     const Tensor_t &activations_backward,  // T x B x D 
                                                     std::vector<Matrix_t> &inp1,
                                                     std::vector<Matrix_t> &inp2)   
-> void
{
   // activations backward is input
   // gradients_backward is activationGradients of layer before it, which is input layer
   // currently gradient_backward is for input(x) and not for state
   // we also need the one for state as
   Matrix_t state_gradients_backward(this->GetBatchSize(), fStateSize);  // B x H
   DNN::initialize<Architecture_t>(state_gradients_backward,  DNN::EInitialization::kZero);

   Matrix_t initState(this->GetBatchSize(), fStateSize);  // B x H
   DNN::initialize<Architecture_t>(initState,   DNN::EInitialization::kZero);

   for (size_t t = fTimeSteps; t > 0; t--) {
      const Matrix_t & currStateActivations = this->GetOutputAt(t - 1);
      Architecture_t::ScaleAdd(state_gradients_backward, this->GetActivationGradientsAt(t - 1));
      if (t > 1) {
         const Matrix_t & precStateActivations = this->GetOutputAt(t - 2);
         CellBackward(state_gradients_backward, precStateActivations, currStateActivations, activations_backward[t - 1],
               gradients_backward[t - 1]);
      } else {
         const Matrix_t & precStateActivations = initState;
         CellBackward(state_gradients_backward, precStateActivations, currStateActivations, activations_backward[t - 1],
               gradients_backward[t - 1]);
      }
   }
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicRNNLayer<Architecture_t>::CellBackward(Matrix_t & state_gradients_backward,
                                                     const Matrix_t & precStateActivations, const Matrix_t & currStateActivations,
                                                     const Matrix_t & input, Matrix_t & input_gradient)
-> Matrix_t &
{
   DNN::evaluateDerivative<Architecture_t>(fDerivatives, this->GetActivationFunction(), currStateActivations);
   return Architecture_t::RecurrentLayerBackward(state_gradients_backward, fWeightInputGradients, fWeightStateGradients,
                                                 fBiasGradients, fDerivatives, precStateActivations, fWeightsInput,
                                                 fWeightsState, input, input_gradient);
}

} // namespace RNN
} // namespace DNN
} // namespace TMVA

#endif
