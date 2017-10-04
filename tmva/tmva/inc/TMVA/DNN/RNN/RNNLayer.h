// @(#)root/tmva/tmva/dnn/rnn:$Id$
// Author: Saurav Shekhar 19/07/17

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class : BasicRNNLayer                                                          *
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
                  bool training = true, DNN::EInitialization fA = DNN::EInitialization::kZero);

   /** Copy Constructor */
   TBasicRNNLayer(const TBasicRNNLayer &);

   /*! Initialize the weights according to the given initialization
    **  method. */
   //void Initialize(DNN::EInitialization m);

   /*! Initialize the state
    **  method. */
   void InitState(DNN::EInitialization m = DNN::EInitialization::kZero);

   /*! Compute and return the next state with given input
   *  matrix */
   void Forward(Tensor_t &input, bool isTraining = true);

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
   Matrix_t        & GetDerivatives()        {return fDerivatives;}
   const Matrix_t & GetDerivatives()   const {return fDerivatives;}
   //Matrix_t        & GetBiases()              {return fBiases;}
   //const Matrix_t & GetBiases()         const {return fBiases;}
   //Matrix_t        & GetBiasGradients()            {return fBiasGradients;}
   //const Matrix_t & GetBiasGradients() const {return fBiasGradients;}
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
                                              bool training, DNN::EInitialization fA)
   : VGeneralLayer<Architecture_t>(batchSize, 1, 1, inputSize, 1, 1, stateSize, 2, {stateSize, stateSize}, {inputSize, stateSize},
   1, {stateSize}, {1}, batchSize, timeSteps, stateSize, fA),
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
//template<typename Architecture_t>
//auto TBasicRNNLayer<Architecture_t>::Initialize(DNN::EInitialization m)
//-> void
//{
//   DNN::initialize<Architecture_t>(fWeightsInput, m);
//   DNN::initialize<Architecture_t>(fWeightsState, m);
//   DNN::initialize<Architecture_t>(fBiases,  DNN::EInitialization::kZero);
//}

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

template <typename Architecture_t>
auto debugMatrix(const typename Architecture_t::Matrix_t &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (size_t i = 0; i < A.GetNrows(); ++i) {
    for (size_t j = 0; j < A.GetNcols(); ++j) {
        std::cout << A(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "********\n";
}


//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicRNNLayer<Architecture_t>::Forward(Tensor_t &input, bool isTraining)   // B x T x D
-> void
{
   Tensor_t arrInput;
   for (size_t t = 0; t < fTimeSteps; ++t) arrInput.emplace_back(this->GetBatchSize(), this->GetInputWidth()); // T x B x D
   Architecture_t::Rearrange(arrInput, input);
   Tensor_t arrOutput;
   for (size_t t = 0; t < fTimeSteps;++t) arrOutput.emplace_back(this->GetBatchSize(), fStateSize); // T x B x H 

   if (!this->fRememberState) InitState(DNN::EInitialization::kZero);
   for (size_t t = 0; t < fTimeSteps; ++t) {
      CellForward(arrInput[t]);
      Architecture_t::Copy(arrOutput[t], fState);
   }
   Architecture_t::Rearrange(this->GetOutput(), arrOutput);  // B x T x D
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
auto inline TBasicRNNLayer<Architecture_t>::Backward(Tensor_t &gradients_backward,          // B x T x D
                                                     const Tensor_t &activations_backward,  // B x T x D 
                                                     std::vector<Matrix_t> &inp1,
                                                     std::vector<Matrix_t> &inp2)   
-> void
{
   // activations backward is input
   // gradients_backward is activationGradients of layer before it, which is input layer
   // currently gradient_backward is for input(x) and not for state
   // TODO use this to change initial state??
   
  bool dummy = false;
  if (gradients_backward.size() == 0 || gradients_backward[0].GetNrows() == 0 || gradients_backward[0].GetNcols() == 0) {
     dummy = true;
  }
  Tensor_t arr_gradients_backward;
  for (size_t t = 0; t < fTimeSteps; ++t) arr_gradients_backward.emplace_back(this->GetBatchSize(), this->GetInputSize()); // T x B x D

  if (!dummy) {
      // TODO gradients_backward will be written back on the matrix
     //Architecture_t::Rearrange(arr_gradients_backward, gradients_backward);
  }
  Tensor_t arr_activations_backward;
  for (size_t t = 0; t < fTimeSteps; ++t) arr_activations_backward.emplace_back(this->GetBatchSize(), this->GetInputSize());  // T x B x D
  Architecture_t::Rearrange(arr_activations_backward, activations_backward);
   
   Matrix_t state_gradients_backward(this->GetBatchSize(), fStateSize);  // B x H
   DNN::initialize<Architecture_t>(state_gradients_backward,  DNN::EInitialization::kZero);

   Matrix_t initState(this->GetBatchSize(), fStateSize);  // B x H
   DNN::initialize<Architecture_t>(initState,   DNN::EInitialization::kZero);

   Tensor_t arr_output;
   for (size_t t = 0; t < fTimeSteps; ++t) arr_output.emplace_back(this->GetBatchSize(), fStateSize);
   Architecture_t::Rearrange(arr_output, this->GetOutput());

   Tensor_t arr_actgradients;
   for (size_t t = 0; t < fTimeSteps; ++t) arr_actgradients.emplace_back(this->GetBatchSize(), fStateSize);
   Architecture_t::Rearrange(arr_actgradients, this->GetActivationGradients());

   for (size_t t = fTimeSteps; t > 0; t--) {
      const Matrix_t & currStateActivations = arr_output[t - 1];
      Architecture_t::ScaleAdd(state_gradients_backward, arr_actgradients[t - 1]);
      if (t > 1) {
         const Matrix_t & precStateActivations = arr_output[t - 2];
         CellBackward(state_gradients_backward, precStateActivations, currStateActivations, arr_activations_backward[t - 1],
               arr_gradients_backward[t - 1]);
      } else {
         const Matrix_t & precStateActivations = initState;
         CellBackward(state_gradients_backward, precStateActivations, currStateActivations, arr_activations_backward[t - 1],
               arr_gradients_backward[t - 1]);
      }
   }
   if (!dummy) {
      Architecture_t::Rearrange(arr_gradients_backward, gradients_backward);
   }
   //Architecture_t::Rearrange(arr_activations_backward, activations_backward);
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
