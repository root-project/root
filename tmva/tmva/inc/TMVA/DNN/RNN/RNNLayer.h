// @(#)root/tmva/tmva/dnn/rnn:$Id$
// Author: Saurav Shekhar 14/06/17

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RNNLayer                                                               *
 *                                                                                *
 * Description:                                                                   *
 *      NeuralNetwork                                                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Saurav Shekhar    <sauravshekhar01@gmail.com> - ETH Zurich, Switzerland   *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 * All rights reserved.                                                           *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * For the licensing terms see $ROOTSYS/LICENSE.                                  *
 * For the list of contributors see $ROOTSYS/README/CREDITS.                      *
 **********************************************************************************/

//#pragma once

//////////////////////////////////////////////////////////////////////
// <Description> //
//////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_RNN_CELL
#define TMVA_DNN_RNN_CELL

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
//  The cell class
//______________________________________________________________________________

/** \class RNNLayer
    Generic implementation
*/
template<typename Architecture_t>
    class TRNNLayer
{
  
public:

  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;

private:
  
  size_t fBatchSize;  ///< Batch size used for training and evaluation.
  size_t fStateSize;  ///< Hidden state size vector
  size_t fInputSize;  ///< Input Size
  bool   isTraining;  ///< Training or not

  DNN::EActivationFunction fF;  ///< Activation function of the hidden state

  Matrix_t fState;   ///< Hidden State

public:

  /** Constructor */
  TRNNLayer(size_t batchSize, size_t stateSize, size_t inputSize,
            DNN::EActivationFunction f = DNN::EActivationFunction::kTanh,
            bool training = true);

  /** Copy Constructor */
  TRNNLayer(const TRNNLayer &);

  /*! Initialize the weights according to the given initialization
   **  method. */
  virtual void Initialize(DNN::EInitialization m) = 0;
  
  /*! Initialize the state
   **  method. */
  virtual void InitState(DNN::EInitialization m) = 0;

  /*! Compute and return the next state with given input
  *  matrix */
  virtual inline Matrix_t& Forward(Matrix_t &input) = 0;

  /*! Must only be called directly
   *  a the corresponding call to Forward(...). */
  virtual inline Matrix_t Backward(Matrix_t & gradients_backward,
                             const Matrix_t & activations_backward) = 0;

  /** Prints the info about the layer */
  virtual void Print() const = 0;

  /** Getters */
  inline bool   IsTraining()       const {return isTraining;}
  inline size_t GetBatchSize()     const {return fBatchSize;}
  inline size_t GetStateSize()     const {return fStateSize;}
  inline size_t GetInputSize()     const {return fInputSize;}
  inline DNN::EActivationFunction GetActivationFunction()  const {return fF;} 
  Matrix_t       & GetState()        {return fState;}
  const Matrix_t & GetState() const  {return fState;}

};

//______________________________________________________________________________
//
//  RNNLayer Class - Implementation
//______________________________________________________________________________

template<typename Architecture_t>
TRNNLayer<Architecture_t>::TRNNLayer(size_t batchSize, size_t stateSize, size_t inputSize,
                                     DNN::EActivationFunction f, bool training)
  : fBatchSize(batchSize), fStateSize(stateSize), fInputSize(inputSize), 
  fF(f), fState(batchSize, stateSize), isTraining(training) 
{
  // Nothing
}

//______________________________________________________________________________
template<typename Architecture_t>
TRNNLayer<Architecture_t>::TRNNLayer(const TRNNLayer &layer)
  : fBatchSize(layer.GetBatchSize()), fStateSize(layer.GetStateSize()),
  fF(layer.GetActivationFunction()), fState(layer.GetBatchSize(), layer.GetStateSize()),
  isTraining(layer.IsTraining())
{
  Architecture_t::Copy(fState, layer.GetState());
}


//______________________________________________________________________________
//
//  Basic RNN Layer
//______________________________________________________________________________

/** \class BasicRNNLayer
    Generic implementation
*/
template<typename Architecture_t>
    class TBasicRNNLayer : public TRNNLayer<Architecture_t>
{
  
public:

  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;

private:
  
  Matrix_t fWeightsInput;         ///< Input weights 
  Matrix_t fWeightsState;         ///< Prev state weights
  Matrix_t fBiases;               ///< Biases 

  Matrix_t fDerivatives;          ///< First fDerivatives of the activations 
  Matrix_t fWeightInputGradients; ///< Gradients w.r.t. the input weights 
  Matrix_t fWeightStateGradients; ///< Gradients w.r.t. the recurring weights 
  Matrix_t fBiasGradients;        ///< Gradients w.r.t. the bias values 
  Matrix_t fActivationGradients;  ///< Gradients w.r.t. the activations

public:

  /** Constructor */
  TBasicRNNLayer(size_t batchSize, size_t stateSize, size_t inputSize,
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
  inline Matrix_t& Forward(Matrix_t &input);

  /*! Must only be called directly
   *  a the corresponding call to Forward(...). */
  inline Matrix_t & Backward(Matrix_t & state_gradients_backward, 
                             const Matrix_t & precStateActivations, const Matrix_t & currStateActivations,
                             const Matrix_t & input, Matrix_t & input_gradient);

  /** Prints the info about the layer */
  //virtual void Print() const = 0;

  /** Getters */
  Matrix_t       & GetWeightsInput()       {return fWeightsInput;}
  const Matrix_t & GetWeightsInput() const {return fWeightsInput;}
  Matrix_t       & GetWeightsState()       {return fWeightsState;}
  const Matrix_t & GetWeightsState() const {return fWeightsState;}
  Matrix_t       & GetBiases()       {return fBiases;}
  const Matrix_t & GetBiases() const {return fBiases;}
  Matrix_t       & GetActivationGradients()       {return fActivationGradients;}
  const Matrix_t & GetActivationGradients() const {return fActivationGradients;}
  Matrix_t       & GetBiasGradients()       {return fBiasGradients;}
  const Matrix_t & GetBiasGradients() const {return fBiasGradients;}
  Matrix_t       & GetWeightInputGradients()       {return fWeightInputGradients;}
  const Matrix_t & GetWeightInputGradients() const {return fWeightInputGradients;}
  Matrix_t       & GetWeightStateGradients()       {return fWeightStateGradients;}
  const Matrix_t & GetWeightStateGradients() const {return fWeightStateGradients;}
};

//______________________________________________________________________________
//
//  BasicRNNLayer Implementation
//______________________________________________________________________________

template<typename Architecture_t>
TBasicRNNLayer<Architecture_t>::TBasicRNNLayer(size_t batchSize, size_t stateSize, size_t inputSize,
                                               DNN::EActivationFunction f, bool training)
  : TRNNLayer<Architecture_t>(batchSize, stateSize, inputSize, f, training), 
  fWeightsInput(stateSize, inputSize), fWeightsState(stateSize, stateSize), 
  fBiases(stateSize, 1), fDerivatives(stateSize, inputSize), fWeightInputGradients(stateSize, inputSize),
  fWeightStateGradients(stateSize, stateSize), fBiasGradients(stateSize, 1)
{
  // Nothing
}

//______________________________________________________________________________
template <typename Architecture_t>
TBasicRNNLayer<Architecture_t>::TBasicRNNLayer(const TBasicRNNLayer &layer)
  : TRNNLayer<Architecture_t>(layer), fWeightsInput(layer.GetStateSize(), layer.GetInputSize()),
  fWeightsState(layer.GetStateSize(), layer.GetStateSize()), fBiases(layer.GetStateSize(), 1),
  fDerivatives(layer.GetStateSize(), layer.GetInputSize()), fWeightInputGradients(layer.GetStateSize(), 
      layer.GetInputSize()), fWeightStateGradients(layer.GetStateSize(), layer.GetStateSize()), 
  fBiasGradients(layer.GetStateSize(), 1)
{
  // Gradient matrices not copied
  Architecture_t::Copy(fWeightsInput, layer.GetWeightsInput());
  Architecture_t::Copy(fWeightsState, layer.GetWeightsState());
  Architecture_t::Copy(fBiases, layer.GetBiases());
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
template <typename Architecture_t>
auto inline TBasicRNNLayer<Architecture_t>::Forward(Matrix_t &input)
-> Matrix_t & 
{
  // State = act(W_input . input + W_state . state + bias) 
  const Matrix_t & fState = this->GetState(); 
  const DNN::EActivationFunction fF = this->GetActivationFunction(); 
  Matrix_t tmpState(fState.GetNrows(), fState.GetNcols());
  Architecture_t::MultiplyTranspose(&tmpState, &fState, &fWeightsState);
  Architecture_t::MultiplyTranspose(&fState, &input, &fWeightsInput);
  Architecture_t::ScaleAdd(fState, tmpState);
  Architecture_t::AddRowWise(fState, fBiases);
  DNN::evaluate<Architecture_t>(fState, fF);
  return fState;
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicRNNLayer<Architecture_t>::Backward(Matrix_t & state_gradients_backward, 
                                      const Matrix_t & precStateActivations, const Matrix_t & currStateActivations,
                                      const Matrix_t & input, Matrix_t & input_gradient)
-> Matrix_t & 
{
  DNN::evaluateDerivative<Architecture_t>(fDerivatives, this->GetActivationFunction(), currStateActivations);  
  return Architecture_t::Backward(state_gradients_backward, fWeightInputGradients, fWeightStateGradients,
                                  fBiasGradients, fDerivatives, precStateActivations, fWeightsInput,
                                  fWeightsState, input, input_gradient);
}

} // namespace RNN
} // namespace DNN
} // namespace TMVA

#endif
