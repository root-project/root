// @(#)root/tmva/tmva/dnn/rnn:$Id$
// Author: Saurav Shekhar 23/06/17

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RecurrentNet                                                          *
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
// RecurrentNet class //
//////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_RNN_NET
#define TMVA_DNN_RNN_NET

#include <cmath>
#include <iostream>
#include <vector>

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"
#include "RNNLayer.h"

namespace TMVA
{
namespace DNN
{
namespace RNN
{

//______________________________________________________________________________
//
//  The network class
//______________________________________________________________________________

/** \class RecurrentNet
*/
template<typename Architecture_t>
class TRecurrentNet
{
  
public:

  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;
  using Layer_t  = TRNNLayer<Architecture_t>;

private:
  
  size_t fBatchSize;  ///< Batch size
  size_t fStateSize;  ///< Hidden state size vector
  size_t fInputSize;  ///< Input Size
  size_t fTimeSteps;  ///< Timesteps to backprop
  bool isTraining;    ///< Training or not

  DNN::EActivationFunction fF;  ///< Activation function of the hidden state

  Layer_t  fRNNLayer; 
  bool rememberState;            ///< Remember state for multiple runs
  std::vector<Matrix_t> fOutput; ///< Output (fTimeSteps, fBatchSize, fStateSize)
  
public:

  /** Constructor */
  TRecurrentNet(size_t batchSize, size_t stateSize, size_t inputSize,
                size_t timeSteps, DNN::EActivationFunction f = DNN::EActivationFunction::kTanh,
                bool training = true, bool _rememberState = false);

  /*! Initialize the weights according to the given initialization
   **  method. */
  void Initialize(DNN::EInitialization m);

  /*! Compute and return the next state with given input
  *  matrix, returns Matrix of output states */
  inline Matrix_t& Forward(Matrix_t &input);

  /*! Must only be called directly
   *  a the corresponding call to Forward(...). 
   *  returns gradients w.r.t input*/
  inline Matrix_t& Backward(Matrix_t & gradients_backward,
                           const Matrix_t & activations_backward);

  /** Prints the info about the layer */
  //virtual void Print() const = 0;

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
//  RecurrentNet Implementation
//______________________________________________________________________________

template<typename Architecture_t>
TRecurrentNet<Architecture_t>::TRecurrentNet(size_t batchSize, size_t stateSize, size_t inputSize,
                                             size_t timeSteps, DNN::EActivationFunction f, bool training,
                                             bool _rememberState)
  : fRNNLayer(batchSize, stateSize, inputSize, f, training), fBatchSize(batchSize) , fStateSize(stateSize), 
  fInputSize(inputSize), fTimeSteps(timeSteps), fF(f), rememberState(_rememberState), fOutput(timeSteps, Matrix_t(fBatchSize, fStateSize));
{
  // TODO agree on output size, if output is vector of states then
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TRecurrentNet<Architecture_t>::Initialize(DNN::EInitialization m)
-> void
{
  this->fRNNLayer.Initialize(m);
}

//______________________________________________________________________________
template <typename Architecture_t>
// TODO decide format for input, one matrix or many
auto inline TRecurrentNet<Architecture_t>::Forward(Matrix_t &input)
-> Matrix_t & 
{
  if (!this->rememberState) fRNNLayer.InitState(DNN::EInitialization::kZero);
  for (size_t i = 0; i < timeSteps; ++i) {
    Architecture_t::Copy(fOutput[i], fRNNLayer.Forward(input));
  }
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TRecurrentNet<Architecture_t>::Backward(const Matrix_t & input, std::vector<Matrix_t> & gradients_output)
-> Matrix_t & 
{
  // TODO output not available, getting fActivationGradients from layer before
  //evaluateGradients<Architecture_t>(fLayers.back().GetActivationGradients(),
  //                                   fJ, Y, fLayers.back().GetOutput());
  // see https://github.com/jcjohnson/torch-rnn/blob/master/VanillaRNN.lua
  Matrix_t state_gradients_backward(fBatchSize, fStateSize);  // B x H
  DNN::initialize<Architecture_t>(state_gradients_backward,  DNN::EInitialization::kZero);
  
  // send back to prev layer   T x B x D
  std::vector<Matrix_t> input_gradients(fTimeSteps, Matrix_t(fBatchSize, fInputSize)); 

  Matrix_t initState(fStateSize, 1);  // H x 1
  DNN::initialize<Architecture_t>(initState,  DNN::EInitialization::kZero);

  for (size_t t = fTimeSteps - 1; t >= 0; t--) {
    const Matrix_t & currStateActivations = fOutput[t];
    Architecture_t::ScaleAdd(state_gradients_backward, gradients_output[t]);
    if (t > 0) {
      const Matrix_t & precStateActivations = fOutput[t - 1];
      fRNNLayer.Backward(state_gradients_backward, precStateActivations, currStateActivations, input,
          input_gradients[t]);
    } else {
      const Matrix_t & precStateActivations = initState;
      fRNNLayer.Backward(state_gradients_backward, precStateActivations, currStateActivations, input, 
          input_gradients[t]);
    }
  }
  //FIXME fix data format and then this
  return input_gradients[0];
}

} // namespace RNN
} // namespace DNN
} // namespace TMVA

#endif

