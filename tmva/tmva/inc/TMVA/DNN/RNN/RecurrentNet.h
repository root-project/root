// @(#)root/tmva/tmva/dnn/rnn:$Id$
// Author: Saurav Shekhar 23/06/17

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class : RecurrentNet                                                           *
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
// The network class
//______________________________________________________________________________

/** \class RecurrentNet
*/
template<typename Architecture_t>
class TRecurrentNet
{
   
public:

   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;
   using Layer_t  = VRNNLayer<Architecture_t>;
   using Tensor_t = std::vector<Matrix_t>;

private:
   
   size_t fTimeSteps;   ///< Timesteps to backprop
   bool isTraining;     ///< Training or not

   Layer_t *fRNNLayer; ///< RNNLayer element for this network
   bool rememberState; ///< Remember state for multiple runs
   Tensor_t fOutput;    ///< Output (fTimeSteps, fBatchSize, fStateSize)
   
public:

   /** Constructor */
   TRecurrentNet(Layer_t *rnnLayer, size_t timeSteps, bool training = true, 
                 bool _rememberState = false);

   /*! Initialize the weights according to the given initialization
    **  method. */
   void Initialize(DNN::EInitialization m);

   /*! Compute and return the next state with given input
   *  matrix, returns Matrix of output states */
   inline Tensor_t& Forward(Tensor_t &input);

   /*! Must only be called directly
    * a the corresponding call to Forward(...). 
    * returns gradients w.r.t input*/
   inline Tensor_t Backward(const Matrix_t & input,
                            const Tensor_t & gradients_output);

   /** Prints the info about the layer */
   //virtual void Print() const = 0;

   /** Getters */
   inline bool IsTraining()       const {return isTraining;}
   inline bool IsRememberState()  const {return rememberState;} 
   Tensor_t  & GetOutput()              {return fOutput;}
   const Tensor_t & GetOutput()   const {return fOutput;}
   Layer_t   & GetRNNLayer()            {return *fRNNLayer;}
   const Layer_t  & GetRNNLayer() const {return *fRNNLayer;}
    
};

//______________________________________________________________________________
//
// RecurrentNet Implementation
//______________________________________________________________________________

template<typename Architecture_t>
TRecurrentNet<Architecture_t>::TRecurrentNet(Layer_t *rnnLayer, size_t timeSteps, bool training,
                                             bool _rememberState)
   : fRNNLayer(rnnLayer), fTimeSteps(timeSteps), isTraining(training), rememberState(_rememberState), 
   fOutput(timeSteps, Matrix_t(rnnLayer->GetBatchSize(), rnnLayer->GetStateSize()))
{
   // TODO agree on output size, if output is vector of states then
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TRecurrentNet<Architecture_t>::Initialize(DNN::EInitialization m)
-> void
{
   fRNNLayer->Initialize(m);
}

//______________________________________________________________________________
template <typename Architecture_t>
// TODO decide format for input, one matrix or many
auto inline TRecurrentNet<Architecture_t>::Forward(Tensor_t &input)
-> Tensor_t & 
{
   if (!this->rememberState) fRNNLayer->InitState(DNN::EInitialization::kZero);
   for (size_t t = 0; t < fTimeSteps; ++t) {
      Architecture_t::Copy(fOutput[t], fRNNLayer->Forward(input[t]));
   }
   return fOutput;
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TRecurrentNet<Architecture_t>::Backward(const Matrix_t & input, const Tensor_t & gradients_output)
-> Tensor_t  
{
   // TODO output not available, getting fActivationGradients from layer before
   //evaluateGradients<Architecture_t>(fLayers.back().GetActivationGradients(),
   //                                                  fJ, Y, fLayers.back().GetOutput());
   Matrix_t state_gradients_backward(fRNNLayer->GetBatchSize(), fRNNLayer->GetStateSize());  // B x H
   DNN::initialize<Architecture_t>(state_gradients_backward,  DNN::EInitialization::kZero);
   
   // send back to prev layer  T x B x D
   Tensor_t input_gradients(fTimeSteps, Matrix_t(fRNNLayer->GetBatchSize(), fRNNLayer->GetInputSize())); 

   Matrix_t initState(fRNNLayer->GetStateSize(), 1);  // H x 1
   DNN::initialize<Architecture_t>(initState,   DNN::EInitialization::kZero);

   for (size_t t = fTimeSteps - 1; t >= 0; t--) {
      const Matrix_t & currStateActivations = fOutput[t];
      Architecture_t::ScaleAdd(state_gradients_backward, gradients_output[t]);
      if (t > 0) {
         const Matrix_t & precStateActivations = fOutput[t - 1];
         fRNNLayer->Backward(state_gradients_backward, precStateActivations, currStateActivations, input,
               input_gradients[t]);
      } else {
         const Matrix_t & precStateActivations = initState;
         fRNNLayer->Backward(state_gradients_backward, precStateActivations, currStateActivations, input, 
               input_gradients[t]);
      }
   }
   //FIXME fix data format and then this
   return input_gradients;
}

} // namespace RNN
} // namespace DNN
} // namespace TMVA

#endif

