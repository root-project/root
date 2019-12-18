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

   using Tensor_t = typename Architecture_t::Tensor_t;
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

private:

   size_t fTimeSteps;              ///< Timesteps for RNN
   size_t fStateSize;              ///< Hidden state size of RNN
   bool   fRememberState;          ///< Remember state in next pass

   DNN::EActivationFunction fF;  ///< Activation function of the hidden state

   Matrix_t fState;                ///< Hidden State
   Matrix_t &fWeightsInput;         ///< Input weights, fWeights[0]
   Matrix_t &fWeightsState;         ///< Prev state weights, fWeights[1]
   Matrix_t &fBiases;               ///< Biases

   Tensor_t fDerivatives; ///< First fDerivatives of the activations
   Matrix_t &fWeightInputGradients; ///< Gradients w.r.t. the input weights
   Matrix_t &fWeightStateGradients; ///< Gradients w.r.t. the recurring weights
   Matrix_t &fBiasGradients;        ///< Gradients w.r.t. the bias values

   typename Architecture_t::ActivationDescriptor_t fActivationDesc;

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
   void CellForward(const Matrix_t &input, Matrix_t & dF);

   /*! Backpropagates the error. Must only be called directly at the corresponding
    *  call to Forward(...). */
   void Backward(Tensor_t &gradients_backward,
                 const Tensor_t &activations_backward);

   /* Updates weights and biases, given the learning rate */
   void Update(const Scalar_t learningRate);

   /*! Backward for a single time unit
    * a the corresponding call to Forward(...). */
   inline Matrix_t & CellBackward(Matrix_t & state_gradients_backward,
                              const Matrix_t & precStateActivations,
                              const Matrix_t & input, Matrix_t & input_gradient, Matrix_t &dF);

   /** Prints the info about the layer */
   void Print() const;

   /*! Writes the information and the weights about the layer in an XML node. */
   virtual void AddWeightsXMLTo(void *parent);

   /*! Read the information and the weights about the layer from XML node. */
   virtual void ReadWeightsFromXML(void *parent);


   /** Getters */
   size_t GetTimeSteps() const { return fTimeSteps; }
   size_t GetStateSize() const { return fStateSize; }
   size_t GetInputSize() const { return this->GetInputWidth(); }
   inline bool IsRememberState()  const {return fRememberState;}
   inline DNN::EActivationFunction GetActivationFunction()  const {return fF;}
   Matrix_t        & GetState()            {return fState;}
   const Matrix_t & GetState()       const  {return fState;}
   Matrix_t        & GetWeightsInput()        {return fWeightsInput;}
   const Matrix_t & GetWeightsInput()   const {return fWeightsInput;}
   Matrix_t        & GetWeightsState()        {return fWeightsState;}
   const Matrix_t & GetWeightsState()   const {return fWeightsState;}
   Tensor_t       & GetDerivatives()        {return fDerivatives;}
   const Tensor_t & GetDerivatives()   const {return fDerivatives;}
   // Matrix_t &GetDerivativesAt(size_t i) { return fDerivatives[i]; }
   // const Matrix_t &GetDerivativesAt(size_t i) const { return fDerivatives[i]; }

   Matrix_t        & GetBiasesState()              {return fBiases;}
   const Matrix_t & GetBiasesState()         const {return fBiases;}
   Matrix_t        & GetBiasStateGradients()            {return fBiasGradients;}
   const Matrix_t & GetBiasStateGradients() const {return fBiasGradients;}
   Matrix_t        & GetWeightInputGradients()         {return fWeightInputGradients;}
   const Matrix_t & GetWeightInputGradients()    const {return fWeightInputGradients;}
   Matrix_t        & GetWeightStateGradients()         {return fWeightStateGradients;}
   const Matrix_t & GetWeightStateGradients()    const {return fWeightStateGradients;}
};

//______________________________________________________________________________
//
// BasicRNNLayer Implementation
//______________________________________________________________________________
template <typename Architecture_t>
TBasicRNNLayer<Architecture_t>::TBasicRNNLayer(size_t batchSize, size_t stateSize, size_t inputSize, size_t timeSteps,
                                               bool rememberState, DNN::EActivationFunction f, bool /*training*/,
                                               DNN::EInitialization fA)
   // TODO inputDepth and outputDepth changed to batchSize??
   : VGeneralLayer<Architecture_t>(batchSize, 1, timeSteps, inputSize, 1, timeSteps, stateSize, 2,
                                   {stateSize, stateSize}, {inputSize, stateSize}, 1, {stateSize}, {1}, batchSize,
                                   timeSteps, stateSize, fA),
     fTimeSteps(timeSteps),
     fStateSize(stateSize),
     fRememberState(rememberState),
     fF(f),
     fState(batchSize, stateSize),
     fWeightsInput(this->GetWeightsAt(0)),
     fWeightsState(this->GetWeightsAt(1)),
     fBiases(this->GetBiasesAt(0)),
     fDerivatives( timeSteps, batchSize, stateSize),   // create tensor time x bs x S
     fWeightInputGradients(this->GetWeightGradientsAt(0)),
     fWeightStateGradients(this->GetWeightGradientsAt(1)),
     fBiasGradients(this->GetBiasGradientsAt(0))
{
   // Nothing
}

//______________________________________________________________________________
template <typename Architecture_t>
TBasicRNNLayer<Architecture_t>::TBasicRNNLayer(const TBasicRNNLayer &layer)
   : VGeneralLayer<Architecture_t>(layer), fTimeSteps(layer.fTimeSteps), fStateSize(layer.fStateSize),
     fRememberState(layer.fRememberState), fF(layer.GetActivationFunction()),
     fState(layer.GetBatchSize(), layer.GetStateSize()), fWeightsInput(this->GetWeightsAt(0)),
     fWeightsState(this->GetWeightsAt(1)), fBiases(this->GetBiasesAt(0)),
     fDerivatives( layer.GetDerivatives().GetShape() ), fWeightInputGradients(this->GetWeightGradientsAt(0)),
     fWeightStateGradients(this->GetWeightGradientsAt(1)), fBiasGradients(this->GetBiasGradientsAt(0))
{

   Architecture_t::Copy(fDerivatives, layer.GetDerivatives() );

   // Gradient matrices not copied
   Architecture_t::Copy(fState, layer.GetState());
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
template <typename Architecture_t>
auto TBasicRNNLayer<Architecture_t>::InitState(DNN::EInitialization /*m*/) -> void
{
   DNN::initialize<Architecture_t>(this->GetState(),  DNN::EInitialization::kZero);

   Architecture_t::InitializeActivationDescriptor(fActivationDesc,this->GetActivationFunction());
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TBasicRNNLayer<Architecture_t>::Print() const
-> void
{
   std::cout << " RECURRENT Layer: \t ";
   std::cout << " (NInput = " << this->GetInputSize();  // input size
   std::cout << ", NState = " << this->GetStateSize();  // hidden state size
   std::cout << ", NTime  = " << this->GetTimeSteps() << " )";  // time size
   std::cout << "\tOutput = ( " << this->GetOutput().GetFirstSize() << " , " << this->GetOutput().GetHSize() << " , " << this->GetOutput().GetWSize() << " )\n";
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
auto inline TBasicRNNLayer<Architecture_t>::Forward(Tensor_t &input, bool /*isTraining*/) // B x T x D
   -> void
{
   // D : input size
   // H : state size
   // T : time size
   // B : batch size

   Tensor_t arrInput (fTimeSteps, this->GetBatchSize(), this->GetInputWidth() );
   //for (size_t t = 0; t < fTimeSteps; ++t) arrInput.emplace_back(this->GetBatchSize(), this->GetInputWidth()); // T x B x D
   Architecture_t::Rearrange(arrInput, input);
   Tensor_t arrOutput ( fTimeSteps, this->GetBatchSize(), fStateSize);
   //for (size_t t = 0; t < fTimeSteps;++t) arrOutput.emplace_back(this->GetBatchSize(), fStateSize); // T x B x H

   if (!this->fRememberState) InitState(DNN::EInitialization::kZero);
   for (size_t t = 0; t < fTimeSteps; ++t) {
      Matrix_t arrInput_m = arrInput.At(t).GetMatrix();
      Matrix_t df_m = fDerivatives.At(t).GetMatrix();
      CellForward(arrInput_m, df_m );
      Matrix_t arrOutput_m = arrOutput.At(t).GetMatrix();
      Architecture_t::Copy(arrOutput_m, fState);
   }
   Architecture_t::Rearrange(this->GetOutput(), arrOutput);  // B x T x D
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicRNNLayer<Architecture_t>::CellForward(const Matrix_t &input, Matrix_t &dF)
-> void
{
   // State = act(W_input . input + W_state . state + bias)
   const DNN::EActivationFunction fAF = this->GetActivationFunction();
   Matrix_t tmpState(fState.GetNrows(), fState.GetNcols());
   Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsState);
   Architecture_t::MultiplyTranspose(fState, input, fWeightsInput);
   Architecture_t::ScaleAdd(fState, tmpState);
   Architecture_t::AddRowWise(fState, fBiases);
   Tensor_t inputActivFunc(dF);
   Tensor_t tState(fState);

   // DNN::evaluateDerivative<Architecture_t>(dFt, fAF, fState);
   // DNN::evaluate<Architecture_t>(tState, fAF);

   Architecture_t::Copy(inputActivFunc, tState);
   Architecture_t::ActivationFunctionForward(tState, fAF, fActivationDesc);

}

//____________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicRNNLayer<Architecture_t>::Backward(Tensor_t &gradients_backward,         // B x T x D
                                                     const Tensor_t &activations_backward) -> void  // B x T x D
                                                   //   std::vector<Matrix_t> & /*inp1*/, std::vector<Matrix_t> &
                                                   //   /*inp2*/) -> void
{
   // activations backward is input
   // gradients_backward is activationGradients of layer before it, which is input layer
   // currently gradient_backward is for input(x) and not for state
   // TODO use this to change initial state??


  bool dummy = false;
  if (gradients_backward.GetSize() == 0) {
     dummy = true;
  }
  Tensor_t arr_gradients_backward ( fTimeSteps, this->GetBatchSize(), this->GetInputSize());
  //for (size_t t = 0; t < fTimeSteps; ++t) arr_gradients_backward.emplace_back(this->GetBatchSize(), this->GetInputSize()); // T x B x D

  if (!dummy) {
      // TODO gradients_backward will be written back on the matrix
     //Architecture_t::Rearrange(arr_gradients_backward, gradients_backward);
  }
  Tensor_t arr_activations_backward ( fTimeSteps, this->GetBatchSize(), this->GetInputSize());
  //for (size_t t = 0; t < fTimeSteps; ++t) arr_activations_backward.emplace_back(this->GetBatchSize(), this->GetInputSize());  // T x B x D
  Architecture_t::Rearrange(arr_activations_backward, activations_backward);

   Matrix_t state_gradients_backward(this->GetBatchSize(), fStateSize);  // B x H
   DNN::initialize<Architecture_t>(state_gradients_backward,  DNN::EInitialization::kZero);

   Matrix_t initState(this->GetBatchSize(), fStateSize);  // B x H
   DNN::initialize<Architecture_t>(initState,   DNN::EInitialization::kZero);

   Tensor_t arr_output (  fTimeSteps, this->GetBatchSize(), fStateSize);
   //for (size_t t = 0; t < fTimeSteps; ++t) arr_output.emplace_back(this->GetBatchSize(), fStateSize);
   Architecture_t::Rearrange(arr_output, this->GetOutput());

   Tensor_t arr_actgradients ( fTimeSteps, this->GetBatchSize(), fStateSize);
   //for (size_t t = 0; t < fTimeSteps; ++t) arr_actgradients.emplace_back(this->GetBatchSize(), fStateSize);
   Architecture_t::Rearrange(arr_actgradients, this->GetActivationGradients());

   // reinitialize weights and biases gradients to 0
   fWeightInputGradients.Zero();
   fWeightStateGradients.Zero();
   fBiasGradients.Zero();

   for (size_t t = fTimeSteps; t > 0; t--) {
      //const Matrix_t & currStateActivations = arr_output[t - 1];
      Matrix_t actgrad_m = arr_actgradients.At(t - 1).GetMatrix();
      Architecture_t::ScaleAdd(state_gradients_backward, actgrad_m);

      Matrix_t actbw_m = arr_activations_backward.At(t - 1).GetMatrix();
      Matrix_t gradbw_m = arr_gradients_backward.At(t - 1).GetMatrix();

      // Architecture_t::PrintTensor(arr_actgradients.At(t - 1), "act grad");
      // Architecture_t::PrintTensor(Tensor_t(state_gradients_backward), "state grad before");

      // compute derivatives of activations
      Tensor_t  df = fDerivatives.At(t-1);
      Tensor_t dy =  Tensor_t(state_gradients_backward);
      //Tensor_t dy =  arr_actgradients.At(t - 1);
      Tensor_t y = arr_output.At(t-1);
      Architecture_t::ActivationFunctionBackward(df, y,
                                                 dy, df, //do in place (should work)
                                              this->GetActivationFunction(), fActivationDesc);

      Matrix_t df_m = df.GetMatrix();

      // Architecture_t::PrintTensor(df, "dy before");
      if (t > 1) {
         Matrix_t precStateActivations = arr_output.At(t - 2).GetMatrix();
         CellBackward(state_gradients_backward, precStateActivations, actbw_m, gradbw_m, df_m);

         // std::cout << "at time " << t << std::endl;
         // Architecture_t::PrintTensor(Tensor_t(state_gradients_backward), "state grad after");
         // Architecture_t::PrintTensor(arr_gradients_backward.At(t-1),"dx");
         // Architecture_t::PrintTensor(arr_activations_backward.At(t - 1),"x");
         // Architecture_t::PrintTensor(df, "dy after");
      } else {
         const Matrix_t & precStateActivations = initState;
         CellBackward(state_gradients_backward, precStateActivations, actbw_m, gradbw_m, df_m);

         // std::cout << "at time " << t << std::endl;
         // Architecture_t::PrintTensor(Tensor_t(state_gradients_backward), "state grad after");
         // Architecture_t::PrintTensor(arr_gradients_backward.At(t - 1), "dx");
         // Architecture_t::PrintTensor(arr_activations_backward.At(t - 1), "x");
         // Architecture_t::PrintTensor(df, "dy");
      }
   }
   if (!dummy) {
      Architecture_t::Rearrange(gradients_backward, arr_gradients_backward );
   }
   //Architecture_t::Rearrange(arr_activations_backward, activations_backward);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicRNNLayer<Architecture_t>::CellBackward(Matrix_t & state_gradients_backward,
                                                     const Matrix_t & precStateActivations,
                                                     const Matrix_t & input, Matrix_t & input_gradient, Matrix_t &dF)
-> Matrix_t &
{
   return Architecture_t::RecurrentLayerBackward(state_gradients_backward, fWeightInputGradients, fWeightStateGradients,
                                                 fBiasGradients, dF, precStateActivations, fWeightsInput,
                                                 fWeightsState, input, input_gradient);
}

//______________________________________________________________________________
template <typename Architecture_t>
void TBasicRNNLayer<Architecture_t>::AddWeightsXMLTo(void *parent)
{
   auto layerxml = gTools().xmlengine().NewChild(parent, 0, "RNNLayer");

   // write All other info like stateSize, inputSize, timeSteps,rememberState
   gTools().xmlengine().NewAttr(layerxml, 0, "StateSize", gTools().StringFromInt(this->GetStateSize()));
   gTools().xmlengine().NewAttr(layerxml, 0, "InputSize", gTools().StringFromInt(this->GetInputSize()));
   gTools().xmlengine().NewAttr(layerxml, 0, "TimeSteps", gTools().StringFromInt(this->GetTimeSteps()));
   gTools().xmlengine().NewAttr(layerxml, 0, "RememberState", gTools().StringFromInt(this->IsRememberState()));

   // write weights and bias matrices
   this->WriteMatrixToXML(layerxml, "InputWeights", this -> GetWeightsAt(0));
   this->WriteMatrixToXML(layerxml, "StateWeights", this -> GetWeightsAt(1));
   this->WriteMatrixToXML(layerxml, "Biases",  this -> GetBiasesAt(0));


}

//______________________________________________________________________________
template <typename Architecture_t>
void TBasicRNNLayer<Architecture_t>::ReadWeightsFromXML(void *parent)
{
   // Read weights and biases
   this->ReadMatrixXML(parent,"InputWeights", this -> GetWeightsAt(0));
   this->ReadMatrixXML(parent,"StateWeights", this -> GetWeightsAt(1));
   this->ReadMatrixXML(parent,"Biases", this -> GetBiasesAt(0));

}


} // namespace RNN
} // namespace DNN
} // namespace TMVA

#endif
