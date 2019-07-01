// @(#)root/tmva/tmva/dnn/lstm:$Id$
// Author: Surya S Dwivedi 27/05/19

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class : BasicLSTMLayer                                                         *
 *                                                                                *
 * Description:                                                                   *
 *       NeuralNetwork                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *       Surya S Dwivedi  <surya2191997@gmail.com> - IIT Kharagpur, India         *
 *                                                                                *
 * Copyright (c) 2005-2019:                                                       *
 * All rights reserved.                                                           *
 *       CERN, Switzerland                                                        *
 *                                                                                *
 * For the licensing terms see $ROOTSYS/LICENSE.                                  *
 * For the list of contributors see $ROOTSYS/README/CREDITS.                      *
 **********************************************************************************/

//#pragma once

//////////////////////////////////////////////////////////////////////
// This class implements the LSTM layer. LSTM is a variant of vanilla 
// RNN which is capable of learning long range dependencies. 
//////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_LSTM_LAYER
#define TMVA_DNN_LSTM_LAYER

#include <cmath>
#include <iostream>
#include <vector>

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"

namespace TMVA
{
namespace DNN
{
namespace LSTM
{

//______________________________________________________________________________
//
// Basic LSTM Layer
//______________________________________________________________________________

/** \class BasicLSTMLayer
      Generic implementation
*/
template<typename Architecture_t>
      class TBasicLSTMLayer : public VGeneralLayer<Architecture_t>
{

public:

   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;
   using Tensor_t = std::vector<Matrix_t>;

  

private:

   size_t fStateSize;                           ///< Hidden state size for LSTM
   size_t fCellSize;                            ///< Cell state size of LSTM
   size_t fTimeSteps;                           ///< Timesteps for LSTM

   bool fRememberState;                         ///< Remember state in next pass

   DNN::EActivationFunction fF1;                ///< Activation function: sigmoid
   DNN::EActivationFunction fF2;                ///< Activaton function: tanh

   Matrix_t fInputValue;                        ///< Computed input gate values
   Matrix_t fCandidateValue;                    ///< Computed candidate values
   Matrix_t fForgetValue;                       ///< Computed forget gate values
   Matrix_t fOutputValue;                       ///< Computed output gate values
   Matrix_t fState;                             ///< Hidden state of LSTM
   Matrix_t fCell;                              ///< Cell state of LSTM

   Matrix_t &fWeightsInputGate;                 ///< Input Gate weights for input, fWeights[0]
   Matrix_t &fWeightsInputGateState;            ///< Input Gate weights for prev state, fWeights[1]
   Matrix_t &fInputGateBias;                    ///< Input Gate bias

   Matrix_t &fWeightsForgetGate;                ///< Forget Gate weights for input, fWeights[0]
   Matrix_t &fWeightsForgetGateState;           ///< Forget Gate weights for prev state, fWeights[1]
   Matrix_t &fForgetGateBias;                   ///< Forget Gate bias

   Matrix_t &fWeightsCandidate;                 ///< Candidate Gate weights for input, fWeights[0]
   Matrix_t &fWeightsCandidateState;            ///< Candidate Gate weights for prev state, fWeights[1]
   Matrix_t &fCandidateBias;                    ///< Candidate Gate bias

   Matrix_t &fWeightsOutputGate;                ///< Output Gate weights for input, fWeights[0]
   Matrix_t &fWeightsOutputGateState;           ///< Output Gate weights for prev state, fWeights[1]
   Matrix_t &fOutputGateBias;                   ///< Output Gate bias


   std::vector<Matrix_t> input_gate_value;      ///< input gate value for every time step
   std::vector<Matrix_t> forget_gate_value;     ///< forget gate value for every time step
   std::vector<Matrix_t> candidate_gate_value;  ///< candidate gate value for every time step
   std::vector<Matrix_t> output_gate_value;     ///< output gate value for every time step
   std::vector<Matrix_t> cell_value;            ///< cell value for every time step
   std::vector<Matrix_t> fDerivativesInput;     ///< First fDerivatives of the activations input gate
   std::vector<Matrix_t> fDerivativesForget;    ///< First fDerivatives of the activations forget gate
   std::vector<Matrix_t> fDerivativesCandidate; ///< First fDerivatives of the activations candidate gate
   std::vector<Matrix_t> fDerivativesOutput;    ///< First fDerivatives of the activations output gate

   Matrix_t &fWeightsInputGradients;            ///< Gradients w.r.t the input gate - input weights
   Matrix_t &fWeightsInputStateGradients;       ///< Gradients w.r.t the input gate - hidden state weights
   Matrix_t &fInputBiasGradients;               ///< Gradients w.r.t the input gate - bias weights
   Matrix_t &fWeightsForgetGradients;           ///< Gradients w.r.t the forget gate - input weights
   Matrix_t &fWeightsForgetStateGradients;      ///< Gradients w.r.t the forget gate - hidden state weights
   Matrix_t &fForgetBiasGradients;              ///< Gradients w.r.t the forget gate - bias weights
   Matrix_t &fWeightsCandidateGradients;        ///< Gradients w.r.t the candidate gate - input weights
   Matrix_t &fWeightsCandidateStateGradients;   ///< Gradients w.r.t the candidate gate - hidden state weights
   Matrix_t &fCandidateBiasGradients;           ///< Gradients w.r.t the candidate gate - bias weights
   Matrix_t &fWeightsOutputGradients;           ///< Gradients w.r.t the output gate - input weights
   Matrix_t &fWeightsOutputStateGradients;      ///< Gradients w.r.t the output gate - hidden state weights
   Matrix_t &fOutputBiasGradients;              ///< Gradients w.r.t the output gate - bias weights

public:

   /*! Constructor */
   TBasicLSTMLayer(size_t batchSize, size_t stateSize, size_t inputSize,
                   size_t timeSteps, bool rememberState = false,
                   DNN::EActivationFunction f1 = DNN::EActivationFunction::kSigmoid,
                   DNN::EActivationFunction f2 = DNN::EActivationFunction::kTanh,
                   bool training = true, DNN::EInitialization fA = DNN::EInitialization::kZero);

   /*! Copy Constructor */
   TBasicLSTMLayer(const TBasicLSTMLayer &);

   /*! Initialize the hidden state and cell state method. */
   void InitState(DNN::EInitialization m = DNN::EInitialization::kZero);
    
   /*! Computes the next hidden state 
    *  and next cell state with given input matrix. */
   void Forward(Tensor_t &input, bool isTraining = true);

   /*! Forward for a single cell (time unit) */
   void CellForward(Matrix_t &inputGateValues, Matrix_t &forgetGateValues,
                     Matrix_t &candidateValues, Matrix_t &outputGateValues);

   /*! Backpropagates the error. Must only be called directly at the corresponding
    *  call to Forward(...). */
   void Backward(Tensor_t &gradients_backward,
                 const Tensor_t &activations_backward,
                 std::vector<Matrix_t> &inp1,
                 std::vector<Matrix_t> &inp2);
    
   /* Updates weights and biases, given the learning rate */
   void Update(const Scalar_t learningRate);
    
   /*! Backward for a single time unit
    *  a the corresponding call to Forward(...). */
   Matrix_t & CellBackward(Matrix_t & state_gradients_backward,
                           Matrix_t & cell_gradients_backward,
                           const Matrix_t & precStateActivations, const Matrix_t & precCellActivations,
                           const Matrix_t & input_gate, const Matrix_t & forget_gate,   
                           const Matrix_t & candidate_gate, const Matrix_t & output_gate,
                           const Matrix_t & input, Matrix_t & input_gradient,
                           Matrix_t &di, Matrix_t &df, Matrix_t &dc, Matrix_t &dout, size_t t);

  
    
   /*! Decides the values we'll update (NN with Sigmoid) */
   void InputGate(const Matrix_t &input, Matrix_t &di);
    
   /*! Forgets the past values (NN with Sigmoid) */
   void ForgetGate(const Matrix_t &input, Matrix_t &df);
    
   /*! Decides the new candidate values (NN with Tanh) */
   void CandidateValue(const Matrix_t &input, Matrix_t &dc);
    
   /*! Computes output values (NN with Sigmoid) */
   void OutputGate(const Matrix_t &input, Matrix_t &dout);
    
   /*! Prints the info about the layer */
   void Print() const;
    
   /*! Writes the information and the weights about the layer in an XML node. */
   void AddWeightsXMLTo(void *parent) override;
    
   /*! Read the information and the weights about the layer from XML node. */
   void ReadWeightsFromXML(void *parent) override;
    
   /*! Getters */
   size_t GetInputSize()               const { return this->GetInputWidth(); }
   size_t GetTimeSteps()               const { return fTimeSteps; }
   size_t GetStateSize()               const { return fStateSize; }
   size_t GetCellSize()                const { return fCellSize; }

   inline bool DoesRememberState()       const { return fRememberState; }
   
   inline DNN::EActivationFunction     GetActivationFunctionF1()        const { return fF1; }
   inline DNN::EActivationFunction     GetActivationFunctionF2()        const { return fF2; }

   const Matrix_t                    & GetInputGateValue()                const { return fInputValue; }
   Matrix_t                          & GetInputGateValue()                      { return fInputValue; }
   const Matrix_t                    & GetCandidateValue()                const { return fCandidateValue; }
   Matrix_t                          & GetCandidateValue()                      { return fCandidateValue; }
   const Matrix_t                    & GetForgetGateValue()               const { return fForgetValue; }
   Matrix_t                          & GetForgetGateValue()                     { return fForgetValue; }
   const Matrix_t                    & GetOutputGateValue()               const { return fOutputValue; }
   Matrix_t                          & GetOutputGateValue()                     { return fOutputValue; }

   const Matrix_t                    & GetState()                   const { return fState; }
   Matrix_t                          & GetState()                         { return fState; }
   const Matrix_t                    & GetCell()                    const { return fCell; }
   Matrix_t                          & GetCell()                          { return fCell; }

   const Matrix_t                    & GetWeightsInputGate()              const { return fWeightsInputGate; }
   Matrix_t                          & GetWeightsInputGate()                    { return fWeightsInputGate; }
   const Matrix_t                    & GetWeightsCandidate()              const { return fWeightsCandidate; }
   Matrix_t                          & GetWeightsCandidate()                    { return fWeightsCandidate; }
   const Matrix_t                    & GetWeightsForgetGate()             const { return fWeightsForgetGate; }
   Matrix_t                          & GetWeightsForgetGate()                   { return fWeightsForgetGate; }
   const Matrix_t                    & GetWeightsOutputGate()             const { return fWeightsOutputGate; }
   Matrix_t                          & GetWeightsOutputGate()                   { return fWeightsOutputGate; }
   const Matrix_t                    & GetWeightsInputGateState()         const { return fWeightsInputGateState; }
   Matrix_t                          & GetWeightsInputGateState()               { return fWeightsInputGateState; }
   const Matrix_t                    & GetWeightsForgetGateState()        const { return fWeightsForgetGateState; }
   Matrix_t                          & GetWeightsForgetGateState()              { return fWeightsForgetGateState; }
   const Matrix_t                    & GetWeightsCandidateState()         const { return fWeightsCandidateState; }
   Matrix_t                          & GetWeightsCandidateState()               { return fWeightsCandidateState; }
   const Matrix_t                    & GetWeightsOutputGateState()        const { return fWeightsOutputGateState; }
   Matrix_t                          & GetWeightsOutputGateState()              { return fWeightsOutputGateState; }

   const std::vector<Matrix_t>       & GetDerivativesInput()              const { return fDerivativesInput; }
   std::vector<Matrix_t>             & GetDerivativesInput()                    { return fDerivativesInput; }
   const Matrix_t                    & GetInputDerivativesAt(size_t i)    const { return fDerivativesInput[i]; }
   Matrix_t                          & GetInputDerivativesAt(size_t i)           { return fDerivativesInput[i]; }
   const std::vector<Matrix_t>       & GetDerivativesForget()              const { return fDerivativesForget; }
   std::vector<Matrix_t>             & GetDerivativesForget()                    { return fDerivativesForget; }
   const Matrix_t                    & GetForgetDerivativesAt(size_t i)    const { return fDerivativesForget[i]; }
   Matrix_t                          & GetForgetDerivativesAt(size_t i)          { return fDerivativesForget[i]; }
   const std::vector<Matrix_t>       & GetDerivativesCandidate()           const { return fDerivativesCandidate; }
   std::vector<Matrix_t>             & GetDerivativesCandidate()                 { return fDerivativesCandidate; }
   const Matrix_t                    & GetCandidateDerivativesAt(size_t i) const { return fDerivativesCandidate[i]; }
   Matrix_t                          & GetCandidateDerivativesAt(size_t i)       { return fDerivativesCandidate[i]; }
   const std::vector<Matrix_t>       & GetDerivativesOutput()              const { return fDerivativesOutput; }
   std::vector<Matrix_t>             & GetDerivativesOutput()                    { return fDerivativesOutput; }
   const Matrix_t                    & GetOutputDerivativesAt(size_t i)    const { return fDerivativesOutput[i]; }
   Matrix_t                          & GetOutputDerivativesAt(size_t i)          { return fDerivativesOutput[i]; }

   const std::vector<Matrix_t>       & GetInputGateTensor()              const { return input_gate_value; }
   std::vector<Matrix_t>             & GetInputGateTensor()                    { return input_gate_value; }
   const Matrix_t                    & GetInputGateTensorAt(size_t i)    const { return input_gate_value[i]; }
   Matrix_t                          & GetInputGateTensorAt(size_t i)           { return input_gate_value[i]; }
   const std::vector<Matrix_t>       & GetForgetGateTensor()              const { return forget_gate_value; }
   std::vector<Matrix_t>             & GetForgetGateTensor()                    { return forget_gate_value; }
   const Matrix_t                    & GetForgetGateTensorAt(size_t i)    const { return forget_gate_value[i]; }
   Matrix_t                          & GetForgetGateTensorAt(size_t i)          { return forget_gate_value[i]; }
   const std::vector<Matrix_t>       & GetCandidateGateTensor()           const { return candidate_gate_value; }
   std::vector<Matrix_t>             & GetCandidateGateTensor()                 { return candidate_gate_value; }
   const Matrix_t                    & GetCandidateGateTensorAt(size_t i) const { return candidate_gate_value[i]; }
   Matrix_t                          & GetCandidateGateTensorAt(size_t i)       { return candidate_gate_value[i]; }
   const std::vector<Matrix_t>       & GetOutputGateTensor()              const { return output_gate_value; }
   std::vector<Matrix_t>             & GetOutputGateTensor()                    { return output_gate_value; }
   const Matrix_t                    & GetOutputGateTensorAt(size_t i)    const { return output_gate_value[i]; }
   Matrix_t                          & GetOutputGateTensorAt(size_t i)          { return output_gate_value[i]; }
   const std::vector<Matrix_t>       & GetCellTensor()                    const { return cell_value; }
   std::vector<Matrix_t>             & GetCellTensor()                          { return cell_value; }
   const Matrix_t                    & GetCellTensorAt(size_t i)          const { return cell_value[i]; }
   Matrix_t                          & GetCellTensorAt(size_t i)                { return cell_value[i]; }
   


   const Matrix_t                   & GetInputGateBias()         const { return fInputGateBias; }
   Matrix_t                         & GetInputGateBias()               { return fInputGateBias; }
   const Matrix_t                   & GetForgetGateBias()        const { return fForgetGateBias; }
   Matrix_t                         & GetForgetGateBias()              { return fForgetGateBias; }
   const Matrix_t                   & GetCandidateBias()         const { return fCandidateBias; }
   Matrix_t                         & GetCandidateBias()               { return fCandidateBias; }
   const Matrix_t                   & GetOutputGateBias()        const { return fOutputGateBias; }
   Matrix_t                         & GetOutputGateBias()              { return fOutputGateBias; }
   const Matrix_t                   & GetWeightsInputGradients()        const { return fWeightsInputGradients; }
   Matrix_t                         & GetWeightsInputGradients()              { return fWeightsInputGradients; }
   const Matrix_t                   & GetWeightsInputStateGradients()   const { return fWeightsInputStateGradients; }
   Matrix_t                         & GetWeightsInputStateGradients()         { return fWeightsInputStateGradients; }
   const Matrix_t                   & GetInputBiasGradients()           const { return fInputBiasGradients; }
   Matrix_t                         & GetInputBiasGradients()                 { return fInputBiasGradients; }
   const Matrix_t                   & GetWeightsForgetGradients()      const { return fWeightsForgetGradients; }
   Matrix_t                         & GetWeightsForgetGradients()            { return fWeightsForgetGradients; }
   const Matrix_t                   & GetWeigthsForgetStateGradients() const { return fWeightsForgetStateGradients; }
   Matrix_t                         & GetWeightsForgetStateGradients()       { return fWeightsForgetStateGradients; }
   const Matrix_t                   & GetForgetBiasGradients()         const { return fForgetBiasGradients; }
   Matrix_t                         & GetForgetBiasGradients()               { return fForgetBiasGradients; }
   const Matrix_t                   & GetWeightsCandidateGradients()      const { return fWeightsCandidateGradients; }
   Matrix_t                         & GetWeightsCandidateGradients()            { return fWeightsCandidateGradients; }
   const Matrix_t                   & GetWeightsCandidateStateGradients() const { return fWeightsCandidateStateGradients; }
   Matrix_t                         & GetWeightsCandidateStateGradients()       { return fWeightsCandidateStateGradients; }
   const Matrix_t                   & GetCandidateBiasGradients()         const { return fCandidateBiasGradients; }
   Matrix_t                         & GetCandidateBiasGradients()               { return fCandidateBiasGradients; }
   const Matrix_t                   & GetWeightsOutputGradients()        const { return fWeightsOutputGradients; }
   Matrix_t                         & GetWeightsOutputGradients()              { return fWeightsOutputGradients; }
   const Matrix_t                   & GetWeightsOutputStateGradients()   const { return fWeightsOutputStateGradients; }
   Matrix_t                         & GetWeightsOutputStateGradients()         { return fWeightsOutputStateGradients; }
   const Matrix_t                   & GetOutputBiasGradients()           const { return fOutputBiasGradients; }
   Matrix_t                         & GetOutputBiasGradients()                 { return fOutputBiasGradients; }

};

//______________________________________________________________________________
//
// Basic LSTM-Layer Implementation
//______________________________________________________________________________

template <typename Architecture_t>
TBasicLSTMLayer<Architecture_t>::TBasicLSTMLayer(size_t batchSize, size_t stateSize, size_t inputSize, size_t timeSteps,
                                                 bool rememberState, DNN::EActivationFunction f1, DNN::EActivationFunction f2,
                                                 bool /* training */, DNN::EInitialization fA)
   : VGeneralLayer<Architecture_t>(batchSize, 1, timeSteps, inputSize, 1, timeSteps, stateSize, 8,
                                   {stateSize, stateSize, stateSize, stateSize, stateSize, stateSize, stateSize, stateSize}, 
                                   {inputSize, stateSize, inputSize, stateSize, inputSize, stateSize, inputSize, stateSize}, 
                                   4, {stateSize, stateSize, stateSize, stateSize}, {1, 1, 1, 1}, batchSize, timeSteps, stateSize, fA),
   fStateSize(stateSize),
   fCellSize(stateSize),
   fTimeSteps(timeSteps),
   fRememberState(rememberState),
   fF1(f1),
   fF2(f2),
   fInputValue(batchSize, stateSize),
   fCandidateValue(batchSize, stateSize),
   fForgetValue(batchSize, stateSize),
   fOutputValue(batchSize, stateSize),
   fState(batchSize, stateSize),
   fCell(batchSize, stateSize),
   fWeightsInputGate(this->GetWeightsAt(0)),
   fWeightsInputGateState(this->GetWeightsAt(1)),
   fInputGateBias(this->GetBiasesAt(0)),
   fWeightsForgetGate(this->GetWeightsAt(2)),
   fWeightsForgetGateState(this->GetWeightsAt(3)),
   fForgetGateBias(this->GetBiasesAt(1)),
   fWeightsCandidate(this->GetWeightsAt(4)),
   fWeightsCandidateState(this->GetWeightsAt(5)),
   fCandidateBias(this->GetBiasesAt(2)),
   fWeightsOutputGate(this->GetWeightsAt(6)),
   fWeightsOutputGateState(this->GetWeightsAt(7)),
   fOutputGateBias(this->GetBiasesAt(3)),
   fWeightsInputGradients(this->GetWeightGradientsAt(0)),
   fWeightsInputStateGradients(this->GetWeightGradientsAt(1)),
   fInputBiasGradients(this->GetBiasGradientsAt(0)),
   fWeightsForgetGradients(this->GetWeightGradientsAt(2)),
   fWeightsForgetStateGradients(this->GetWeightGradientsAt(3)),
   fForgetBiasGradients(this->GetBiasGradientsAt(1)),
   fWeightsCandidateGradients(this->GetWeightGradientsAt(4)),
   fWeightsCandidateStateGradients(this->GetWeightGradientsAt(5)),
   fCandidateBiasGradients(this->GetBiasGradientsAt(2)),
   fWeightsOutputGradients(this->GetWeightGradientsAt(6)),
   fWeightsOutputStateGradients(this->GetWeightGradientsAt(7)),
   fOutputBiasGradients(this->GetBiasGradientsAt(3))
{
   for (size_t i = 0; i < timeSteps; ++i) {
      fDerivativesInput.emplace_back(batchSize, stateSize);
      fDerivativesForget.emplace_back(batchSize, stateSize);
      fDerivativesCandidate.emplace_back(batchSize, stateSize);
      fDerivativesOutput.emplace_back(batchSize, stateSize);
      input_gate_value.emplace_back(batchSize, stateSize);
      forget_gate_value.emplace_back(batchSize, stateSize);
      candidate_gate_value.emplace_back(batchSize, stateSize);
      output_gate_value.emplace_back(batchSize, stateSize);
      cell_value.emplace_back(batchSize, stateSize);
   }
}

 //______________________________________________________________________________
template <typename Architecture_t>
TBasicLSTMLayer<Architecture_t>::TBasicLSTMLayer(const TBasicLSTMLayer &layer)
   : VGeneralLayer<Architecture_t>(layer), 
      fStateSize(layer.fStateSize),
      fCellSize(layer.fCellSize),
      fTimeSteps(layer.fTimeSteps),
      fRememberState(layer.fRememberState),
      fF1(layer.GetActivationFunctionF1()),
      fF2(layer.GetActivationFunctionF2()),
      fInputValue(layer.GetBatchSize(), layer.GetStateSize()),
      fCandidateValue(layer.GetBatchSize(), layer.GetStateSize()),
      fForgetValue(layer.GetBatchSize(), layer.GetStateSize()),
      fOutputValue(layer.GetBatchSize(), layer.GetStateSize()),
      fState(layer.GetBatchSize(), layer.GetStateSize()),
      fCell(layer.GetBatchSize(), layer.GetCellSize()),
      fWeightsInputGate(this->GetWeightsAt(0)),
      fWeightsInputGateState(this->GetWeightsAt(1)),
      fInputGateBias(this->GetBiasesAt(0)),
      fWeightsForgetGate(this->GetWeightsAt(2)),
      fWeightsForgetGateState(this->GetWeightsAt(3)),
      fForgetGateBias(this->GetBiasesAt(1)),
      fWeightsCandidate(this->GetWeightsAt(4)),
      fWeightsCandidateState(this->GetWeightsAt(5)),
      fCandidateBias(this->GetBiasesAt(2)),
      fWeightsOutputGate(this->GetWeightsAt(6)),
      fWeightsOutputGateState(this->GetWeightsAt(7)),
      fOutputGateBias(this->GetBiasesAt(3)),
      fWeightsInputGradients(this->GetWeightGradientsAt(0)),
      fWeightsInputStateGradients(this->GetWeightGradientsAt(1)),
      fInputBiasGradients(this->GetBiasGradientsAt(0)),
      fWeightsForgetGradients(this->GetWeightGradientsAt(2)),
      fWeightsForgetStateGradients(this->GetWeightGradientsAt(3)),
      fForgetBiasGradients(this->GetBiasGradientsAt(1)),
      fWeightsCandidateGradients(this->GetWeightGradientsAt(4)),
      fWeightsCandidateStateGradients(this->GetWeightGradientsAt(5)),
      fCandidateBiasGradients(this->GetBiasGradientsAt(2)),
      fWeightsOutputGradients(this->GetWeightGradientsAt(6)),
      fWeightsOutputStateGradients(this->GetWeightGradientsAt(7)),
      fOutputBiasGradients(this->GetBiasGradientsAt(3)) 
{
   for (size_t i = 0; i < fTimeSteps; ++i) {
      fDerivativesInput.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
      Architecture_t::Copy(fDerivativesInput[i], layer.GetInputDerivativesAt(i));
        
      fDerivativesForget.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
      Architecture_t::Copy(fDerivativesForget[i], layer.GetForgetDerivativesAt(i));
        
      fDerivativesCandidate.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
      Architecture_t::Copy(fDerivativesCandidate[i], layer.GetCandidateDerivativesAt(i));
        
      fDerivativesOutput.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
      Architecture_t::Copy(fDerivativesOutput[i], layer.GetOutputDerivativesAt(i));

      input_gate_value.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
      Architecture_t::Copy(input_gate_value[i], layer.GetInputGateTensorAt(i));

      forget_gate_value.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
      Architecture_t::Copy(forget_gate_value[i], layer.GetForgetGateTensorAt(i));

      candidate_gate_value.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
      Architecture_t::Copy(candidate_gate_value[i], layer.GetCandidateGateTensorAt(i));

      output_gate_value.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
      Architecture_t::Copy(output_gate_value[i], layer.GetOutputGateTensorAt(i));

      cell_value.emplace_back(layer.GetBatchSize(), layer.GetStateSize());
      Architecture_t::Copy(cell_value[i], layer.GetCellTensorAt(i));
   }
    
   // Gradient matrices not copied
   Architecture_t::Copy(fState, layer.GetState());
   Architecture_t::Copy(fCell, layer.GetCell());

   // Copy each gate values.
   Architecture_t::Copy(fInputValue, layer.GetInputGateValue());
   Architecture_t::Copy(fCandidateValue, layer.GetCandidateValue());
   Architecture_t::Copy(fForgetValue, layer.GetForgetGateValue());
   Architecture_t::Copy(fOutputValue, layer.GetOutputGateValue());
}
 
//______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::InputGate(const Matrix_t &input, Matrix_t &di)
-> void
{
   /*! Computes input gate values according to equation:
    *  input = act(W_input . input + W_state . state + bias)
    *  activation function: sigmoid. */
   const DNN::EActivationFunction fInp = this->GetActivationFunctionF1();
   Matrix_t tmpState(fInputValue.GetNrows(), fInputValue.GetNcols());
   Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsInputGateState);
   Architecture_t::MultiplyTranspose(fInputValue, input, fWeightsInputGate);
   Architecture_t::ScaleAdd(fInputValue, tmpState);
   Architecture_t::AddRowWise(fInputValue, fInputGateBias);
   DNN::evaluateDerivative<Architecture_t>(di, fInp, fInputValue);
   DNN::evaluate<Architecture_t>(fInputValue, fInp);
}

 //______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::ForgetGate(const Matrix_t &input, Matrix_t &df)
-> void
{
   /*! Computes forget gate values according to equation:
    *  forget = act(W_input . input + W_state . state + bias)
    *  activation function: sigmoid. */
   const DNN::EActivationFunction fFor = this->GetActivationFunctionF1();
   Matrix_t tmpState(fForgetValue.GetNrows(), fForgetValue.GetNcols());
   Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsForgetGateState);
   Architecture_t::MultiplyTranspose(fForgetValue, input, fWeightsForgetGate);
   Architecture_t::ScaleAdd(fForgetValue, tmpState);
   Architecture_t::AddRowWise(fForgetValue, fForgetGateBias);
   DNN::evaluateDerivative<Architecture_t>(df, fFor, fForgetValue);
   DNN::evaluate<Architecture_t>(fForgetValue, fFor);
}

 //______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::CandidateValue(const Matrix_t &input, Matrix_t &dc)
-> void
{
   /*! Candidate value will be used to scale input gate values followed by Hadamard product.
    *  candidate_value = act(W_input . input + W_state . state + bias)
    *  activation function = tanh. */
   const DNN::EActivationFunction fCan = this->GetActivationFunctionF2();
   Matrix_t tmpState(fCandidateValue.GetNrows(), fCandidateValue.GetNcols());
   Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsCandidateState);
   Architecture_t::MultiplyTranspose(fCandidateValue, input, fWeightsCandidate);
   Architecture_t::ScaleAdd(fCandidateValue, tmpState);
   Architecture_t::AddRowWise(fCandidateValue, fCandidateBias);
   DNN::evaluateDerivative<Architecture_t>(dc, fCan, fCandidateValue);
   DNN::evaluate<Architecture_t>(fCandidateValue, fCan);
}

 //______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::OutputGate(const Matrix_t &input, Matrix_t &dout)
-> void
{
   /*! Output gate values will be used to calculate next hidden state and output values.
    *  output = act(W_input . input + W_state . state + bias)
    *  activation function = sigmoid. */
   const DNN::EActivationFunction fOut = this->GetActivationFunctionF1();
   Matrix_t tmpState(fOutputValue.GetNrows(), fOutputValue.GetNcols());
   Architecture_t::MultiplyTranspose(tmpState, fState, fWeightsOutputGateState);
   Architecture_t::MultiplyTranspose(fOutputValue, input, fWeightsOutputGate);
   Architecture_t::ScaleAdd(fOutputValue, tmpState);
   Architecture_t::AddRowWise(fOutputValue, fOutputGateBias);
   DNN::evaluateDerivative<Architecture_t>(dout, fOut, fOutputValue);
   DNN::evaluate<Architecture_t>(fOutputValue, fOut);
}

 //______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::Forward(Tensor_t &input, bool /* isTraining = true */)
-> void
{
   // D : input size
   // H : state size
   // T : time size
   // B : batch size

   Tensor_t arrInput;
   for (size_t t = 0; t < fTimeSteps; ++t) {
      arrInput.emplace_back(this->GetBatchSize(), this->GetInputWidth()); // T x B x D
   }
   Architecture_t::Rearrange(arrInput, input); // B x T x D

   Tensor_t arrOutput;
   for (size_t t = 0; t < fTimeSteps;++t) {
      arrOutput.emplace_back(this->GetBatchSize(), fStateSize); // T x B x H 
   }
  
   if (!this->fRememberState) {
      InitState(DNN::EInitialization::kZero);
   }

   /*! Pass each gate values to CellForward() to calculate
    *  next hidden state and next cell state. */
   for (size_t t = 0; t < fTimeSteps; ++t) {
      /* Feed forward network: value of each gate being computed at each timestep t. */
      InputGate(arrInput[t], fDerivativesInput[t]);
      ForgetGate(arrInput[t], fDerivativesForget[t]);
      CandidateValue(arrInput[t], fDerivativesCandidate[t]);
      OutputGate(arrInput[t], fDerivativesOutput[t]);

      Architecture_t::Copy(this->GetInputGateTensorAt(t), fInputValue);
      Architecture_t::Copy(this->GetForgetGateTensorAt(t), fForgetValue);
      Architecture_t::Copy(this->GetCandidateGateTensorAt(t), fCandidateValue);
      Architecture_t::Copy(this->GetOutputGateTensorAt(t), fOutputValue);
       
      CellForward(fInputValue, fForgetValue, fCandidateValue, fOutputValue);
      Architecture_t::Copy(arrOutput[t], fState);
      Architecture_t::Copy(this->GetCellTensorAt(t), fCell);
   }

   Architecture_t::Rearrange(this->GetOutput(), arrOutput);  // B x T x D
}

 //______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::CellForward(Matrix_t &inputGateValues, Matrix_t &forgetGateValues,
                                                         Matrix_t &candidateValues, Matrix_t &outputGateValues)
-> void
{
    
   // Update cell state.
   Architecture_t::Hadamard(fCell, forgetGateValues);
   Architecture_t::Hadamard(inputGateValues, candidateValues);
   Architecture_t::ScaleAdd(fCell, inputGateValues);

   Matrix_t cache(fCell.GetNrows(), fCell.GetNcols());
   Architecture_t::Copy(cache, fCell);
    
   // Update hidden state.
   const DNN::EActivationFunction fAT = this->GetActivationFunctionF2();
   DNN::evaluate<Architecture_t>(cache, fAT);

   /*! The Hadamard product of output_gate_value . tanh(cell_state)
    *  will be copied to next hidden state (passed to next LSTM cell)
    *  and we will update our outputGateValues also. */
   Architecture_t::Copy(fState, cache);
   Architecture_t::Hadamard(fState, outputGateValues);
}

 //____________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::Backward(Tensor_t &gradients_backward,           // B x T x D
                                                      const Tensor_t &activations_backward,   // B x T x D
                                                      std::vector<Matrix_t> & /*inp1*/,
                                                      std::vector<Matrix_t> & /*inp2*/)
-> void
{
   // gradients_backward is activationGradients of layer before it, which is input layer.
   // Currently, gradients_backward is for input(x) and not for state.
   // For the state it can be:
   Matrix_t state_gradients_backward(this->GetBatchSize(), fStateSize); // B x H
   DNN::initialize<Architecture_t>(state_gradients_backward, DNN::EInitialization::kZero); // B x H


   Matrix_t cell_gradients_backward(this->GetBatchSize(), fStateSize); // B x H
   DNN::initialize<Architecture_t>(cell_gradients_backward, DNN::EInitialization::kZero); // B x H

   // if dummy is false gradients_backward will be written back on the matrix
   bool dummy = false;
   if (gradients_backward.size() == 0 || gradients_backward[0].GetNrows() == 0 || gradients_backward[0].GetNcols() == 0) {
      dummy = true;
   }


   Tensor_t arr_gradients_backward;
   for (size_t t = 0; t < fTimeSteps; ++t) {
      arr_gradients_backward.emplace_back(this->GetBatchSize(), this->GetInputSize()); // T x B x D
   }
   
   //Architecture_t::Rearrange(arr_gradients_backward, gradients_backward); // B x T x D
   // activations_backward is input.
   Tensor_t arr_activations_backward;
   for (size_t t = 0; t < fTimeSteps; ++t) {
      arr_activations_backward.emplace_back(this->GetBatchSize(), this->GetInputSize()); // T x B x D
   }
   Architecture_t::Rearrange(arr_activations_backward, activations_backward); // B x T x D

   /*! For backpropagation, we need to calculate loss. For loss, output must be known.
    *  We obtain outputs during forward propagation and place the results in arr_output tensor. */
   Tensor_t arr_output;
   for (size_t t = 0; t < fTimeSteps; ++t) {
      arr_output.emplace_back(this->GetBatchSize(), fStateSize); // B x H
   }
   Architecture_t::Rearrange(arr_output, this->GetOutput());

   Matrix_t initState(this->GetBatchSize(), fCellSize); // B x H
   DNN::initialize<Architecture_t>(initState, DNN::EInitialization::kZero); // B x H

   // This will take partial derivative of state[t] w.r.t state[t-1]
   Tensor_t arr_actgradients;
   for (size_t t = 0; t < fTimeSteps; ++t) {
      arr_actgradients.emplace_back(this->GetBatchSize(), fCellSize);
   }
   Architecture_t::Rearrange(arr_actgradients, this->GetActivationGradients());

   /*! There are total 8 different weight matrices and 4 bias vectors.
    *  Re-initialize them with zero because it should have some value. (can't be garbage values) */

   // Input Gate.
   fWeightsInputGradients.Zero();
   fWeightsInputStateGradients.Zero();
   fInputBiasGradients.Zero();

   // Forget Gate.
   fWeightsForgetGradients.Zero();
   fWeightsForgetStateGradients.Zero();
   fForgetBiasGradients.Zero();

   // Candidate Gate.
   fWeightsCandidateGradients.Zero();
   fWeightsCandidateStateGradients.Zero();
   fCandidateBiasGradients.Zero();

   // Output Gate.
   fWeightsOutputGradients.Zero();
   fWeightsOutputStateGradients.Zero();
   fOutputBiasGradients.Zero();


   for (size_t t = fTimeSteps; t > 0; t--) {
      // Store the sum of gradients obtained at each timestep during backward pass.
      Architecture_t::ScaleAdd(state_gradients_backward, arr_actgradients[t-1]);
      if (t > 1) {
         const Matrix_t &prevStateActivations = arr_output[t-2];
         const Matrix_t &prevCellActivations = this->GetCellTensorAt(t-2);
         // During forward propagation, each gate value calculates their gradients.
         CellBackward(state_gradients_backward, cell_gradients_backward, 
         	          prevStateActivations, prevCellActivations,
                      this->GetInputGateTensorAt(t-1), this->GetForgetGateTensorAt(t-1),
                      this->GetCandidateGateTensorAt(t-1), this->GetOutputGateTensorAt(t-1),  
                      arr_activations_backward[t-1], arr_gradients_backward[t-1],
                      fDerivativesInput[t-1], fDerivativesForget[t-1],
                      fDerivativesCandidate[t-1], fDerivativesOutput[t-1], t-1);
      } else {
         const Matrix_t &prevStateActivations = initState;
         const Matrix_t &prevCellActivations = initState;
         CellBackward(state_gradients_backward, cell_gradients_backward, 
         	          prevStateActivations, prevCellActivations, 
                      this->GetInputGateTensorAt(t-1), this->GetForgetGateTensorAt(t-1),
                      this->GetCandidateGateTensorAt(t-1), this->GetOutputGateTensorAt(t-1), 
                      arr_activations_backward[t-1], arr_gradients_backward[t-1],
                      fDerivativesInput[t-1], fDerivativesForget[t-1],
                      fDerivativesCandidate[t-1], fDerivativesOutput[t-1], t-1);
        }
   }

   if (!dummy) {
      Architecture_t::Rearrange(gradients_backward, arr_gradients_backward );
   }

}

 //______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::CellBackward(Matrix_t & state_gradients_backward,
                                                          Matrix_t & cell_gradients_backward,
                                                          const Matrix_t & precStateActivations, const Matrix_t & precCellActivations, 
                                                          const Matrix_t & input_gate, const Matrix_t & forget_gate,   
                                                          const Matrix_t & candidate_gate, const Matrix_t & output_gate,
                                                          const Matrix_t & input, Matrix_t & input_gradient,
                                                          Matrix_t &di, Matrix_t &df, Matrix_t &dc, Matrix_t &dout,
                                                          size_t t)
-> Matrix_t &
{   
   /*! Call here LSTMLayerBackward() to pass parameters i.e. gradient
    *  values obtained from each gate during forward propagation. */

    
   // Update hidden state.
   const DNN::EActivationFunction fAT = this->GetActivationFunctionF2();   
   Matrix_t cell_gradient(this->GetCellTensorAt(t).GetNrows(), this->GetCellTensorAt(t).GetNcols());
   DNN::evaluateDerivative<Architecture_t>(cell_gradient, fAT, this->GetCellTensorAt(t));

   Matrix_t cell_tanh(this->GetCellTensorAt(t).GetNrows(), this->GetCellTensorAt(t).GetNcols());
   Architecture_t::Copy(cell_tanh, this->GetCellTensorAt(t));
   DNN::evaluate<Architecture_t>(cell_tanh, fAT);

   return Architecture_t::LSTMLayerBackward(state_gradients_backward, cell_gradients_backward,
                                            fWeightsInputGradients, fWeightsForgetGradients, fWeightsCandidateGradients,
                                            fWeightsOutputGradients, fWeightsInputStateGradients, fWeightsForgetStateGradients,
                                            fWeightsCandidateStateGradients, fWeightsOutputStateGradients, fInputBiasGradients, fForgetBiasGradients,
                                            fCandidateBiasGradients, fOutputBiasGradients, di, df, dc, dout, 
                                            precStateActivations, precCellActivations,
                                            input_gate, forget_gate, candidate_gate, output_gate,
                                            fWeightsInputGate, fWeightsForgetGate, fWeightsCandidate, fWeightsOutputGate,
                                            fWeightsInputGateState, fWeightsForgetGateState, fWeightsCandidateState,
                                            fWeightsOutputGateState, input, input_gradient, 
                                            cell_gradient, cell_tanh);
}
	
 //______________________________________________________________________________
template <typename Architecture_t>
auto TBasicLSTMLayer<Architecture_t>::InitState(DNN::EInitialization /* m */) 
-> void
{
   DNN::initialize<Architecture_t>(this->GetState(),  DNN::EInitialization::kZero);
   DNN::initialize<Architecture_t>(this->GetCell(),  DNN::EInitialization::kZero);
}

 //______________________________________________________________________________
template<typename Architecture_t>
auto TBasicLSTMLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "Batch Size: " << this->GetBatchSize() << "\n"
             << "Input Size: " << this->GetInputSize() << "\n"
             << "Hidden State Size: " << this->GetStateSize() << "\n"
             << "Cell State Size: " << this->GetCellSize() << "\n"
             << "Timesteps: " << this->GetTimeSteps() << "\n";
}

 //______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::AddWeightsXMLTo(void *parent) 
-> void
{
   auto layerxml = gTools().xmlengine().NewChild(parent, 0, "LSTMLayer");
    
   // Write all other info like outputSize, cellSize, inputSize, timeSteps, rememberState
   gTools().xmlengine().NewAttr(layerxml, 0, "OutputSize", gTools().StringFromInt(this->GetStateSize()));
   gTools().xmlengine().NewAttr(layerxml, 0, "CellSize", gTools().StringFromInt(this->GetCellSize()));
   gTools().xmlengine().NewAttr(layerxml, 0, "InputSize", gTools().StringFromInt(this->GetInputSize()));
   gTools().xmlengine().NewAttr(layerxml, 0, "TimeSteps", gTools().StringFromInt(this->GetTimeSteps()));
   gTools().xmlengine().NewAttr(layerxml, 0, "RememberState", gTools().StringFromInt(this->DoesRememberState()));
   
   // write weights and bias matrices
   this->WriteMatrixToXML(layerxml, "InputWeights", this->GetWeightsAt(0));
   this->WriteMatrixToXML(layerxml, "InputStateWeights", this->GetWeightsAt(1));
   this->WriteMatrixToXML(layerxml, "InputBiases", this->GetBiasesAt(0));
   this->WriteMatrixToXML(layerxml, "ForgetWeights", this->GetWeightsAt(2));
   this->WriteMatrixToXML(layerxml, "ForgetStateWeights", this->GetWeightsAt(3));
   this->WriteMatrixToXML(layerxml, "ForgetBiases", this->GetBiasesAt(1));
   this->WriteMatrixToXML(layerxml, "Candidateeights", this->GetWeightsAt(4));
   this->WriteMatrixToXML(layerxml, "CandidateStateWeights", this->GetWeightsAt(5));
   this->WriteMatrixToXML(layerxml, "CandidateBiases", this->GetBiasesAt(2));
   this->WriteMatrixToXML(layerxml, "OuputWeights", this->GetWeightsAt(6));
   this->WriteMatrixToXML(layerxml, "OutputStateWeights", this->GetWeightsAt(7));
   this->WriteMatrixToXML(layerxml, "OutputBiases", this->GetBiasesAt(3));
}

 //______________________________________________________________________________
template <typename Architecture_t>
auto inline TBasicLSTMLayer<Architecture_t>::ReadWeightsFromXML(void *parent) 
-> void
{
	// Read weights and biases
   this->ReadMatrixXML(parent, "InputWeights", this->GetWeightsAt(0));
   this->ReadMatrixXML(parent, "InputStateWeights", this->GetWeightsAt(1));
   this->ReadMatrixXML(parent, "InputBiases", this->GetBiasesAt(0));
   this->ReadMatrixXML(parent, "ForgetWeights", this->GetWeightsAt(2));
   this->ReadMatrixXML(parent, "ForgetStateWeights", this->GetWeightsAt(3));
   this->ReadMatrixXML(parent, "ForgetBiases", this->GetBiasesAt(1));
   this->ReadMatrixXML(parent, "Candidateeights", this->GetWeightsAt(4));
   this->ReadMatrixXML(parent, "CandidateStateWeights", this->GetWeightsAt(5));
   this->ReadMatrixXML(parent, "CandidateBiases", this->GetBiasesAt(2));
   this->ReadMatrixXML(parent, "OuputWeights", this->GetWeightsAt(6));
   this->ReadMatrixXML(parent, "OutputStateWeights", this->GetWeightsAt(7));
   this->ReadMatrixXML(parent, "OutputBiases", this->GetBiasesAt(3));
}

} // namespace RNN
} // namespace DNN
} // namespace TMVA

#endif // LSTM_LAYER_H