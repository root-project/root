
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TDenseLayer                                                           *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Dense Layer Class                                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
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

#ifndef TMVA_DNN_DENSELAYER
#define TMVA_DNN_DENSELAYER

#include "TMatrix.h"

#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Functions.h"

namespace TMVA {
namespace DNN {
/** \class TDenseLayer

Generic layer class.

This generic layer class represents a dense layer of a neural network with
a given width n and activation function f. The activation function of each
layer is given by \f$\mathbf{u} = \mathbf{W}\mathbf{x} + \boldsymbol{\theta}\f$.

In addition to the weight and bias matrices, each layer allocates memory
for its activations and the corresponding first partial fDerivatives of
the activation function as well as the gradients of the weights and biases.

The layer provides member functions for the forward propagation of
activations through the given layer.
*/
template <typename Architecture_t>
class TDenseLayer : public VGeneralLayer<Architecture_t> {
public:
   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;

private:
   std::vector<Matrix_t> fDerivatives; ///< First fDerivatives of the activations of this layer.

   Scalar_t fDropoutProbability; ///< Probability that an input is active.

   EActivationFunction fF; ///< Activation function of the layer.
   ERegularization fReg;   ///< The regularization method.
   Scalar_t fWeightDecay;  ///< The weight decay.

public:
   /*! Constructor */
   TDenseLayer(size_t BatchSize, size_t InputWidth, size_t Width, EInitialization init, Scalar_t DropoutProbability,
               EActivationFunction f, ERegularization reg, Scalar_t weightDecay);

   /*! Copy the dense layer provided as a pointer */
   TDenseLayer(TDenseLayer<Architecture_t> *layer);

   /*! Copy Constructor */
   TDenseLayer(const TDenseLayer &);

   /*! Destructor */
   ~TDenseLayer();

   /*! Compute activation of the layer for the given input. The input
    * must be in 3D tensor form with the different matrices corresponding to
    * different events in the batch. Computes activations as well as
    * the first partial derivative of the activation function at those
    * activations. */
   void Forward(std::vector<Matrix_t> input, bool applyDropout = false);

   /*! Compute weight, bias and activation gradients. Uses the precomputed
    *  first partial derviatives of the activation function computed during
    *  forward propagation and modifies them. Must only be called directly
    *  a the corresponding call to Forward(...). */
   void Backward(std::vector<Matrix_t> &gradients_backward, const std::vector<Matrix_t> &activations_backward,
                 std::vector<Matrix_t> &inp1, std::vector<Matrix_t> &inp2);

   /*! Printing the layer info. */
   void Print() const;

   /*! Getters */
   Scalar_t GetDropoutProbability() const { return fDropoutProbability; }

   const std::vector<Matrix_t> &GetDerivatives() const { return fDerivatives; }
   std::vector<Matrix_t> &GetDerivatives() { return fDerivatives; }

   Matrix_t &GetDerivativesAt(size_t i) { return fDerivatives[i]; }
   const Matrix_t &GetDerivativesAt(size_t i) const { return fDerivatives[i]; }

   EActivationFunction GetActivationFunction() const { return fF; }
   ERegularization GetRegularization() const { return fReg; }
   Scalar_t GetWeightDecay() const { return fWeightDecay; }
};

//
//
//  The Dense Layer Class - Implementation
//______________________________________________________________________________
template <typename Architecture_t>
TDenseLayer<Architecture_t>::TDenseLayer(size_t batchSize, size_t inputWidth, size_t width, EInitialization init,
                                         Scalar_t dropoutProbability, EActivationFunction f, ERegularization reg,
                                         Scalar_t weightDecay)
   : VGeneralLayer<Architecture_t>(batchSize, 1, 1, inputWidth, 1, 1, width, 1, width, inputWidth, 1, width, 1, 1,
                                   batchSize, width, init),
     fDerivatives(), fDropoutProbability(dropoutProbability), fF(f), fReg(reg), fWeightDecay(weightDecay)
{
   fDerivatives.emplace_back(batchSize, width);
}

//______________________________________________________________________________
template <typename Architecture_t>
TDenseLayer<Architecture_t>::TDenseLayer(TDenseLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer), fDerivatives(), fDropoutProbability(layer->GetDropoutProbability()),
     fF(layer->GetActivationFunction()), fReg(layer->GetRegularization()), fWeightDecay(layer->GetWeightDecay())
{
   fDerivatives.emplace_back(layer->GetBatchSize(), layer->GetWidth());
}

//______________________________________________________________________________
template <typename Architecture_t>
TDenseLayer<Architecture_t>::TDenseLayer(const TDenseLayer &layer)
   : VGeneralLayer<Architecture_t>(layer), fDropoutProbability(layer.fDropoutProbability), fDerivatives(), fF(layer.fF),
     fReg(layer.fReg), fWeightDecay(layer.fWeightDecay)
{
   fDerivatives.emplace_back(layer.fBatchSize, layer.fWidth);
}

//______________________________________________________________________________
template <typename Architecture_t>
TDenseLayer<Architecture_t>::~TDenseLayer()
{
   // Nothing to do here.
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TDenseLayer<Architecture_t>::Forward(std::vector<Matrix_t> input, bool applyDropout) -> void
{
   if (applyDropout && (this->GetDropoutProbability() != 1.0)) {
      Architecture_t::Dropout(input[0], this->GetDropoutProbability());
   }

   Architecture_t::MultiplyTranspose(this->GetOutputAt(0), input[0], this->GetWeightsAt(0));
   Architecture_t::AddRowWise(this->GetOutputAt(0), this->GetBiasesAt(0));
   evaluateDerivative<Architecture_t>(this->GetDerivativesAt(0), this->GetActivationFunction(), this->GetOutputAt(0));
   evaluate<Architecture_t>(this->GetOutputAt(0), this->GetActivationFunction());
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TDenseLayer<Architecture_t>::Backward(std::vector<Matrix_t> &gradients_backward,
                                           const std::vector<Matrix_t> &activations_backward,
                                           std::vector<Matrix_t> &inp1, std::vector<Matrix_t> &inp2) -> void
{
   Architecture_t::Backward(gradients_backward[0], this->GetWeightGradientsAt(0), this->GetBiasGradientsAt(0),
                            this->GetDerivativesAt(0), this->GetActivationGradientsAt(0), this->GetWeightsAt(0),
                            activations_backward[0]);

   addRegularizationGradients<Architecture_t>(this->GetWeightGradientsAt(0), this->GetWeightsAt(0),
                                              this->GetWeightDecay(), this->GetRegularization());
}

//______________________________________________________________________________
template <typename Architecture_t>
void TDenseLayer<Architecture_t>::Print() const
{
   std::cout << "Rows = " << this->GetWeightsAt(0).GetNrows();
   std::cout << "Cols = " << this->GetWeightsAt(0).GetNcols();
   std::cout << ", Activation Function = ";
   std::cout << static_cast<int>(fF) << std::endl;
}
} // namespace DNN
} // namespace TMVA

#endif
