// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/06/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
// Contains Layer and SharedLayer classes, that represent layers in //
// neural networks.                                                 //
//////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_DENSELAYER
#define TMVA_DNN_DENSELAYER

#include <iostream>

#include "TMatrix.h"
#include "Functions.h"
#include "GeneralLayer.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
//
//  The Dense Layer Class
//______________________________________________________________________________

/** \class TDenseLayer

    Generic layer class.

    This generic layer class represents a layer of a neural network with
    a given width n and activation function f. The activation
    function of each layer is given by \f$\mathbf{u} =
    \mathbf{W}\mathbf{x} + \boldsymbol{\theta}\f$.

    In addition to the weight and bias matrices, each layer allocates memory
    for its activations and the corresponding first partial fDerivatives of
    the activation function as well as the gradients of the fWeights and fBiases.

    The layer provides member functions for the forward propagation of
    activations through the given layer.
*/
template<typename Architecture_t>
    class TDenseLayer : public VGeneralLayer<Architecture_t>
{

public:
   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;

private:
   EActivationFunction fF; ///< Activation function of the layer.

public:

   /*! Constructor */
   TDenseLayer(size_t BatchSize,
               size_t InputWidth,
               size_t Width,
               Scalar_t DropoutProbability,
               EActivationFunction f);
    
   /*! Copy Constructor */
   TDenseLayer(const TDenseLayer &);

   /*! Compute activation of the layer for the given input. The input
    * must be in matrix form with the different rows corresponding to
    * different events in the batch. Computes activations as well as
    * the first partial derivative of the activation function at those
    * activations. */
   void Forward(Matrix_t & input, bool applyDropout = false);
    
   /*! Compute weight, bias and activation gradients. Uses the precomputed
    *  first partial derviatives of the activation function computed during
    *  forward propagation and modifies them. Must only be called directly
    *  a the corresponding call to Forward(...). */
   void Backward(Matrix_t & gradients_backward,
                 const Matrix_t & activations_backward,
                 ERegularization r,
                 Scalar_t weightDecay);

   void Print() const;
    
   EActivationFunction GetActivationFunction() const {return fF;}

};


//______________________________________________________________________________
template<typename Architecture_t>
   TDenseLayer<Architecture_t>::TDenseLayer(size_t batchSize,
                                            size_t inputWidth,
                                            size_t width,
                                            Scalar_t dropoutProbability,
                                            EActivationFunction f)
   : VGeneralLayer<Architecture_t>(batchSize, 1, 1, inputWidth, 1, 1, width,
                                   dropoutProbability, width, inputWidth,
                                   width, 1, 1, batchSize, width) ,fF(f)
{
   // Nothing to do here.
}

//______________________________________________________________________________
template<typename Architecture_t>
TDenseLayer<Architecture_t>::TDenseLayer(const TDenseLayer &layer)
    : VGeneralLayer<Architecture_t>(layer), fF(layer.fF)
{
    
}

//______________________________________________________________________________
template<typename Architecture_t>
   auto inline TDenseLayer<Architecture_t>::Forward(Matrix_t & input,
                                                    bool applyDropout)
-> void
{
   if (applyDropout && (this -> GetDropoutProbability() != 1.0)) {
      Architecture_t::Dropout(input, this -> GetDropoutProbability());
   }
    
   Architecture_t::MultiplyTranspose(this -> GetOutputAt(0), input, this -> GetWeights());
   Architecture_t::AddRowWise(this -> GetOutputAt(0), this -> GetBiases());
   evaluateDerivative<Architecture_t>(this -> GetDerivativesAt(0), fF, this -> GetOutputAt(0));
   evaluate<Architecture_t>(this -> GetOutputAt(0), fF);
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TDenseLayer<Architecture_t>::Backward(Matrix_t & gradients_backward,
                                           const Matrix_t & activations_backward,
                                           ERegularization r,
                                           Scalar_t weightDecay)
-> void
{
   Architecture_t::Backward(gradients_backward,
                            this -> GetWeightGradients(),
                            this -> GetBiasGradients(),
                            this -> GetDerivativesAt(0),
                            this -> GetActivationGradientsAt(0),
                            this -> GetWeights(),
                            activations_backward);
    
   addRegularizationGradients<Architecture_t>(this -> GetWeightGradients(),
                                              this -> GetWeights(),
                                              weightDecay, r);
}

//______________________________________________________________________________
template<typename Architecture_t>
   void TDenseLayer<Architecture_t>::Print() const
{
   std::cout << "Width = " << fWeights.GetNrows();
   std::cout << ", Activation Function = ";
   std::cout << static_cast<int>(fF) << std::endl;
}

} // namespace DNN
} // namespace TMVA

#endif
