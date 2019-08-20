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

#ifndef TMVA_DNN_LAYER
#define TMVA_DNN_LAYER

#include <iostream>

#include "TMatrix.h"
#include "Functions.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
//
//  The Layer Class
//______________________________________________________________________________

/** \class TLayer

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
   class TLayer
{

public:
   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Tensor_t = typename Architecture_t::Tensor_t;


private:

   size_t fBatchSize;  ///< Batch size used for training and evaluation.
   size_t fInputWidth; ///< Number of neurons of the previous layer.
   size_t fWidth;      ///< Number of neurons of this layer.

   Scalar_t fDropoutProbability;  ///< Probability that an input is active.

   Matrix_t fWeights;             ///< The fWeights of this layer.
   Matrix_t fBiases;              ///< The bias values of this layer.
   Matrix_t fOutput;              ///< Activations of this layer.
   Matrix_t fDerivatives;         ///< First fDerivatives of the activations of this layer.
   Matrix_t fWeightGradients;     ///< Gradients w.r.t. the weigths of this layer.
   Matrix_t fBiasGradients;       ///< Gradients w.r.t. the bias values of this layer.
   Matrix_t fActivationGradients; ///< Gradients w.r.t. the activations of this layer.

   EActivationFunction fF; ///< Activation function of the layer.

public:

   TLayer(size_t             BatchSize,
          size_t             InputWidth,
          size_t             Width,
          EActivationFunction f,
          Scalar_t           dropoutProbability);
   TLayer(const TLayer &);

   /*! Initialize fWeights according to the given initialization
    *  method. */
   void Initialize(EInitialization m);
   /*! Compute activation of the layer for the given input. The input
    * must be in matrix form with the different rows corresponding to
    * different events in the batch. Computes activations as well as
    * the first partial derivative of the activation function at those
    * activations. */
   void inline Forward(Matrix_t & input, bool applyDropout = false);
   /*! Compute weight, bias and activation gradients. Uses the precomputed
    *  first partial derviatives of the activation function computed during
    *  forward propagation and modifies them. Must only be called directly
    *  a the corresponding call to Forward(...). */
   void inline Backward(Matrix_t & gradients_backward,
                        const Matrix_t & activations_backward,
                        ERegularization r,
                        Scalar_t weightDecay);

   void Print() const;

   size_t GetBatchSize()          const {return fBatchSize;}
   size_t GetInputWidth()         const {return fInputWidth;}
   size_t GetWidth()              const {return fWidth;}
   size_t GetDropoutProbability() const {return fDropoutProbability;}

   void SetDropoutProbability(Scalar_t p) {fDropoutProbability = p;}

   EActivationFunction GetActivationFunction() const {return fF;}

   Matrix_t       & GetOutput()        {return fOutput;}
   const Matrix_t & GetOutput() const  {return fOutput;}
   Matrix_t       & GetWeights()       {return fWeights;}
   const Matrix_t & GetWeights() const {return fWeights;}
   Matrix_t       & GetBiases()       {return fBiases;}
   const Matrix_t & GetBiases() const {return fBiases;}
   Matrix_t       & GetActivationGradients()       {return fActivationGradients;}
   const Matrix_t & GetActivationGradients() const {return fActivationGradients;}
   Matrix_t       & GetBiasGradients()       {return fBiasGradients;}
   const Matrix_t & GetBiasGradients() const {return fBiasGradients;}
   Matrix_t       & GetWeightGradients()       {return fWeightGradients;}
   const Matrix_t & GetWeightGradients() const {return fWeightGradients;}

};

//______________________________________________________________________________
//
//  The Shared Layer Class
//______________________________________________________________________________

/** \class TSharedLayer

    Layer class width shared weight and bias layers.

    Like the Layer class only that weight matrices are shared between
    different instances of the net, which can be used to implement
    multithreading 'Hogwild' style.
*/

template<typename Architecture_t>
class TSharedLayer
{

public:

   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Tensor_t = typename Architecture_t::Tensor_t;


private:

   size_t fBatchSize;  ///< Batch size used for training and evaluation.
   size_t fInputWidth; ///< Number of neurons of the previous layer.
   size_t fWidth;      ///< Number of neurons of this layer.

   Scalar_t fDropoutProbability;  ///< Probability that an input is active.

   Matrix_t & fWeights;           ///< Reference to the weight matrix of this layer.
   Matrix_t & fBiases;            ///< Reference to the bias vectors of this layer.
   Matrix_t fOutput;              ///< Activations of this layer.
   Matrix_t fDerivatives;         ///< First fDerivatives of the activations of this layer.
   Matrix_t fWeightGradients;     ///< Gradients w.r.t. the weigths of this layer.
   Matrix_t fBiasGradients;       ///< Gradients w.r.t. the bias values of this layer.
   Matrix_t fActivationGradients; ///< Gradients w.r.t. the activations of this layer.

   EActivationFunction fF; ///< Activation function of the layer.

public:

   TSharedLayer(size_t fBatchSize,
                TLayer<Architecture_t> & layer);
   TSharedLayer(const TSharedLayer & layer);

   /*! Compute activation of the layer for the given input. The input
    * must be in matrix form with the different rows corresponding to
    * different events in the batch. Computes activations as well as
    * the first partial derivative of the activation function at those
    * activations. */
   void inline Forward(Matrix_t & input, bool applyDropout = false);
   /*! Compute weight, bias and activation gradients. Uses the precomputed
    *  first partial derviatives of the activation function computed during
    *  forward propagation and modifies them. Must only be called directly
    *  a the corresponding call to Forward(...). */
   void inline Backward(Matrix_t & gradients_backward,
                        const Matrix_t & activations_backward,
                        ERegularization r,
                        Scalar_t weightDecay);

   void Print() const;

   size_t GetBatchSize()          const {return fBatchSize;}
   size_t GetInputWidth()         const {return fInputWidth;}
   size_t GetWidth()              const {return fWidth;}
   size_t GetDropoutProbability() const {return fDropoutProbability;}

   void SetDropoutProbability(Scalar_t p) {fDropoutProbability = p;}

   EActivationFunction GetActivationFunction() const {return fF;}

   Matrix_t       & GetOutput()        {return fOutput;}
   const Matrix_t & GetOutput() const  {return fOutput;}
   Matrix_t       & GetWeights() const {return fWeights;}
   Matrix_t       & GetBiases()       {return fBiases;}
   const Matrix_t & GetBiases() const {return fBiases;}
   Matrix_t       & GetActivationGradients()       {return fActivationGradients;}
   const Matrix_t & GetActivationGradients() const {return fActivationGradients;}
   Matrix_t       & GetBiasGradients()       {return fBiasGradients;}
   const Matrix_t & GetBiasGradients() const {return fBiasGradients;}
   Matrix_t       & GetWeightGradients()       {return fWeightGradients;}
   const Matrix_t & GetWeightGradients() const {return fWeightGradients;}

};

//______________________________________________________________________________
//
//  The Layer Class - Implementation
//______________________________________________________________________________

template<typename Architecture_t>
   TLayer<Architecture_t>::TLayer(size_t batchSize,
                                  size_t inputWidth,
                                  size_t width,
                                  EActivationFunction f,
                                  Scalar_t dropoutProbability)
   : fBatchSize(batchSize), fInputWidth(inputWidth), fWidth(width),
     fDropoutProbability(dropoutProbability), fWeights(width, fInputWidth),
     fBiases(width, 1), fOutput(fBatchSize, width), fDerivatives(fBatchSize, width),
     fWeightGradients(width, fInputWidth), fBiasGradients(width, 1),
     fActivationGradients(fBatchSize, width), fF(f)
{
   // Nothing to do here.
}

//______________________________________________________________________________
template<typename Architecture_t>
TLayer<Architecture_t>::TLayer(const TLayer &layer)
    : fBatchSize(layer.fBatchSize), fInputWidth(layer.fInputWidth),
    fWidth(layer.fWidth), fDropoutProbability(layer.fDropoutProbability),
    fWeights(layer.fWidth, layer.fInputWidth), fBiases(layer.fWidth, 1),
    fOutput(layer.fBatchSize, layer.fWidth),
    fDerivatives(layer.fBatchSize, layer.fWidth),
    fWeightGradients(layer.fWidth, layer.fInputWidth),
    fBiasGradients(layer.fWidth, 1),
    fActivationGradients(layer.fBatchSize, layer.fWidth),
    fF(layer.fF)
{
   Architecture_t::Copy(fWeights, layer.GetWeights());
   Architecture_t::Copy(fBiases,  layer.GetBiases());
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TLayer<Architecture_t>::Initialize(EInitialization m)
-> void
{
   initialize<Architecture_t>(fWeights, m);
   initialize<Architecture_t>(fBiases,  EInitialization::kZero);
}

//______________________________________________________________________________
template<typename Architecture_t>
auto inline TLayer<Architecture_t>::Forward(Matrix_t & input,
                                            bool applyDropout)
-> void
{
   if (applyDropout && (fDropoutProbability != 1.0)) {
      Architecture_t::Dropout(input, fDropoutProbability);
   }
   Architecture_t::MultiplyTranspose(fOutput, input, fWeights);
   Architecture_t::AddRowWise(fOutput, fBiases);
   Tensor_t tOutput(fOutput); 
   Tensor_t tDerivatives(fDerivatives); 
   evaluateDerivative<Architecture_t>(tDerivatives, fF, tOutput);
  
   evaluate<Architecture_t>(tOutput, fF);
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TLayer<Architecture_t>::Backward(Matrix_t & gradients_backward,
                                    const Matrix_t & activations_backward,
                                    ERegularization r,
                                    Scalar_t weightDecay)
-> void
{

   Tensor_t tGradBw(gradients_backward);
   Tensor_t tActBw(activations_backward);
   Tensor_t tActGrad(fActivationGradients);
   Tensor_t tDeriv(fDerivatives);

   Architecture_t::Backward( tGradBw,
                            fWeightGradients,
                            fBiasGradients,
                            tDeriv,
                            tActGrad,
                            fWeights,
                            tActBw);
   addRegularizationGradients<Architecture_t>(fWeightGradients,
                                              fWeights,
                                              weightDecay, r);
}

//______________________________________________________________________________
template<typename Architecture_t>
   void TLayer<Architecture_t>::Print() const
{
   std::cout << "Width = " << fWeights.GetNrows();
   std::cout << ", Activation Function = ";
   std::cout << static_cast<int>(fF) << std::endl;
}

//______________________________________________________________________________
//
//  The Shared Layer Class - Implementation
//______________________________________________________________________________

//______________________________________________________________________________
template<typename Architecture_t>
TSharedLayer<Architecture_t>::TSharedLayer(size_t BatchSize,
                                         TLayer<Architecture_t> &layer)
: fBatchSize(BatchSize),
fInputWidth(layer.GetInputWidth()), fWidth(layer.GetWidth()),
fDropoutProbability(layer.GetDropoutProbability()),
fWeights(layer.GetWeights()), fBiases(layer.GetBiases()),
fOutput(fBatchSize, fWidth), fDerivatives(fBatchSize, fWidth),
fWeightGradients(fWidth, fInputWidth), fBiasGradients(fWidth, 1),
fActivationGradients(fBatchSize, fWidth), fF(layer.GetActivationFunction())
{
   // Nothing to do here.
}

//______________________________________________________________________________
template<typename Architecture_t>
TSharedLayer<Architecture_t>::TSharedLayer(const TSharedLayer &layer)
    : fBatchSize(layer.fBatchSize),
    fInputWidth(layer.GetInputWidth()), fWidth(layer.GetWidth()),
    fDropoutProbability(layer.fDropoutProbability), fWeights(layer.fWeights),
    fBiases(layer.fBiases), fOutput(layer.fBatchSize, fWidth),
    fDerivatives(layer.fBatchSize, fWidth), fWeightGradients(fWidth, fInputWidth),
    fBiasGradients(fWidth, 1), fActivationGradients(layer.fBatchSize, fWidth),
    fF(layer.fF)
{
}

//______________________________________________________________________________
template<typename Architecture_t>
auto inline TSharedLayer<Architecture_t>::Forward(Matrix_t & input,
                                                  bool applyDropout)
-> void
{
   if (applyDropout && (fDropoutProbability != 1.0)) {
      Architecture_t::Dropout(input, fDropoutProbability);
   }
   Architecture_t::MultiplyTranspose(fOutput, input, fWeights);
   Architecture_t::AddRowWise(fOutput, fBiases);
   Tensor_t tOutput(fOutput); 
   Tensor_t tDerivatives(fDerivatives); 
   evaluateDerivative<Architecture_t>(tDerivatives, fF, tOutput);
   evaluate<Architecture_t>(tOutput, fF);
}

//______________________________________________________________________________
template<typename Architecture_t>
auto inline TSharedLayer<Architecture_t>::Backward(Matrix_t & gradients_backward,
                                                 const Matrix_t & activations_backward,
                                                 ERegularization r,
                                                 Scalar_t weightDecay)
-> void
{
   Architecture_t::Backward(gradients_backward,
                            fWeightGradients,
                            fBiasGradients,
                            fDerivatives,
                            fActivationGradients,
                            fWeights,
                            activations_backward);
   addRegularizationGradients<Architecture_t>(fWeightGradients,
                                              fWeights,
                                              weightDecay, r);
}

//______________________________________________________________________________
template<typename Architecture_t>
void TSharedLayer<Architecture_t>::Print() const
{
   std::cout << "Width = " << fWeights.GetNrows();
   std::cout << ", Activation Function = ";
   std::cout << static_cast<int>(fF) << std::endl;
}

} // namespace DNN
} // namespace TMVA

#endif
