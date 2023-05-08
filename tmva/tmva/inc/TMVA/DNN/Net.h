// @(#)root/tmva: $Id$
// Author: Simon Pfreundschuh 20/06/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_DNN_NET
#define TMVA_DNN_NET

#include <vector>
#include <iostream>

#include "Layer.h"

namespace TMVA {
namespace DNN  {

/** \class TNet

    Generic neural network class.

    This generic neural network class represents a concrete neural
    network through a vector of layers and coordinates the forward
    and backward propagation through the net.

    The net takes as input a batch from the training data given in
    matrix form, with each row corresponding to a certain training
    event.

    On construction, the neural network allocates all the memory
    required for the training of the neural net and keeps it until
    its destruction.

    The Architecture type argument simply holds the
    architecture-specific data types, which are just the matrix type
    Matrix_t and the used scalar type Scalar_t.

    \tparam Architecture The Architecture type that holds the
    \tparam Layer_t The type used for the layers. Can be either
    Layer<Architecture> or SharedWeightLayer<Architecture>.
    datatypes for a given architecture.
*/
template<typename Architecture_t, typename Layer_t = TLayer<Architecture_t>>
   class TNet {

public:
   using Matrix_t         = typename Architecture_t::Matrix_t;
   using Scalar_t         = typename Architecture_t::Scalar_t;
   using LayerIterator_t  = typename std::vector<Layer_t>::iterator;

private:
   size_t fBatchSize;  ///< Batch size for training and evaluation of the Network.
   size_t fInputWidth; ///< Number of features in a single input event.

   std::vector<Layer_t> fLayers; ///< Layers in the network.

   Matrix_t fDummy;       ///< Empty matrix for last step in back propagation.
   ELossFunction fJ;      ///< The loss function of the network.
   ERegularization fR;    ///< The regularization used for the network.
   Scalar_t fWeightDecay; ///< The weight decay factor.

public:
   TNet();
   TNet(const TNet & other);
   template<typename OtherArchitecture_t>
   TNet(size_t batchSize, const TNet<OtherArchitecture_t> &);
   /*! Construct a neural net for a given batch size with
    *  given output function * and regularization. */
   TNet(size_t batchSize,
        size_t inputWidth,
        ELossFunction fJ,
        ERegularization fR = ERegularization::kNone,
        Scalar_t fWeightDecay = 0.0);
   /*! Create a clone that uses the same weight and biases matrices but
    *  potentially a difference batch size. */
   TNet<Architecture_t, TSharedLayer<Architecture_t>> CreateClone(size_t batchSize);

   /*! Add a layer of the given size to the neural net. */
   void AddLayer(size_t width, EActivationFunction f,
                 Scalar_t dropoutProbability = 1.0);

   /*! Remove all layers from the network.*/
   void Clear();

   /*! Add a layer which shares its weights with another TNet instance. */
   template <typename SharedLayer>
   void AddLayer(SharedLayer & layer);

   /*! Iterator to the first layer of the net. */
   LayerIterator_t LayersBegin() {return fLayers;}

   /*! Iterator to the last layer of the net. */
   LayerIterator_t LayersEnd() {return fLayers;}

   /*! Initialize the weights in the net with the
    *  initialization method. */
   inline void Initialize(EInitialization m);

   /*! Initialize the gradients in the net to zero. Required if net is
    *  used to store velocities of momentum-based minimization techniques. */
   inline void InitializeGradients();

   /*! Forward a given input through the neural net. Computes
    *  all layer activations up to the output layer */
   inline void Forward(Matrix_t& X, bool applyDropout = false);

   /*! Compute the weight gradients in the net from the given training
    * samples X and training labels Y. */
   inline void Backward(const Matrix_t &X, const Matrix_t &Y, const Matrix_t &weights);

   /*! Evaluate the loss function of the net using the activations
    *  that are currently stored in the output layer. */
   inline Scalar_t Loss(const Matrix_t &Y, const Matrix_t &weights, bool includeRegularization = true) const;

   /*! Propagate the input batch X through the net and evaluate the
    *  error function for the resulting activations of the output
    *  layer */
   inline Scalar_t Loss(Matrix_t &X, const Matrix_t &Y, const Matrix_t &weights, bool applyDropout = false,
                        bool includeRegularization = true);

   /*! Compute the neural network prediction obtained from forwarding the
    *  batch X through the neural network and applying the output function
    *  f to the activation of the last layer in the network. */
   inline void Prediction(Matrix_t &Y_hat, Matrix_t &X, EOutputFunction f);

   /*! Compute the neural network prediction obtained from applying the output
    * function f to the activation of the last layer in the network. */
   inline void Prediction(Matrix_t &Y_hat, EOutputFunction f) const;

   Scalar_t            GetNFlops();

   size_t              GetDepth() const          {return fLayers.size();}
   size_t              GetBatchSize() const      {return fBatchSize;}
   Layer_t &           GetLayer(size_t i)        {return fLayers[i];}
   const Layer_t &     GetLayer(size_t i) const  {return fLayers[i];}
   ELossFunction       GetLossFunction() const   {return fJ;}
   Matrix_t &          GetOutput()               {return fLayers.back().GetOutput();}
   size_t              GetInputWidth() const     {return fInputWidth;}
   size_t              GetOutputWidth() const    {return fLayers.back().GetWidth();}
   ERegularization     GetRegularization() const {return fR;}
   Scalar_t            GetWeightDecay() const    {return fWeightDecay;}

   void SetBatchSize(size_t batchSize)       {fBatchSize = batchSize;}
   void SetInputWidth(size_t inputWidth)     {fInputWidth = inputWidth;}
   void SetRegularization(ERegularization R) {fR = R;}
   void SetLossFunction(ELossFunction J)     {fJ = J;}
   void SetWeightDecay(Scalar_t weightDecay) {fWeightDecay = weightDecay;}
   void SetDropoutProbabilities(const std::vector<Double_t> & probabilities);

   void Print();
};

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   TNet<Architecture_t, Layer_t>::TNet()
    : fBatchSize(0), fInputWidth(0), fLayers(), fDummy(0,0),
    fJ(ELossFunction::kMeanSquaredError), fR(ERegularization::kNone),
    fWeightDecay(0.0)
{
   // Nothing to do here.
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   TNet<Architecture_t, Layer_t>::TNet(const TNet & other)
   : fBatchSize(other.fBatchSize), fInputWidth(other.fInputWidth),
    fLayers(other.fLayers), fDummy(0,0), fJ(other.fJ), fR(other.fR),
    fWeightDecay(other.fWeightDecay)
{
   // Nothing to do here.
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
template<typename OtherArchitecture_t>
TNet<Architecture_t, Layer_t>::TNet(size_t batchSize,
                                    const TNet<OtherArchitecture_t> & other)
    : fBatchSize(batchSize), fInputWidth(other.GetInputWidth()), fLayers(),
    fDummy(0,0), fJ(other.GetLossFunction()), fR(other.GetRegularization()),
    fWeightDecay(other.GetWeightDecay())
{
   fLayers.reserve(other.GetDepth());
   for (size_t i = 0; i < other.GetDepth(); i++) {
      AddLayer(other.GetLayer(i).GetWidth(),
               other.GetLayer(i).GetActivationFunction(),
               other.GetLayer(i).GetDropoutProbability());
      fLayers[i].GetWeights() = (TMatrixT<Scalar_t>) other.GetLayer(i).GetWeights();
      fLayers[i].GetBiases()  = (TMatrixT<Scalar_t>) other.GetLayer(i).GetBiases();
   }
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   TNet<Architecture_t, Layer_t>::TNet(size_t        batchSize,
                                       size_t        inputWidth,
                                       ELossFunction J,
                                       ERegularization R,
                                       Scalar_t weightDecay)
    : fBatchSize(batchSize), fInputWidth(inputWidth), fLayers(), fDummy(0,0),
    fJ(J), fR(R), fWeightDecay(weightDecay)
{
   // Nothing to do here.
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TNet<Architecture_t, Layer_t>::CreateClone(size_t BatchSize)
   -> TNet<Architecture_t, TSharedLayer<Architecture_t>>
{
   TNet<Architecture_t, TSharedLayer<Architecture_t>> other(BatchSize, fInputWidth,
                                                            fJ, fR);
   for (auto &l : fLayers) {
      other.AddLayer(l);
   }
   return other;
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   void TNet<Architecture_t, Layer_t>::AddLayer(size_t width,
                                                EActivationFunction f,
                                                Scalar_t dropoutProbability)
{
   if (fLayers.size() == 0) {
      fLayers.emplace_back(fBatchSize, fInputWidth, width, f, dropoutProbability);
   } else {
      size_t prevWidth = fLayers.back().GetWidth();
      fLayers.emplace_back(fBatchSize, prevWidth, width, f, dropoutProbability);
   }
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   void TNet<Architecture_t, Layer_t>::Clear()
{
   fLayers.clear();
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   template<typename SharedLayer_t>
   inline void TNet<Architecture_t, Layer_t>::AddLayer(SharedLayer_t & layer)
{
   fLayers.emplace_back(fBatchSize, layer);
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   inline void TNet<Architecture_t, Layer_t>::Initialize(EInitialization m)
{
   for (auto &l : fLayers) {
      l.Initialize(m);
   }
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   inline void TNet<Architecture_t, Layer_t>::InitializeGradients()
{
   for (auto &l : fLayers) {
      initialize<Architecture_t>(l.GetWeightGradients(), EInitialization::kZero);
      initialize<Architecture_t>(l.GetBiasGradients(),   EInitialization::kZero);
   }
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
inline void TNet<Architecture_t, Layer_t>::Forward(Matrix_t &input,
                                                   bool applyDropout)
{
   fLayers.front().Forward(input, applyDropout);

   for (size_t i = 1; i < fLayers.size(); i++) {
      fLayers[i].Forward(fLayers[i-1].GetOutput(), applyDropout);
   }
}

//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
inline void TNet<Architecture_t, Layer_t>::Backward(const Matrix_t &X, const Matrix_t &Y, const Matrix_t &weights)
{

   evaluateGradients<Architecture_t>(fLayers.back().GetActivationGradients(), fJ, Y, fLayers.back().GetOutput(),
                                     weights);

   for (size_t i = fLayers.size()-1; i > 0; i--) {
      auto & activation_gradient_backward
         = fLayers[i-1].GetActivationGradients();
      auto & activations_backward
         = fLayers[i-1].GetOutput();
      fLayers[i].Backward(activation_gradient_backward,
                          activations_backward, fR, fWeightDecay);
   }
   fLayers[0].Backward(fDummy, X, fR, fWeightDecay);

}

//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
inline auto TNet<Architecture_t, Layer_t>::Loss(const Matrix_t &Y, const Matrix_t &weights,
                                                bool includeRegularization) const -> Scalar_t
{
   auto loss = evaluate<Architecture_t>(fJ, Y, fLayers.back().GetOutput(), weights);
   includeRegularization &= (fR != ERegularization::kNone);
   if (includeRegularization) {
      for (auto &l : fLayers) {
         loss += fWeightDecay * regularization<Architecture_t>(l.GetWeights(), fR);
      }
   }
   return loss;
}

//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
inline auto TNet<Architecture_t, Layer_t>::Loss(Matrix_t &X, const Matrix_t &Y, const Matrix_t &weights,
                                                bool applyDropout, bool includeRegularization) -> Scalar_t
{
   Forward(X, applyDropout);
   return Loss(Y, weights, includeRegularization);
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   inline void TNet<Architecture_t, Layer_t>::Prediction(Matrix_t &Yhat,
                                                         Matrix_t &X,
                                                         EOutputFunction f)
{
   Forward(X, false);
   evaluate<Architecture_t>(Yhat, f, fLayers.back().GetOutput());
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   inline void TNet<Architecture_t, Layer_t>::Prediction(Matrix_t &Y_hat,
                                                         EOutputFunction f) const
{
   evaluate<Architecture_t>(Y_hat, f, fLayers.back().GetOutput());
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
auto TNet<Architecture_t, Layer_t>::GetNFlops()
   -> Scalar_t
{
   Scalar_t flops = 0;

   Scalar_t nb  = (Scalar_t) fBatchSize;
   Scalar_t nlp = (Scalar_t) fInputWidth;

   for(size_t i = 0; i < fLayers.size(); i++) {
      Layer_t & layer = fLayers[i];
      Scalar_t nl = (Scalar_t) layer.GetWidth();

      // Forward propagation.
      flops += nb * nl * (2.0 * nlp - 1); // Matrix mult.
      flops += nb * nl;                   // Add bias values.
      flops += 2 * nb * nl;               // Apply activation function and compute
                                          // derivative.
      // Backward propagation.
      flops += nb * nl;                      // Hadamard
      flops += nlp * nl * (2.0 * nb - 1.0);  // Weight gradients
      flops += nl * (nb - 1);                // Bias gradients
      if (i > 0) {
         flops += nlp * nb * (2.0 * nl  - 1.0); // Previous layer gradients.
      }
      nlp = nl;
   }
   return flops;
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
void TNet<Architecture_t, Layer_t>::SetDropoutProbabilities(
    const std::vector<Double_t> & probabilities)
{
   for (size_t i = 0; i < fLayers.size(); i++) {
      if (i < probabilities.size()) {
         fLayers[i].SetDropoutProbability(probabilities[i]);
      } else {
         fLayers[i].SetDropoutProbability(1.0);
      }
   }
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   void TNet<Architecture_t, Layer_t>::Print()
{
   std::cout << "DEEP NEURAL NETWORK:";
   std::cout << " Loss function = " << static_cast<char>(fJ);
   std::cout << ", Depth = " << fLayers.size() << std::endl;

   size_t i = 1;
   for (auto & l : fLayers) {
      std::cout << "DNN Layer " << i << ":" << std::endl;
      l.Print();
      i++;
   }

}

} // namespace DNN
} // namespace TMVA

#endif
