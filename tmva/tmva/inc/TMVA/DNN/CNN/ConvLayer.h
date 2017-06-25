// @(#)root/tmva/tmva/dnn/cnn:$Id$
// Author: Vladimir Ilievski 04/06/17

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////////
// Contains ConvLayer class that represents the convolutional layer in //
// the Convolutional Neural Networks                                   //
/////////////////////////////////////////////////////////////////////////

#ifndef TMVA_CNN_CONVLAYER
#define TMVA_CNN_CONVLAYER

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"
#include "CNNLayer.h"

#include <cmath>
#include <iostream>


namespace TMVA
{
namespace DNN
{
namespace CNN
{

/** \class TConvLayer
    
    Generic Convolutional Layer class.
    
    This generic Convolutional Layer class represents a convolutional layer of
    a CNN. It inherits all of the properties of the generic virtual base class
    TCNNLayer. In addition to that, it contains an activation function.
 
*/
template<typename Architecture_t>
   class TConvLayer : public VCNNLayer<Architecture_t>
{

public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;
    

private:
   EActivationFunction fF;      ///< Activation function of the layer.
    
public:

   /*! Constructor. */
    TConvLayer(size_t BatchSize,
               size_t InputDepth,
               size_t InputHeight,
               size_t InputWidth,
               size_t Depth,
               size_t Height,
               size_t Width,
               Scalar_t DropoutProbability,
               size_t WeightsNRows,
               size_t WeightsNCols,
               size_t BiasesNRows,
               size_t BiasesNCols,
               size_t OutputNSlices,
               size_t OutputNRows,
               size_t OutputNCols,
               size_t FilterHeight,
               size_t FilterWidth,
               size_t StrideRows,
               size_t StrideCols,
               size_t ZeroPaddingHeight,
               size_t ZeroPaddingWidth,
               EActivationFunction f);

   /*! Copy constructor. */
   TConvLayer(const TConvLayer &);
    
   /*! Destructor. */
   ~TConvLayer();
    
    
   /*! Computes activation of the layer for the given input. The input
    * must be in tensor form with the different matrices corresponding to
    * different events in the batch. Computes activations as well as
    * the first partial derivative of the activation function at those
    * activations. */
   void Forward(std::vector<Matrix_t> input,
                bool applyDropout);
    

   /*! Compute weight, bias and activation gradients. Uses the precomputed
    *  first partial derviatives of the activation function computed during
    *  forward propagation and modifies them. Must only be called directly
    *  at the corresponding call to Forward(...). */
   void Backward(std::vector<Matrix_t> &gradients_backward,
                 const std::vector<Matrix_t> &activations_backward,
                 ERegularization r,
                 Scalar_t weightDecay);

   /*! Prints the info about the layer. */
   void Print() const;
  
   /** Getters */
   EActivationFunction GetActivationFunction() const {return fF;}
};
    

//______________________________________________________________________________
template<typename Architecture_t>
    TConvLayer<Architecture_t>::TConvLayer(size_t batchSize,
                                           size_t inputDepth,
                                           size_t inputHeight,
                                           size_t inputWidth,
                                           size_t depth,
                                           size_t height,
                                           size_t width,
                                           Scalar_t dropoutProbability,
                                           size_t weightsNRows,
                                           size_t weightsNCols,
                                           size_t biasesNRows,
                                           size_t biasesNCols,
                                           size_t outputNSlices,
                                           size_t outputNRows,
                                           size_t outputNCols,
                                           size_t filterHeight,
                                           size_t filterWidth,
                                           size_t strideRows,
                                           size_t strideCols,
                                           size_t zeroPaddingHeight,
                                           size_t zeroPaddingWidth,
                                           EActivationFunction f)
    
   : VCNNLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, depth,
                               height, width, dropoutProbability, weightsNRows, weightsNCols,
                               biasesNRows, biasesNCols, outputNSlices, outputNRows,
                               outputNCols, inputDepth, filterHeight, filterWidth,
                               strideRows, strideCols, zeroPaddingHeight, zeroPaddingWidth), fF(f)
{
   // Nothing to do here.
}
    
//______________________________________________________________________________
template<typename Architecture_t>
   TConvLayer<Architecture_t>::TConvLayer(const TConvLayer &convLayer)
   : VCNNLayer<Architecture_t>(convLayer), fF(convLayer.fF)
{
    
}

//______________________________________________________________________________
template<typename Architecture_t>
  TConvLayer<Architecture_t>::~TConvLayer()
{
        
}


//______________________________________________________________________________
template<typename Architecture_t>
   auto TConvLayer<Architecture_t>::Forward(std::vector<Matrix_t> input,
                                            bool applyDropout)
-> void
{
   for(size_t i = 0; i < this -> GetBatchSize(); i++) {
       
      if (applyDropout && (this -> GetDropoutProbability() != 1.0)) {
         Architecture_t::Dropout(input[i], this -> GetDropoutProbability());
      }
      
      Matrix_t inputTr(this -> GetNLocalViews(), this -> GetNLocalViewPixels());
      Architecture_t::Im2col(inputTr, input[i], this -> GetInputHeight(),
                             this -> GetInputWidth(), this -> GetFilterHeight(),
                             this -> GetFilterWidth(), this -> GetStrideRows(),
                             this -> GetStrideCols(), this -> GetZeroPaddingHeight(),
                             this -> GetZeroPaddingWidth());
      
      Architecture_t::MultiplyTranspose(this -> GetOutputAt(i), this -> GetWeights(),
                                        inputTr);
      Architecture_t::AddConvBiases(this -> GetOutputAt(i), this -> GetBiases());
     
      evaluateDerivative<Architecture_t>(this -> GetDerivativesAt(i), fF, this -> GetOutputAt(i));
      evaluate<Architecture_t>(this -> GetOutputAt(i), fF);
   }
}

//______________________________________________________________________________
template<typename Architecture_t>
   auto TConvLayer<Architecture_t>::Backward(std::vector<Matrix_t> &gradients_backward,
                                             const std::vector<Matrix_t> &activations_backward,
                                             ERegularization r,
                                             Scalar_t weightDecay)
-> void
{
 
   Architecture_t::ConvLayerBackward(gradients_backward, this -> GetWeightGradients(),
                                     this -> GetBiasGradients(), this -> GetDerivatives(),
                                     this -> GetActivationGradients(), this -> GetWeights(),
                                     activations_backward, this -> GetBatchSize(),
                                     this -> GetInputHeight(), this -> GetInputWidth(),
                                     this -> GetDepth(), this -> GetHeight(), this -> GetWidth(),
                                     this -> GetFilterDepth(), this -> GetFilterHeight(),
                                     this -> GetFilterWidth(), this -> GetNLocalViews());
    
   addRegularizationGradients<Architecture_t>(this -> GetWeightGradients(), this -> GetWeights(),
                                              weightDecay, r);
}

//______________________________________________________________________________
template<typename Architecture_t>
   auto TConvLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "\t\t CONV LAYER: " << std::endl;
   std::cout << "\t\t\t Width = " <<  this -> GetWidth() << std::endl;
   std::cout << "\t\t\t Height = " << this -> GetHeight() << std::endl;
   std::cout << "\t\t\t Depth = " << this -> GetDepth() << std::endl;
    
   std::cout << "\t\t\t Filter Width = " << this -> GetFilterWidth() << std::endl;
   std::cout << "\t\t\t Filter Height = " << this -> GetFilterHeight() << std::endl;
   std::cout << "\t\t\t Activation Function = " << static_cast<int>(fF) << std::endl;
}

} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif
