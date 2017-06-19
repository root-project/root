// @(#)root/tmva/tmva/dnn/cnn:$Id$
// Author: Vladimir Ilievski 03/06/2017

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
// Contains CNNLayer class that represents the base class for all layers//
// in the Convolutional Neural Networks.                                //
//////////////////////////////////////////////////////////////////////////


#ifndef CNNLAYER_H_
#define CNNLAYER_H_

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"

#include <iostream>

namespace TMVA
{
namespace DNN
{
namespace CNN
{

/** \class VCNNLayer
    
     Generic CNN layer virtual base class.
 
     This generic CNN layer virtual class represents a base class of the layers
     in a Convolutional Neural Network. It contains the its own depth, height
     and witdth as well as of the input and filters. It also contains the padding
     and zero padding in both spatial dimensions.
 
     In addition to the weight and bias matrices, each layer allocates memory
     for its activations and the corresponding first partial fDerivatives of
     the activation function as well as the gradients of the fWeights and fBiases.
 
     The class provides member functions for the initialization, and the forward
     and backward propagation of activations through the given layer.
 */
template<typename Architecture_t>
   class VCNNLayer
{

public:
    
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;
    
protected:
    
   size_t fBatchSize;                ///< Batch size used for training and evaluation.

   size_t fInputDepth;               ///< The depth of the previous layer.
   size_t fInputHeight;              ///< The height of the previous layer.
   size_t fInputWidth;               ///< The width of the previous layer.
    
   size_t fFilterDepth;              ///< The depth of the filter.
   size_t fFilterHeight;             ///< The height of the filter.
   size_t fFilterWidth;              ///< The width of the filter.
    
   size_t fDepth;                    ///< The depth of the layer.
   size_t fHeight;                   ///< The height of the layer.
   size_t fWidth;                    ///< The width of this layer.
    
   size_t fStrideRows;               ///< The number of row pixels to slid the filter each step.
   size_t fStrideCols;               ///< The number of column pixels to slid the filter each step.
   size_t fZeroPaddingHeight;        ///< The number of zero layers added top and bottom of the input.
   size_t fZeroPaddingWidth;         ///< The number of zero layers left and right of the input.
    
   size_t fNLocalViewPixels;         ///< The number of pixels in one local image view.
   size_t fNLocalViews;              ///< The number of local views in one image.
    
   Scalar_t fDropoutProbability;     ///< Probability that an input is active.
   
   Matrix_t fWeights;                ///< The weights of the layer.
   Matrix_t fWeightGradients;        ///< Gradients w.r.t. the weights of this layer.
    
    
   Matrix_t fBiases;                 ///< The bias values of this layer.
   Matrix_t fBiasGradients;          ///< Gradients w.r.t. the bias values of this layer.
    
   std::vector<Matrix_t> fOutput;                ///< Activations of this layer.
   std::vector<Matrix_t> fDerivatives;           ///< First fDerivatives of the activations of this layer.
   std::vector<Matrix_t> fActivationGradients;   ///< Gradients w.r.t. the activations of this layer.

public:

   /*! Constructor, such that the height and the weight are derived. */
   VCNNLayer(size_t   BatchSize,
             size_t   InputDepth,
             size_t   InputHeight,
             size_t   InputWidth,
             size_t   FilterDepth,
             size_t   FilterHeight,
             size_t   FilterWidth,
             size_t   Depth,
             size_t   Height,
             size_t   Width,
             size_t   WeightsNRows,
             size_t   WeightsNCols,
             size_t   BiasesNRows,
             size_t   BiasesNCols,
             size_t   StrideRows,
             size_t   StrideCols,
             size_t   ZeroPaddingHeight,
             size_t   ZeroPaddingWidth,
             Scalar_t DropoutProbability);
    
   /*! Copy Constructor. */
   VCNNLayer(const VCNNLayer &);
    
   /*! Virtual Destructor. */
   virtual ~VCNNLayer();
    
   /*! Initialize the weights according to the given initialization method. */
   virtual void Initialize(EInitialization m);
    
    
   // virtual std::shared_ptr<VCNNLayer<Architecture_t>> clone() = 0;
    
   /*! Computes activation of the layer for the given input. The input
    * must be in tensor form with the different matrices corresponding to
    * different events in the batch. Computes activations as well as
    * the first partial derivative of the activation function at those
    * activations. */
   virtual void Forward(std::vector<Matrix_t> input,
                        bool applyDropout) = 0;
    
   /*! Compute weight, bias and activation gradients. Uses the precomputed
    *  first partial derviatives of the activation function computed during
    *  forward propagation and modifies them. Must only be called directly
    *  at the corresponding call to Forward(...). */
   virtual void Backward(std::vector<Matrix_t> gradients_backward,
                         const std::vector<Matrix_t> activations_backward,
                         ERegularization r,
                         Scalar_t weightDecay) = 0;

   /*! Prints the info about the layer. */
   virtual void Print() const = 0;
   
   /*! Prints the weights of the layer. */
   void PrintWeights() const;
  
   /** Getters */
   size_t GetBatchSize()                    const {return fBatchSize;}
    
   size_t GetInputDepth()                   const {return fInputDepth;}
   size_t GetInputHeight()                  const {return fInputHeight;}
   size_t GetInputWidth()                   const {return fInputWidth;}
    
   size_t GetFilterDepth()                  const {return fFilterDepth;}
   size_t GetFilterHeight()                 const {return fFilterHeight;}
   size_t GetFilterWidth()                  const {return fFilterWidth;}
    
   size_t GetDepth()                        const {return fDepth;}
   size_t GetHeight()                       const {return fHeight;}
   size_t GetWidth()                        const {return fWidth;}
    
   size_t GetStrideRows()                   const {return fStrideRows;}
   size_t GetStrideCols()                   const {return fStrideCols;}
   size_t GetZeroPaddingHeight()            const {return fZeroPaddingHeight;}
   size_t GetZeroPaddingWidth()             const {return fZeroPaddingWidth;}
    
   size_t GetNLocalViewPixels()             const {return fNLocalViewPixels;}
   size_t GetNLocalViews()                  const {return fNLocalViews;}

   Scalar_t GetDropoutProbability()         const {return fDropoutProbability;}


    
   const Matrix_t& GetWeights()                            const {return fWeights;}
   Matrix_t& GetWeights()                                        {return fWeights;}
    
   const Matrix_t& GetBiases()                             const {return fBiases;}
   Matrix_t& GetBiases()                                         {return fBiases;}
    
   const Matrix_t& GetWeightGradients()                    const {return fWeightGradients;}
   Matrix_t& GetWeightGradients()                                {return fWeightGradients;}
    
   const Matrix_t & GetBiasGradients()                     const {return fBiasGradients;}
   Matrix_t & GetBiasGradients()                                 {return fBiasGradients;}

   const std::vector<Matrix_t>& GetOutput()                const {return fOutput;}
   std::vector<Matrix_t>& GetOutput()                            {return fOutput;}

   const std::vector<Matrix_t>& GetDerivatives()           const {return fDerivatives;}
   std::vector<Matrix_t>& GetDerivatives()                       {return fDerivatives;}

   const std::vector<Matrix_t>& GetActivationGradients()   const {return fActivationGradients;}
   std::vector<Matrix_t>& GetActivationGradients()               {return fActivationGradients;}


   Matrix_t& GetOutputAt(size_t i) {return fOutput[i];}
   const Matrix_t& GetOutputAt(size_t i) const {return fOutput[i];}

   Matrix_t& GetDerivativesAt(size_t i) {return fDerivatives[i];}
   const Matrix_t& GetDerivativesAt(size_t i) const {return fDerivatives[i];}

   Matrix_t& GetActivationGradientsAt(size_t i) {return fActivationGradients[i];}
   const Matrix_t& GetActivationGradientsAt(size_t i) const {return fActivationGradients[i];}
};
    
//_________________________________________________________________________________________________
template<typename Architecture_t>
   VCNNLayer<Architecture_t>::VCNNLayer(size_t batchSize,
                                        size_t inputDepth,
                                        size_t inputHeight,
                                        size_t inputWidth,
                                        size_t filterDepth,
                                        size_t filterHeight,
                                        size_t filterWidth,
                                        size_t depth,
                                        size_t height,
                                        size_t width,
                                        size_t weightsNRows,
                                        size_t weightsNCols,
                                        size_t biasesNRows,
                                        size_t biasesNCols,
                                        size_t strideRows,
                                        size_t strideCols,
                                        size_t zeroPaddingHeight,
                                        size_t zeroPaddingWidth,
                                        Scalar_t dropoutProbability)
    
   : fBatchSize(batchSize), fInputDepth(inputDepth),
     fInputHeight(inputHeight), fInputWidth(inputWidth),
     fFilterDepth(filterDepth), fFilterHeight(filterHeight),
     fFilterWidth(filterWidth), fDepth(depth), fHeight(height), fWidth(width),
     fStrideRows(strideRows), fStrideCols(strideCols),
     fZeroPaddingHeight(zeroPaddingHeight), fZeroPaddingWidth(zeroPaddingWidth),
     fNLocalViewPixels(fFilterWidth * fFilterHeight * fFilterDepth), fNLocalViews(height * width),
     fDropoutProbability(dropoutProbability), fWeights(weightsNRows, weightsNCols),
     fWeightGradients(weightsNRows, weightsNCols), fBiases(biasesNRows, biasesNCols),
     fBiasGradients(biasesNRows, biasesNCols),
     fOutput(), fDerivatives(), fActivationGradients()
{
   for(size_t i = 0; i < fBatchSize; i++) {
      fOutput.emplace_back(fDepth, fNLocalViews);
      fDerivatives.emplace_back(fDepth, fNLocalViews);
      fActivationGradients.emplace_back(fDepth, fNLocalViews);
    }
}

//_________________________________________________________________________________________________
template<typename Architecture_t>
    VCNNLayer<Architecture_t>::VCNNLayer(const VCNNLayer &CNNLayer)
   : fBatchSize(CNNLayer.fBatchSize), fInputDepth(CNNLayer.fInputDepth),
     fInputHeight(CNNLayer.fInputHeight), fInputWidth(CNNLayer.fInputWidth),
     fFilterDepth(CNNLayer.fFilterDepth), fFilterHeight(CNNLayer.fFilterHeight),
     fFilterWidth(CNNLayer.fFilterWidth), fDepth(CNNLayer.fDepth),
     fHeight(CNNLayer.fHeight), fWidth(CNNLayer.fWidth), fStrideRows(CNNLayer.fStrideRows),
     fStrideCols(CNNLayer.fStrideCols), fZeroPaddingHeight(CNNLayer.fZeroPaddingHeight),
     fZeroPaddingWidth(CNNLayer.fZeroPaddingWidth), fNLocalViewPixels(CNNLayer.fNLocalViewPixels),
     fNLocalViews(CNNLayer.fNLocalViews), fDropoutProbability(CNNLayer.fDropoutProbability),
     fWeights(CNNLayer.fWeights.GetNrows(), CNNLayer.fWeights.GetNcols()),
     fWeightGradients(CNNLayer.fWeightGradients.GetNrows(), CNNLayer.fWeightGradients.GetNcols()),
     fBiases(CNNLayer.fBiases.GetNrows(), CNNLayer.fBiases.GetNcols()),
     fBiasGradients(CNNLayer.fBiasGradients.GetNrows(), CNNLayer.fBiasGradients.GetNcols()),
     fOutput(), fDerivatives(), fActivationGradients()
{

   for(size_t i = 0; i < CNNLayer.fBatchSize; i++) {
      fOutput.emplace_back(CNNLayer.fDepth, CNNLayer.fNLocalViews);
      fDerivatives.emplace_back(CNNLayer.fDepth, CNNLayer.fNLocalViews);
      fActivationGradients.emplace_back(CNNLayer.fDepth, CNNLayer.fNLocalViews);
   }
}

//______________________________________________________________________________
template<typename Architecture_t>
   VCNNLayer<Architecture_t>::~VCNNLayer()
{

}
    
//______________________________________________________________________________
template<typename Architecture_t>
   auto VCNNLayer<Architecture_t>::Initialize(EInitialization m)
-> void
{
   initialize<Architecture_t>(fWeights, m);
   initialize<Architecture_t>(fBiases,  EInitialization::kZero);
}

//______________________________________________________________________________
template<typename Architecture_t>
   auto VCNNLayer<Architecture_t>::PrintWeights() const
-> void
{
   std::cout << "\t\t\t Layer Weights: " << std::endl;
   for(size_t i = 0; i < fWeights.GetNrows(); i++) {
      for(size_t j = 0; j < fWeights.GetNcols(); j++) {
         std::cout<< fWeights(i, j) << "  ";
      }
      std::cout<<""<<std::endl;
   }
    
   std::cout << "Layer Biases: " << std::endl;
   for(size_t i = 0; i < fBiases.GetNrows(); i++) {
      for(size_t j = 0; j < fBiases.GetNcols(); j++) {
         std::cout<< fBiases(i, j) << "  ";
      }
      std::cout<<""<<std::endl;
   }
}

} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif /* CNNLAYER_H_ */
