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
#include "TMVA/DNN/GeneralLayer.h"
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
   class VCNNLayer : public VGeneralLayer<Architecture_t>
{

public:
    
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;
    
protected:
    
   size_t fFilterDepth;              ///< The depth of the filter.
   size_t fFilterHeight;             ///< The height of the filter.
   size_t fFilterWidth;              ///< The width of the filter.
    
   size_t fStrideRows;               ///< The number of row pixels to slid the filter each step.
   size_t fStrideCols;               ///< The number of column pixels to slid the filter each step.
   size_t fZeroPaddingHeight;        ///< The number of zero layers added top and bottom of the input.
   size_t fZeroPaddingWidth;         ///< The number of zero layers left and right of the input.
    
   size_t fNLocalViewPixels;         ///< The number of pixels in one local image view.
   size_t fNLocalViews;              ///< The number of local views in one image.

public:

   /*! Constructor */
   VCNNLayer(size_t BatchSize,
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
             size_t FilterDepth,
             size_t FilterHeight,
             size_t FilterWidth,
             size_t StrideRows,
             size_t StrideCols,
             size_t ZeroPaddingHeight,
             size_t ZeroPaddingWidth);
    
   /*! Copy Constructor. */
   VCNNLayer(const VCNNLayer &);
    
   /*! Virtual Destructor. */
   virtual ~VCNNLayer();
    
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
   virtual void Backward(std::vector<Matrix_t> &gradients_backward,
                         const std::vector<Matrix_t> &activations_backward,
                         ERegularization r,
                         Scalar_t weightDecay) = 0;

   /*! Prints the info about the layer. */
   virtual void Print() const = 0;
  
   /** Getters */
   size_t GetFilterDepth()                  const {return fFilterDepth;}
   size_t GetFilterHeight()                 const {return fFilterHeight;}
   size_t GetFilterWidth()                  const {return fFilterWidth;}
    
   size_t GetStrideRows()                   const {return fStrideRows;}
   size_t GetStrideCols()                   const {return fStrideCols;}
   size_t GetZeroPaddingHeight()            const {return fZeroPaddingHeight;}
   size_t GetZeroPaddingWidth()             const {return fZeroPaddingWidth;}
    
   size_t GetNLocalViewPixels()             const {return fNLocalViewPixels;}
   size_t GetNLocalViews()                  const {return fNLocalViews;}
};
    
//_________________________________________________________________________________________________
template<typename Architecture_t>
    VCNNLayer<Architecture_t>::VCNNLayer(size_t batchSize,
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
                                         size_t filterDepth,
                                         size_t filterHeight,
                                         size_t filterWidth,
                                         size_t strideRows,
                                         size_t strideCols,
                                         size_t zeroPaddingHeight,
                                         size_t zeroPaddingWidth)
    
   : VGeneralLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, depth, height, width,
                                   dropoutProbability, weightsNRows, weightsNCols, biasesNRows,
                                   biasesNCols, outputNSlices, outputNRows, outputNCols),
     fFilterDepth(filterDepth), fFilterHeight(filterHeight), fFilterWidth(filterWidth),
     fStrideRows(strideRows), fStrideCols(strideCols), fZeroPaddingHeight(zeroPaddingHeight),
     fZeroPaddingWidth(zeroPaddingWidth), fNLocalViewPixels(fFilterWidth * fFilterHeight * fFilterDepth),
     fNLocalViews(height * width)
{
}

//_________________________________________________________________________________________________
template<typename Architecture_t>
    VCNNLayer<Architecture_t>::VCNNLayer(const VCNNLayer &CNNLayer)
   : VGeneralLayer<Architecture_t>(CNNLayer),
     fFilterDepth(CNNLayer.fFilterDepth), fFilterHeight(CNNLayer.fFilterHeight),
     fFilterWidth(CNNLayer.fFilterWidth), fStrideRows(CNNLayer.fStrideRows),
     fStrideCols(CNNLayer.fStrideCols), fZeroPaddingHeight(CNNLayer.fZeroPaddingHeight),
     fZeroPaddingWidth(CNNLayer.fZeroPaddingWidth), fNLocalViewPixels(CNNLayer.fNLocalViewPixels),
     fNLocalViews(CNNLayer.fNLocalViews)
{
    
}

//______________________________________________________________________________
template<typename Architecture_t>
   VCNNLayer<Architecture_t>::~VCNNLayer()
{

}

} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif /* CNNLAYER_H_ */
