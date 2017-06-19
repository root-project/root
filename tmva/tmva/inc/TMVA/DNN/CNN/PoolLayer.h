// @(#)root/tmva/tmva/dnn/cnn:$Id$
// Author: Vladimir Ilievski 05/06/17

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/////////////////////////////////////////////////////////////////////////
// Contains PoolLayer class that represents the pooling layer in       //
// the Convolutional Neural Networks                                   //
/////////////////////////////////////////////////////////////////////////


#ifndef POOLLAYER_H_
#define POOLLAYER_H_

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"
#include "CNNLayer.h"

#include <cmath>
#include <iostream>
#include <limits>


namespace TMVA
{
namespace DNN
{
namespace CNN
{

/** \class TPoolLayer
 
    Generic Pooling Layer class.
 
    This generic Pooling Layer c=Class represents a pooling layer of
    a CNN. It inherits all of the properties of the generic virtual base class
    TCNNLayer. In addition to that, it contains a matrix of winning units.
 
    The height and width of the weights and biases is set to 0, since this
    layer does not contain any weights.

 */
template<typename Architecture_t>
   class TPoolLayer : public VCNNLayer<Architecture_t>
{

public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

private:
   std::vector<Matrix_t> indexMatrix;      ///< Matrix of indices for the backward pass.
    
public:
    
   /*! Constructor. */
   TPoolLayer(size_t BatchSize,
              size_t InputDepth,
              size_t InputHeight,
              size_t InputWidth,
              size_t FilterHeight,
              size_t FilterWidth,
              size_t Height,
              size_t Width,
              size_t WeightsNRows,
              size_t WeightsNCols,
              size_t BiasesNRows,
              size_t BiasesNCols,
              size_t StrideRows,
              size_t StrideCols,
              Scalar_t DropoutProbability);
    
   /*! Copy constructor. */
   TPoolLayer(const TPoolLayer &);

   /*! Destructor. */
   ~TPoolLayer();
  
  // virtual std::shared_ptr<VCNNLayer<Architecture_t>> clone();
    
   /*! Computes activation of the layer for the given input. The input
    *  must be in tensor form with the different matrices corresponding to
    *  different events in the batch. It spatially downsamples the input
    *  matrices. */
   void inline Forward(std::vector<Matrix_t> input,
                       bool applyDropout);
   
    
   /*! Depending on the winning units determined during the Forward pass,
    *  it only forwards the derivatives to the right units in the previous
    *  layer. Must only be called directly at the corresponding call
    *  to Forward(...). */
   void inline Backward(std::vector<Matrix_t> gradients_backward,
                        const std::vector<Matrix_t> activations_backward,
                        ERegularization r,
                        Scalar_t weightDecay);
    
    
   /*! Prints the info about the layer. */
   void Print() const;
    
   /*! Getters */
   const std::vector<Matrix_t>& GetIndexMatrix() const {return indexMatrix;}
   std::vector<Matrix_t>& GetIndexMatrix() {return indexMatrix;}

};

    
//______________________________________________________________________________
template<typename Architecture_t>
   TPoolLayer<Architecture_t>::TPoolLayer(size_t batchSize,
                                          size_t inputDepth,
                                          size_t inputHeight,
                                          size_t inputWidth,
                                          size_t filterHeight,
                                          size_t filterWidth,
                                          size_t height,
                                          size_t width,
                                          size_t weightsNRows,
                                          size_t weightsNCols,
                                          size_t biasesNRows,
                                          size_t biasesNCols,
                                          size_t strideRows,
                                          size_t strideCols,
                                          Scalar_t dropoutProbability)
   : VCNNLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, inputDepth,
                               filterHeight, filterWidth, inputDepth, height, width,
                               weightsNRows, weightsNCols, biasesNRows, biasesNCols,
                               strideRows, strideCols, 0, 0, dropoutProbability), indexMatrix()
{
   for(size_t i = 0; i < this -> GetBatchSize(); i++) {
      indexMatrix.emplace_back(this -> GetDepth(), this -> GetNLocalViews());
   }
}

//______________________________________________________________________________
template<typename Architecture_t>
   TPoolLayer<Architecture_t>::TPoolLayer(const TPoolLayer &poolLayer)
   : VCNNLayer<Architecture_t>(poolLayer), indexMatrix()
{
   for(size_t i = 0; i < poolLayer.fBatchSize; i++) {
      indexMatrix.emplace_back(poolLayer.fDepth, poolLayer.fNLocalViews);
      Architecture_t::Copy(indexMatrix[i], poolLayer.indexMatrix[i]);
   }
}

//______________________________________________________________________________
template<typename Architecture_t>
  TPoolLayer<Architecture_t>::~TPoolLayer()
{
        
}

    
//______________________________________________________________________________
template<typename Architecture_t>
   auto TPoolLayer<Architecture_t>::Forward(std::vector<Matrix_t> input,
                                            bool applyDropout)
-> void
{
   for(size_t i = 0; i < this -> GetBatchSize(); i++) {
       
      if (applyDropout && (this -> GetDropoutProbability()  != 1.0)) {
         Architecture_t::Dropout(input[i], this -> GetDropoutProbability() );
      }
       
      Architecture_t::Downsample(this -> GetOutputAt(i), indexMatrix[i], input[i],
                                 this -> GetInputHeight(), this -> GetInputWidth(),
                                 this -> GetFilterHeight(), this -> GetFilterWidth(),
                                 this -> GetStrideRows(), this -> GetStrideCols());
   }
}

//______________________________________________________________________________
template<typename Architecture_t>
   auto TPoolLayer<Architecture_t>::Backward(std::vector<Matrix_t> gradients_backward,
                                             const std::vector<Matrix_t> activations_backward,
                                             ERegularization r,
                                             Scalar_t weightDecay)
-> void
{
    
   Architecture_t::PoolLayerBackward(gradients_backward,
                                     this -> GetActivationGradients(),
                                     indexMatrix,
                                     this -> GetBatchSize(),
                                     this -> GetDepth(),
                                     this -> GetNLocalViews());
}

//______________________________________________________________________________
template<typename Architecture_t>
   auto TPoolLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "\t\t POOL LAYER: " << std::endl;
   std::cout << "\t\t\t Width = " <<  this -> GetWidth() << std::endl;
   std::cout << "\t\t\t Height = " << this -> GetHeight() << std::endl;
   std::cout << "\t\t\t Depth = " << this -> GetDepth() << std::endl;
    
   std::cout << "\t\t\t Frame Width = " << this -> GetFilterWidth() << std::endl;
   std::cout << "\t\t\t Frame Height = " << this -> GetFilterHeight() << std::endl;
}
    
} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif /* POOLLAYER_H_ */
