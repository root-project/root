// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski 01/06/17

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef CONVNET_H_
#define CONVNET_H_



#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/Net.h"
#include "CNNLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"

#include <iostream>
#include <vector>
#include <cmath>


namespace TMVA
{
namespace DNN
{
namespace CNN
{

    
/** \class TConvNet
    
    Generic Convolutional Neural Network class.
 
    This generic Convolutional Neural Network class represents a concrete
    convolutional neural network through a vector of three types of layers,
    namely Convolutional, Max Pooling and Fully Connected Layer, such that
    it coordinates the forward and backward propagation through the net.
    The network must contain a number of Convolutional and Pooling layers,
    and then in the end it must end with a Fully Connected Layer.
 
    The net takes as input a batch from the training data given in 3 
    dimensional tensor form, with each matrix corresponding to a certain
    training event.
 
    On construction, the convolutional neural network allocates all the
    memory required for the training of the neural net and keeps it until
    its destruction.
 
    The Architecture type argument simply holds the
    architecture-specific data types, which are just the matrix type
    Matrix_t and the used scalar type Scalar_t.
 
    \tparam Architecture The Architecture type that holds the
    architecture-specific data types.
 
    \tparam Layer_t The type used for the layers. Can be either
    ConvLayer<Architecture> or PoolLayer<Architecture>.
    datatypes for a given architecture.
 
*/

template<typename Architecture_t, typename Layer_t = VCNNLayer<Architecture_t>>
class TConvNet
{
    
public:
   using Net_t    = TNet<Architecture_t>;
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

private:
   bool inline isInteger(double x) const {return x == floor(x);}
   size_t inline calculateDimension(int imgDim, int fltDim, int padding, int stride);

private:
    
   size_t fBatchSize;                  ///< Batch size used for training and evaluation.
   size_t fImageDepth;                 ///< The depth of the input images.
   size_t fImageHeight;                ///< The height of the input images.
   size_t fImageWidth;                 ///< The width of the input images.
    
   std::vector<std::shared_ptr<Layer_t>> fLayers;     ///< The array of convolutional and pooling layers.
    
    
   Net_t fNet;                         ///< The array of Fully-Connected layers in the end.
   ELossFunction fJ;                   ///< The loss function of the network.
   ERegularization fR;                 ///< The regularization used for the network.
   Scalar_t fWeightDecay;              ///< The weight decay factor.
    
   std::vector<Matrix_t> fDummy;       ///< Empty 3D tensor for last step in back propagation.

public:
    
   /*! Default Constructor */
   TConvNet();
    
   /*! Constructor */
   TConvNet(size_t BatchSize,
            size_t ImageDepth,
            size_t ImageHeight,
            size_t ImageWidth,
            ELossFunction fJ,
            ERegularization fR = ERegularization::kNone,
            Scalar_t fWeightDecay = 0.0);
    
   /*! Copy-constructor */
   TConvNet(const TConvNet &tCnn);
    
   /*! Function for adding Convolution layer in the Convolutional Neural Network,
    *  with a given depth, filter height and width, striding in rows and columns,
    *  the zero paddings, as well as the activation function and the dropout
    *  probability. Based on these parameters, it calculates the width and height
    *  of the convolutional layer. */
   void AddConvLayer(size_t depth,
                     size_t filterHeight,
                     size_t filterWidth,
                     size_t strideRows,
                     size_t strideCols,
                     size_t zeroPaddingHeight,
                     size_t zeroPaddingWidth,
                     EActivationFunction f,
                     Scalar_t dropoutProbability = 1.0);
    
   /*! Function for adding Pooling layer in the Convolutional Neural Network,
    *  with a given filter height and width, striding in rows and columns as
    *  well as the dropout probability. The depth is same as the previous
    *  layer depth. Based on these parameters, it calculates the width and
    *  height of the pooling layer. */
   void AddPoolLayer(size_t filterHeight,
                     size_t filterWidth,
                     size_t strideRows,
                     size_t strideCols,
                     Scalar_t dropoutProbability = 1.0);
    
   /*! Function for adding Fully Connected Layer in the Convolutional Neural Network,
    *  with a given width, activation function and dropout probability.
    *  Based on the previous layer dimensions, it calculates the input width
    *  of the fully connected layer. */
   void AddFullyConnLayer(size_t width,
                          EActivationFunction f,
                          Scalar_t dropoutProbability = 1.0);

   /*! Remove all layers from the network. */
   void Clear();
    
   /*! Function for initialization of the CNN. */
   void Initialize(EInitialization m);
    
   /*! Initialize the gradients of the CNN to zero. Required if the CNN
    *  is optimized by the momentum-based techniques. */
   void InitializeGradients();
    
   /*! Function that executes the entire forward pass in the network. */
   void Forward(std::vector<Matrix_t> input, bool applyDropout = false);
    
   /*! Function that executes the entire backward pass in the network. */
   void Backward(std::vector<Matrix_t> input, const Matrix_t &groundTruth);
   
   /*! Function for evaluating the loss, based on the activations stored
    *  in the last Fully Connected Layer. */
   Scalar_t Loss(const Matrix_t &groundTruth, bool includeRegularization = true) const;
    
   /*! Function for evaluating the loss, based on the propagation of the given input. */
   Scalar_t Loss(std::vector<Matrix_t> input, const Matrix_t &groundTruth, bool applyDropout = false);

   /*! Prediction for the given inputs, based on what network learned. */
   void Prediction(Matrix_t &predictions, std::vector<Matrix_t> input, EOutputFunction f);
    
   /*! Prediction based on activations stored in the last Fully Connected Layer. */
   void Prediction(Matrix_t &predictions, EOutputFunction f) const;

   void Print();
    
   /*! Getters */
   inline size_t GetBatchSize()   const {return fBatchSize;}
   inline size_t GetImageDepth()  const {return fImageDepth;}
   inline size_t GetImageHeight() const {return fImageHeight;}
   inline size_t GetImageWidth()  const {return fImageWidth;}
   inline size_t GetDepth()       const {return fLayers.size() + fNet.GetDepth();}

    
   inline size_t GetOutputWidth()              const {return fNet.GetOutputWidth();}
   inline ELossFunction GetLossFunction()      const {return fJ;}
   inline ERegularization GetRegularization()  const {return fR;}
   inline Scalar_t GetWeightDecay()            const {return fWeightDecay;}
};


//______________________________________________________________________________
//
//  The ConvNet Class - Implementation
//______________________________________________________________________________

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   TConvNet<Architecture_t, Layer_t>::TConvNet()
   : fBatchSize(0), fImageDepth(0), fImageHeight(0), fImageWidth(0),
   fLayers(), fNet(), fJ(ELossFunction::kMeanSquaredError),
   fR(ERegularization::kNone), fWeightDecay(0.0), fDummy()

{
    // Nothing to do here.
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   TConvNet<Architecture_t, Layer_t>::TConvNet(size_t batchSize,
                                               size_t imageDepth,
                                               size_t imageHeight,
                                               size_t imageWidth,
                                               ELossFunction J,
                                               ERegularization R,
                                               Scalar_t weightDecay)
   : fBatchSize(batchSize), fImageDepth(imageDepth),fImageHeight(imageHeight),
     fImageWidth(imageWidth), fLayers(), fNet(), fJ(J), fR(R),
     fWeightDecay(weightDecay), fDummy()
{
   fNet.SetBatchSize(batchSize);
   fNet.SetLossFunction(J);
   fNet.SetRegularization(R);
   fNet.SetWeightDecay(weightDecay);
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   TConvNet<Architecture_t, Layer_t>::TConvNet(const TConvNet &tCnn)
   : fBatchSize(tCnn.fBatchSize), fImageDepth(tCnn.fImageDepth),
    fImageHeight(tCnn.fImageHeight), fImageWidth(tCnn.fImageWidth),
    fLayers(tCnn.fLayers.size()), fNet(tCnn.fNet),  fJ(tCnn.fJ),
    fR(tCnn.fR), fWeightDecay(tCnn.fWeightDecay), fDummy()
{
    // Have to copy the layers
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::AddConvLayer(size_t depth,
                                                        size_t filterHeight,
                                                        size_t filterWidth,
                                                        size_t strideRows,
                                                        size_t strideCols,
                                                        size_t zeroPaddingHeight,
                                                        size_t zeroPaddingWidth,
                                                        EActivationFunction f,
                                                        Scalar_t dropoutProbability)
->void
{
   size_t height;
   size_t width;
   size_t weightsNRows = depth;
   size_t weightsNCols;
   size_t biasesNRows = depth;
   size_t biasesNCols = 1;
    
   if(fLayers.size() == 0) {
       
      height = calculateDimension(fImageHeight, filterHeight, zeroPaddingHeight, strideRows);
      width = calculateDimension(fImageWidth, filterWidth, zeroPaddingWidth, strideCols);
      weightsNCols = fImageDepth * filterHeight * filterWidth;
       
      fLayers.emplace_back(std::make_shared<TConvLayer<Architecture_t>>(fBatchSize, fImageDepth,
                                                                        fImageHeight, fImageWidth,
                                                                        filterHeight, filterWidth, depth,
                                                                        height, width, weightsNRows,
                                                                        weightsNCols, biasesNRows, biasesNCols,
                                                                        strideRows, strideCols,
                                                                        zeroPaddingHeight, zeroPaddingWidth,
                                                                        dropoutProbability, f));
            
   } else {
       
      std::shared_ptr<Layer_t> lastLayer = fLayers.back();
       
      height = calculateDimension(lastLayer -> GetHeight(), filterHeight, zeroPaddingHeight, strideRows);
      width = calculateDimension(lastLayer -> GetWidth(), filterWidth, zeroPaddingWidth, strideCols);
      weightsNCols = lastLayer -> GetDepth() * filterHeight * filterWidth;

       
      fLayers.emplace_back(std::make_shared<TConvLayer<Architecture_t>>(fBatchSize, lastLayer -> GetDepth(),
                                                                        lastLayer -> GetHeight(), lastLayer -> GetWidth(),
                                                                        filterHeight, filterWidth, depth,
                                                                        height, width, weightsNRows,
                                                                        weightsNCols, biasesNRows, biasesNCols,
                                                                        strideRows, strideCols,
                                                                        zeroPaddingHeight, zeroPaddingWidth,
                                                                        dropoutProbability, f));
   }
}
    
//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::AddPoolLayer(size_t filterHeight,
                                                        size_t filterWidth,
                                                        size_t strideRows,
                                                        size_t strideCols,
                                                        Scalar_t dropoutProbability)
-> void
{
   size_t height;
   size_t width;
   size_t weightsNRows = 0;
   size_t weightsNCols = 0;
   size_t biasesNRows = 0;
   size_t biasesNCols = 0;
    
   if(fLayers.size() == 0) {
       
      height = calculateDimension(fImageHeight, filterHeight, 0, strideRows);
      width = calculateDimension(fImageWidth, filterWidth, 0, strideCols);
       
      fLayers.emplace_back(std::make_shared<TPoolLayer<Architecture_t>>(fBatchSize, fImageDepth,
                                                                        fImageHeight, fImageWidth,
                                                                        filterHeight, filterWidth,
                                                                        height, width, weightsNRows,
                                                                        weightsNCols, biasesNRows,biasesNCols,
                                                                        strideRows, strideCols,
                                                                        dropoutProbability));
   } else {
      std::shared_ptr<Layer_t> lastLayer = fLayers.back();
       
      height = calculateDimension(lastLayer -> GetHeight(), filterHeight, 0, strideRows);
      width = calculateDimension(lastLayer -> GetWidth(), filterWidth, 0, strideCols);
       
      fLayers.emplace_back(std::make_shared<TPoolLayer<Architecture_t>>(fBatchSize, lastLayer -> GetDepth(),
                                                                        lastLayer -> GetHeight(), lastLayer -> GetWidth(),
                                                                        filterHeight, filterWidth,
                                                                        height, width, weightsNRows,
                                                                        weightsNCols, biasesNRows,biasesNCols,
                                                                        strideRows, strideCols,
                                                                        dropoutProbability));
   }
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::AddFullyConnLayer(size_t width,
                                                             EActivationFunction f,
                                                             Scalar_t dropoutProbability)
-> void
{
    
   if(fNet.GetDepth() == 0) {
      size_t inputWidth;
            
      if(fLayers.size() == 0) {
         inputWidth = fImageDepth * fImageHeight * fImageWidth;
      } else {
         std::shared_ptr<Layer_t> lastLayer = fLayers.back();
         inputWidth = lastLayer -> GetDepth() * lastLayer -> GetHeight() * lastLayer -> GetWidth();
      }

      fNet.SetInputWidth(inputWidth);
   }
    
   fNet.AddLayer(width, f, dropoutProbability);
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::Clear()
-> void
{
   fNet.Clear();
   fLayers.clear();
}

    
    
//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::Initialize(EInitialization m)
-> void
{
   for(size_t i = 0; i < fLayers.size(); i++) {
      fLayers[i] -> Initialize(m);
   }
        
   fNet.Initialize(m);
}
    
//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::InitializeGradients()
-> void
{
   for (size_t i = 0; i < fLayers.size(); i++) {
      initialize<Architecture_t>(fLayers[i] -> GetWeightGradients(), EInitialization::kZero);
      initialize<Architecture_t>(fLayers[i] -> GetBiasGradients(),   EInitialization::kZero);
    }
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::Forward(std::vector<Matrix_t> input,
                                                   bool applyDropout)
-> void
{
   fLayers.front() -> Forward(input, applyDropout);
    
   for(size_t i = 1; i < fLayers.size(); i++) {
      fLayers[i] -> Forward(fLayers[i - 1] -> GetOutput(), applyDropout);
   }
    
   std::shared_ptr<Layer_t> lastLayer = fLayers.back();
   std::vector<Matrix_t> lastOutput = lastLayer -> GetOutput();
    
   size_t nRows = lastLayer -> GetDepth();
   size_t nCols = lastLayer -> GetNLocalViews();
   
   Matrix_t fFlatOutput(this -> GetBatchSize(), fNet.GetInputWidth());
   Architecture_t::Flatten(fFlatOutput, lastOutput, this -> GetBatchSize(), nRows, nCols);
   fNet.Forward(fFlatOutput, applyDropout);
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::Backward(std::vector<Matrix_t> input,
                                                    const Matrix_t &groundTruth)
-> void
{
    
   std::shared_ptr<Layer_t> lastLayer = fLayers.back();
    
   std::vector<Matrix_t> lastOutput = lastLayer -> GetOutput();
   size_t nRows = lastLayer -> GetDepth();
   size_t nCols = lastLayer -> GetNLocalViews();
    
   // Backward in the MLP
   Matrix_t fFlatOutput(this -> GetBatchSize(), fNet.GetInputWidth());
   Architecture_t::Flatten(fFlatOutput, lastOutput, this -> GetBatchSize(), nRows, nCols);
   fNet.Backward(fFlatOutput, groundTruth);
    
    
   // Deflattening the backward activation gradients
   Matrix_t activation_gradients_backward_flat = fNet.GetDummy();
    
   
   std::vector<Matrix_t> activation_gradients_backward = lastLayer -> GetActivationGradients();
   Architecture_t::Deflatten(activation_gradients_backward,
                             activation_gradients_backward_flat,
                             this -> GetBatchSize(), nRows, nCols);
    
    
   // Backward in the rest of the network
   for(size_t i = fLayers.size() - 1; i > 0; i--) {
      std::vector<Matrix_t> activation_gradient_backward
           = fLayers[i-1] -> GetActivationGradients();
      std::vector<Matrix_t> activations_backward
           = fLayers[i-1] -> GetOutput();
      fLayers[i] -> Backward(activation_gradient_backward,
                             activations_backward, fR, fWeightDecay);
   }
    
   fLayers[0] -> Backward(fDummy, input, fR, fWeightDecay);
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::Loss(const Matrix_t &groundTruth,
                                                bool includeRegularization) const
-> Scalar_t
{
   auto loss = fNet.Loss(groundTruth, includeRegularization);
   includeRegularization &= (fR != ERegularization::kNone);
    
   if(includeRegularization) {
      for(size_t i = 0; i < fLayers.size(); i++) {
         loss += fWeightDecay * regularization<Architecture_t>(fLayers[i] -> GetWeights(), fR);
      }
   }
    
   return loss;
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::Loss(std::vector<Matrix_t> input,
                                                const Matrix_t &groundTruth,
                                                bool applyDropout)
-> Scalar_t
{
   Forward(input, applyDropout);
   return Loss(groundTruth);
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::Prediction(Matrix_t &predictions,
                                                      std::vector<Matrix_t> input,
                                                      EOutputFunction f)
-> void
{
   Forward(input, false);
   evaluate<Architecture_t>(predictions, f, fNet.GetLayer(fNet.GetDepth() - 1).GetOutput());
}
    
//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::Prediction(Matrix_t &predictions,
                                                      EOutputFunction f) const
-> void
{
   evaluate<Architecture_t>(predictions, f, fNet.GetLayer(fNet.GetDepth() - 1).GetOutput());
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::Print()
-> void
{
   std::cout << "CONVOLUTIONAL NEURAL NETWORK:" << std::endl;
   std::cout << "\t Loss function = " << static_cast<char>(fJ) << std::endl;
   std::cout << "\t Network Depth = " << GetDepth() << std::endl;
   std::cout << "\t Input images depth = " << GetImageDepth() << std::endl;
   std::cout << "\t Input images height = " << GetImageHeight() << std::endl;
   std::cout << "\t Input images width = " << GetImageWidth() << std::endl;
   std::cout << "\t Batch size = " << GetBatchSize() << std::endl;

   std::cout << "\t Layers: " << std::endl;
    
   for(size_t i = 0; i < fLayers.size(); i++) {
      fLayers[i] -> Print();
   }
    
    
   std::cout << "\t Multylayer Percepton: " << std::endl;
   fNet.Print();
}

//______________________________________________________________________________
template<typename Architecture_t, typename Layer_t>
   auto TConvNet<Architecture_t, Layer_t>::calculateDimension(int imgDim,
                                                              int fltDim,
                                                              int padding,
                                                              int stride)
-> size_t
{
   double dimension = ((imgDim - fltDim + 2 * padding) / stride) + 1;
   if(!isInteger(dimension)) {
      std::cout << "Not compatible hyper parameters" << std::endl;
      std::exit(EXIT_FAILURE);
   }
    
   return (size_t) dimension;
}

} // namesoace CNN
} // namespace DNN
} // namespace TMVA


#endif
