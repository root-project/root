// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMaxPoolLayer                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Max Pool Deep Neural Network Layer                                        *
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

#ifndef MAXPOOLLAYER_H_
#define MAXPOOLLAYER_H_

#include "TMatrix.h"

#include "TMVA/DNN/CNN/ConvLayer.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/CNN/ContextHandles.h"

#include <iostream>

namespace TMVA {
namespace DNN {
namespace CNN {


/** \class TMaxPoolLayer

    Generic Max Pooling Layer class.

    This generic Max Pooling Layer Class represents a pooling layer of
    a CNN. It inherits all of the properties of the convolutional layer
    TConvLayer, but it overrides the propagation methods. In a sense, max pooling
    can be seen as non-linear convolution: a filter slides over the input and produces
    one element as a function of the the elements within the receptive field.
    In addition to that, it contains a matrix of winning units.

    The height and width of the weights and biases is set to 0, since this
    layer does not contain any weights.

 */
template <typename Architecture_t>
class TMaxPoolLayer : public TConvLayer<Architecture_t> {

public:
    using Tensor_t = typename Architecture_t::Tensor_t;
    using Matrix_t = typename Architecture_t::Matrix_t;
    using Scalar_t = typename Architecture_t::Scalar_t;
    
    using LayerDescriptor_t   = typename Architecture_t::PoolingDescriptor_t;
    using WeightsDescriptor_t = typename Architecture_t::EmptyDescriptor_t;
    using HelperDescriptor_t  = typename Architecture_t::DropoutDescriptor_t;

private:
   Tensor_t fIndexTensor; ///< Matrix of indices for the backward pass.
   
   void InitializeDescriptors();
   void ReleaseDescriptors();
   void InitializeWorkspace();
   void FreeWorkspace();  
public:
   /*! Constructor. */
   TMaxPoolLayer(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t FilterHeight,
                 size_t FilterWidth, size_t StrideRows, size_t StrideCols, Scalar_t DropoutProbability);

   /*! Copy the max pooling layer provided as a pointer */
   TMaxPoolLayer(TMaxPoolLayer<Architecture_t> *layer);

   /*! Copy constructor. */
   TMaxPoolLayer(const TMaxPoolLayer &);

   /*! Destructor. */
   ~TMaxPoolLayer();

   /*! Computes activation of the layer for the given input. The input
    *  must be in 3D tensor form with the different matrices corresponding to
    *  different events in the batch. It spatially downsamples the input
    *  matrices. */
   void Forward(Tensor_t &input, bool applyDropout = false);

   /*! Depending on the winning units determined during the Forward pass,
    *  it only forwards the derivatives to the right units in the previous
    *  layer. Must only be called directly at the corresponding call
    *  to Forward(...). */
   void Backward(Tensor_t &gradients_backward, const Tensor_t &activations_backward);
    //             Tensor_t &inp1, Tensor_t &inp2);

   /*! Writes the information and the weights about the layer in an XML node. */
   virtual void AddWeightsXMLTo(void *parent);

   /*! Read the information and the weights about the layer from XML node. */
   virtual void ReadWeightsFromXML(void *parent);

   /*! Prints the info about the layer. */
   void Print() const;

   /*! Getters */
   const Tensor_t & GetIndexTensor() const { return fIndexTensor; }
   Tensor_t & GetIndexTensor() { return fIndexTensor; }

};

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::TMaxPoolLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                             size_t filterHeight, size_t filterWidth, size_t strideRows,
                                             size_t strideCols, Scalar_t dropoutProbability)

        : TConvLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, inputDepth, EInitialization::kZero,
                                     filterHeight, filterWidth, strideRows, strideCols, 0, 0, dropoutProbability,
                                     EActivationFunction::kIdentity, ERegularization::kNone, 0),
          fIndexTensor(  this->GetBatchSize(), this->GetDepth(), this->GetNLocalViews() )
{
   InitializeDescriptors();
   InitializeWorkspace();
}

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::TMaxPoolLayer(TMaxPoolLayer<Architecture_t> *layer)
   : TConvLayer<Architecture_t>(layer), 
   fIndexTensor( layer->GetIndexTensor().GetShape())
{
   InitializeDescriptors();
   InitializeWorkspace();
}

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::TMaxPoolLayer(const TMaxPoolLayer &layer)
   : TConvLayer<Architecture_t>(layer), 
   fIndexTensor( layer.GetIndexTensor().GetShape() )
{
   InitializeDescriptors();
   InitializeWorkspace();
}

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::~TMaxPoolLayer()
{
   if (TConvLayer<Architecture_t>::fDescriptors) {
      ReleaseDescriptors();
      delete TConvLayer<Architecture_t>::fDescriptors;
      TConvLayer<Architecture_t>::fDescriptors = nullptr;     // Prevents double release in the TConvLayer Destructor
   }   

   if (TConvLayer<Architecture_t>::fWorkspace) {
      FreeWorkspace();
      delete TConvLayer<Architecture_t>::fWorkspace;
      TConvLayer<Architecture_t>::fWorkspace = nullptr;
   }
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TMaxPoolLayer<Architecture_t>::Forward(Tensor_t &input, bool applyDropout) -> void
{
   

   if (applyDropout && (this->GetDropoutProbability() != 1.0)) {
         Architecture_t::Dropout(input, this->GetDropoutProbability());
   }

   

   Architecture_t::Downsample(this->GetOutput(), fIndexTensor, input, 
                              (TCNNDescriptors<TMaxPoolLayer<Architecture_t>> &) (*TConvLayer<Architecture_t>::fDescriptors),
                              (TCNNWorkspace<TMaxPoolLayer<Architecture_t>> &) (*TConvLayer<Architecture_t>::fWorkspace),
                              this->GetInputHeight(), this->GetInputWidth(), 
                              this->GetFilterHeight(), this->GetFilterWidth(),
                              this->GetStrideRows(), this->GetStrideCols());
   
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TMaxPoolLayer<Architecture_t>::Backward(Tensor_t &gradients_backward,
                                             const Tensor_t & /*activations_backward*/) -> void
//                                             Tensor_t & /*inp1*/, Tensor_t &
{
   Architecture_t::MaxPoolLayerBackward(gradients_backward, this->GetActivationGradients(), fIndexTensor, this->GetInputActivation(), this->GetOutput(),
                                        (TCNNDescriptors<TMaxPoolLayer<Architecture_t>> &) (*TConvLayer<Architecture_t>::fDescriptors),
                                        (TCNNWorkspace<TMaxPoolLayer<Architecture_t>> &) (*TConvLayer<Architecture_t>::fWorkspace),
                                        this->GetInputHeight(), this->GetInputWidth(),      
                                        this->GetFilterHeight(), this->GetFilterWidth(),
                                        this->GetStrideRows(), this->GetStrideCols(), this->GetNLocalViews());
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TMaxPoolLayer<Architecture_t>::Print() const -> void
{
   std::cout << " POOL Layer: \t";
   std::cout << "( W = " << this->GetWidth() << " , ";
   std::cout << " H = " << this->GetHeight() << " , ";
   std::cout << " D = " << this->GetDepth() << " ) ";

   std::cout << "\t Filter ( W = " << this->GetFilterWidth() << " , ";
   std::cout << " H = " << this->GetFilterHeight() << " ) ";

   if (this->GetOutput().GetSize() > 0) {
      std::cout << "\tOutput = ( " << this->GetOutput().GetFirstSize() << " , " << this->GetOutput().GetHSize() << " , " << this->GetOutput().GetWSize() << " ) ";
   }
   std::cout << std::endl;
}

//______________________________________________________________________________
template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::AddWeightsXMLTo(void *parent)
{
   auto layerxml = gTools().xmlengine().NewChild(parent, 0, "MaxPoolLayer");

   // write  maxpool layer info
   gTools().xmlengine().NewAttr(layerxml, 0, "FilterHeight", gTools().StringFromInt(this->GetFilterHeight()));
   gTools().xmlengine().NewAttr(layerxml, 0, "FilterWidth", gTools().StringFromInt(this->GetFilterWidth()));
   gTools().xmlengine().NewAttr(layerxml, 0, "StrideRows", gTools().StringFromInt(this->GetStrideRows()));
   gTools().xmlengine().NewAttr(layerxml, 0, "StrideCols", gTools().StringFromInt(this->GetStrideCols()));

}

//______________________________________________________________________________
template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::ReadWeightsFromXML(void * /*parent */)
{
   // all info is read before - nothing to do 
}

//______________________________________________________________________________
template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::InitializeDescriptors() {
   Architecture_t::InitializePoolDescriptors(TConvLayer<Architecture_t>::fDescriptors, this);
}

template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::ReleaseDescriptors() {
   Architecture_t::ReleasePoolDescriptors(TConvLayer<Architecture_t>::fDescriptors, this);
}

//______________________________________________________________________________
template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::InitializeWorkspace() {
   TConvParams params(this->GetBatchSize(), this->GetInputDepth(), this->GetInputHeight(), this->GetInputWidth(),
                      this->GetDepth(), this->GetFilterHeight(), this->GetFilterWidth(),
                      this->GetStrideRows(), this->GetStrideCols(), this->GetPaddingHeight(), this->GetPaddingWidth());

   Architecture_t::InitializePoolWorkspace(TConvLayer<Architecture_t>::fWorkspace, TConvLayer<Architecture_t>::fDescriptors, params, this);
}

template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::FreeWorkspace() {
   Architecture_t::FreePoolWorkspace(TConvLayer<Architecture_t>::fWorkspace, this);
}

} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif
