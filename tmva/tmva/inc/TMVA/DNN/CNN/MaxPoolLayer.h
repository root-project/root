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
class TMaxPoolLayer : public VGeneralLayer<Architecture_t>  {

public:
   using Tensor_t = typename Architecture_t::Tensor_t;
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

   using LayerDescriptor_t = typename Architecture_t::PoolingDescriptor_t;
   using WeightsDescriptor_t = typename Architecture_t::EmptyDescriptor_t;
   using HelperDescriptor_t = typename Architecture_t::DropoutDescriptor_t;

   // do we need thse ???
   using AlgorithmForward_t = typename Architecture_t::AlgorithmForward_t;   // Forward layer operation
   using AlgorithmBackward_t = typename Architecture_t::AlgorithmBackward_t; // Backward layer operation
   using AlgorithmHelper_t = typename Architecture_t::AlgorithmHelper_t;     // Used for weight grad backward pass

   // FIXME: Add other cudnn types (algorithm preference etc.)
   using AlgorithmDataType_t = typename Architecture_t::AlgorithmDataType_t;

protected:
   size_t fFilterDepth;  ///< The depth of the filter.
   size_t fFilterHeight; ///< The height of the filter.
   size_t fFilterWidth;  ///< The width of the filter.

   size_t fStrideRows; ///< The number of row pixels to slid the filter each step.
   size_t fStrideCols; ///< The number of column pixels to slid the filter each step.

   size_t fNLocalViewPixels; ///< The number of pixels in one local image view.
   size_t fNLocalViews;      ///< The number of local views in one image.

   Scalar_t fDropoutProbability; ///< Probability that an input is active.

   TDescriptors *fDescriptors = nullptr; ///< Keeps the convolution, activations and filter descriptors

   TWorkspace *fWorkspace = nullptr;

private:
   Tensor_t fIndexTensor; ///< Matrix of indices for the backward pass.

   virtual void InitializeDescriptors();
   virtual void ReleaseDescriptors();
   virtual void InitializeWorkspace();
   virtual void FreeWorkspace();

public:
   /*! Constructor. */
   TMaxPoolLayer(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t FilterHeight,
                 size_t FilterWidth, size_t StrideRows, size_t StrideCols, Scalar_t DropoutProbability);

   /*! Copy the max pooling layer provided as a pointer */
   TMaxPoolLayer(TMaxPoolLayer<Architecture_t> *layer);

   /*! Copy constructor. */
   TMaxPoolLayer(const TMaxPoolLayer &);

   /*! Destructor. */
   virtual ~TMaxPoolLayer();

   // virtual void Initialize();

   /*! Computes activation of the layer for the given input. The input
    *  must be in 3D tensor form with the different matrices corresponding to
    *  different events in the batch. It spatially downsamples the input
    *  matrices. */
   void Forward(Tensor_t &input, bool applyDropout = true);

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
   size_t GetFilterDepth() const { return fFilterDepth; }
   size_t GetFilterHeight() const { return fFilterHeight; }
   size_t GetFilterWidth() const { return fFilterWidth; }

   size_t GetStrideRows() const { return fStrideRows; }
   size_t GetStrideCols() const { return fStrideCols; }

   size_t GetNLocalViews() const { return fNLocalViews; }

   Scalar_t GetDropoutProbability() const { return fDropoutProbability; }
 
   const Tensor_t & GetIndexTensor() const { return fIndexTensor; }
   Tensor_t & GetIndexTensor() { return fIndexTensor; }

   // The following getters are used for testing
   TDescriptors *GetDescriptors() { return fDescriptors; }
   const TDescriptors *GetDescriptors() const { return fDescriptors; }

   TWorkspace *GetWorkspace() { return fWorkspace; }
   const TWorkspace *GetWorkspace() const { return fWorkspace; }
};

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::TMaxPoolLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                             size_t filterHeight, size_t filterWidth, size_t strideRows,
                                             size_t strideCols, Scalar_t dropoutProbability)
   : VGeneralLayer<Architecture_t>(
        batchSize, inputDepth, inputHeight, inputWidth, inputDepth,
        TConvLayer<Architecture_t>::calculateDimension(inputHeight, filterHeight, 0, strideRows),
        TConvLayer<Architecture_t>::calculateDimension(inputWidth, filterWidth, 0, strideCols), 0, 0, 0, 0, 0,
        0, // weights dimensions
        batchSize, inputDepth,
        TConvLayer<Architecture_t>::calculateNLocalViews(inputHeight, filterHeight, 0, strideRows, inputWidth,
                                                         filterWidth, 0, strideCols),
        EInitialization::kZero),
     fFilterDepth(inputDepth), fFilterHeight(filterHeight), fFilterWidth(filterWidth), fStrideRows(strideRows),
     fStrideCols(strideCols),
     fNLocalViews(TConvLayer<Architecture_t>::calculateNLocalViews(inputHeight, filterHeight, 0, strideRows,
                                                                   inputWidth, filterWidth, 0, strideCols)),
     fDropoutProbability(dropoutProbability), fIndexTensor(batchSize, inputDepth, fNLocalViews)
{
   InitializeDescriptors();
   InitializeWorkspace();
}

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::TMaxPoolLayer(TMaxPoolLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer), fFilterDepth(layer->GetFilterDepth()),
     fFilterHeight(layer->GetFilterHeight()), fFilterWidth(layer->GetFilterWidth()),
     fStrideRows(layer->GetStrideRows()), fStrideCols(layer->GetStrideCols()), fNLocalViews(layer->GetNLocalViews()),
     fDropoutProbability(layer->GetDropoutProbability()), fIndexTensor(layer->GetIndexTensor().GetShape())
{
   InitializeDescriptors();
   InitializeWorkspace();
}

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::TMaxPoolLayer(const TMaxPoolLayer &layer)
   : VGeneralLayer<Architecture_t>(layer), fFilterDepth(layer.fFilterDepth), fFilterHeight(layer.fFilterHeight),
     fFilterWidth(layer.fFilterWidth), fStrideRows(layer.fStrideRows), fStrideCols(layer.fStrideCols),
     fNLocalViews(layer.fNLocalViews), fDropoutProbability(layer.fDropoutProbability),
     fIndexTensor(layer.GetIndexTensor().GetShape())
{
   InitializeDescriptors();
   InitializeWorkspace();
}

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::~TMaxPoolLayer()
{
   if (fDescriptors) {
      ReleaseDescriptors();
      delete fDescriptors;
      fDescriptors = nullptr;
   }

   if (fWorkspace) {
      FreeWorkspace();
      delete fWorkspace;
      fWorkspace = nullptr;
   }
}

//______________________________________________________________________________
// template <typename Architecture_t>
// void TMaxPoolLayer<Architecture_t>::Initialize() { 
//    InitializeDescriptors();
//    InitializeWorkspace();  
// }
 
//______________________________________________________________________________
template <typename Architecture_t>
auto TMaxPoolLayer<Architecture_t>::Forward(Tensor_t &input, bool applyDropout) -> void
{
   if (applyDropout && (this->GetDropoutProbability() != 1.0)) {
      Architecture_t::DropoutForward(input, fDescriptors, fWorkspace, this->GetDropoutProbability());
   }

   Architecture_t::Downsample(
      this->GetOutput(), fIndexTensor, input, (TCNNDescriptors<TMaxPoolLayer<Architecture_t>> &)*fDescriptors,
      (TCNNWorkspace<TMaxPoolLayer<Architecture_t>> &)*fWorkspace, this->GetInputHeight(), this->GetInputWidth(),
      this->GetFilterHeight(), this->GetFilterWidth(), this->GetStrideRows(), this->GetStrideCols());
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TMaxPoolLayer<Architecture_t>::Backward(Tensor_t &gradients_backward, const Tensor_t &activations_backward) -> void
//                                             Tensor_t & /*inp1*/, Tensor_t &
{

   if (this->GetDropoutProbability() != 1.0) {
      Architecture_t::DropoutBackward(this->GetActivationGradients(), fDescriptors, fWorkspace);
   }
   Architecture_t::MaxPoolLayerBackward(
      gradients_backward, this->GetActivationGradients(), fIndexTensor, activations_backward, this->GetOutput(),
      (TCNNDescriptors<TMaxPoolLayer<Architecture_t>> &)(*fDescriptors),
      (TCNNWorkspace<TMaxPoolLayer<Architecture_t>> &)(*fWorkspace), this->GetInputHeight(), this->GetInputWidth(),
      this->GetFilterHeight(), this->GetFilterWidth(), this->GetStrideRows(), this->GetStrideCols(),
      this->GetNLocalViews());
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
   Architecture_t::InitializePoolDescriptors(fDescriptors, this);
}

template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::ReleaseDescriptors() {
   Architecture_t::ReleasePoolDescriptors(fDescriptors, this);
}

//______________________________________________________________________________
template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::InitializeWorkspace() {
#if 0  // this is for dropout and needs to be fixed
   TConvParams params(this->GetBatchSize(), this->GetInputDepth(), this->GetInputHeight(), this->GetInputWidth(),
                      this->GetDepth(), this->GetFilterHeight(), this->GetFilterWidth(),
                      this->GetStrideRows(), this->GetStrideCols(), this->GetPaddingHeight(), this->GetPaddingWidth());

   Architecture_t::InitializePoolDropoutWorkspace(fWorkspace, fDescriptors, params, this);
#endif
}

template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::FreeWorkspace() {
   //Architecture_t::FreePoolDropoutWorkspace(fWorkspace, this);
}

} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif
