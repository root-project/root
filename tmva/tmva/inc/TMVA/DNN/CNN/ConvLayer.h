// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TConvLayer                                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Convolutional Deep Neural Network Layer                                   *
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

#ifndef TMVA_CNN_CONVLAYER
#define TMVA_CNN_CONVLAYER

#include "TMatrix.h"

#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Functions.h"

#include <vector>
#include <iostream>

namespace TMVA {
namespace DNN {
namespace CNN {

template <typename Architecture_t>
class TConvLayer : public VGeneralLayer<Architecture_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

private:
   size_t fFilterDepth;  ///< The depth of the filter.
   size_t fFilterHeight; ///< The height of the filter.
   size_t fFilterWidth;  ///< The width of the filter.

   size_t fStrideRows; ///< The number of row pixels to slid the filter each step.
   size_t fStrideCols; ///< The number of column pixels to slid the filter each step.

   size_t fPaddingHeight; ///< The number of zero layers added top and bottom of the input.
   size_t fPaddingWidth;  ///< The number of zero layers left and right of the input.

   size_t fNLocalViewPixels; ///< The number of pixels in one local image view.
   size_t fNLocalViews;      ///< The number of local views in one image.

   Scalar_t fDropoutProbability; ///< Probability that an input is active.

   std::vector<Matrix_t> fDerivatives; ///< First fDerivatives of the activations of this layer.

   EActivationFunction fF; ///< Activation function of the layer.
   ERegularization fReg;   ///< The regularization method.
   Scalar_t fWeightDecay;  ///< The weight decay.

public:
   /*! Constructor. */
   TConvLayer(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t Depth, size_t Height,
              size_t Width, size_t WeightsNRows, size_t WeightsNCols, size_t BiasesNRows, size_t BiasesNCols,
              size_t OutputNSlices, size_t OutputNRows, size_t OutputNCols, EInitialization Init, size_t FilterDepth,
              size_t FilterHeight, size_t FilterWidth, size_t StrideRows, size_t StrideCols, size_t PaddingHeight,
              size_t PaddingWidth, Scalar_t DropoutProbability, EActivationFunction f, ERegularization Reg,
              Scalar_t WeightDecay);

   /*! Copy the conv layer provided as a pointer */
   TConvLayer(TConvLayer<Architecture_t> *layer);

   /*! Copy constructor. */
   TConvLayer(const TConvLayer &);

   /*! Destructor. */
   ~TConvLayer();

   /*! Computes activation of the layer for the given input. The input
   * must be in 3D tensor form with the different matrices corresponding to
   * different events in the batch. Computes activations as well as
   * the first partial derivative of the activation function at those
   * activations. */
   void Forward(std::vector<Matrix_t> &input, bool applyDropout = false);

   /*! Compute weight, bias and activation gradients. Uses the precomputed
    *  first partial derviatives of the activation function computed during
    *  forward propagation and modifies them. Must only be called directly
    *  at the corresponding call to Forward(...). */
   void Backward(std::vector<Matrix_t> &gradients_backward, const std::vector<Matrix_t> &activations_backward,
                 std::vector<Matrix_t> &inp1, std::vector<Matrix_t> &inp2);

   /*! Prints the info about the layer. */
   void Print() const;

   /*! Getters */
   size_t GetFilterDepth() const { return fFilterDepth; }
   size_t GetFilterHeight() const { return fFilterHeight; }
   size_t GetFilterWidth() const { return fFilterWidth; }

   size_t GetStrideRows() const { return fStrideRows; }
   size_t GetStrideCols() const { return fStrideCols; }

   size_t GetPaddingHeight() const { return fPaddingHeight; }
   size_t GetPaddingWidth() const { return fPaddingWidth; }

   size_t GetNLocalViewPixels() const { return fNLocalViewPixels; }
   size_t GetNLocalViews() const { return fNLocalViews; }

   Scalar_t GetDropoutProbability() const { return fDropoutProbability; }

   const std::vector<Matrix_t> &GetDerivatives() const { return fDerivatives; }
   std::vector<Matrix_t> &GetDerivatives() { return fDerivatives; }

   Matrix_t &GetDerivativesAt(size_t i) { return fDerivatives[i]; }
   const Matrix_t &GetDerivativesAt(size_t i) const { return fDerivatives[i]; }

   EActivationFunction GetActivationFunction() const { return fF; }
   ERegularization GetRegularization() const { return fReg; }
   Scalar_t GetWeightDecay() const { return fWeightDecay; }
};

//
//
//  Conv Layer Class - Implementation
//______________________________________________________________________________
template <typename Architecture_t>
TConvLayer<Architecture_t>::TConvLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                       size_t depth, size_t height, size_t width, size_t weightsNRows,
                                       size_t weightsNCols, size_t biasesNRows, size_t biasesNCols,
                                       size_t outputNSlices, size_t outputNRows, size_t outputNCols,
                                       EInitialization init, size_t filterDepth, size_t filterHeight,
                                       size_t filterWidth, size_t strideRows, size_t strideCols, size_t paddingHeight,
                                       size_t paddingWidth, Scalar_t dropoutProbability, EActivationFunction f,
                                       ERegularization reg, Scalar_t weightDecay)
   : VGeneralLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, depth, height, width, 1,
                                   weightsNRows, weightsNCols, 1, biasesNRows, biasesNCols, outputNSlices, outputNRows,
                                   outputNCols, init),
     fFilterDepth(filterDepth), fFilterHeight(filterHeight), fFilterWidth(filterWidth), fStrideRows(strideRows),
     fStrideCols(strideCols), fPaddingHeight(paddingHeight), fPaddingWidth(paddingWidth),
     fNLocalViewPixels(filterDepth * filterHeight * filterWidth), fNLocalViews(height * width),
     fDropoutProbability(dropoutProbability), fDerivatives(), fF(f), fReg(reg), fWeightDecay(weightDecay)
{
   for (size_t i = 0; i < outputNSlices; i++) {
      fDerivatives.emplace_back(outputNRows, outputNCols);
   }
}

//______________________________________________________________________________
template <typename Architecture_t>
TConvLayer<Architecture_t>::TConvLayer(TConvLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer), fFilterDepth(layer->GetFilterDepth()),
     fFilterHeight(layer->GetFilterHeight()), fFilterWidth(layer->GetFilterWidth()),
     fStrideRows(layer->GetStrideRows()), fStrideCols(layer->GetStrideCols()),
     fPaddingHeight(layer->GetPaddingHeight()), fPaddingWidth(layer->GetPaddingWidth()),
     fNLocalViewPixels(layer->GetNLocalViewPixels()), fNLocalViews(layer->GetNLocalViews()),
     fDropoutProbability(layer->GetDropoutProbability()), fF(layer->GetActivationFunction()),
     fReg(layer->GetRegularization()), fWeightDecay(layer->GetWeightDecay())
{
   size_t outputNSlices = (layer->GetDerivatives()).size();
   size_t outputNRows = 0;
   size_t outputNCols = 0;

   for (size_t i = 0; i < outputNSlices; i++) {
      outputNRows = (layer->GetDerivativesAt(i)).GetNrows();
      outputNCols = (layer->GetDerivativesAt(i)).GetNcols();

      fDerivatives.emplace_back(outputNRows, outputNCols);
   }
}

//______________________________________________________________________________
template <typename Architecture_t>
TConvLayer<Architecture_t>::TConvLayer(const TConvLayer &convLayer)
   : VGeneralLayer<Architecture_t>(convLayer), fFilterDepth(convLayer.fFilterDepth),
     fFilterHeight(convLayer.fFilterHeight), fFilterWidth(convLayer.fFilterWidth), fStrideRows(convLayer.fStrideRows),
     fStrideCols(convLayer.fStrideCols), fPaddingHeight(convLayer.fPaddingHeight),
     fPaddingWidth(convLayer.fPaddingWidth), fNLocalViewPixels(convLayer.fNLocalViewPixels),
     fNLocalViews(convLayer.fNLocalViews), fDropoutProbability(convLayer.fDropoutProbability), fF(convLayer.fF),
     fReg(convLayer.fReg), fWeightDecay(convLayer.fWeightDecay)
{
   size_t outputNSlices = convLayer.fDerivatives.size();
   size_t outputNRows = convLayer.GetDerivativesAt(0).GetNrows();
   size_t outputNCols = convLayer.GetDerivativesAt(0).GetNcols();

   for (size_t i = 0; i < outputNSlices; i++) {
      fDerivatives.emplace_back(outputNRows, outputNCols);
   }
}

//______________________________________________________________________________
template <typename Architecture_t>
TConvLayer<Architecture_t>::~TConvLayer()
{
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TConvLayer<Architecture_t>::Forward(std::vector<Matrix_t> &input, bool applyDropout) -> void
{
   for (size_t i = 0; i < this->GetBatchSize(); i++) {

      if (applyDropout && (this->GetDropoutProbability() != 1.0)) {
         Architecture_t::Dropout(input[i], this->GetDropoutProbability());
      }

      Matrix_t inputTr(this->GetNLocalViews(), this->GetNLocalViewPixels());
      Architecture_t::Im2col(inputTr, input[i], this->GetInputHeight(), this->GetInputWidth(), this->GetFilterHeight(),
                             this->GetFilterWidth(), this->GetStrideRows(), this->GetStrideCols(),
                             this->GetPaddingHeight(), this->GetPaddingWidth());

      Architecture_t::MultiplyTranspose(this->GetOutputAt(i), this->GetWeightsAt(0), inputTr);
      Architecture_t::AddConvBiases(this->GetOutputAt(i), this->GetBiasesAt(0));

      evaluateDerivative<Architecture_t>(this->GetDerivativesAt(i), fF, this->GetOutputAt(i));
      evaluate<Architecture_t>(this->GetOutputAt(i), fF);
   }
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TConvLayer<Architecture_t>::Backward(std::vector<Matrix_t> &gradients_backward,
                                          const std::vector<Matrix_t> &activations_backward,
                                          std::vector<Matrix_t> & /*inp1*/, std::vector<Matrix_t> &
                                          /*inp2*/) -> void
{
   Architecture_t::ConvLayerBackward(
      gradients_backward, this->GetWeightGradientsAt(0), this->GetBiasGradientsAt(0), this->GetDerivatives(),
      this->GetActivationGradients(), this->GetWeightsAt(0), activations_backward, this->GetBatchSize(),
      this->GetInputHeight(), this->GetInputWidth(), this->GetDepth(), this->GetHeight(), this->GetWidth(),
      this->GetFilterDepth(), this->GetFilterHeight(), this->GetFilterWidth(), this->GetNLocalViews());

   addRegularizationGradients<Architecture_t>(this->GetWeightGradientsAt(0), this->GetWeightsAt(0),
                                              this->GetWeightDecay(), this->GetRegularization());
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TConvLayer<Architecture_t>::Print() const -> void
{
   std::cout << "\t\t CONV LAYER: " << std::endl;
   std::cout << "\t\t\t Width = " << this->GetWidth() << std::endl;
   std::cout << "\t\t\t Height = " << this->GetHeight() << std::endl;
   std::cout << "\t\t\t Depth = " << this->GetDepth() << std::endl;

   std::cout << "\t\t\t Filter Width = " << this->GetFilterWidth() << std::endl;
   std::cout << "\t\t\t Filter Height = " << this->GetFilterHeight() << std::endl;
   std::cout << "\t\t\t Local Views = " << this->GetNLocalViews()  << std::endl;
   std::cout << "\t\t\t Activation Function = " << static_cast<int>(fF) << std::endl;
}

} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif
