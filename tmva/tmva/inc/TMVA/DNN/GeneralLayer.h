// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TGeneralLayer                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      General Deep Neural Network Layer                                         *
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

#ifndef TMVA_DNN_GENERALLAYER
#define TMVA_DNN_GENERALLAYER

namespace TMVA {
namespace DNN {

/** \class VGeneralLayer
    Generic General Layer class.

    This class represents the general class for all layers in the Deep Learning
    Module.
 */
template <typename Architecture_t>
class VGeneralLayer {
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

protected:
   size_t fBatchSize; ///< Batch size used for training and evaluation

   size_t fInputDepth;  ///< The depth of the previous layer or input.
   size_t fInputHeight; ///< The height of the previous layer or input.
   size_t fInputWidth;  ///< The width of the previous layer or input.

   size_t fDepth;  ///< The depth of the layer.
   size_t fHeight; ///< The height of the layer.
   size_t fWidth;  ///< The width of this layer.

   bool fIsTraining; ///< Flag indicatig the mode

   Matrix_t fWeights; ///< The weights associated to the layer.
   Matrix_t fBiases;  ///< The biases associated to the layer.

   Matrix_t fWeightGradients; ///< Gradients w.r.t. the weights of the layer.
   Matrix_t fBiasGradients;   ///< Gradients w.r.t. the bias values of the layer.

   std::vector<Matrix_t> fOutput;              ///< Activations of this layer.
   std::vector<Matrix_t> fActivationGradients; ///< Gradients w.r.t. the activations of this layer.

   EInitialization fInit; ///< The initialization method.

public:
   /*! Constructor */
   VGeneralLayer(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t Depth,
                 size_t Height, size_t Width, size_t WeightsNRows, size_t WeightsNCols, size_t BiasesNRows,
                 size_t BiasesNCols, size_t OutputNSlices, size_t OutputNRows, size_t OutputNCols,
                 EInitialization Init);

   /*! Copy the layer provided as a pointer */
   VGeneralLayer(VGeneralLayer<Architecture_t> *layer);

   /*! Copy Constructor */
   VGeneralLayer(const VGeneralLayer &);

   /*! Virtual Destructor. */
   virtual ~VGeneralLayer();

   /*! Initialize the weights and biases according to the given initialization method. */
   void Initialize();

   /*! Computes activation of the layer for the given input. The input
    * must be in 3D tensor form with the different matrices corresponding to
    * different events in the batch.  */
   virtual void Forward(std::vector<Matrix_t> input, bool applyDropout = false) = 0;

   /*! Backpropagates the error. Must only be called directly at the corresponding
    *  call to Forward(...). */
   virtual void Backward(std::vector<Matrix_t> &gradients_backward,
                         const std::vector<Matrix_t> &activations_backward) = 0;

   /*! Updates the weights and biases, given the learning rate */
   void Update(const Scalar_t learningRate);

   /*! Updates the weights, given the gradients and the learning rate, */
   void UpdateWeights(const Matrix_t &weightGradients, const Scalar_t learningRate);

   /*! Updates the biases, given the gradients and the learning rate. */
   void UpdateBiases(const Matrix_t &biasGradients, const Scalar_t learningRate);

   /*! Prints the info about the layer. */
   virtual void Print() const = 0;

   /*! Getters */
   size_t GetBatchSize() const { return fBatchSize; }
   size_t GetInputDepth() const { return fInputDepth; }
   size_t GetInputHeight() const { return fInputHeight; }
   size_t GetInputWidth() const { return fInputWidth; }
   size_t GetDepth() const { return fDepth; }
   size_t GetHeight() const { return fHeight; }
   size_t GetWidth() const { return fWidth; }
   bool IsTraining() const { return fIsTraining; }

   const Matrix_t &GetWeights() const { return fWeights; }
   Matrix_t &GetWeights() { return fWeights; }

   const Matrix_t &GetBiases() const { return fBiases; }
   Matrix_t &GetBiases() { return fBiases; }

   const Matrix_t &GetWeightGradients() const { return fWeightGradients; }
   Matrix_t &GetWeightGradients() { return fWeightGradients; }

   const Matrix_t &GetBiasGradients() const { return fBiasGradients; }
   Matrix_t &GetBiasGradients() { return fBiasGradients; }

   const std::vector<Matrix_t> &GetOutput() const { return fOutput; }
   std::vector<Matrix_t> &GetOutput() { return fOutput; }

   const std::vector<Matrix_t> &GetActivationGradients() const { return fActivationGradients; }
   std::vector<Matrix_t> &GetActivationGradients() { return fActivationGradients; }

   Matrix_t &GetOutputAt(size_t i) { return fOutput[i]; }
   const Matrix_t &GetOutputAt(size_t i) const { return fOutput[i]; }

   Matrix_t &GetActivationGradientsAt(size_t i) { return fActivationGradients[i]; }
   const Matrix_t &GetActivationGradientsAt(size_t i) const { return fActivationGradients[i]; }

   EInitialization GetInitialization() const { return fInit; }

   /*! Setters */
   void SetBatchSize(size_t batchSize) { fBatchSize = batchSize; }
   void SetInputDepth(size_t inputDepth) { fInputDepth = inputDepth; }
   void SetInputHeight(size_t inputHeight) { fInputHeight = inputHeight; }
   void SetInputWidth(size_t inputWidth) { fInputWidth = inputWidth; }
   void SetDepth(size_t depth) { fDepth = depth; }
   void SetHeight(size_t height) { fHeight = height; }
   void SetWidth(size_t width) { fWidth = width; }
   void SetIsTraining(bool isTraining) { fIsTraining = isTraining; }
};

//
//
//  The General Layer Class - Implementation
//_________________________________________________________________________________________________
template <typename Architecture_t>
VGeneralLayer<Architecture_t>::VGeneralLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                             size_t depth, size_t height, size_t width, size_t weightsNRows,
                                             size_t weightsNCols, size_t biasesNRows, size_t biasesNCols,
                                             size_t outputNSlices, size_t outputNRows, size_t outputNCols,
                                             EInitialization init)
   : fBatchSize(batchSize), fInputDepth(inputDepth), fInputHeight(inputHeight), fInputWidth(inputWidth), fDepth(depth),
     fHeight(height), fWidth(width), fIsTraining(true), fWeights(weightsNRows, weightsNCols),
     fBiases(biasesNRows, biasesNCols), fWeightGradients(weightsNRows, weightsNCols),
     fBiasGradients(biasesNRows, biasesNCols), fOutput(), fActivationGradients(), fInit(init)
{
   for (size_t i = 0; i < outputNSlices; i++) {
      fOutput.emplace_back(outputNRows, outputNCols);
      fActivationGradients.emplace_back(outputNRows, outputNCols);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
VGeneralLayer<Architecture_t>::VGeneralLayer(VGeneralLayer<Architecture_t> *layer)
   : fBatchSize(layer->GetBatchSize()), fInputDepth(layer->GetInputDepth()), fInputHeight(layer->GetInputHeight()),
     fInputWidth(layer->GetInputWidth()), fDepth(layer->GetDepth()), fHeight(layer->GetHeight()),
     fWidth(layer->GetWidth()), fIsTraining(layer->IsTraining()),
     fWeights((layer->GetWeights()).GetNrows(), (layer->GetWeights()).GetNcols()),
     fBiases((layer->GetBiases()).GetNrows(), (layer->GetBiases()).GetNcols()),
     fWeightGradients((layer->GetWeightGradients()).GetNrows(), (layer->GetWeightGradients()).GetNcols()),
     fBiasGradients((layer->GetBiasGradients()).GetNrows(), (layer->GetBiasGradients()).GetNcols()), fOutput(),
     fActivationGradients(), fInit(layer->GetInitialization())
{
   Architecture_t::Copy(fWeights, layer->GetWeights());
   Architecture_t::Copy(fBiases, layer->GetBiases());

   size_t outputNSlices = (layer->GetOutput()).size();
   size_t outputNRows = (layer->GetOutputAt(0)).GetNrows();
   size_t outputNCols = (layer->GetOutputAt(0)).GetNcols();

   for (size_t i = 0; i < outputNSlices; i++) {
      fOutput.emplace_back(outputNRows, outputNCols);
      fActivationGradients.emplace_back(outputNRows, outputNCols);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
VGeneralLayer<Architecture_t>::VGeneralLayer(const VGeneralLayer &layer)
   : fBatchSize(layer.fBatchSize), fInputDepth(layer.fInputDepth), fInputHeight(layer.fInputHeight),
     fInputWidth(layer.fInputWidth), fDepth(layer.fDepth), fHeight(layer.fHeight), fWidth(layer.fWidth),
     fIsTraining(layer.fIsTraining), fWeights(layer.fWeights.GetNrows(), layer.fWeights.GetNcols()),
     fBiases(layer.fBiases.GetNrows(), layer.fBiases.GetNcols()),
     fWeightGradients(layer.fWeightGradients.GetNrows(), layer.fWeightGradients.GetNcols()),
     fBiasGradients(layer.fBiasGradients.GetNrows(), layer.fBiasGradients.GetNcols()), fOutput(),
     fActivationGradients(), fInit(layer.fInit)
{
   Architecture_t::Copy(fWeights, layer.fWeights);
   Architecture_t::Copy(fBiases, layer.fBiases);

   size_t outputNSlices = layer.fOutput.size();
   size_t outputNRows = layer.GetOutputAt(0).GetNrows();
   size_t outputNCols = layer.GetOutputAt(0).GetNcols();

   for (size_t i = 0; i < outputNSlices; i++) {
      fOutput.emplace_back(outputNRows, outputNCols);
      fActivationGradients.emplace_back(outputNRows, outputNCols);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
VGeneralLayer<Architecture_t>::~VGeneralLayer()
{
   // Nothing to do here.
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::Initialize() -> void
{
   initialize<Architecture_t>(fWeights, this->GetInitialization());
   initialize<Architecture_t>(fBiases, this->GetInitialization());

   initialize<Architecture_t>(fWeightGradients, EInitialization::kZero);
   initialize<Architecture_t>(fBiasGradients, EInitialization::kZero);
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::Update(const Scalar_t learningRate) -> void
{
   this->UpdateWeights(this->GetWeightGradients(), learningRate);
   this->UpdateBiases(this->GetBiasGradients(), learningRate);
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::UpdateWeights(const Matrix_t &weightGradients, const Scalar_t learningRate) -> void
{
   Architecture_t::ScaleAdd(this->GetWeights(), weightGradients, -learningRate);
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::UpdateBiases(const Matrix_t &biasGradients, const Scalar_t learningRate) -> void
{
   Architecture_t::ScaleAdd(this->GetBiases(), biasGradients, -learningRate);
}

} // namespace DNN
} // namespace TMVA

#endif