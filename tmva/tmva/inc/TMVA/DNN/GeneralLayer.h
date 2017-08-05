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

   size_t fLayerType; ///< Type of Layer in Network

   std::vector<Matrix_t> fWeights; ///< The weights associated to the layer.
   std::vector<Matrix_t> fBiases;  ///< The biases associated to the layer.

   std::vector<Matrix_t> fWeightGradients; ///< Gradients w.r.t. the weights of the layer.
   std::vector<Matrix_t> fBiasGradients;   ///< Gradients w.r.t. the bias values of the layer.

   std::vector<Matrix_t> fOutput;              ///< Activations of this layer.
   std::vector<Matrix_t> fActivationGradients; ///< Gradients w.r.t. the activations of this layer.

   EInitialization fInit; ///< The initialization method.

public:
   /*! Constructor */
   VGeneralLayer(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t Depth,
                 size_t Height, size_t Width, size_t WeightsNSlices, size_t WeightsNRows, size_t WeightsNCols,
                 size_t BiasesNSlices, size_t BiasesNRows, size_t BiasesNCols, size_t OutputNSlices, size_t OutputNRows,
                 size_t OutputNCols, size_t LayerType, EInitialization Init);

   /*! General Constructor with different weights dimension */
   VGeneralLayer(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t Depth,
                 size_t Height, size_t Width, size_t WeightsNSlices, std::vector<size_t> WeightsNRows,
                 std::vector<size_t> WeightsNCols, size_t BiasesNSlices, std::vector<size_t> BiasesNRows,
                 std::vector<size_t> BiasesNCols, size_t OutputNSlices, size_t OutputNRows, size_t OutputNCols,
                 size_t LayerType, EInitialization Init);

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
   virtual void Backward(std::vector<Matrix_t> &gradients_backward, const std::vector<Matrix_t> &activations_backward,
                         std::vector<Matrix_t> &inp1, std::vector<Matrix_t> &inp2) = 0;

   /*! Updates the weights and biases, given the learning rate */
   void Update(const Scalar_t learningRate);

   /*! Updates the weights, given the gradients and the learning rate, */
   void UpdateWeights(const std::vector<Matrix_t> &weightGradients, const Scalar_t learningRate);

   /*! Updates the biases, given the gradients and the learning rate. */
   void UpdateBiases(const std::vector<Matrix_t> &biasGradients, const Scalar_t learningRate);

   /*! Updates the weight gradients, given some other weight gradients and learning rate. */
   void UpdateWeightGradients(const std::vector<Matrix_t> &weightGradients, const Scalar_t learningRate);

   /*! Updates the bias gradients, given some other weight gradients and learning rate. */
   void UpdateBiasGradients(const std::vector<Matrix_t> &biasGradients, const Scalar_t learningRate);

   /*! Copies the weights provided as an input.  */
   void CopyWeights(const std::vector<Matrix_t> &otherWeights);

   /*! Copies the biases provided as an input. */
   void CopyBiases(const std::vector<Matrix_t> &otherBiases);

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
   size_t GetLayerType() const { return fLayerType; }
   bool IsTraining() const { return fIsTraining; }

   const std::vector<Matrix_t> &GetWeights() const { return fWeights; }
   std::vector<Matrix_t> &GetWeights() { return fWeights; }

   const Matrix_t &GetWeightsAt(size_t i) const { return fWeights[i]; }
   Matrix_t &GetWeightsAt(size_t i) { return fWeights[i]; }

   const std::vector<Matrix_t> &GetBiases() const { return fBiases; }
   std::vector<Matrix_t> &GetBiases() { return fBiases; }

   const Matrix_t &GetBiasesAt(size_t i) const { return fBiases[i]; }
   Matrix_t &GetBiasesAt(size_t i) { return fBiases[i]; }

   const std::vector<Matrix_t> &GetWeightGradients() const { return fWeightGradients; }
   std::vector<Matrix_t> &GetWeightGradients() { return fWeightGradients; }

   const Matrix_t &GetWeightGradientsAt(size_t i) const { return fWeightGradients[i]; }
   Matrix_t &GetWeightGradientsAt(size_t i) { return fWeightGradients[i]; }

   const std::vector<Matrix_t> &GetBiasGradients() const { return fBiasGradients; }
   std::vector<Matrix_t> &GetBiasGradients() { return fBiasGradients; }

   const Matrix_t &GetBiasGradientsAt(size_t i) const { return fBiasGradients[i]; }
   Matrix_t &GetBiasGradientsAt(size_t i) { return fBiasGradients[i]; }

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
   void SetLayerType(size_t layerType) { fLayerType = layerType; }
   void SetIsTraining(bool isTraining) { fIsTraining = isTraining; }
};

//
//
//  The General Layer Class - Implementation
//_________________________________________________________________________________________________
template <typename Architecture_t>
VGeneralLayer<Architecture_t>::VGeneralLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                             size_t depth, size_t height, size_t width, size_t weightsNSlices,
                                             size_t weightsNRows, size_t weightsNCols, size_t biasesNSlices,
                                             size_t biasesNRows, size_t biasesNCols, size_t outputNSlices,
                                             size_t outputNRows, size_t outputNCols, size_t layerType,
                                             EInitialization init)
   : fBatchSize(batchSize), fInputDepth(inputDepth), fInputHeight(inputHeight), fInputWidth(inputWidth), fDepth(depth),
     fHeight(height), fWidth(width), fIsTraining(true), fWeights(), fBiases(), fWeightGradients(), fBiasGradients(),
     fOutput(), fActivationGradients(), fInit(init), fLayerType(layerType)
{

   for (size_t i = 0; i < weightsNSlices; i++) {
      fWeights.emplace_back(weightsNRows, weightsNCols);
      fWeightGradients.emplace_back(weightsNRows, weightsNCols);
   }

   for (size_t i = 0; i < biasesNSlices; i++) {
      fBiases.emplace_back(biasesNRows, biasesNCols);
      fBiasGradients.emplace_back(biasesNRows, biasesNCols);
   }

   for (size_t i = 0; i < outputNSlices; i++) {
      fOutput.emplace_back(outputNRows, outputNCols);
      fActivationGradients.emplace_back(outputNRows, outputNCols);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
VGeneralLayer<Architecture_t>::VGeneralLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                             size_t depth, size_t height, size_t width, size_t weightsNSlices,
                                             std::vector<size_t> weightsNRows, std::vector<size_t> weightsNCols,
                                             size_t biasesNSlices, std::vector<size_t> biasesNRows,
                                             std::vector<size_t> biasesNCols, size_t outputNSlices, size_t outputNRows,
                                             size_t outputNCols, size_t layerType, EInitialization init)
   : fBatchSize(batchSize), fInputDepth(inputDepth), fInputHeight(inputHeight), fInputWidth(inputWidth), fDepth(depth),
     fHeight(height), fWidth(width), fIsTraining(true), fWeights(), fBiases(), fWeightGradients(), fBiasGradients(),
     fOutput(), fActivationGradients(), fInit(init), fLayerType(layerType)
{

   for (size_t i = 0; i < weightsNSlices; i++) {
      fWeights.emplace_back(weightsNRows[i], weightsNCols[i]);
      fWeightGradients.emplace_back(weightsNRows[i], weightsNCols[i]);
   }

   for (size_t i = 0; i < biasesNSlices; i++) {
      fBiases.emplace_back(biasesNRows[i], biasesNCols[i]);
      fBiasGradients.emplace_back(biasesNRows[i], biasesNCols[i]);
   }

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
     fWidth(layer->GetWidth()), fIsTraining(layer->IsTraining()), fWeights(), fBiases(), fWeightGradients(),
     fBiasGradients(), fOutput(), fActivationGradients(), fInit(layer->GetInitialization()),
     fLayerType(layer->GetLayerType())
{
   size_t weightsNSlices = (layer->GetWeights()).size();
   size_t weightsNRows = 0;
   size_t weightsNCols = 0;

   for (size_t i = 0; i < weightsNSlices; i++) {
      weightsNRows = (layer->GetWeightsAt(i)).GetNrows();
      weightsNCols = (layer->GetWeightsAt(i)).GetNcols();

      fWeights.emplace_back(weightsNRows, weightsNCols);
      fWeightGradients.emplace_back(weightsNRows, weightsNCols);

      Architecture_t::Copy(fWeights[i], layer->GetWeightsAt(i));
   }

   size_t biasesNSlices = (layer->GetBiases()).size();
   size_t biasesNRows = 0;
   size_t biasesNCols = 0;

   for (size_t i = 0; i < biasesNSlices; i++) {
      biasesNRows = (layer->GetBiasesAt(i)).GetNrows();
      biasesNCols = (layer->GetBiasesAt(i)).GetNcols();

      fBiases.emplace_back(biasesNRows, biasesNCols);
      fBiasGradients.emplace_back(biasesNRows, biasesNCols);

      Architecture_t::Copy(fBiases[i], layer->GetBiasesAt(i));
   }

   size_t outputNSlices = (layer->GetOutput()).size();
   size_t outputNRows = 0;
   size_t outputNCols = 0;

   for (size_t i = 0; i < outputNSlices; i++) {
      outputNRows = (layer->GetOutputAt(i)).GetNrows();
      outputNCols = (layer->GetOutputAt(i)).GetNcols();

      fOutput.emplace_back(outputNRows, outputNCols);
      fActivationGradients.emplace_back(outputNRows, outputNCols);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
VGeneralLayer<Architecture_t>::VGeneralLayer(const VGeneralLayer &layer)
   : fBatchSize(layer.fBatchSize), fInputDepth(layer.fInputDepth), fInputHeight(layer.fInputHeight),
     fInputWidth(layer.fInputWidth), fDepth(layer.fDepth), fHeight(layer.fHeight), fWidth(layer.fWidth),
     fIsTraining(layer.fIsTraining), fWeights(), fBiases(), fWeightGradients(), fBiasGradients(), fOutput(),
     fActivationGradients(), fInit(layer.fInit), fLayerType(layer.fLayerType)
{
   size_t weightsNSlices = layer.fWeights.size();
   size_t weightsNRows = 0;
   size_t weightsNCols = 0;

   for (size_t i = 0; i < weightsNSlices; i++) {
      weightsNRows = (layer.fWeights[i]).GetNrows();
      weightsNCols = (layer.fWeights[i]).GetNcols();

      fWeights.emplace_back(weightsNRows, weightsNCols);
      fWeightGradients.emplace_back(weightsNRows, weightsNCols);

      Architecture_t::Copy(fWeights[i], layer.fWeights[i]);
   }

   size_t biasesNSlices = layer.fBiases.size();
   size_t biasesNRows = 0;
   size_t biasesNCols = 0;

   for (size_t i = 0; i < biasesNSlices; i++) {
      biasesNRows = (layer.fBiases[i]).GetNrows();
      biasesNCols = (layer.fBiases[i]).GetNcols();

      fBiases.emplace_back(biasesNRows, biasesNCols);
      fBiasGradients.emplace_back(biasesNRows, biasesNCols);

      Architecture_t::Copy(fBiases[i], layer.fBiases[i]);
   }

   size_t outputNSlices = layer.fOutput.size();
   size_t outputNRows = 0;
   size_t outputNCols = 0;

   for (size_t i = 0; i < outputNSlices; i++) {
      outputNRows = (layer.fOutput[i]).GetNrows();
      outputNCols = (layer.fOutput[i]).GetNcols();

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
   for (size_t i = 0; i < fWeights.size(); i++) {
      initialize<Architecture_t>(fWeights[i], this->GetInitialization());
      initialize<Architecture_t>(fWeightGradients[i], EInitialization::kZero);
   }

   for (size_t i = 0; i < fBiases.size(); i++) {
      initialize<Architecture_t>(fBiases[i], this->GetInitialization());
      initialize<Architecture_t>(fBiasGradients[i], EInitialization::kZero);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::Update(const Scalar_t learningRate) -> void
{
   this->UpdateWeights(fWeightGradients, learningRate);
   this->UpdateBiases(fBiasGradients, learningRate);
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::UpdateWeights(const std::vector<Matrix_t> &weightGradients,
                                                  const Scalar_t learningRate) -> void
{
   for (size_t i = 0; i < fWeights.size(); i++) {
      Architecture_t::ScaleAdd(fWeights[i], weightGradients[i], -learningRate);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::UpdateBiases(const std::vector<Matrix_t> &biasGradients,
                                                 const Scalar_t learningRate) -> void
{

   for (size_t i = 0; i < fBiases.size(); i++) {
      Architecture_t::ScaleAdd(fBiases[i], biasGradients[i], -learningRate);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::UpdateWeightGradients(const std::vector<Matrix_t> &weightGradients,
                                                          const Scalar_t learningRate) -> void
{
   for (size_t i = 0; i < fWeightGradients.size(); i++) {
      Architecture_t::ScaleAdd(fWeightGradients[i], weightGradients[i], -learningRate);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::UpdateBiasGradients(const std::vector<Matrix_t> &biasGradients,
                                                        const Scalar_t learningRate) -> void
{
   for (size_t i = 0; i < fBiasGradients.size(); i++) {
      Architecture_t::ScaleAdd(fBiasGradients[i], biasGradients[i], -learningRate);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::CopyWeights(const std::vector<Matrix_t> &otherWeights) -> void
{
   for (size_t i = 0; i < fWeights.size(); i++) {
      Architecture_t::Copy(fWeights[i], otherWeights[i]);
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto VGeneralLayer<Architecture_t>::CopyBiases(const std::vector<Matrix_t> &otherBiases) -> void
{
   for (size_t i = 0; i < fBiases.size(); i++) {
      Architecture_t::Copy(fBiases[i], otherBiases[i]);
   }
}

} // namespace DNN
} // namespace TMVA

#endif
