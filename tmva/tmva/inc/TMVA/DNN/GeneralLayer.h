// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski 24/06/17

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////
// Contains the General Layer class, which represents the virtual   //
// base class for all Layer classes in the Deep Learning Module.    //
//////////////////////////////////////////////////////////////////////


#ifndef TMVA_DNN_GENERALLAYER
#define TMVA_DNN_GENERALLAYER

#include "Functions.h"

namespace TMVA
{
namespace DNN
{

/** \class VGeneralLayer
 
 Generic General Layer class.
 */
template<typename Architecture_t>
class VGeneralLayer
{
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

private:
    
   size_t fBatchSize;                ///< Batch size used for training and evaluation.
    
   size_t fInputDepth;               ///< The depth of the previous layer.
   size_t fInputHeight;              ///< The height of the previous layer.
   size_t fInputWidth;               ///< The width of the previous layer.
    
   size_t fDepth;                    ///< The depth of the layer.
   size_t fHeight;                   ///< The height of the layer.
   size_t fWidth;                    ///< The width of this layer.
    
   Scalar_t fDropoutProbability;     ///< Probability that an input is active.
    
   Matrix_t fWeights;                ///< The weights of the layer.
   Matrix_t fWeightGradients;        ///< Gradients w.r.t. the weights of this layer.
    
   Matrix_t fBiases;                 ///< The bias values of this layer.
   Matrix_t fBiasGradients;          ///< Gradients w.r.t. the bias values of this layer.
    
   std::vector<Matrix_t> fOutput;                ///< Activations of this layer.
   std::vector<Matrix_t> fDerivatives;           ///< First fDerivatives of the activations of this layer.
   std::vector<Matrix_t> fActivationGradients;   ///< Gradients w.r.t. the activations of this layer.

public:

   /*! Constructor, such that the number of rows and columns of the
    * weights and biases are passed as arguments. */
   VGeneralLayer(size_t BatchSize,
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
                 size_t OutputNCols);

   /*! Copy Constructor */
   VGeneralLayer(const VGeneralLayer &);
    
   /*! Virtual Destructor. */
   virtual ~VGeneralLayer();

   /*! Initialize the weights according to the given initialization method. */
   virtual void Initialize(EInitialization m);
    
   /*! Initialize the gratients to zero */
   virtual void InitializeGradients();
    
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
   size_t GetBatchSize()                    const {return fBatchSize;}
    
   size_t GetInputDepth()                   const {return fInputDepth;}
   size_t GetInputHeight()                  const {return fInputHeight;}
   size_t GetInputWidth()                   const {return fInputWidth;}
    
   size_t GetDepth()                        const {return fDepth;}
   size_t GetHeight()                       const {return fHeight;}
   size_t GetWidth()                        const {return fWidth;}
    
   Scalar_t GetDropoutProbability()         const {return fDropoutProbability;}
   void SetDropoutProbability(Scalar_t p)         {fDropoutProbability = p;}
    
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
   VGeneralLayer<Architecture_t>::VGeneralLayer(size_t batchSize,
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
                                                size_t outputNCols)
    : fBatchSize(batchSize), fInputDepth(inputDepth), fInputHeight(inputHeight),
      fInputWidth(inputWidth), fDepth(depth), fHeight(height), fWidth(width),
      fDropoutProbability(dropoutProbability), fWeights(weightsNRows, weightsNCols),
      fWeightGradients(weightsNRows, weightsNCols), fBiases(biasesNRows, biasesNCols),
      fBiasGradients(biasesNRows, biasesNCols), fOutput(), fDerivatives(), fActivationGradients()
{

   for(size_t i = 0; i < outputNSlices; i++) {
      fOutput.emplace_back(outputNRows, outputNCols);
      fDerivatives.emplace_back(outputNRows, outputNCols);
      fActivationGradients.emplace_back(outputNRows, outputNCols);
   }
}

//_________________________________________________________________________________________________
template<typename Architecture_t>
   VGeneralLayer<Architecture_t>::VGeneralLayer(const VGeneralLayer &layer)
   : fBatchSize(layer.fBatchSize), fInputDepth(layer.fInputDepth), fInputHeight(layer.fInputHeight),
     fInputWidth(layer.fInputWidth), fDepth(layer.fDepth), fHeight(layer.fHeight),
     fWidth(layer.fWidth), fDropoutProbability(layer.fDropoutProbability),
     fWeights(layer.fWeights.GetNrows(), layer.fWeights.GetNcols()),
     fWeightGradients(layer.fWeightGradients.GetNrows(), layer.fWeightGradients.GetNcols()),
     fBiases(layer.fBiases.GetNrows(), layer.fBiases.GetNcols()),
     fBiasGradients(layer.fBiasGradients.GetNrows(), layer.fBiasGradients.GetNcols()),
     fOutput(), fDerivatives(), fActivationGradients()
{
    
   Architecture_t::Copy(fBiases, layer.fBiases);
   Architecture_t::Copy(fWeights, layer.fWeights);
    
   size_t outputNSlices = layer.fOutput.size();
   size_t outputNRows = layer.GetOutputAt(0).GetNrows();
   size_t outputNCols = layer.GetOutputAt(0).GetNcols();
    
   for(size_t i = 0; i < outputNSlices; i++){
      fOutput.emplace_back(outputNRows, outputNCols);
      fDerivatives.emplace_back(outputNRows, outputNCols);
      fActivationGradients.emplace_back(outputNRows, outputNCols);
   }
}

//_________________________________________________________________________________________________
template<typename Architecture_t>
   VGeneralLayer<Architecture_t>::~VGeneralLayer()
{
    
}

//_________________________________________________________________________________________________
template<typename Architecture_t>
   auto VGeneralLayer<Architecture_t>::Initialize(EInitialization m)
-> void
{
   initialize<Architecture_t>(fWeights, m);
   initialize<Architecture_t>(fBiases,  EInitialization::kZero);
}

//_________________________________________________________________________________________________
template<typename Architecture_t>
   auto VGeneralLayer<Architecture_t>::InitializeGradients()
-> void
{
   initialize<Architecture_t>(fWeightGradients, EInitialization::kZero);
   initialize<Architecture_t>(fBiasGradients,  EInitialization::kZero);
}
    
} // namespace DNN
} // namespace TMVA


#endif
