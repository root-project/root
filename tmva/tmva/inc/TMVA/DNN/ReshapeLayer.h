// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TReshapeLayer                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Reshape Deep Neural Network Layer                                         *
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

#ifndef TMVA_DNN_RESHAPELAYER
#define TMVA_DNN_RESHAPELAYER

#include "TMatrix.h"

#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Functions.h"

#include <iostream>

namespace TMVA {
namespace DNN {

template <typename Architecture_t>
class TReshapeLayer : public VGeneralLayer<Architecture_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

private:
   bool fFlattening; ///< Whather the layer is doing flattening

public:
   /*! Constructor */
   TReshapeLayer(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t Depth,
                 size_t Height, size_t Width, size_t OutputNSlices, size_t OutputNRows, size_t OutputNCols,
                 bool Flattening);

   /*! Copy the reshape layer provided as a pointer */
   TReshapeLayer(TReshapeLayer<Architecture_t> *layer);

   /*! Copy Constructor */
   TReshapeLayer(const TReshapeLayer &);

   /*! Destructor. */
   ~TReshapeLayer();

   /*! The input must be in 3D tensor form with the different matrices
    *  corresponding to different events in the batch. It transforms the
    *  input matrices. */
   void Forward(std::vector<Matrix_t> &input, bool applyDropout = false);

   void Backward(std::vector<Matrix_t> &gradients_backward, const std::vector<Matrix_t> &activations_backward,
                 std::vector<Matrix_t> &inp1, std::vector<Matrix_t> &inp2);

   /*! Prints the info about the layer. */
   void Print() const;

   /*! TODO Add documentation 
    * Does this layer flatten? (necessary for DenseLayer)
    * B x D1 x D2 --> 1 x B x (D1 * D2) */
   bool isFlattening() const { return fFlattening; }
};

//
//
//  The Reshape Layer Class - Implementation
//_________________________________________________________________________________________________
template <typename Architecture_t>
TReshapeLayer<Architecture_t>::TReshapeLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                             size_t depth, size_t height, size_t width, size_t outputNSlices,
                                             size_t outputNRows, size_t outputNCols, bool flattening)
   : VGeneralLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, depth, height, width, 0, 0, 0, 0, 0,
                                   0, outputNSlices, outputNRows, outputNCols, EInitialization::kZero),
     fFlattening(flattening)
{
   if (this->GetInputDepth() * this->GetInputHeight() * this->GetInputWidth() !=
       this->GetDepth() * this->GetHeight() * this->GetWidth()) {
      std::cout << "Reshape Dimensions not compatible \n" 
        << this->GetInputDepth() << " x " << this->GetInputHeight() << " x " << this->GetInputWidth() << " --> "
        << this->GetDepth() << " x " << this->GetHeight() << " x " << this->GetWidth() << std::endl;
      return;
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
TReshapeLayer<Architecture_t>::TReshapeLayer(TReshapeLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer), fFlattening(layer->isFlattening())
{
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
TReshapeLayer<Architecture_t>::TReshapeLayer(const TReshapeLayer &layer)
   : VGeneralLayer<Architecture_t>(layer), fFlattening(layer.fFlattening)
{
   // Nothing to do here.
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
TReshapeLayer<Architecture_t>::~TReshapeLayer()
{
   // Nothing to do here.
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto TReshapeLayer<Architecture_t>::Forward(std::vector<Matrix_t> &input, bool applyDropout) -> void
{
   if (fFlattening) {
      size_t size = input.size();
      size_t nRows = input[0].GetNrows();
      size_t nCols = input[0].GetNcols();
      Architecture_t::Flatten(this->GetOutputAt(0), input, size, nRows, nCols);
   } else {
      for (size_t i = 0; i < this->GetBatchSize(); i++) {
         Architecture_t::Reshape(this->GetOutputAt(i), input[i]);
      }
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto TReshapeLayer<Architecture_t>::Backward(std::vector<Matrix_t> &gradients_backward,
                                             const std::vector<Matrix_t> &activations_backward,
                                             std::vector<Matrix_t> &inp1, std::vector<Matrix_t> &inp2) -> void
{
   if (fFlattening) {
      size_t size = gradients_backward.size();
      size_t nRows = gradients_backward[0].GetNrows();
      size_t nCols = gradients_backward[0].GetNcols();
      Architecture_t::Deflatten(gradients_backward, this->GetActivationGradientsAt(0), size, nRows, nCols);
   } else {
      for (size_t i = 0; i < this->GetBatchSize(); i++) {
         Architecture_t::Reshape(gradients_backward[i], this->GetActivationGradientsAt(i));
      }
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto TReshapeLayer<Architecture_t>::Print() const -> void
{
}

} // namespace DNN
} // namespace TMVA

#endif
