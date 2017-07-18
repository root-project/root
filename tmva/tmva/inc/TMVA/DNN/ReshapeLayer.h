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

namespace TMVA {
namespace DNN {

template <typename Architecture_t>
class TReshapeLayer : public VGeneralLayer<Architecture_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

   /*! Constructor */
   TReshapeLayer(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t Depth,
                 size_t Height, size_t Width, size_t OutputNSlices, size_t OutputNRows, size_t OutputNCols);

   /*! Copy the reshape layer provided as a pointer */
   TReshapeLayer(TReshapeLayer<Architecture_t> *layer);

   /*! Copy Constructor */
   TReshapeLayer(const TReshapeLayer &);

   /*! Destructor. */
   ~TReshapeLayer();

   /*! The input must be in 3D tensor form with the different matrices
    *  corresponding to different events in the batch. It transforms the
    *  input matrices. */
   void Forward(std::vector<Matrix_t> input, bool applyDropout = false);

   void Backward(std::vector<Matrix_t> &gradients_backward, const std::vector<Matrix_t> &activations_backward);

   /*! Prints the info about the layer. */
   void Print() const;
};

//
//
//  The Reshape Layer Class - Implementation
//_________________________________________________________________________________________________
template <typename Architecture_t>
TReshapeLayer<Architecture_t>::TReshapeLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                             size_t depth, size_t height, size_t width, size_t outputNSlices,
                                             size_t outputNRows, size_t outputNCols)
   : VGeneralLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, depth, height, width, 0, 0, 0, 0,
                                   outputNSlices, outputNRows, outputNCols, EInitialization::kZero)
{
   // Nothing to do here.
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
TReshapeLayer<Architecture_t>::TReshapeLayer(TReshapeLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer)
{
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
TReshapeLayer<Architecture_t>::TReshapeLayer(const TReshapeLayer &layer) : VGeneralLayer<Architecture_t>(layer)
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
auto TReshapeLayer<Architecture_t>::Forward(std::vector<Matrix_t> input, bool applyDropout) -> void
{
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto TReshapeLayer<Architecture_t>::Backward(std::vector<Matrix_t> &gradients_backward,
                                             const std::vector<Matrix_t> &activations_backward) -> void
{
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto TReshapeLayer<Architecture_t>::Print() const -> void
{
}

} // namespace DNN
} // namespace TMVA

#endif