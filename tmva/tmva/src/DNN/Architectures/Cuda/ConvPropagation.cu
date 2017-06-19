// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski 31/05/17

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"

//////////////////////////////////////////////////////////////////////////
// Implementation of the Convolution functions for TCuda architectures. //
//////////////////////////////////////////////////////////////////////////

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Im2col(TCudaMatrix<AFloat> &A,
                           TCudaMatrix<AFloat> &B,
                           size_t imgHeight,
                           size_t imgWidth,
                           size_t fltHeight,
                           size_t fltWidth,
                           size_t strideRows,
                           size_t strideCols,
                           size_t zeroPaddingHeight,
                           size_t zeroPaddingWidth)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::RotateWeights(TCudaMatrix<AFloat> &A,
                                  const TCudaMatrix<AFloat> &B,
                                  size_t filterDepth,
                                  size_t filterHeight,
                                  size_t filterWidth,
                                  size_t numFilters)
{

}


//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Flatten(TCudaMatrix<AFloat> &A,
                            const std::vector<TCudaMatrix<AFloat>> B,
                            size_tt size,
                            size_t nRows,
                            size_t nCols)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Deflatten(std::vector<TCudaMatrix<AFloat>> A,
                              const TCudaMatrix<AFloat> &B,
                              size_t index,
                              size_t nRows,
                              size_t nCols)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ConvLayerBackward(std::vector<TCudaMatrix<AFloat>> activation_gradients_backward,
                                      TCudaMatrix<AFloat> & weight_gradients,
                                      TCudaMatrix<AFloat> & bias_gradients,
                                      std::vector<TCudaMatrix<AFloat>> df,
                                      const std::vector<TCudaMatrix<AFloat>> activation_gradients,
                                      const TCudaMatrix<AFloat> & weights,
                                      const std::vector<TCudaMatrix<AFloat>> activation_backward,
                                      size_t batchSize,
                                      size_t inputHeight,
                                      size_t inputWidth,
                                      size_t depth,
                                      size_t height,
                                      size_t width,
                                      size_t filterDepth,
                                      size_t filterHeight,
                                      size_t filterWidth,
                                      size_t nLocalViews)
{


}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvActivationGradients(
                                    std::vector<TCudaMatrix<AFloat>> activation_gradients_backward,
                                    std::vector<TCudaMatrix<AFloat>> df,
                                    const TCudaMatrix<AFloat> & weights,
                                    size_t batchSize,
                                    size_t inputHeight,
                                    size_t inputWidth,
                                    size_t depth,
                                    size_t height,
                                    size_t width,
                                    size_t filterDepth,
                                    size_t filterHeight,
                                    size_t filterWidth)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvWeightGradients(TCudaMatrix<AFloat> & weight_gradients,
                                                 std::vector<TCudaMatrix<AFloat>> df,
                                                 const TCudaMatrix<AFloat> *activations_backward,
                                                 size_t batchSize,
                                                 size_t inputHeight,
                                                 size_t inputWidth,
                                                 size_t depth,
                                                 size_t height,
                                                 size_t width,
                                                 size_t filterDepth,
                                                 size_t filterHeight,
                                                 size_t filterWidth,
                                                 size_t nLocalViews)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvBiasGradients(TCudaMatrix<AFloat> & bias_gradients,
                                               std::vector<TCudaMatrix<AFloat>> df,
                                               size_t batchSize,
                                               size_t depth,
                                               size_t nLocalViews)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::AddConvBiases(TCudaMatrix<AFloat> &output,
                                  const TCudaMatrix<AFloat> &biases)
{

}


//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Downsample(TCudaMatrix<AFloat> &A,
                               TCudaMatrix<AFloat> &B,
                               const TCudaMatrix<AFloat> &C,
                               size_t imgHeight,
                               size_t imgWidth,
                               size_t fltHeight,
                               size_t fltWidth,
                               size_t strideRows,
                               size_t strideCols)
{

}

/____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::PoolLayerBackward(std::vector<TCudaMatrix<AFloat>> activationGradientsBackward,
                                      const std::vector<TCudaMatrix<AFloat>> activationGradients,
                                      const std::vector<TCudaMatrix<AFloat>> indexMatrix,
                                      size_t batchSize,
                                      size_t depth,
                                      size_t nLocalViews)
{

}

} // namespace DNN
} // namespace TMVA
