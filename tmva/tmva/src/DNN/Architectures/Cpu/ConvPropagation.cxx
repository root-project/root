// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski 31/05/17

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/DNN/Architectures/Cpu.h"

///////////////////////////////////////////////////////////////////////////////////
// Implementation of Convolution functions for multi-threaded CPU architectures. //
///////////////////////////////////////////////////////////////////////////////////

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
template<typename AFloat>
    void TCpu<AFloat>::Im2col(TCpuMatrix<AFloat> &A,
                              TCpuMatrix<AFloat> &B,
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
void TCpu<AFloat>::RotateWeights(TCpuMatrix<AFloat> &A,
                             const TCpuMatrix<AFloat> &B,
                             size_t filterDepth,
                             size_t filterHeight,
                             size_t filterWidth,
                             size_t numFilters)
{

}
    
//____________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::Flatten(TCpuMatrix<AFloat> &A,
                           const std::vector<TCpuMatrix<AFloat>> B,
                           size_t size,
                           size_t nRows,
                           size_t nCols)
{


}

//____________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::Deflatten(std::vector<TCpuMatrix<AFloat>> A,
                             const TCpuMatrix<AFloat> &B,
                             size_t index,
                             size_t nRows,
                             size_t nCols)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::ConvLayerBackward(std::vector<TCpuMatrix<AFloat>> activationGradientsBackward,
                                     TCpuMatrix<AFloat> & weightGradients,
                                     TCpuMatrix<AFloat> & biasGradients,
                                     std::vector<TCpuMatrix<AFloat>> df,
                                     const std::vector<TCpuMatrix<AFloat>> activationGradients,
                                     const TCpuMatrix<AFloat> & weights,
                                     const std::vector<TCpuMatrix<AFloat>> activationsBackward,
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
void TCpu<AFloat>::CalculateConvActivationGradients(std::vector<TCpuMatrix<AFloat>> activationGradientsBackward,
                                                    std::vector<TCpuMatrix<AFloat>> df,
                                                    const TCpuMatrix<AFloat> & weights,
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
void TCpu<AFloat>::CalculateConvWeightGradients(TCpuMatrix<AFloat> & weightGradients,
                                                std::vector<TCpuMatrix<AFloat>> df,
                                                const std::vector<TCpuMatrix<AFloat>> activations_backward,
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
void TCpu<AFloat>::CalculateConvBiasGradients(TCpuMatrix<AFloat> & biasGradients,
                                              std::vector<TCpuMatrix<AFloat>> df,
                                              size_t batchSize,
                                              size_t depth,
                                              size_t nLocalViews)
{
        
}

//____________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::AddConvBiases(TCpuMatrix<AFloat> &output,
                                 const TCpuMatrix<AFloat> &biases)
{
        
}

//____________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::Downsample(TCpuMatrix<AFloat> &A,
                              TCpuMatrix<AFloat> &B,
                              const TCpuMatrix<AFloat> &C,
                              size_t imgHeight,
                              size_t imgWidth,
                              size_t fltHeight,
                              size_t fltWidth,
                              size_t strideRows,
                              size_t strideCols)
{
        
}

//____________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::PoolLayerBackward(std::vector<TCpuMatrix<AFloat>> activationGradientsBackward,
                                     const std::vector<TCpuMatrix<AFloat>> activationGradients,
                                     const std::vector<TCpuMatrix<AFloat>> indexMatrix,
                                     size_t batchSize,
                                     size_t depth,
                                     size_t nLocalViews)
{

}
    
} // namespace DNN
} // namespace TMVA
