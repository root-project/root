// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 14/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/DNN/Architectures/TCudnn.h"
/*#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"*/

/////////////////////////////////////////////////////////////////////
// Implementation of the Dropout function for TCudnn architectures.//
/////////////////////////////////////////////////////////////////////

namespace TMVA {
namespace DNN  {

// FIXME: Do testing!!!
//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::DropoutForward(TCudaTensor<AFloat> &A, 
                                    TDescriptors * descriptors,
                                    TWorkspace   * workspace, 
                                    AFloat /*dropoutProbability*/)
{
    if (!workspace || !descriptors) return;
    auto poolWorkspace = static_cast<ConvWorkspace_t *>(workspace);
    auto poolDescriptors = static_cast<PoolingDescriptors_t *>(descriptors);

    //TCudaTensor<AFloat> tmp (A);

    // Write the output into A      
    CUDNNCHECK(cudnnDropoutForward(A.GetCudnnHandle(),
                                   poolDescriptors->HelperDescriptor,
                                   A.GetTensorDescriptor(),// use tmp, if inplace op fails
                                   A.GetDataPointer(),
                                   A.GetTensorDescriptor(),
                                   A.GetDataPointer(),
                                   poolWorkspace->HelperWorkspace,
                                   poolWorkspace->HelperWorkspaceSize));
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::DropoutBackward(TCudaTensor<AFloat> &A,
                                     TDescriptors * descriptors,
                                     TWorkspace   * workspace)
{
    if (!workspace || !descriptors) return;
    auto poolWorkspace = static_cast<ConvWorkspace_t *>(workspace);
    auto poolDescriptors = static_cast<PoolingDescriptors_t *>(descriptors);

    //TCudaTensor<AFloat> tmp (A);

    // Write the output into A
    CUDNNCHECK(cudnnDropoutBackward(A.GetCudnnHandle(),
                                   poolDescriptors->HelperDescriptor,
                                   A.GetTensorDescriptor(),// use tmp, if inplace op fails
                                   A.GetDataPointer(),
                                   A.GetTensorDescriptor(),
                                   A.GetDataPointer(),
                                   poolWorkspace->HelperWorkspace,
                                   poolWorkspace->HelperWorkspaceSize));
}

} // namespace DNN
} // namespace TMVA
