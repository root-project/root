// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 14/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"

////////////////////////////////////////////////////////////////////
// Implementation of the Dropout function for TCuda architectures. //
////////////////////////////////////////////////////////////////////

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::Dropout(TCudaMatrix &A,
                                 CudaDouble_t dropoutProbability)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);

   tick();
   dropout<<<gridDims, blockDims>>>(A.GetDataPointer(),
                                    (int) A.GetNrows(),
                                    (int) A.GetNcols(),
                                    dropoutProbability,
                                    TCudaMatrix::GetCurandStatesPointer());
   tock(fTimings.TimeDropout);
}

} // namespace DNN
} // namespace TMVA
