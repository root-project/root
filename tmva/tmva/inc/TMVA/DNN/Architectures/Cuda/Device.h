// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
// Declares the DeviceSettings struct which holdsx device specific      //
// information to be defined at compile time.                           //
//////////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CUDA_DEVICE
#define TMVA_DNN_ARCHITECTURES_CUDA_DEVICE

#include "cuda.h"
#include "vector_types.h" // definition of dim3
#include "CudaMatrix.h"

namespace TMVA
{
namespace DNN
{

class TDevice
{
public:
   static constexpr int BlockDimX = 1;
   static constexpr int BlockDimY = 32;
   static constexpr int BlockSize = BlockDimX * BlockDimY;

   static dim3 BlockDims()
   {
      return dim3(BlockDimX, BlockDimY);
   }

   static dim3 GridDims(const TCudaMatrix &A)
   {
      int gridDimX = A.GetNcols() / TDevice::BlockDimX;
      if ((A.GetNcols() % TDevice::BlockDimX) != 0)
          gridDimX += 1;
      int gridDimY = A.GetNrows() / TDevice::BlockDimY;
      if ((A.GetNrows() % TDevice::BlockDimY) != 0)
          gridDimY += 1;
      return dim3(gridDimX, gridDimY);
   }

   static int NThreads(const TCudaMatrix &A)
   {
      int gridDimX = A.GetNcols() / TDevice::BlockDimX;
      if ((A.GetNcols() % TDevice::BlockDimX) != 0)
          gridDimX += 1;
      int gridDimY = A.GetNrows() / TDevice::BlockDimY;
      if ((A.GetNrows() % TDevice::BlockDimY) != 0)
          gridDimY += 1;
      return gridDimX * gridDimY * TDevice::BlockDimX * TDevice::BlockDimY;
   }

};


} // namespace DNN
} // namespace TMVA

#endif
