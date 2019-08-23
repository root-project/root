// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 11/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////
// Explicit instantiation of the Reference architecture class //
// template for Double_t scalar types.                        //
////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/TCudnn.h"
#include "TMVA/DNN/Architectures/Cuda.h"


namespace TMVA
{
namespace DNN
{

template<typename AFloat>
void TCudnn<AFloat>::Sigmoid(TCudaTensor<AFloat> & B,
                            const TCudaTensor<AFloat> & A)
{
   TCudaMatrix<AFloat> mB = B.GetMatrix(); 
   TCuda<AFloat>::Sigmoid(mB,A.GetMatrix());
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::Softmax(TCudaTensor<AFloat> & B,
                            const TCudaTensor<AFloat> & A)
{
   TCudaMatrix<AFloat> mB = B.GetMatrix(); 
   TCuda<AFloat>::Softmax(mB,A.GetMatrix());
}

} // namespace DNN
} // namespace TMVA
