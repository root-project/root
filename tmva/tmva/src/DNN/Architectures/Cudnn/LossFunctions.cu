// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////
// Implementation of the loss functions for the TCuda implementation //
// of the low-level interface.                                       //
///////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/TCudnn.h"
#include "TMVA/DNN/Architectures/Cuda.h"


namespace TMVA
{
namespace DNN
{

//____________________________________________________________________________
template<typename AFloat>
AFloat TCudnn<AFloat>::MeanSquaredError(const TCudaTensor<AFloat> & Y,
                                       const TCudaTensor<AFloat> & output,
                                       const TCudaTensor<AFloat> & weights)
{
    return TCuda<AFloat>::MeanSquaredError(Y.GetMatrix(), output.GetMatrix(), weights.GetMatrix());
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::MeanSquaredErrorGradients(TCudaTensor<AFloat> & dY,
                                              const TCudaTensor<AFloat> & Y,
                                              const TCudaTensor<AFloat> & output,
                                              const TCudaTensor<AFloat> &weights)
{
   TCudaMatrix<AFloat> mdY = dY.GetMatrix(); 
   TCuda<AFloat>::MeanSquaredErrorGradients(mdY, Y.GetMatrix(), output.GetMatrix(), weights.GetMatrix());
}

//____________________________________________________________________________
template<typename AFloat>
AFloat TCudnn<AFloat>::CrossEntropy(const TCudaTensor<AFloat> & Y,
                                   const TCudaTensor<AFloat> & output,
                                   const TCudaTensor<AFloat> &weights)
{
   return TCuda<AFloat>::CrossEntropy(Y.GetMatrix(), output.GetMatrix(), weights.GetMatrix());
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::CrossEntropyGradients(TCudaTensor<AFloat> & dY,
                                          const TCudaTensor<AFloat> & Y,
                                          const TCudaTensor<AFloat> & output,
                                          const TCudaTensor<AFloat> &weights)
{
   TCudaMatrix<AFloat> mdY = dY.GetMatrix(); 
   TCuda<AFloat>::CrossEntropyGradients(mdY, Y.GetMatrix(), output.GetMatrix(), weights.GetMatrix());
}

//____________________________________________________________________________
template<typename AFloat>
AFloat TCudnn<AFloat>::SoftmaxCrossEntropy(const TCudaTensor<AFloat> & Y,
                                          const TCudaTensor<AFloat> & output,
                                          const TCudaTensor<AFloat> &weights)
{
   return TCuda<AFloat>::SoftmaxCrossEntropy(Y.GetMatrix(), output.GetMatrix(), weights.GetMatrix());    
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::SoftmaxCrossEntropyGradients(TCudaTensor<AFloat> & dY,
                                                 const TCudaTensor<AFloat> & Y,
                                                 const TCudaTensor<AFloat> & output,
                                                 const TCudaTensor<AFloat> &weights)
{
   TCudaMatrix<AFloat> mdY = dY.GetMatrix(); 
   TCuda<AFloat>::SoftmaxCrossEntropyGradients(mdY, Y.GetMatrix(), output.GetMatrix(), weights.GetMatrix());
}

} // namespace DNN
} // namespace TMVA
