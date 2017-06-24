// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha(ajatgd)

/*************************************************************************
 * Copyright (C) 2017 ajatgd                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
 // Implementation of the Denoise Autoencoder functions for the  //
 // Cuda implementation.                                         //
///////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::AddBiases(TCudaMatrix<AFloat> &A,
                              const TCudaMatrix<AFloat> &biases)
{

}

//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::UpdateParams(TCudaMatrix<AFloat> &x,
                                TCudaMatrix<AFloat> &z,
                                TCudaMatrix<AFloat> &fVBiases,
                                TCudaMatrix<AFloat> &fHBiases,
                                TCudaMatrix<AFloat> &fWeights,
                                AFloat learningRate,
                                AFloat corruptionLevel,
                                size_t fBatchSize)
{


}
//______________________________________________________________________________

//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::SoftmaxAE(TCudaMatrix<AFloat> & A)
{

}

//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::CorruptInput(TCudaMatrix<AFloat> & input,
                                 TCudaMatrix<AFloat> & corruptedInput,
                                 AFloat corruptionLevel)
{

}


//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::EncodeInput(TCudaMatrix<AFloat> & input,
                                TCudaMatrix<AFloat> & compressedInput,
                                TCudaMatrixT<AFloat> &fWeights)
{


}
//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ReconstructInput(TCudaMatrix<AFloat> & compressedInput,
                                     TCudaMatrix<AFloat> & reconstructedInput,
                                     TCudaMatrix<AFloat> &fWeights)
{


}


}
}
