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
 // reference implementation.                                    //
//////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu.h"


namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::AddBiases(TCpuMatrix<AFloat> &A,
                             const TCpuMatrix<AFloat> &biases)
{

}

//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::UpdateParams(TCpuMatrix<AFloat> &x,
                                TCpuMatrix<AFloat> &z,
                                TCpuMatrix<AFloat> &fVBiases,
                                TCpuMatrix<AFloat> &fHBiases,
                                TCpuMatrix<AFloat> &fWeights,
                                Double_t learningRate,
                                Double_t corruptionLevel,
                                size_t fBatchSize)
{


}
//______________________________________________________________________________

//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::SoftmaxAE(TCpuMatrix<AFloat> & A)
{

}

//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::CorruptInput(TCpuMatrix<AFloat> & input,
                                 TCpuMatrix<AFloat> & corruptedInput,
                                 AFloat corruptionLevel)
{

}


//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::EncodeInput(TCpuMatrix<AFloat> & input,
                                TCpuMatrix<AFloat> & compressedInput,
                                TCpuMatrixT<AFloat> &fWeights)
{


}
//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::ReconstructInput(TCpuMatrix<AFloat> & compressedInput,
                                     TCpuMatrix<AFloat> & reconstructedInput,
                                     TCpuMatrix<AFloat> &fWeights)
{


}

//______________________________________________________________________________
// Logistic Regression Layer Methods
//
//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::ForwardLogReg(TCpuMatrix<AFloat> &input,
                                 TCpuMatrix<AFloat> &p,
                                 TCpuMatrix<AFloat> &fWeights)
{

}

//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::UpdateParamsLogReg(TCpuMatrix<AFloat> &input,
                                      TCpuMatrix<AFloat> &output,
                                      TCpuMatrix<AFloat> &difference,
                                      TCpuMatrix<AFloat> &p,
                                      TCpuMatrix<AFloat> &fWeights,
                                      TCpuMatrix<AFloat> &fBiases,
                                      AFloat learningRate,
                                      size_t fBatchSize)
{

}
//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::Transform(TCpuMatrix<AFloat> &input,
                              TCpuMatrix<AFloat> &transformed,
                              TCpuMatrix<AFloat> &fWeights,
                              TCpuMatrix<AFloat> &fBiases)
{
}
//______________________________________________________________________________

}
}
