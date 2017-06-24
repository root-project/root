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

#include "TMVA/DNN/Architectures/Reference.h"


namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________

template<typename Real_t>
void TReference<Real_t>::AddBiases(TMatrixT<Real_t> &A,
                                   const TMatrixT<Real_t> &biases)
{
  size_t m,n;
  m = A.GetNrows();
  n= A.GetNcols();
  for(size_t i = 0; i<m; i++)
  {
    for(size_t j=0; j<n; j++)
    {
      A(i,j) += biases(i,j);
    }
  }
}

//______________________________________________________________________________

template<typename Real_t>
void TReference<Real_t>::UpdateParams(TMatrixT<Real_t> &x,
                                      TMatrixT<Real_t> &z,
                                      TMatrixT<Real_t> &fVBiases,
                                      TMatrixT<Real_t> &fHBiases,
                                      TMatrixT<Real_t> &fWeights,
                                      Real_t learningRate,
                                      Real_t corruptionLevel,
                                      size_t fBatchSize)
{

  //updating fVBiases
  for (size_t i = 0; i < (size_t)fVBiases.GetNrows(); i++)
  {
    VBiasError(i,1) = x(i,1)-z(i,1);
    fVBiases(i,1) += learningRate * VBiasError(i,1)/ fBatchSize;
  }

  //updating fHBiases
  for(size_t i = 0; i < fHBiases.GetNrows(); i++)
  {
    HBiasError(i,0) = 0;
    for(size_t j = 0; j < fVBiases.GetNrows(); j++)
    {
      HBiasError(i,1) += fWeights(i,j) * VBiasError(j,1);
    }
    HBiasError(i,1) *= y(i,1) * (1-y(i,1));
    fHBiases(i,1) += learningRate * HBiasError(i,1)/ fBatchSize;
  }

  //updating weights
  for(size_t i = 0; i < fHBiases.GetNrows(); i++)
  {
    for(size_t j = 0; j< fVBiases.GetNrows(); j++)
    {
      fWeights(i,j) += learningRate * (HBiasError(i,1)*tildeX(j,1) + VBiasError(j,1)*y(i,1)) / fBatchSize;
    }
  }

}

//______________________________________________________________________________

template<typename Real_t>
void TReference<Real_t>::SoftmaxAE(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();
   Real_t sum = 0.0;
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
         sum += exp(A(i,j));
      }
   }

   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
         A(i,j) = exp(A(i,j)) / sum;
      }
   }
}

//______________________________________________________________________________

template<typename Real_t>
void TReference<Real_t>::CorruptInput(TMatrixT<Real_t> & input,
                                         TMatrixT<Real_t> & corruptedInput,
                                         Real_t corruptionLevel)
{
  for(size_t i=0; i< (size_t)input.GetNrows(); i++)
  {
    for(size_t j=0; j<(size_t)input.GetNcols(); j++ )
    {
      if(size_t((rand()/(RAND_MAX + 1.0))*100) % ((size_t)(corruptionLevel)*10)
      {
          corruptedInput(i,j) = 0;
      }
      else
      {
          corruptedInput(i,j) == x(i,j);
      }
    }
  }
}


//______________________________________________________________________________

template<typename Real_t>
void TReference<Real_t>::EncodeInput(TMatrixT<Real_t> & input,
                                          TMatrixT<Real_t> & compressedInput,
                                          TMatrixT<Real_t> &fWeights)
{
  for (size_t i = 0; i<(size_t)compressedInput.GetNrows(); i++)
  {
    compressedInput(i,1)=0;
    for (size_t j= 0; j<(size_t)input.GetNrows(); j++)
    {
      compressedInput(i,1) += fWeights(i,j) * input(j,1);
    }
  }
}
//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::ReconstructInput(TMatrixT<Real_t> & compressedInput,
                                               TMatrixT<Real_t> & reconstructedInput,
                                               TMatrixT<Real_t> &fWeights)
{
  for (size_t i=0; i<(size_t)reconstructedInput.GetNrows(); i++)
  {
    reconstructedInput(i,1) = 0;
    for(size_t j=0; j<(size_t)compressedInput.GetNrows();j++)
    {
      reconstructedInput(i,1) += fWeights(j,i) * compressedInput(j,1);
    }
  }

}

//______________________________________________________________________________
// Logistic Regression Layer Methods
//
//______________________________________________________________________________

template<typename Real_t>
void TReference<Real_t>::ForwardLogReg(TMatrixT<Real_t> &input,
                                       TMatrixT<Real_t> &p,
                                       TMatrixT<Real_t> &fWeights)
{
  size_t m,n;
  m = p.GetNrows();
  n = input.GetNrows();
  for(size_t i= 0; i < m; i++)
  {
    p(i,1)=0;
    for(size_t j=0; j < n; j++)
    {
      p(i,1) += fWeights(i,j) * input(j,1);
    }
  }
}

//______________________________________________________________________________

template<typename Real_t>
void TReference<Real_t>::UpdateParamsLogReg(TMatrixT<Real_t> &input,
                                            TMatrixT<Real_t> &output,
                                            TMatrixT<Real_t> &difference,
                                            TMatrixT<Real_t> &p,
                                            TMatrixT<Real_t> &fWeights,
                                            TMatrixT<Real_t> &fBiases,
                                            Real_t learningRate,
                                            size_t fBatchSize)
{
  size_t m,n;
  m = p.GetNrows();
  n = input.GetNrows();

  for(size_t i= 0; i<m; i++)
  {
    difference(i,1) = output(i,1) - p(i,1);
    for(size_t j=0; j<n; j++)
    {
      fWeights(i,j) += learningRate * difference(i,1) *input(j,1) / fBatchSize;

    }

    fBiases(i,1) += learningRate * difference(i,1) / fBatchSize;

  }
}
//______________________________________________________________________________
// Transform input from input dimensions to transformed dimensions
//
//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::Transform(TMatrixT<Real_t> &input,
                                   TMatrixT<Real_t> &transformed,
                                   TMatrixT<Real_t> &fWeights,
                                   TMatrixT<Real_t> &fBiases)
{
  size_t m,n;
  m=fWeights.GetNrows();
  n=fWeights.GetNcols();

  for (size_t i = 0; i < m ; i++)
  {
    Double_t output = 0.0;
    for(size_t j = 0; j < n; j++)
      {
        output += fWeights(i,j) * input(j,1)
      }
    output+=fBiases(i,1);
    transformed(i,1)=output;
  }


}
}
