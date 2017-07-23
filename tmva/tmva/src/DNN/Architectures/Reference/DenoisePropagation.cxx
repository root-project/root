// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha(ajatgd)

/*************************************************************************
 * Copyright (C) 2017 ajatgd                                             *
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

#include <cmath>
#include <cstdlib>
#include <iostream>
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
      A(i, j) += biases(i, 0);
    }
  }
}

//______________________________________________________________________________

template <typename Real_t>
void TReference<Real_t>::UpdateParams(
    TMatrixT<Real_t> &x, TMatrixT<Real_t> &tildeX, TMatrixT<Real_t> &y,
    TMatrixT<Real_t> &z, TMatrixT<Real_t> &fVBiases, TMatrixT<Real_t> &fHBiases,
    TMatrixT<Real_t> &fWeights, TMatrixT<Real_t> &VBiasError,
    TMatrixT<Real_t> &HBiasError, Real_t learningRate, size_t fBatchSize) {

  //updating fVBiases
  for (size_t i = 0; i < (size_t)fVBiases.GetNrows(); i++)
  {
    for (size_t j = 0; j < (size_t)fVBiases.GetNcols(); j++) {
      VBiasError(i, j) = x(i, j) - z(i, j);
      fVBiases(i, j) += learningRate * VBiasError(i, j) / fBatchSize;
    }
  }

  /*for (size_t i = 0; i < (size_t)fVBiases.GetNrows(); i++)
  {
    for(size_t j=0; j<(size_t)fVBiases.GetNcols();j++)
    {
      std::cout<<fVBiases(i,j)<<std::endl;
    }
  }*/

  //updating fHBiases
  for(size_t i = 0; i < fHBiases.GetNrows(); i++)
  {
    HBiasError(i,0) = 0;
    for(size_t j = 0; j < fVBiases.GetNrows(); j++)
    {
      HBiasError(i, 0) += fWeights(i, j) * VBiasError(j, 0);
    }
    HBiasError(i, 0) *= y(i, 0) * (1 - y(i, 0));
    fHBiases(i, 0) += learningRate * HBiasError(i, 0) / fBatchSize;
  }

  /*for (size_t i = 0; i < (size_t)fHBiases.GetNrows(); i++)
  {
    for(size_t j=0; j<(size_t)fHBiases.GetNcols();j++)
    {
      std::cout<<fHBiases(i,j)<<std::endl;
    }
  }*/

  //updating weights
  for(size_t i = 0; i < fHBiases.GetNrows(); i++)
  {
    for(size_t j = 0; j< fVBiases.GetNrows(); j++)
    {
      fWeights(i, j) += learningRate * (HBiasError(i, 0) * tildeX(j, 0) +
                                        VBiasError(j, 0) * y(i, 0)) /
                        fBatchSize;
    }
  }

  /*for (size_t i = 0; i < (size_t)fWeights.GetNrows(); i++)
  {
    for(size_t j=0; j<(size_t)fWeights.GetNcols();j++)
    {
      std::cout<<fWeights(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }*/
}

//______________________________________________________________________________

template<typename Real_t>
void TReference<Real_t>::SoftmaxAE(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   Real_t sum = 0.0;
   for (size_t i = 0; i < m; i++) {
     for (size_t j = 0; j < n; j++) {
       sum += exp(A(i, j));
     }
   }

   for (size_t i = 0; i < m; i++) {
     for (size_t j = 0; j < n; j++) {
       A(i, j) = exp(A(i, j)) / sum;
     }
   }
}

//______________________________________________________________________________

template <typename Real_t>
void TReference<Real_t>::CorruptInput(TMatrixT<Real_t> &input,
                                      TMatrixT<Real_t> &corruptedInput,
                                      Real_t corruptionLevel) {
  for(size_t i=0; i< (size_t)input.GetNrows(); i++)
  {
    for(size_t j=0; j<(size_t)input.GetNcols(); j++ )
    {

      if ((size_t)((rand() / (RAND_MAX + 1.0)) * 100) %
              ((size_t)(corruptionLevel * 10)) ==
          0) {
        corruptedInput(i, j) = 0;
      }
      else
      {
        corruptedInput(i, j) = input(i, j);
      }
    }
  }
}


//______________________________________________________________________________

template <typename Real_t>
void TReference<Real_t>::EncodeInput(TMatrixT<Real_t> &input,
                                     TMatrixT<Real_t> &compressedInput,
                                     TMatrixT<Real_t> &Weights) {

  size_t m, a;
  m = compressedInput.GetNrows();
  a = input.GetNrows();

  for (size_t i = 0; i < m; i++) {
    compressedInput(i, 0) = 0;
    for (size_t j = 0; j < a; j++) {
      compressedInput(i, 0) =
          compressedInput(i, 0) + (Weights(i, j) * input(j, 0));
    }
  }
}
//______________________________________________________________________________
template <typename Real_t>
void TReference<Real_t>::ReconstructInput(TMatrixT<Real_t> &compressedInput,
                                          TMatrixT<Real_t> &reconstructedInput,
                                          TMatrixT<Real_t> &fWeights) {
  for (size_t i=0; i<(size_t)reconstructedInput.GetNrows(); i++)
  {
    reconstructedInput(i, 0) = 0;
    for(size_t j=0; j<(size_t)compressedInput.GetNrows();j++)
    {
      reconstructedInput(i, 0) += fWeights(j, i) * compressedInput(j, 0);
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
    p(i, 0) = 0;
    for(size_t j=0; j < n; j++)
    {
      p(i, 0) += fWeights(i, j) * input(j, 0);
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
    difference(i, 0) = output(i, 0) - p(i, 0);
    for(size_t j=0; j<n; j++)
    {
      fWeights(i, j) +=
          learningRate * difference(i, 0) * input(j, 0) / fBatchSize;
    }

    fBiases(i, 0) += learningRate * difference(i, 0) / fBatchSize;
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
      output += fWeights(i, j) * input(j, 0);
      }
      output += fBiases(i, 0);
      transformed(i, 0) = output;
  }
}
}
}
