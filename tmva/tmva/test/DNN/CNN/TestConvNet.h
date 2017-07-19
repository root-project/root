// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Conv Net Features                                                 *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_TEST_DNN_TEST_CNN_TEST_CONV_NET_H
#define TMVA_TEST_DNN_TEST_CNN_TEST_CONV_NET_H

#include "../Utility.h"

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;

/** Testing the image to column function. Check wheether the matrix A after
 *  the transformation will be equal to the matrix B. */
//______________________________________________________________________________
template <typename Architecture>
auto testIm2col(typename Architecture::Matrix_t &A, typename Architecture::Matrix_t &B, size_t imgHeight,
                size_t imgWidth, size_t fltHeight, size_t fltWidth, size_t strideRows, size_t strideCols,
                size_t zeroPaddingHeight, size_t zeroPaddingWidth) -> bool
{
   using Matrix_t = typename Architecture::Matrix_t;

   size_t m, n;
   m = B.GetNrows();
   n = B.GetNcols();

   Matrix_t ATr(m, n);
   Architecture::Im2col(ATr, A, imgHeight, imgWidth, fltHeight, fltWidth, strideRows, strideCols, zeroPaddingHeight,
                        zeroPaddingWidth);

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         if (ATr(i, j) != B(i, j)) {
            return false;
         }
      }
   }

   return true;
}

/** Testing the rotation of the weights function. Check whether the rotated
 *  weight matrix A, will be equal to the matrix B. */
//______________________________________________________________________________
template <typename Architecture>
auto testRotateWeights(typename Architecture::Matrix_t &A, typename Architecture::Matrix_t &B, size_t filterDepth,
                       size_t filterHeight, size_t filterWidth, size_t numFilters) -> bool
{
   using Matrix_t = typename Architecture::Matrix_t;

   size_t m, n;
   m = B.GetNrows();
   n = B.GetNcols();

   Matrix_t ARot(m, n);
   Architecture::RotateWeights(ARot, A, filterDepth, filterHeight, filterWidth, numFilters);

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         if (ARot(i, j) != B(i, j)) {
            return false;
         }
      }
   }

   return true;
}

/** Downsample the matrix A and check whether the downsampled version
 *  is equal to B, and if the winning indices are equal to the matrix ind. */
//______________________________________________________________________________
template <typename Architecture>
auto testDownsample(const typename Architecture::Matrix_t &A, const typename Architecture::Matrix_t &ind,
                    const typename Architecture::Matrix_t &B, size_t imgHeight, size_t imgWidth, size_t fltHeight,
                    size_t fltWidth, size_t strideRows, size_t strideCols) -> bool
{

   using Matrix_t = typename Architecture::Matrix_t;

   size_t m1, n1;
   m1 = B.GetNrows();
   n1 = B.GetNcols();

   Matrix_t ADown(m1, n1);

   size_t m2, n2;
   m2 = ind.GetNrows();
   n2 = ind.GetNcols();

   Matrix_t AInd(m2, n2);

   Architecture::Downsample(ADown, AInd, A, imgHeight, imgWidth, fltHeight, fltWidth, strideRows, strideCols);

   for (size_t i = 0; i < m1; i++) {
      for (size_t j = 0; j < n1; j++) {
         if (ADown(i, j) != B(i, j)) {
            return false;
         }
      }
   }

   for (size_t i = 0; i < m2; i++) {
      for (size_t j = 0; j < n2; j++) {
         if (AInd(i, j) != ind(i, j)) {
            return false;
         }
      }
   }

   return true;
}

/*! Generate a conv net, perform forward pass */
//______________________________________________________________________________
template <typename Architecture>
auto testConvForwardPass(size_t batchSize, size_t imgDepth, size_t imgHeight, size_t imgWidth, size_t batchDepth,
                         size_t batchHeight, size_t batchWidth) -> void
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t = TDeepNet<Architecture>;

   Net_t convNet(batchSize, imgDepth, imgHeight, imgWidth, batchDepth, batchHeight, batchWidth,
                 ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   constructConvNet(convNet);
   convNet.Initialize();

   std::vector<Matrix_t> X;
   for (size_t i = 0; i < batchSize; i++) {
      X.emplace_back(imgDepth, imgHeight * imgWidth);
      randomMatrix(X[i]);
   }

   convNet.Forward(X);
}

/*! Generate a conv net, get the loss. */
//______________________________________________________________________________
template <typename Architecture>
auto testConvLossFunction(size_t batchSize, size_t imgDepth, size_t imgHeight, size_t imgWidth, size_t batchDepth,
                          size_t batchHeight, size_t batchWidth) -> void
{
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t = TDeepNet<Architecture>;

   Net_t convNet(batchSize, imgDepth, imgHeight, imgWidth, batchDepth, batchHeight, batchWidth,
                 ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   constructConvNet(convNet);
   convNet.Initialize();

   std::vector<Matrix_t> X;
   for (size_t i = 0; i < batchSize; i++) {
      X.emplace_back(imgDepth, imgHeight * imgWidth);
      randomMatrix(X[i]);
   }

   Matrix_t Y(batchSize, convNet.GetOutputWidth());
   Matrix_t W(batchSize, 1);
   randomMatrix(Y);
   randomMatrix(W);

   Scalar_t loss = convNet.Loss(X, Y, W);
   std::cout << "The loss is: " << loss << std::endl;
}

/*! Generate a conv net, get the predictios */
//______________________________________________________________________________
template <typename Architecture>
auto testConvPrediction(size_t batchSize, size_t imgDepth, size_t imgHeight, size_t imgWidth, size_t batchDepth,
                        size_t batchHeight, size_t batchWidth, EOutputFunction f) -> void
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t = TDeepNet<Architecture>;

   Net_t convNet(batchSize, imgDepth, imgHeight, imgWidth, batchDepth, batchHeight, batchWidth,
                 ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   constructConvNet(convNet);
   convNet.Initialize();

   std::vector<Matrix_t> X;
   for (size_t i = 0; i < batchSize; i++) {
      X.emplace_back(imgDepth, imgHeight * imgWidth);
      randomMatrix(X[i]);
   }

   Matrix_t Predictions(batchSize, convNet.GetOutputWidth());
   convNet.Prediction(Predictions, X, f);

   for (size_t i = 0; i < batchSize; i++) {
      for (size_t j = 0; j < convNet.GetOutputWidth(); j++) {
         std::cout << Predictions(i, j) << " ";
      }
      std::cout << "" << std::endl;
   }
}

#endif