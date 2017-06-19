// @(#)root/tmva $Id$
// Author: Vladimir Ilievski, 14/06/2017

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef TMVA_TEST_DNN_TEST_CNN_TEST_CONV_BACKPROPAGATION_H
#define TMVA_TEST_DNN_TEST_CNN_TEST_CONV_BACKPROPAGATION_H

////////////////////////////////////////////////////////////////////
// Generic tests of the CNN functionalities                       //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "../Utility.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/CNN/ConvNet.h"


using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;


/** Testing the image to column function. Check wheether the matrix A after
 *  the transformation will be equal to the matrix B. */
//______________________________________________________________________________
template <typename Architecture>
auto testIm2col(typename Architecture::Matrix_t &A,
                typename Architecture::Matrix_t &B,
                size_t imgHeight,
                size_t imgWidth,
                size_t fltHeight,
                size_t fltWidth,
                size_t strideRows,
                size_t strideCols,
                size_t zeroPaddingHeight,
                size_t zeroPaddingWidth)
-> bool
{
   using Matrix_t = typename Architecture::Matrix_t;
    
   size_t m, n;
   m = B.GetNrows();
   n = B.GetNcols();
    
   Matrix_t ATr(m, n);
   Architecture::Im2col(ATr, A, imgHeight, imgWidth, fltHeight,
                        fltWidth, strideRows, strideCols,
                        zeroPaddingHeight, zeroPaddingWidth);
    
    
   for(size_t i = 0; i < m; i++) {
      for(size_t j = 0; j < n; j++) {
         if(ATr(i, j) != B(i, j)) {
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
auto testRotateWeights(typename Architecture::Matrix_t &A,
                       typename Architecture::Matrix_t &B,
                       size_t filterDepth,
                       size_t filterHeight,
                       size_t filterWidth,
                       size_t numFilters)
-> bool
{
   using Matrix_t = typename Architecture::Matrix_t;
    
   size_t m, n;
   m = B.GetNrows();
   n = B.GetNcols();
    
   Matrix_t ARot(m, n);
   Architecture::RotateWeights(ARot, A, filterDepth, filterHeight,
                               filterWidth, numFilters);
    
   for(size_t i = 0; i < m; i++) {
      for(size_t j = 0; j < n; j++) {
         if(ARot(i, j) != B(i, j)) {
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
auto testDownsample(const typename Architecture::Matrix_t &A,
                    const typename Architecture::Matrix_t &ind,
                    const typename Architecture::Matrix_t &B,
                    size_t imgHeight,
                    size_t imgWidth,
                    size_t fltHeight,
                    size_t fltWidth,
                    size_t strideRows,
                    size_t strideCols)
-> bool
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
    
   Architecture::Downsample(ADown, AInd, A,
                            imgHeight, imgWidth,
                            fltHeight, fltWidth,
                            strideRows, strideCols);
    
   for(size_t i = 0; i < m1; i++) {
      for(size_t j = 0; j < n1; j++) {
         if(ADown(i, j) != B(i, j)) {
            return false;
         }
      }
   }
            
   for(size_t i = 0; i < m2; i++) {
      for(size_t j = 0; j < n2; j++) {
         if(AInd(i, j) != ind(i, j)) {
            return false;
         }
      }
   }
    
   return true;
}


/** Flatten the 3D tensor A using the Flatten function and compare it to
 *  the result in the flat matrix B. */
//______________________________________________________________________________
template <typename Architecture>
auto testFlatten(std::vector<typename Architecture::Matrix_t> &A,
                 const typename Architecture::Matrix_t &B,
                 size_t size,
                 size_t nRows,
                 size_t nCols)
-> bool
{
    
   using Matrix_t = typename Architecture::Matrix_t;
    
   size_t m, n;
   m = B.GetNrows();
   n = B.GetNcols();
    
   Matrix_t AFlat(m, n);
   Architecture::Flatten(AFlat, A, size, nRows, nCols);

   for(size_t i = 0; i < m; i++) {
      for(size_t j = 0; j < n; j++) {
         if(AFlat(i, j) != B(i, j)) {
            return false;
         }
      }
   }
    
   return true;
}




/** Test the instantiation of a Conv Net */
//______________________________________________________________________________
template <typename Architecture>
TConvNet<Architecture>& testConvNetInstantiation(size_t batchSize,
                                                 size_t imgDepth,
                                                 size_t imgHeight,
                                                 size_t imgWidth)
{
   using Net_t    = TConvNet<Architecture>;
   Net_t convNet(batchSize, imgDepth, imgHeight, imgWidth, ELossFunction::kMeanSquaredError);
    
   return convNet;
}

/** Test adding a Conv Layer to a Conv Net. */
//______________________________________________________________________________
template <typename Architecture>
void testAddConvLayer(TConvNet<Architecture>& convNet,
                      size_t depth,
                      size_t filterHeight,
                      size_t filterWidth,
                      size_t strideRows,
                      size_t strideCols,
                      size_t zeroPaddingHeight,
                      size_t zeroPaddingWidth,
                      EActivationFunction f,
                      typename Architecture::Scalar_t dropoutProbability)
{
   convNet.AddConvLayer(depth, filterHeight, filterWidth,
                        strideRows, strideCols, zeroPaddingHeight,
                        zeroPaddingWidth, f, dropoutProbability);
}

/** Test adding a Pool Layer to a Conv Net. */
// ______________________________________________________________________________
template <typename Architecture>
void testAddPoolLayer(TConvNet<Architecture>& convNet,
                      size_t filterHeight,
                      size_t filterWidth,
                      size_t strideRows,
                      size_t strideCols,
                      typename Architecture::Scalar_t dropoutProbability)
{
   convNet.AddPoolLayer(filterHeight, filterWidth, strideRows,
                        strideCols, dropoutProbability);
}


/** Test adding a Fully Connected Layer to a Conv Net */
// ______________________________________________________________________________
template <typename Architecture>
void testAddFullyConnLayer(TConvNet<Architecture>& convNet,
                           size_t width,
                           EActivationFunction f)
{
   convNet.AddFullyConnLayer(width, f);
}

/** Test Conv Net Initialization. */
// ______________________________________________________________________________
template <typename Architecture>
void testConvNetInitialization(TConvNet<Architecture>& convNet)
{
   convNet.Initialize(EInitialization::kGauss);
}


/*! Generate a conv net, perform forward pass */
//______________________________________________________________________________
template <typename Architecture>
auto testConvForwardPass(size_t batchSize,
                         size_t imgDepth,
                         size_t imgHeight,
                         size_t imgWidth,
                         typename Architecture::Scalar_t dropoutProbability = 1.0,
                         bool applyDropout = false)
-> void
{
    
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TConvNet<Architecture>;
    
    
   Net_t convNet(batchSize, imgDepth, imgHeight, imgWidth, ELossFunction::kMeanSquaredError);
   constructConvNet(convNet, dropoutProbability);
   convNet.Initialize(EInitialization::kGauss);
   convNet.InitializeGradients();
    
   std::vector<Matrix_t> X;
   for(size_t i = 0; i < batchSize; i++) {
      X.emplace_back(imgDepth, imgHeight * imgWidth);
      randomMatrix(X[i]);
   }
    
   convNet.Forward(X, applyDropout);
}

/*! Generate a conv net, get the loss. */
//______________________________________________________________________________
template <typename Architecture>
auto testConvLossFunction(size_t batchSize,
                          size_t imgDepth,
                          size_t imgHeight,
                          size_t imgWidth,
                          ERegularization fR = ERegularization::kNone)
-> void
{
    
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TConvNet<Architecture>;
    
   Net_t convNet(batchSize, imgDepth, imgHeight, imgWidth,
                 ELossFunction::kMeanSquaredError, fR);
    
    
   constructConvNet(convNet);
   convNet.Initialize(EInitialization::kGauss);
   convNet.InitializeGradients();
    
   std::vector<Matrix_t> X;
   for(size_t i = 0; i < batchSize; i++) {
      X.emplace_back(imgDepth, imgHeight * imgWidth);
      randomMatrix(X[i]);
   }
    
   Matrix_t Y(batchSize, convNet.GetOutputWidth());
   randomMatrix(Y);
    
   Scalar_t loss = convNet.Loss(X, Y);
   std::cout << "The loss is: " << loss << std::endl;
    
}
    
/*! Generate a conv net, get the predictios */
//______________________________________________________________________________
template <typename Architecture>
auto testConvPrediction(size_t batchSize,
                        size_t imgDepth,
                        size_t imgHeight,
                        size_t imgWidth,
                        EOutputFunction f)
-> void
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TConvNet<Architecture>;
    
   Net_t convNet(batchSize, imgDepth, imgHeight, imgWidth,
                 ELossFunction::kMeanSquaredError);
    
   constructConvNet(convNet);
   convNet.Initialize(EInitialization::kGauss);
   convNet.InitializeGradients();
    
   std::vector<Matrix_t> X;
   for(size_t i = 0; i < batchSize; i++) {
      X.emplace_back(imgDepth, imgHeight * imgWidth);
      randomMatrix(X[i]);
   }
    
   Matrix_t Predictions(batchSize, convNet.GetOutputWidth());
   convNet.Prediction(Predictions, X, f);
    
    
   for(size_t i = 0; i < batchSize; i++){
      for(size_t j = 0; j < convNet.GetOutputWidth(); j++){
         std::cout << Predictions(i, j) << " ";
      }
      std::cout << "" << std::endl;
   }
}


#endif
