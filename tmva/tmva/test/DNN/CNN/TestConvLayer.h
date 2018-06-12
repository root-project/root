// @(#)root/tmva/tmva/cnn:$Id$
// Author: Manos Stergiadis

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Downsample method on a CPU architecture                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Manos Stergiadis       <em.stergiadis@gmail.com>  - CERN, Switzerland     *
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

////////////////////////////////////////////////////////////////////
// Testing the Convolutional Layer                                //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;

inline bool isInteger(double x)
{
   return x == floor(x);
}

size_t calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride)
{
   double dimension = ((imgDim - fltDim + 2 * padding) / stride) + 1;
   if (!isInteger(dimension)) {
      std::cout << "Not compatible hyper parameters" << std::endl;
      std::exit(EXIT_FAILURE);
   }

   return (size_t)dimension;
}

/*************************************************************************
 * Test 1: Forward Propagation
 *  batch size = 1
 *  image depth = 2, image height = 4, image width = 4,
 *  num frames = 3, filter height = 2, filter width = 2,
 *  stride rows = 2, stride cols = 2,
 *  zero-padding height = 0, zero-padding width = 0,
 *************************************************************************/
template<typename Architecture>
bool testForward1()
{
   using Matrix_t = typename Architecture::Matrix_t;
   double img[][16] = {
           {166, 212, 213, 150,
            114, 119, 109, 115,
             88, 144, 227, 208,
            208, 235,  57,  58},

           { 57,  255, 184, 162,
            204,  220,  11, 192,
            183,  174,   2, 153,
            184,  175,  10,  55}
   };

   double weights[][8] = {
           {2.0,  3.0,  0.5, -1.5,
            1.0,  1.5, -2.0, -3.0},

           {-0.5,  1.0,  2.5, -1.0,
             2.0,  1.5, -0.5,  1.0},

           {-1.0, -2.0, 1.5, 0.5,
             2.0, -1.5, 0.5, 1.0}
   };

   double biases[][1] = {
           {45},

           {60},

           {12}
   };

   double expected[][9] = {

           {263.0, 1062.0,  632.0,
            104.0,  224.0,  245.5,
            -44.5,  843.0, 1111.0},

           { 969.5, 1042.5, 1058.5,
            1018.5,  614.0,  942.0,
            1155.0, 1019.0,  522.5},

           {-294.0, -38.0,   42.5,
             207.5, 517.0,    5.5,
             437.5, 237.5, -682.0}
    };



   size_t imgDepth = 2;
   size_t imgHeight = 4;
   size_t imgWidth = 4;
   size_t numberFilters = 3;
   size_t fltHeight = 2;
   size_t fltWidth = 2;
   size_t strideRows = 1;
   size_t strideCols = 1;
   size_t zeroPaddingHeight = 0;
   size_t zeroPaddingWidth = 0;

   Matrix_t inputEvent(imgDepth, imgHeight * imgWidth);

   for (size_t i = 0; i < imgDepth; i++) {
      for (size_t j = 0; j < imgHeight * imgWidth; j++) {
         inputEvent(i, j) = img[i][j];
      }
   }
   std::vector<Matrix_t> input;
   input.push_back(inputEvent);

   Matrix_t weightsMatrix(numberFilters, fltHeight * fltWidth * imgDepth);
   Matrix_t biasesMatrix(numberFilters, 1);
   for (size_t i = 0; i < numberFilters; i++) {
       for (size_t j = 0; j < fltHeight * fltWidth * imgDepth; j++){
           weightsMatrix(i, j) = weights[i][j];
       }
       biasesMatrix(i, 0) = biases[i][0];
   }

   size_t height = calculateDimension(imgHeight, fltHeight, zeroPaddingHeight, strideRows);
   size_t width = calculateDimension(imgWidth, fltWidth, zeroPaddingWidth, strideCols);

   Matrix_t outputEvent(numberFilters, height * width);

   for (size_t i = 0; i < numberFilters; i++) {
      for (size_t j = 0; j < height * width; j++) {
         outputEvent(i, j) = expected[i][j];
      }
   }
   std::vector<Matrix_t> expectedOutput;
   expectedOutput.push_back(outputEvent);

   bool status = testConvLayerForward<Architecture>(input, expectedOutput, weightsMatrix, biasesMatrix, imgHeight,
                                                    imgWidth, imgDepth, fltHeight, fltWidth, numberFilters, strideRows,
                                                    strideCols, zeroPaddingHeight, zeroPaddingWidth);

   return status;
}


