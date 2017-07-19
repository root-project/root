// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Downsample method                                                 *
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



////////////////////////////////////////////////////////////////////
// Testing the Downsample function                                //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

#include "TMVA/DNN/Architectures/Reference.h"
#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;
using Matrix_t = typename TReference<double>::Matrix_t;


inline bool isInteger(double x) {return x == floor(x);}

size_t calculateDimension(size_t imgDim,
                          size_t fltDim,
                          size_t padding,
                          size_t stride)
{
   double dimension = ((imgDim - fltDim + 2 * padding) / stride) + 1;
   if(!isInteger(dimension)) {
      std::cout << "Not compatible hyper parameters" << std::endl;
      std::exit(EXIT_FAILURE);
   }
    
   return (size_t) dimension;
}



/*************************************************************************
 * Test 1:
 *  depth = 2, image height = 4, image width = 5,
 *  frame depth = 2, filter height = 2, filter width = 2,
 *  stride rows = 2, stride cols = 1,
 *  zero-padding height = 0, zero-padding width = 0,
 *************************************************************************/

void test1()
{
    
   double imgTest1[][20] =
      {
        {166,  212,  213,  150,  114,
         119,  109,  115,   88,  144,
         227,  208,  208,  235,   57,
          57,  165,  250,  139,   76},
        
        { 57,  255,  184,  162,  204,
         220,   11,  192,  183,  174,
           2,  153,  183,  175,   10,
          55,  123,  246,  138,   80}
      };
    
    
   double answerTest1[][8] =
      {
        {212,  213,  213,  150,
         227,  250,  250,  235},
        
        {255,  255,  192,  204,
         153,  246,  246,  175}
      };
    
   double answerIdxTest1[][8] =
      {
        {  1,    2,    2,    3,
          10,   17,   17,   13},
        
        {  1,    1,    7,    4,
          11,   17,   17,   13}
      };
    
   size_t imgDepthTest1 = 2;
   size_t imgHeightTest1 = 4;
   size_t imgWidthTest1 = 5;
   size_t fltHeightTest1 = 2;
   size_t fltWidthTest1 = 2;
   size_t strideRowsTest1 = 2;
   size_t strideColsTest1 = 1;
    
    
   Matrix_t A(imgDepthTest1, imgHeightTest1 * imgWidthTest1);
    
   for(size_t i = 0; i < (size_t) A.GetNrows(); i++){
      for(size_t j = 0; j < (size_t) A.GetNcols(); j++){
         A(i, j) = imgTest1[i][j];
      }
   }
    
    
   size_t height = calculateDimension(imgHeightTest1, fltHeightTest1,
                                      0, strideRowsTest1);
    
   size_t width = calculateDimension(imgWidthTest1, fltWidthTest1,
                                     0, strideColsTest1);

    
   Matrix_t idx(imgDepthTest1,  height * width);
   Matrix_t B(imgDepthTest1, height * width);
    
   for(size_t i = 0; i < (size_t) B.GetNrows(); i++){
      for(size_t j = 0; j < (size_t) B.GetNcols(); j++){
         idx(i, j) = answerIdxTest1[i][j];
         B(i, j) = answerTest1[i][j];
      }
   }
    
    
    
   bool status = testDownsample<TReference<double>>(A, idx, B,
                                                    imgHeightTest1, imgWidthTest1,
                                                    fltHeightTest1, fltWidthTest1,
                                                    strideRowsTest1, strideColsTest1);

   if(status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}

/*************************************************************************
 * Test 1:
 *  depth = 1, image height = 6, image width = 6,
 *  frame depth = 1, filter height = 2, filter width = 3,
 *  stride rows = 1, stride cols = 3,
 *  zero-padding height = 0, zero-padding width = 0,
 *************************************************************************/

void test2()
{
    
   double imgTest2[][36] =
      {
        {200,  79,  69,  58,  98, 168,
          49, 230,  21, 141, 218,  38,
          72, 224,  14,  65, 147, 105,
          38,  27, 111, 160, 200,  48,
         109, 104, 153, 149, 233,  11,
          16,  91, 236, 183, 166, 155}
      };
    
    
   double answerTest2[][10] =
      {
        {230, 218,
         230, 218,
         224, 200,
         153, 233,
         236, 233}
      };
    
   double answerIdxTest2[][10] =
      {
        {  7,  10,
           7,  10,
          13,  22,
          26,  28,
          32,  28}
      };
    
   size_t imgDepthTest2 = 1;
   size_t imgHeightTest2 = 6;
   size_t imgWidthTest2 = 6;
   size_t fltHeightTest2 = 2;
   size_t fltWidthTest2 = 3;
   size_t strideRowsTest2 = 1;
   size_t strideColsTest2 = 3;
    
    
   Matrix_t A(imgDepthTest2, imgHeightTest2 * imgWidthTest2);
    
   for(size_t i = 0; i < (size_t) A.GetNrows(); i++){
      for(size_t j = 0; j < (size_t) A.GetNcols(); j++){
         A(i, j) = imgTest2[i][j];
      }
   }
    
    
   size_t height = calculateDimension(imgHeightTest2, fltHeightTest2,
                                      0, strideRowsTest2);
    
   size_t width = calculateDimension(imgWidthTest2, fltWidthTest2,
                                     0, strideColsTest2);
    
    
   Matrix_t idx(imgDepthTest2,  height * width);
   Matrix_t B(imgDepthTest2, height * width);
    
   for(size_t i = 0; i < (size_t) B.GetNrows(); i++){
      for(size_t j = 0; j < (size_t) B.GetNcols(); j++){
         idx(i, j) = answerIdxTest2[i][j];
         B(i, j) = answerTest2[i][j];
      }
   }
    
    
    
   bool status = testDownsample<TReference<double>>(A, idx, B,
                                                    imgHeightTest2, imgWidthTest2,
                                                    fltHeightTest2, fltWidthTest2,
                                                    strideRowsTest2, strideColsTest2);
    
   if(status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}


int main(){
   std::cout << "Testing Downsample function:" << std::endl;
    
   std::cout << "Test 1: " << std::endl;
   test1();
    
   std::cout << "Test 2: " << std::endl;
   test2();
}
