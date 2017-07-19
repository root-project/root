// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Im2Col method                                                     *
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
// Testing the method Im2col                                      //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

#include "TMVA/DNN/Architectures/Reference.h"
#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;
using Matrix_t = typename TReference<double>::Matrix_t;

inline bool isInteger(double x)
{
   return x == floor(x);
}

int calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride)
{
   double dimension = ((imgDim - fltDim + 2 * padding) / stride) + 1;
   if (!isInteger(dimension)) {
      std::cout << "Not compatible hyper parameters" << std::endl;
      std::exit(EXIT_FAILURE);
   }

   return (size_t)dimension;
}

/*************************************************************************
 * Test 1:
 *  depth = 1, image height = 5, image width = 5,
 *  filter depth = 1, filter height = 2, filter width = 2,
 *  stride rows = 1, stride cols = 1,
 *  zero-padding height = 1, zero-padding width = 1,
 *************************************************************************/

void test1()
{
   double imgTest1[][25] = {{244, 198, 134, 194, 86, 104, 156, 52,  126, 39,  56,  250, 68,
                             247, 251, 93,  160, 61, 8,   81,  204, 113, 107, 206, 146}

   };

   double answerTest1[][4] = {{0, 0, 0, 244},      {0, 0, 244, 198},    {0, 0, 198, 134},    {0, 0, 134, 194},
                              {0, 0, 194, 86},     {0, 0, 86, 0},       {0, 244, 0, 104},    {244, 198, 104, 156},
                              {198, 134, 156, 52}, {134, 194, 52, 126}, {194, 86, 126, 39},  {86, 0, 39, 0},
                              {0, 104, 0, 56},     {104, 156, 56, 250}, {156, 52, 250, 68},  {52, 126, 68, 247},
                              {126, 39, 247, 251}, {39, 0, 251, 0},     {0, 56, 0, 93},      {56, 250, 93, 160},
                              {250, 68, 160, 61},  {68, 247, 61, 8},    {247, 251, 8, 81},   {251, 0, 81, 0},
                              {0, 93, 0, 204},     {93, 160, 204, 113}, {160, 61, 113, 107}, {61, 8, 107, 206},
                              {8, 81, 206, 146},   {81, 0, 146, 0},     {0, 204, 0, 0},      {204, 113, 0, 0},
                              {113, 107, 0, 0},    {107, 206, 0, 0},    {206, 146, 0, 0},    {146, 0, 0, 0}};

   size_t imgDepthTest1 = 1;
   size_t imgHeightTest1 = 5;
   size_t imgWidthTest1 = 5;
   size_t fltHeightTest1 = 2;
   size_t fltWidthTest1 = 2;
   size_t strideRowsTest1 = 1;
   size_t strideColsTest1 = 1;
   size_t zeroPaddingHeightTest1 = 1;
   size_t zeroPaddingWidthTest1 = 1;

   Matrix_t A(imgDepthTest1, imgHeightTest1 * imgWidthTest1);

   for (size_t i = 0; i < (size_t)A.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)A.GetNcols(); j++) {
         A(i, j) = imgTest1[i][j];
      }
   }

   size_t height = calculateDimension(imgHeightTest1, fltHeightTest1, zeroPaddingHeightTest1, strideRowsTest1);

   size_t width = calculateDimension(imgWidthTest1, fltWidthTest1, zeroPaddingWidthTest1, strideColsTest1);

   Matrix_t B(height * width, imgDepthTest1 * fltHeightTest1 * fltWidthTest1);

   for (size_t i = 0; i < (size_t)B.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)B.GetNcols(); j++) {
         B(i, j) = answerTest1[i][j];
      }
   }

   bool status =
      testIm2col<TReference<double>>(A, B, imgHeightTest1, imgWidthTest1, fltHeightTest1, fltWidthTest1,
                                     strideRowsTest1, strideColsTest1, zeroPaddingHeightTest1, zeroPaddingWidthTest1);

   if (status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}

/*************************************************************************
 * Test 2:
 *  depth = 2, image height = 5, image width = 5,
 *  filter depth = 2, filter height = 2, filter width = 3,
 *  stride rows = 1, stride cols = 1,
 *  zero-padding height = 1, zero-padding width = 1,
 *************************************************************************/

void test2()
{

   // 2 x 5 x 5 image
   double imgTest2[][25] = {

      {244, 198, 134, 194, 86, 104, 156, 52,  126, 39,  56,  250, 68,
       247, 251, 93,  160, 61, 8,   81,  204, 113, 107, 206, 146},

      {205, 136, 184, 196, 42,  157, 10, 62,  201, 46,  250, 78, 43,
       185, 82,  95,  218, 128, 104, 71, 118, 215, 228, 199, 52}

   };

   double answerTest2[][12] = {{0, 0, 0, 0, 244, 198, 0, 0, 0, 0, 205, 136},
                               {0, 0, 0, 244, 198, 134, 0, 0, 0, 205, 136, 184},
                               {0, 0, 0, 198, 134, 194, 0, 0, 0, 136, 184, 196},
                               {0, 0, 0, 134, 194, 86, 0, 0, 0, 184, 196, 42},
                               {0, 0, 0, 194, 86, 0, 0, 0, 0, 196, 42, 0},
                               {0, 244, 198, 0, 104, 156, 0, 205, 136, 0, 157, 10},
                               {244, 198, 134, 104, 156, 52, 205, 136, 184, 157, 10, 62},
                               {198, 134, 194, 156, 52, 126, 136, 184, 196, 10, 62, 201},
                               {134, 194, 86, 52, 126, 39, 184, 196, 42, 62, 201, 46},
                               {194, 86, 0, 126, 39, 0, 196, 42, 0, 201, 46, 0},
                               {0, 104, 156, 0, 56, 250, 0, 157, 10, 0, 250, 78},
                               {104, 156, 52, 56, 250, 68, 157, 10, 62, 250, 78, 43},
                               {156, 52, 126, 250, 68, 247, 10, 62, 201, 78, 43, 185},
                               {52, 126, 39, 68, 247, 251, 62, 201, 46, 43, 185, 82},
                               {126, 39, 0, 247, 251, 0, 201, 46, 0, 185, 82, 0},
                               {0, 56, 250, 0, 93, 160, 0, 250, 78, 0, 95, 218},
                               {56, 250, 68, 93, 160, 61, 250, 78, 43, 95, 218, 128},
                               {250, 68, 247, 160, 61, 8, 78, 43, 185, 218, 128, 104},
                               {68, 247, 251, 61, 8, 81, 43, 185, 82, 128, 104, 71},
                               {247, 251, 0, 8, 81, 0, 185, 82, 0, 104, 71, 0},
                               {0, 93, 160, 0, 204, 113, 0, 95, 218, 0, 118, 215},
                               {93, 160, 61, 204, 113, 107, 95, 218, 128, 118, 215, 228},
                               {160, 61, 8, 113, 107, 206, 218, 128, 104, 215, 228, 199},
                               {61, 8, 81, 107, 206, 146, 128, 104, 71, 228, 199, 52},
                               {8, 81, 0, 206, 146, 0, 104, 71, 0, 199, 52, 0},
                               {0, 204, 113, 0, 0, 0, 0, 118, 215, 0, 0, 0},
                               {204, 113, 107, 0, 0, 0, 118, 215, 228, 0, 0, 0},
                               {113, 107, 206, 0, 0, 0, 215, 228, 199, 0, 0, 0},
                               {107, 206, 146, 0, 0, 0, 228, 199, 52, 0, 0, 0},
                               {206, 146, 0, 0, 0, 0, 199, 52, 0, 0, 0, 0}};

   size_t imgDepthTest2 = 2;
   size_t imgHeightTest2 = 5;
   size_t imgWidthTest2 = 5;
   size_t fltHeightTest2 = 2;
   size_t fltWidthTest2 = 3;
   size_t strideRowsTest2 = 1;
   size_t strideColsTest2 = 1;
   size_t zeroPaddingHeightTest2 = 1;
   size_t zeroPaddingWidthTest2 = 1;

   Matrix_t A(imgDepthTest2, imgHeightTest2 * imgWidthTest2);

   for (size_t i = 0; i < (size_t)A.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)A.GetNcols(); j++) {
         A(i, j) = imgTest2[i][j];
      }
   }

   size_t height = calculateDimension(imgHeightTest2, fltHeightTest2, zeroPaddingHeightTest2, strideRowsTest2);

   size_t width = calculateDimension(imgWidthTest2, fltWidthTest2, zeroPaddingWidthTest2, strideColsTest2);

   Matrix_t B(height * width, imgDepthTest2 * fltHeightTest2 * fltWidthTest2);

   for (size_t i = 0; i < (size_t)B.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)B.GetNcols(); j++) {
         B(i, j) = answerTest2[i][j];
      }
   }

   bool status =
      testIm2col<TReference<double>>(A, B, imgHeightTest2, imgWidthTest2, fltHeightTest2, fltWidthTest2,
                                     strideRowsTest2, strideColsTest2, zeroPaddingHeightTest2, zeroPaddingWidthTest2);

   if (status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}

/*************************************************************************
 * Test 3:
 *  depth = 3, image height = 2, image width = 3,
 *  filter depth = 3, filter height = 3, filter width = 2,
 *  stride rows = 3, stride cols = 1,
 *  zero-padding height = 2, zero-padding width = 1,
 *************************************************************************/

void test3()
{

   // 3 x 2 x 3 image
   double imgTest3[][6] = {{235, 213, 185, 144, 235, 212},

                           {158, 168, 116, 68, 159, 157},

                           {240, 135, 195, 252, 36, 77}};

   double answerTest3[][18] = {{0, 0, 0, 0, 0, 235, 0, 0, 0, 0, 0, 158, 0, 0, 0, 0, 0, 240},
                               {0, 0, 0, 0, 235, 213, 0, 0, 0, 0, 158, 168, 0, 0, 0, 0, 240, 135},
                               {0, 0, 0, 0, 213, 185, 0, 0, 0, 0, 168, 116, 0, 0, 0, 0, 135, 195},
                               {0, 0, 0, 0, 185, 0, 0, 0, 0, 0, 116, 0, 0, 0, 0, 0, 195, 0},
                               {0, 144, 0, 0, 0, 0, 0, 68, 0, 0, 0, 0, 0, 252, 0, 0, 0, 0},
                               {144, 235, 0, 0, 0, 0, 68, 159, 0, 0, 0, 0, 252, 36, 0, 0, 0, 0},
                               {235, 212, 0, 0, 0, 0, 159, 157, 0, 0, 0, 0, 36, 77, 0, 0, 0, 0},
                               {212, 0, 0, 0, 0, 0, 157, 0, 0, 0, 0, 0, 77, 0, 0, 0, 0, 0}};

   size_t imgDepthTest3 = 3;
   size_t imgHeightTest3 = 2;
   size_t imgWidthTest3 = 3;
   size_t fltHeightTest3 = 3;
   size_t fltWidthTest3 = 2;
   size_t strideRowsTest3 = 3;
   size_t strideColsTest3 = 1;
   size_t zeroPaddingHeightTest3 = 2;
   size_t zeroPaddingWidthTest3 = 1;

   Matrix_t A(imgDepthTest3, imgHeightTest3 * imgWidthTest3);

   for (size_t i = 0; i < (size_t)A.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)A.GetNcols(); j++) {
         A(i, j) = imgTest3[i][j];
      }
   }

   size_t height = calculateDimension(imgHeightTest3, fltHeightTest3, zeroPaddingHeightTest3, strideRowsTest3);

   size_t width = calculateDimension(imgWidthTest3, fltWidthTest3, zeroPaddingWidthTest3, strideColsTest3);

   Matrix_t B(height * width, imgDepthTest3 * fltHeightTest3 * fltWidthTest3);

   for (size_t i = 0; i < (size_t)B.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)B.GetNcols(); j++) {
         B(i, j) = answerTest3[i][j];
      }
   }

   bool status =
      testIm2col<TReference<double>>(A, B, imgHeightTest3, imgWidthTest3, fltHeightTest3, fltWidthTest3,
                                     strideRowsTest3, strideColsTest3, zeroPaddingHeightTest3, zeroPaddingWidthTest3);

   if (status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}

int main()
{
   std::cout << "Testing Im2Col function:" << std::endl;

   std::cout << "Test 1: " << std::endl;
   test1();

   std::cout << "Test 2: " << std::endl;
   test2();

   std::cout << "Test 3: " << std::endl;
   test3();
}
