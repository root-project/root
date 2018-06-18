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

/*************************************************************************
* Test 1: Backward Propagation
*  batch size = 1
*  image depth = 2, image height = 5, image width = 5,
*  num frames = 2, filter height = 3, filter width = 3,
*  stride rows = 1, stride cols = 1,
*  zero-padding height = 0, zero-padding width = 0,
*************************************************************************/
template<typename Architecture>
bool testBackward1()
{
    using Matrix_t = typename Architecture::Matrix_t;

    size_t imgDepth = 2;
    size_t imgHeight = 5;
    size_t imgWidth = 5;
    size_t numberFilters = 2;
    size_t fltHeight = 3;
    size_t fltWidth = 3;
    size_t strideRows = 1;
    size_t strideCols = 1;
    size_t zeroPaddingHeight = 0;
    size_t zeroPaddingWidth = 0;

    size_t height = calculateDimension(imgHeight, fltHeight, zeroPaddingHeight, strideRows);
    size_t width = calculateDimension(imgWidth, fltWidth, zeroPaddingWidth, strideCols);
    size_t nLocalViews =  height * width;
    size_t batchSize = 1;

    double grad[][9] = {
            {0, 1.37122, 0, 0, 0, 0, 0, -0.901251, 0},
            {0, -0.377314, 0, 0, 0, -0.259376, 0.262045, 0, 0}
    };

    Matrix_t gradEvent(numberFilters, height * width);

    for (size_t i = 0; i < numberFilters; i++) {
        for (size_t j = 0; j < height * width; j++) {
            gradEvent(i, j) = grad[i][j];
        }
    }
    std::vector<Matrix_t> activationGradients;
    activationGradients.push_back(gradEvent);

    double derivatives[][9] = {
            {1, 1, 1, 1 , 1, 1, 1, 1, 1},
            {1, 1, 1, 1 , 1, 1, 1, 1, 1}
    };

    Matrix_t dfEvent(numberFilters, height * width);

    for (size_t i = 0; i < numberFilters; i++) {
        for (size_t j = 0; j < height * width; j++) {
            dfEvent(i, j) = derivatives[i][j];
        }
    }
    std::vector<Matrix_t> df;
    df.push_back(dfEvent);

    double W[][18] = {
            {0.432889, 0.313561, -0.352283, -0.338028, 0.403259, 0.261901, -0.303414, 0.296608, -0.319302,
             0.216209, 0.447231, -0.368792, 0.0384685, -0.272119, -0.530435, 0.240504, 0.227794, -0.358768},

            {-0.0605551, -0.353563, 0.10477, -0.493842, -0.887311, 0.352012, 0.033447, -0.00807299, -0.199177,
             -0.0916032, 0.295868, 0.105432, -0.153309, -0.113395, 0.0297921, 0.0887323, 0.174929, 0.358333}
    };

    Matrix_t weights(numberFilters, imgDepth * fltHeight * fltWidth);

    for (size_t i = 0; i < numberFilters; i++) {
        for (size_t j = 0; j < imgDepth * fltHeight * fltWidth; j++) {
            weights(i, j) = W[i][j];
        }
    }

    double activationsPreviousLayer[][25] = {
            {-0.339062, -0.0553435, -0.347659, -0.159283, -0.1376, 0.449175, 0.514769, 0.269177, 1.21222,
             -0.116483, 0.867627, 0.00710033, -0.569898, -1.11606, 1.2248, -0.0225848, 0.291324, 2.3244,
             1.87425, -0.319708, -0.0807289, -0.206493, -1.62712, -1.4058, -0.800368},

            {0.42656, -1.12085, 1.10635, -0.732954, 0.388681, 0.669813, -2.36616, 1.13498, -1.19103, 0.398025,
             0.282946, 0.0433294, 1.01771, 0.315107, -1.87112, -0.0927787, 0.215233, -2.51543, 0.117346, 0.841533,
             -0.0431507, 0.56968, 0.481225, 0.0962775, -0.0885691}
    };

    Matrix_t activationsBackwardEvent(imgDepth, imgHeight * imgWidth);

    for (size_t i = 0; i < imgDepth; i++) {
        for (size_t j = 0; j < imgHeight * imgWidth; j++) {
            activationsBackwardEvent(i, j) = activationsPreviousLayer[i][j];
        }
    }
    std::vector<Matrix_t> activationsBackward;
    activationsBackward.push_back(activationsBackwardEvent);

    /////////////////////// Fill the expected output //////////////////////////

    double expectedActivationGradsBackward[][25] = {
            {0, 0.616434, 0.563365, -0.522588, 0, 0, -0.277178, 0.903457, 0.31801, -0.0271747, -0.0158681, -0.911457,
             0.282709, 0.184961, -0.0913035, -0.129409, 0.0721334, -0.27987, -0.233944, 0.0516617, 0.00876461, 0.271336,
             -0.319512, 0.287772, 0},

            {0, 0.331034, 0.501617, -0.545475, 0, 0, 0.110594, -0.30659, -0.815325, -0.0273465, -0.0240041, 0.178975,
             -0.0893224, -0.265368, -0.00772735, -0.0401738, -0.0643844, 0.23004, 0.432683, -0.092943, 0.0232518,
             -0.170915, -0.1114, 0.32334, 0}
    };

    Matrix_t expectedActivationGradientsBackwardEvent(imgDepth, imgHeight * imgWidth);

    for (size_t i = 0; i < imgDepth; i++) {
        for (size_t j = 0; j < imgHeight * imgWidth; j++) {
            expectedActivationGradientsBackwardEvent(i, j) = expectedActivationGradsBackward[i][j];
        }
    }

    std::vector<Matrix_t> computedActivationGradientsBackward;
    computedActivationGradientsBackward.emplace_back(imgDepth, imgHeight * imgWidth);


    Matrix_t weightGradients(numberFilters, imgDepth * fltHeight * fltWidth);
    Matrix_t biasGradients(numberFilters, 1);

    Architecture::ConvLayerBackward(computedActivationGradientsBackward, weightGradients, biasGradients,
                                    df, activationGradients, weights, activationsBackward,
                                    batchSize, imgHeight, imgWidth, numberFilters, height,
                                    width, imgDepth, fltHeight, fltWidth, nLocalViews);

    printf("---Expected---:\n\n");
    expectedActivationGradientsBackwardEvent.Print();

    printf("---Computed--:\n\n");
    computedActivationGradientsBackward[0].Print();

    double epsilon = 0.001;
    for (size_t i = 0; i < imgDepth; i++) {
        for (size_t j = 0; j < imgHeight * imgWidth; j++) {
            double computed = computedActivationGradientsBackward[0](i, j);
            double expected = expectedActivationGradientsBackwardEvent(i, j);
            if (abs(computed - expected) > epsilon) return false;
        }
    }
    return true;
};