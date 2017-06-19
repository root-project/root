// @(#)root/tmva $Id$
// Author: Vladimir Ilievski, 15/06/2017

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


////////////////////////////////////////////////////////////////////
// Testing the Conv Net Initialization                            //
////////////////////////////////////////////////////////////////////

#include <iostream>


#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/CNN/ConvNet.h"
#include "TestConvNet.h"
#include "TMVA/DNN/Functions.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;


int main(){
    
   std::cout << "Testing CNN Instantiation:" << std::endl;
    
   size_t batchSize = 1;
   size_t imgDepth = 3;
   size_t imgHeight = 32;
   size_t imgWidth = 32;
    
   TConvNet<TReference<double>> convNet =
         testConvNetInstantiation<TReference<double>>(batchSize, imgDepth,
                                                      imgHeight, imgWidth);
    std::cout << "Instantiation successful!" << std::endl;
    
    
    
   std::cout << "Add Conv Layer testing!" << std::endl;
   size_t depthConv1 = 12;
   size_t filterHeightConv1 = 2;
   size_t filterWidthConv1 = 2;
   size_t strideRowsConv1 = 1;
   size_t strideColsConv1 = 1;
   size_t zeroPaddingHeightConv1 = 1;
   size_t zeroPaddingWidthConv1 = 1;
   EActivationFunction fConv1 = EActivationFunction::kSigmoid;
   testAddConvLayer<TReference<double>>(convNet, depthConv1, filterHeightConv1, filterWidthConv1,
                                        strideRowsConv1, strideColsConv1, zeroPaddingHeightConv1,
                                        zeroPaddingWidthConv1, fConv1, 1.0);
   std::cout << "Add Conv Layer successful!" << std::endl;
    
    
   std::cout << "Add Pool Layer testing!" << std::endl;
   size_t filterHeightPool1 = 6;
   size_t filterWidthPool1 = 6;
   size_t strideRowsPool1 = 1;
   size_t strideColsPool1 = 1;
   testAddPoolLayer<TReference<double>>(convNet, filterHeightPool1, filterWidthPool1,
                                        strideRowsPool1, strideColsPool1, 1.0);
   std::cout << "Add Pool Layer successful!" << std::endl;
    
    
   std::cout << "Add Fully Connected Layer testing!" << std::endl;
   size_t widthFC1 = 20;
   EActivationFunction fFC1 = EActivationFunction::kSigmoid;
   testAddFullyConnLayer<TReference<double>>(convNet, widthFC1, fFC1);
   std::cout << "Add Fully Connected Layer successful!" << std::endl;
    
    
   std::cout << "Initializing Conv Net testing" << std::endl;
   testConvNetInitialization<TReference<double>>(convNet);
   std::cout << "Initialization successful" << std::endl;
    
   convNet.Print();
    
    
}
