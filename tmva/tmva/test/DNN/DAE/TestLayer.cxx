// @(#)root/tmva $Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 #include <iostream>
 #include "TMVA/DNN/Architectures/Reference.h"
 #include "TestLayer.h"

 using namespace TMVA::DNN;
 using namespace TMVA::DNN::DAE;

 int main()

 {
   std::cout<<"Testing started"<<std::endl;

   testLayer<TReference<double>>();

   std::cout<<"Testing for parameter updation"<<std::endl;
   testTraining<TReference<double>>();
   return 0;
 }
