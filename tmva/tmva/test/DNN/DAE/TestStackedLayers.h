// @(#)root/tmva $Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2016, ajatgd
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <iostream>
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DAE/StackedNet.h"
#include "../Utility.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

template<typename Architecture>
auto testStacked()
-> void
{
  using Scalar_t = typename Architecture::Scalar_t;
  using Matrix_t = typename Architecture::Matrix_t;
  using TSDAE = TSDAE<Architecture>;

  TSDAE sdae(1,6,2,2,{3,2});
  //sdae.Init();
}
