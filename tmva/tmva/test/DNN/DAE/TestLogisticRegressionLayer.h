// @(#)root/tmva $Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd                                            *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef TMVA_TEST_DNN_TEST_DAE_TEST_LOGISTICREGRESSIONLAYER_H_
#define TMVA_TEST_DNN_TEST_DAE_TEST_LOGISTICREGRESSIONLAYER_H_

////////////////////////////////////////////////////////////////////
// Generic tests of the DAE functionalities                       //
////////////////////////////////////////////////////////////////////


#include <iostream>
#include "../Utility.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DAE/StackedNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;




//_____________________________________________________________________________
template<typename Architecture>
auto testForwardLogReg(typename Architecture::Matrix_t & input,
                      typename Architecture::Matrix_t & p,
                      typename Architecture::Matrix_t & fWeights)
-> void
{
  size_t m,n;
  m=p.GetNrows();
  n=p.GetNcols();
  using Matrix_t = typename Architecture::Matrix_t;

  Architecture::ForwardLogReg(input,p,fWeights);

  /*for(size_t i=0; i<m;i++)
  {
    for(size_t j=0;j<n;j++)
    {
      std::cout<<p(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }*/
}



//______________________________________________________________________________
template<typename Architecture>
auto testUpdateParamsLogReg(typename Architecture::Matrix_t &input,
                            typename Architecture::Matrix_t &output,
                            typename Architecture::Matrix_t &difference,
                            typename Architecture::Matrix_t &p,
                            typename Architecture::Matrix_t &fWeights,
                            typename Architecture::Matrix_t &fBiases,
                            double learningRate,
                            size_t fBatchSize)
-> void
{
  size_t m,n;
  m=fWeights.GetNrows();
  n=fWeights.GetNcols();
  using Matrix_t = typename Architecture::Matrix_t;

  Architecture::UpdateParamsLogReg(input,output,difference,p,fWeights,fBiases,
                                   learningRate,fBatchSize);

  /*for(size_t i=0; i<m;i++)
  {
    for(size_t j=0; j<n;j++)
    {
      std::cout<<fWeights(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }*/

}

#endif
