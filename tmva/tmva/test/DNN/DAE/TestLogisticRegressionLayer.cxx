// @(#)root/tmva $Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


////////////////////////////////////////////////////////////////////
// Testing the Encode function                                    //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Functions.h"
#include "TestLogisticRegressionLayer.h"
#include "TestUtilsAE.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

using Matrix_t = typename TReference<double>::Matrix_t;

void test()
{
  double train_X[][6] = {
    {1, 1, 1, 0, 0, 0},
    {1, 0, 1, 0, 0, 0},
    {1, 1, 1, 0, 0, 0},
    {0, 0, 1, 1, 1, 0},
    {0, 0, 1, 1, 0, 0},
    {0, 0, 1, 1, 1, 0}
};
  double train_Y[][2] = {
    {1, 0},
    {1, 0},
    {1, 0},
    {0, 1},
    {0, 1},
    {0, 1}
};
                    //   {{0.3333}, {0.4000}, {0.7778}, {0.7778}, {0.8000}, {0.9333}};

  double test_X[][6] = {
    {1, 0, 1, 0, 0, 0},
    {0, 0, 1, 1, 1, 0}
};
                  // {{10},{15},{20},{25},{30},{35},{40},{45},{50}};

  double learningRate=0.1;
  size_t fBatchSize =6;
  size_t totalRowsTrainInput= sizeof(train_X)/sizeof(train_X[0]);
  size_t totalColumnsTrainInput = sizeof(train_X[0]) / sizeof(train_X[0][0]);
  std::cout<<"Train inp "<<totalRowsTrainInput<<" "<<totalColumnsTrainInput<<std::endl;
  std::vector<Matrix_t> TrainX;
  for(size_t i=0; i<totalRowsTrainInput;i++)
  {
    TrainX.emplace_back(totalColumnsTrainInput,1);
  }
  std::cout<<"size "<<TrainX.size()<<std::endl;




  size_t totalRowsTrainOutput= sizeof(train_Y)/sizeof(train_Y[0]);
  size_t totalColumnsTrainOutput = sizeof(train_Y[0]) / sizeof(train_Y[0][0]);
  std::cout<<"Train out "<<totalRowsTrainOutput<<" "<<totalColumnsTrainOutput<<std::endl;

  std::vector<Matrix_t> TrainY;
  for(size_t i=0; i<totalRowsTrainOutput;i++)
  {
    TrainY.emplace_back(totalColumnsTrainOutput,1);
  }
  std::cout<<"size "<<TrainY.size()<<std::endl;



  size_t totalRowsTestInput= sizeof(test_X)/sizeof(test_X[0]);
  size_t totalColumnsTestInput = sizeof(test_X[0]) / sizeof(test_X[0][0]);
  std::cout<<"Test Inp "<<totalRowsTestInput<<" "<<totalColumnsTestInput<<std::endl;

  std::vector<Matrix_t> TestX;
  for(size_t i=0; i<totalRowsTestInput;i++)
  {
    TestX.emplace_back(totalColumnsTestInput,1);
  }
  std::cout<<"size "<<TestX.size()<<std::endl;


  size_t totalRowsTestOutput= 2;
  size_t totalColumnsTestOutput = 2;

  std::vector<Matrix_t> TestY;
  for(size_t i=0; i<totalRowsTestOutput;i++)
  {
    TestY.emplace_back(totalColumnsTestOutput,1);
  }
  std::cout<<"size of test out "<<TestY.size()<<std::endl;


  Matrix_t p(totalColumnsTestOutput,1);
  Matrix_t difference(totalColumnsTestOutput,1);
  Matrix_t Weights(totalColumnsTestOutput,totalColumnsTrainInput);
  randomMatrix(Weights);
  std::cout<<"Weights "<<Weights.GetNrows()<<" "<<Weights.GetNcols()<<std::endl;

  /*for( size_t i=0; i<(size_t)Weights.GetNrows(); i++)
  {
    for(size_t j=0; j<(size_t)Weights.GetNcols(); j++)
    {
      std::cout<<Weights(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }*/

  Matrix_t Biases(totalColumnsTestOutput,1);
  for(size_t epoch=0;epoch<1000;epoch++){
  for(size_t i=0; i<totalRowsTrainInput; i++)
  {
    for( size_t j=0; j<(size_t)TrainX[i].GetNrows(); j++)
    {
      for(size_t k=0; k<(size_t)TrainX[i].GetNcols(); k++)
      {
        TrainX[i](j,k) = train_X[i][j];
      }
    }

    /*for( size_t j=0; j<(size_t)TrainX[i].GetNrows(); j++)
    {
      for(size_t k=0; k<(size_t)TrainX[i].GetNcols(); k++)
      {
        std::cout<<TrainX[i](j,k)<<"\t";
      }
      std::cout<<std::endl;
    }*/

    std::cout<<std::endl;
    for(size_t j=0; j<(size_t)TrainY[i].GetNrows(); j++)
    {
      for(size_t k=0; k<(size_t)TrainY[i].GetNcols(); k++)
      {
        TrainY[i](j,k) = train_Y[i][j];
      }
    }

    /*for(size_t j=0; j<(size_t)TrainY[i].GetNrows(); j++)
    {
      for(size_t k=0; k<(size_t)TrainY[i].GetNcols(); k++)
      {
        std::cout<<TrainY[i](j,k)<<"\t";
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;*/

    std::cout<<"crating p matrix"  << std::endl;
    testForwardLogReg<TReference<double>>(TrainX[i],p, Weights);

    std::cout<<"Adding Bias matrix to given Matrix"<<std::endl;
    testAddBiases<TReference<double>>(p,Biases);

    std::cout<<std::endl;
    std::cout<<"Given Matrix after Softmax Operation"<<std::endl;
    testSoftmaxAE<TReference<double>>(p);
    std::cout<<std::endl;

    std::cout<<"updating parameters"  << std::endl;
    testUpdateParamsLogReg<TReference<double>>(TrainX[i],TrainY[i], difference,p,
                                               Weights,Biases,learningRate,fBatchSize);


  }}


  std::cout<<"output"<<std::endl;
  for(size_t i=0;i<totalRowsTestOutput;i++)
  {
    for(size_t j=0;j<(size_t)TestX[i].GetNrows();j++)
    {
      for (size_t k=0;k<(size_t)TestX[i].GetNcols();k++)
      {
        TestX[i](j,k)=test_X[i][j];
      }
    }
    std::cout<<"predicting"  << std::endl;
    testForwardLogReg<TReference<double>>(TestX[i],TestY[i], Weights);
    std::cout<<"Adding Bias matrix to given Matrix"<<std::endl;
    testAddBiases<TReference<double>>(TestY[i],Biases);

    std::cout<<std::endl;
    std::cout<<"Given Matrix after Softmax Operation"<<std::endl;
    testSoftmaxAE<TReference<double>>(TestY[i]);
    std::cout<<std::endl;
  }
}
int main()
{
  test();
}
