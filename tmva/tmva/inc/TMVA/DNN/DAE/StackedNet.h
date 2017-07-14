// @(#)root/tmva/tmva/dnn/dae:$Id$
// Author: Akshay Vashistha(ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd                                            *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////////
// Contains the Layers required for stacking the denoise layer.        //
//                                                                     //
/////////////////////////////////////////////////////////////////////////


#ifndef TMVA_SDAE_H
#define TMVA_SDAE_H

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"
#include "DenoiseAE.h"
#include "AE.h"

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>

namespace TMVA
{
namespace DNN
{
namespace DAE
{

//______________________________________________________________________________
// The logistic Regression layer for supervised training in the finetune step.
//
//______________________________________________________________________________

template<typename Architecture_t>
class LogisticRegressionLayer
{

public:
  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;

  size_t fBatchSize;
  size_t fInputUnits;
  size_t fOutputUnits;
  Matrix_t fWeights;
  Matrix_t fBiases;

  // constructor
  LogisticRegressionLayer(size_t BatchSize, size_t InputUnits, size_t OutputUnits);

  //copy constructor
  LogisticRegressionLayer(const LogisticRegressionLayer &);

  //Some Getter functions
  size_t GetBatchSize()          const {return fBatchSize;}
  size_t GetInputUnits()         const {return fInputUnits;}
  size_t GetOutputUnits()        const {return fOutputUnits;}
  const Matrix_t & GetWeights() const {return fWeights;}
  Matrix_t & GetWeights() {return fWeights;}
  const Matrix_t & GetBiases() const {return fBiases;}
  Matrix_t & GetBiases() {return fBiases;}

  // Initialize the Weights and biases
  void Initialize();

  // Train the Logistic Regression Layer
  void TrainLogReg(Matrix_t &input, Matrix_t &output, Double_t learningRate);

  // Predict output of Logistic Regression Layer, should be used as a
  // successive call  after TrainLogReg()
  void PredictLogReg(Matrix_t &input, Matrix_t &outputLabel, Double_t learningRate);

};


template<typename Architecture_t>
LogisticRegressionLayer<Architecture_t>::LogisticRegressionLayer(size_t batchSize,
                                                       size_t inputUnits,
                                                       size_t outputUnits)
                                                       :fBatchSize(batchSize),
                                                       fInputUnits(inputUnits),
                                                       fOutputUnits(outputUnits),
                                                       fWeights(outputUnits, inputUnits),
                                                       fBiases(outputUnits,1)
{

}
//______________________________________________________________________________
template<typename Architecture_t>
LogisticRegressionLayer<Architecture_t>::LogisticRegressionLayer(const LogisticRegressionLayer &logistic)
                                    :fBatchSize(logistic.fBatchSize),
                                    fInputUnits(logistic.fInputUnits),
                                    fOutputUnits(logistic.fOutputUnits),
                                    fWeights(logistic.fOutputUnits,logistic.fInputUnits),
                                    fBiases(logistic.fOutputUnits,1)
{
  Architecture_t::Copy(fWeights, logistic.GetWeights());
  Architecture_t::Copy(fBiases, logistic.GetBiases());
}
//______________________________________________________________________________
// Initialize Weights and Biases Matrices to zero

template<typename Architecture_t>
auto LogisticRegressionLayer<Architecture_t>::Initialize()
-> void
{
   DNN::initialize<Architecture_t>(fWeights, DNN::EInitialization::kZero);
   DNN::initialize<Architecture_t>(fBiases, DNN::EInitialization::kZero);
}

//______________________________________________________________________________

//______________________________________________________________________________
template<typename Architecture_t>
auto LogisticRegressionLayer<Architecture_t>::TrainLogReg(Matrix_t &input,
                                                          Matrix_t &output,
                                                          Double_t learningRate)
-> void
{
  Matrix_t p(this->GetOutputUnits(),1);
  Matrix_t difference(this->GetOutputUnits(),1);

  Architecture_t::ForwardLogReg(input,p,this->GetWeights());
  Architecture_t::AddBiases(p,this->GetBiases());
  Architecture_t::SoftmaxAE(p);
  Architecture_t::UpdateParamsLogReg(input,output,difference,p,
                                     this->GetWeights(),this->GetBiases(),
                                     learningRate,this->GetBatchSize());

}

//______________________________________________________________________________

template<typename Architecture_t>
auto LogisticRegressionLayer<Architecture_t>::PredictLogReg(Matrix_t &input,
                                                            Matrix_t &output,
                                                            Double_t learningRate)
-> void
{
  Architecture_t::ForwardLogReg(input,output,this->GetWeights());
  Architecture_t::SoftmaxAE(output);
}



//______________________________________________________________________________
// The main function of this layer is to pass the hidden units of denoise layer
// as an input to next denoise layer or finetune layer.
// The weights and biases will be shared i.e weights and biases will be
// same as in the denoise layer.
//______________________________________________________________________________





template<typename Architecture_t>
class TransformLayer
{
public:
  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;

  size_t fBatchSize;
  size_t fInputUnits;
  size_t fOutputUnits;
  Matrix_t fWeights;
  Matrix_t fBiases;

  // constructor
  TransformLayer(size_t BatchSize, size_t InputUnits, size_t OutputUnits);

  // copy constructor
  TransformLayer(const TransformLayer &);

  // Transform the matrix from Input to Transformed.
  // This transformed matrix is same as the hidden layer we get in TDAE class.
  // This will be the input to next layer in stacking steps.
  void Transform(Matrix_t &input, Matrix_t &transformed);

  // Initialize the Weights and Bias Matrices of this layer with same weight
  // and bias as in the previously trained TDAE layer.
  void Initialize(Matrix_t &, Matrix_t &);

  //some Getter Functions
  size_t GetBatchSize()                       const {return fBatchSize;}
  size_t GetInputUnits()                      const {return fInputUnits;}
  size_t GetOutputUnits()                     const {return fOutputUnits;}
  const Matrix_t & GetWeights()               const {return fWeights;}
  Matrix_t & GetWeights()                           {return fWeights;}
  const Matrix_t & GetBiases()                const {return fBiases;}
  Matrix_t & GetBiases()                            {return fBiases;}
};

template<typename Architecture_t>
TransformLayer<Architecture_t>::TransformLayer(size_t batchSize,
                                         size_t inputUnits,
                                         size_t outputUnits):
                                         fBatchSize(batchSize),
                                         fInputUnits(inputUnits),
                                         fOutputUnits(outputUnits),
                                         fWeights(outputUnits,inputUnits),
                                         fBiases(outputUnits,1)

{

}
//______________________________________________________________________________

template<typename Architecture_t>
TransformLayer<Architecture_t>::TransformLayer(const TransformLayer &trans):
                                        fBatchSize(trans.fBatchSize),
                                        fInputUnits(trans.fInputUnits),
                                        fOutputUnits(trans.fOutputUnits),
                                        fWeights(trans.fOutputUnits,trans.fInputUnits),
                                        fBiases(trans.fOutputUnits,1)
{

}
//______________________________________________________________________________
template<typename Architecture_t>
auto TransformLayer<Architecture_t>::Initialize(Matrix_t &Weights, Matrix_t &Biases)
-> void
{
   Architecture_t::Copy(fWeights, Weights);
   Architecture_t::Copy(fBiases, Biases);
   /*std::cout<<"In transformed Layer"<<std::endl;
   for(size_t i=0;i<(size_t)fWeights.GetNrows();i++)
   {
     for(size_t j=0;j<(size_t)fWeights.GetNcols();j++)
     {
       std::cout<<fWeights(i,j)<<"\t";
     }
     std::cout<<std::endl;
   }*/
}



//_____________________________________________________________________________

template<typename Architecture_t>
auto TransformLayer<Architecture_t>::Transform(Matrix_t &input, Matrix_t &transformed)
-> void
{
  Architecture_t::Transform(input,transformed,this->GetWeights(),this->GetBiases());
  Architecture_t::Sigmoid(transformed);
}


//______________________________________________________________________________
// The SDAE layer, this layer comprises of finetune and pretraining steps.
// This layer stacks the denoise layer and predicts the output according to the
// output labels in finetune layer.
//______________________________________________________________________________


template<typename Architecture_t>
class TSDAE
{
public:
  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;

  size_t fBatchSize;
  size_t fInputUnits;
  size_t fOutputUnits;
  size_t fNumHiddenLayers;

  // Creating vector of TDAE class type layers.
  std::vector<TDAE<Architecture_t>> fDae;

  // Creating vector of TransformLayer class type layers.
  std::vector<TransformLayer<Architecture_t>> fTransLayer;

  // Creating vector of LogisticRegressionLayer class type layer.
  std::vector<LogisticRegressionLayer<Architecture_t>>fLogReg;

  // Creating vector that contain number of hidden units in a Stacked network.
  std::vector<size_t> fNumHiddenUnitsPerLayer;

  //Some Getter functions
  size_t GetBatchSize()                        {return fBatchSize;}
  size_t GetInputUnits()                       {return fInputUnits;}
  size_t GetOutputUnits()                      {return fOutputUnits;}
  size_t GetNumHiddenLayers()                  {return fNumHiddenLayers;}



  TSDAE(size_t fBatchSize, size_t fInputUnits, size_t fOutputUnits,
        size_t fNumHiddenLayers, std::vector<size_t> fNumHiddenUnitsPerLayer);

  // To train the network with required number of TDAE class type layers.
  void Pretrain(Matrix_t &input, Double_t learningRate, Double_t corruptionLevel, size_t epochs);

  // To train the Layer with supervised step.
  void Finetune(Matrix_t &input, Matrix_t &outputLabel, Double_t learningRate, size_t epochs);

  void Predict(Matrix_t &input, Matrix_t &output);
};

//______________________________________________________________________________
// Creating the required number of Denoise, transformed and logistic Regression
// layer as soon as the constructor invokes.
// Weights and biases are Initialized for every layer.
// Same weights are passed on to transform layer.
//______________________________________________________________________________
template<typename Architecture_t>
TSDAE<Architecture_t>::TSDAE(size_t batchSize, size_t inputUnits,
                             size_t outputUnits, size_t numHiddenLayers,
                             std::vector<size_t> numHiddenUnitsPerLayer):
                             fBatchSize(batchSize),fInputUnits(inputUnits),
                             fOutputUnits(outputUnits),
                             fNumHiddenLayers(numHiddenLayers),
                             fNumHiddenUnitsPerLayer(numHiddenUnitsPerLayer)
{
  size_t inputSize;
  for(size_t i=0; i<fNumHiddenLayers;i++)
  {
    if(i==0)
    {
      inputSize = fInputUnits;
    }
    else
    {
      inputSize = fNumHiddenUnitsPerLayer[i-1];
    }

    //construct a transform layer and a denoise layer
    fTransLayer.emplace_back(TransformLayer<Architecture_t>(fBatchSize,
                                                    inputSize,
                                                    fNumHiddenUnitsPerLayer[i]));
    fDae.emplace_back(TDAE<Architecture_t>(fBatchSize,
                                           inputSize,
                                           fNumHiddenUnitsPerLayer[i]));
    // Initializing the weights and Biases for every TDAE class
    fDae[i].Initialize(DNN::EInitialization::kUniform);

    fTransLayer[i].Initialize(fDae[i].GetWeights(),fDae[i].GetWeights());

  }
  fLogReg.emplace_back(LogisticRegressionLayer<Architecture_t>(fBatchSize,
                                    fNumHiddenUnitsPerLayer[fNumHiddenLayers -1],
                                    fOutputUnits));
  fLogReg[0].Initialize();

  //  std::cout<<"Size of fdae "<<fDae.size()<<std::endl;
  //  std::cout<<"Size of trans "<<fTransLayer.size()<<std::endl;
  //  std::cout<<"Size of log reg "<<fLogReg.size()<<std::endl;
}


//______________________________________________________________________________
// with this function, we want to train each and every layer with every input and
// for all the epochs. This will train the denoise layers one after other with
// all inputs.
//______________________________________________________________________________

template<typename Architecture_t>
auto TSDAE<Architecture_t>::Pretrain(Matrix_t &input,
                                     Double_t learningRate,
                                     Double_t corruptionLevel,
                                     size_t epochs)
-> void
{
  size_t prevLayerSize;
  Matrix_t trainingInput(fInputUnits,1);
  Matrix_t layerInput;
  Matrix_t prevLayerInput;

  for(size_t i =0; i < fNumHiddenLayers; i++)//for every layer
  {
    for(size_t epoch = 0; epoch<epochs; epoch++)//for every epoch
    {
      for(size_t n=0; n < fBatchSize; n++)//for every example
      {
        for(size_t m =0; m < fInputUnits; m++)//every unit of input
        {
          trainingInput(m,1) = input(m,1); //getting each training value to pretrain via dae layer, trainX get new input set for every new n
        }

        //input in layer
        for(size_t l=0; l<=i; l++)
        {
          if(l==0)//for first layer
          {
            Matrix_t layerInput(fInputUnits,1);
            for(size_t j=0; j < fInputUnits; j++)
            {
              layerInput(j,1)=trainingInput(j,1);
            }
          }
          else
          {
            if(l == 1)
            {
              prevLayerSize = fInputUnits;
            }
            else
            {
              prevLayerSize = fNumHiddenUnitsPerLayer[l-2];

            }

            Matrix_t prevLayerInput(prevLayerSize,1);
            for(size_t j = 0; j<prevLayerSize; j++)
            {
            //  Matrix_t layerInput;//**************************************************************************
              prevLayerInput(j,1)= layerInput(j,1);
            }
            Matrix_t layerInput(fNumHiddenUnitsPerLayer[l-1],1);
            //Matrix_t prevLayerInput;//**********************************************************************
            //Matrix_t layerInput(fNumHiddenUnitsPerLayer[l-1],1);//************************************************
            fTransLayer[l-1]->Transform(prevLayerInput,layerInput);

          }

        }
        //Matrix_t layerInput;//*******************************************************************************
        fDae[i]->TrainLayer(layerInput, learningRate, corruptionLevel);
      }
    }
  }
}

//______________________________________________________________________________
// This is the supervised step of training. It takes Output Labels for training
// and updating weights and biases. This layer takes input from last denoise
// layer of network and passes it to the Transform layer. The output of this
// transform layer is passed as an input to Logistic Regression Layer for
// supervised traing of network.
//______________________________________________________________________________
template<typename Architecture_t>
auto TSDAE<Architecture_t>::Finetune(Matrix_t &input, Matrix_t &outputLabel, Double_t learningRate, size_t epochs)
-> void
{

  size_t prevLayerSize;
  Matrix_t trainingInput(fInputUnits,1);
  Matrix_t trainingOutputLabel(fOutputUnits,1);

  for(size_t epoch=0; epoch<epochs; epoch++)
  {
    for(size_t n=0; n<fBatchSize; n++)
    {
      for(size_t m=0; m<fInputUnits; m++)
      {
        trainingInput(m,1)=input(m,1);
      }
      for(size_t m=0; m<fOutputUnits; m++)
      {
        trainingOutputLabel(m,1)=outputLabel(m,1);
      }

      //input in layer
      for(size_t i=0; i<fNumHiddenLayers; i++)
      {
        if(i==0)
        {
          Matrix_t prevLayerInput(fInputUnits,1);
          for(size_t j=0; j<fInputUnits;j++)
          {
            prevLayerInput(j,1)=trainingInput(j,1);
          }
        }
        else
        {
          Matrix_t prevLayerInput(fNumHiddenUnitsPerLayer[i-1],1);
          for(size_t k=0; k<fNumHiddenUnitsPerLayer[i-1];k++)
          {
            Matrix_t layerInput;//*****************************************************************************************
            prevLayerInput(k,1)=layerInput(k,1);
          }

        }
        Matrix_t prevLayerInput;//******************************************************************************************
        Matrix_t layerInput(fNumHiddenUnitsPerLayer[i],1);
        fTransLayer[i]->Transform(prevLayerInput,layerInput);


      }
      Matrix_t layerInput;//**********************************************************************************************
      fLogReg->TrainLogReg(layerInput, trainingOutputLabel, learningRate);
    }
  }

}

//______________________________________________________________________________
template<typename Architecture_t>
auto TSDAE<Architecture_t>::Predict(Matrix_t &input, Matrix_t &output)
-> void
{

  Matrix_t prevLayerInput(fInputUnits,1);
  for(size_t j=0;j<fInputUnits;j++)
  {
    prevLayerInput(j,1)=input(j,1);
  }

  for (size_t i = 0; i<fNumHiddenLayers; i++)
  {

    Matrix_t layerInput(fTransLayer[i]->fOutputUnits,1);
    for(size_t k=0; k<fTransLayer[i]->fOutputUnits;k++)
    {
      output=0.0;
      for(size_t j=0; j<fTransLayer[i]->fInputUnits;j++)
      {
        output+= fTransLayer[i]->fWeights(k,j) * prevLayerInput(j,1);

      }
      output += fTransLayer[i]->fBiases(k,1);
      layerInput(k,1) = output;
    }
    Architecture_t::Sigmoid(layerInput);
    if(i<fNumHiddenLayers-1)
    {
      Matrix_t prevLayerInput(fTransLayer[i]->fOutputUnits,1);
    }
  }

  for(size_t i=0; i<fLogReg->fOutputUnits;i++)
  {
    output(i,1) = 0;
    for(size_t j=0;j< fLogReg->fInputUnits;j++)
    {
      Matrix_t layerInput;//**********************************************************************************************************
      output(i,1)+=fLogReg->fWeights(i,j) * layerInput(j,1);
    }
    output(i,1) += fLogReg->fBiases(i,1);
  }

  Architecture_t::SoftmaxAE(output);
}


}// namespace DAE
}// namespace DNN
}// namespace TMVA
#endif
