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
#include "denoiseAE.h"

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
  size_t fBatchSize;
  size_t fInputUnits;
  size_t fOutputUnits;
  Matrix_t fWeights;
  Matrix_t fBiases;
  LogisticRegressionLayer(size_t BatchSize, size_t InputUnits, size_t OutputUnits);
  LogisticRegressionLayer(const LogisticRegressionLayer &);
  size_t GetBatchSize()          const {return fBatchSize;}
  size_t GetInputUnits()         const {return fInputUnits;}
  size_t GetOutputUnits()        const {return fOutputUnits;}
  const Matrix_t & GetWeights() const {return fWeights;}
  Matrix_t & GetWeights() {return fWeights;}
  const Matrix_t & GetBiases() const {return fBiases;}
  Matrix_t & GetBiases() {return fBiases;}
  void TrainLogReg(Matrix_t &input, Matrix_t &output, Double_t learningRate);
  void PredictLogReg(Matrix_t &input, Matrix_t &output);

};


template<Architecture_t>
LogisticRegressionLayer<Architecture_t>::LogisticRegressionLayer(size_t batchSize,
                                                       size_t inputUnits,
                                                       size_t outputUnits):
                                                       fBatchSize(batchSize),
                                                       fInputUnits(inputUnits),
                                                       fOutputUnits(outputUnits),
                                                       fWeights(outputUnits, inputUnits),
                                                       fBiases(outputUnits,1)
{

}
//______________________________________________________________________________
template<Architecture_t>
LogisticRegressionLayer<Architecture_t>::LogisticRegressionLayer(const LogisticRegression &log)
                                    :fBatchSize(log.fBatchSize),
                                    fInputUnits(log.fInputUnits),
                                    fOutputUnits(log.fOutputUnits),
                                    fWeights(log.fOutputUnits,log.fInputUnits),
                                    fBiases(log.fOutputUnits,1)
{
  Architecture_t::Copy(fWeights, layer.GetWeights());
  Architecture_t::Copy(fBiases, layer.GetBiases());
}
//______________________________________________________________________________
template<typename Architecture_t>
LogisticRegressionLayer<Architecture_t>::Initialize(EInitialization m)
-> void
{
   initialize<Architecture_t>(fWeights, EInitialization::kZero);
   initialize<Architecture_t>(fBiases,  EInitialization::kZero);
}

//______________________________________________________________________________

//______________________________________________________________________________

template<typename Architecture_t>
LogisticRegressionLayer<Architecture_t>::TrainLogReg(Matrix_t &input, Matrix_t &output,Double_t learningRate)
-> void
{
  Matrix_t p(fOutputUnits,1);
  Matrix_t difference(fOutputUnits,1);

  Architecture_t::ForwardLogReg(input,p,fWeights);
  Architecture_t::AddBiases(p,fBiases);
  Architecture_t::SoftmaxAE(p);
  Architecture_t::UpdateParamsLogReg(input,output,difference,p,fWeights,fBiases,learningRate,fBatchSize);

}

//______________________________________________________________________________

template<typename Architecture_t>
LogisticRegressionLayer<Architecture_t>::PredictLogReg(Matrix_t &input, Matrix_t &outputLabel,Double_t learningRate)
-> void
{
  Architecture_t::ForwardLogReg(input,output,fWeights);
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
  size_t fBatchSize;
  size_t fInputUnits;
  size_t fOutputUnits;
  Matrix_t fWeights;
  Matrix_t fBiases;
  TransformLayer(size_t BatchSize, size_t InputUnits, size_t OutputUnits);
  TransformLayer(const TransformLayer &);
  void Transform(Matrix_t &input, Matrix_t &fWeights, Matrix_t fBiases);
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
  Architecture_t::Copy(fWeights,trans.GetWeights());
  Architecture_t::Copy(fBiases,trans.GetBiases());
}
//______________________________________________________________________________




//_____________________________________________________________________________

template<typename Architecture_t>
TransformLayer<Architecture_t>::Transform(Matrix_t &input, Matrix_t &transformed)
-> void
{
  Architecture_t::Transform(input,transformed,fWeights,fBiases);
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
  size_t fBatchSize;
  size_t fInputUnits;
  size_t fOutputUnits;
  size_t fNumHiddenLayers;
  std::vector<TDAE> fDae;
  std::vector<TransformLayer> fTransLayer;
  std::vector<LogisticRegressionLayer>fLogReg;
  std::vector<size_t> fNumHiddenUnitsPerLayer;


  TSDAE(size_t fBatchSize, size_t fInputUnits, size_t fOutputUnits,
        size_t fNumHiddenLayers, std::vector<size_t> &fNumHiddenUnitsPerLayer);

  TSDAE(const TSDAE &);
  void Pretrain(Matrix_t &input, Double_t learningRate, Double_t corruptionLevel, size_t epochs);
  void Finetune(Matrix_t &input, Matrix_t &outputLabel, Double_t learningRate, Int_t epochs);
};

//______________________________________________________________________________

template<typename Architecture_t>
TSDAE<Architecture_t>::TSDAE(size_t batchSize, size_t inputUnits,
                             size_t outputUnits, size_t numHiddenLayers,
                             std::vector<size_t> &numHiddenUnitsPerLayer):
                             fBatchSize(batchSize),fInputUnits(inputUnits),
                             fOutputUnits(outputUnits),
                             fNumHiddenLayers(numHiddenLayers),
                             fNumHiddenUnitsPerLayer(numHiddenUnitsPerLayer)
{

std::vector<TDAE>fDae(fNumHiddenLayers);

std::vector<TransformLayer>fTransLayer(fNumHiddenLayers);

std::vector<LogisticRegression>fLogReg;
//constructing multiple layers
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
  fTransLayer[i] = new TransLayer(fBatchSize, inputSize,numHiddenUnitsPerLayer[i]);
  fDae[i] = new TDAE(fBatchSize, inputSize, numHiddenUnitsPerLayer[i]);

}
fLogReg = new LogisticRegression(fBatchSize, fNumHiddenUnitsPerLayer[fNumHiddenLayers -1], fOutputUnits);

}

template<typename Architecture_t>
TSDAE<Architecture_t>::TSDAE(const TSDAE &sdae):
                       fBatchSize(sdae.fBatchSize),
                       fInputUnits(sdae.fInputUnits),
                       fOutputUnits(sdae.fOutputUnits),
                       fNumHiddenLayers(sdae.fNumHiddenLayers),
                       fNumHiddenUnitsPerLayer(sdae.fNumHiddenUnitsPerLayer)

{

}

//______________________________________________________________________________
// with this function, we want to train each and every layer with every input and
// for all the epochs. This will train the denoise layers one after other with
// all inputs.
//______________________________________________________________________________

template<typename Architecture_t>
TSDAE<Architecture_t>::Pretrain(Matrix_t &input, Double_t learningRate, Double_t corruptionLevel, size_t epochs)
-> void
{
  size_t prevLayerSize;
  Matrix_t trainingInput(fInputUnits,1);

  for(size_t i =0; i < fNumHiddenLayers; i++)//for every layer
  {
    for(size_t epoch = 0; epoch<epochs; epoch++)//for every epoch
    {
      for(size_t n=0; n < fBatchSize; n++)//for every example
      {
        for(size_t m =0; m < fInputUnits; m++)//every unit of input
        {
          trainingInput(m,1) = input(,); //getting each training value to pretrain via dae layer, trainX get new input set for every new n
        }

        //input in layer
        for(size_t l=0; l<=i; l++)
        {
          if(l==0)//for first layer
          {
            Matrix_t layerInput(fInputUnits,1);
            for(size_t j=0; j < fInputUnits; j++)
            {
              layerInput[j]=trainingInput[j];
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
              prevLayerInput(j,1)= layerInput(j,1);
            }

            layerInput(fNumHiddenUnitsPerLayer[l-1],1);
            fTransLayer[l-1]->Transform(prevLayerInput,layerInput);
          }

        }
        fDae[i]->train(layerInput, learningRate, corruptionLevel);

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
TSDAE<Architecture_t>::Finetune(Matrix_t &input, Matrix_t &outputLabel, Double_t learningRate, Int_t epochs)
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
        trainingInput(m,1)=input();
      }
      for(size_t m=0; m<fOutputUnits; m++)
      {
        trainingOutputLabel(m,1)=outputLabel();
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
            prevLayerInput(k,1)=layerInput(k,1);
          }

        }
        Matrix_t layerInput(fNumHiddenUnitsPerLayer[i],1);
        fTransLayer[i]->Transform(prevLayerInput,layerInput);


      }
      fLogReg->train(layerInput, trainingOutputLabel, learningRate);
    }
  }

}

//______________________________________________________________________________



}// namespace DAE
}// namespace DNN
}// namespace TMVA
#endif
