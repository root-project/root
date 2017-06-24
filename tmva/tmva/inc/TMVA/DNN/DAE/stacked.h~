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
template<typename Architecture_t>
class HiddenLayer
{
public:
  size_t fBatchsize;
  size_t fInputUnits;
  size_t fOutputUnits;
  Matrix_t fWeights;
  Matrix_t fBiases;
  HiddenLayer(size_t Batchsize, size_t InputUnits, size_t OutputUnits);
  HiddenLayer(const HiddenLayer &);
  void Output(Matrix_t &input, Matrix_t &fWeights, Matrix_t fBiases);
  size_t GetBatchSize()          const {return fBatchSize;}
  size_t GetInputUnits()         const {return fInputUnits;}
  size_t GetOutputUnits()        const {return fOutputUnits;}
  const Matrix_t & GetWeights() const {return fWeights;}
  Matrix_t & GetWeights() {return fWeights;}
  const Matrix_t & GetBiases() const {return fBiases;}
  Matrix_t & GetBiases() {return fBiases;}
};//Hidden Layer

template<typename Architecture_t>
HiddenLayer<Architecture_t>::HiddenLayer(size_t batchSize,
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
HiddenLayer<Architecture_t>::HiddenLayer(const HiddenLayer &hid):
                                        fBatchSize(hid.fBatchSize),
                                        fInputUnits(hid.fInputUnits),
                                        fOutputUnits(hid.fOutputUnits),
                                        fWeights(hid.fOutputUnits,hid.fInputUnits),
                                        fBiases(hid.fOutputUnits,1)
{
  Architecture_t::Copy(fWeights,hid.GetWeights());
  Architecture_t::Copy(fBiases,hid.GetBiases());
}
//initialize weights uniformly
//______________________________________________________________________________

template<typename Architecture_t>
HiddenLayer<Architecture_t>::Initialize(EInitialization m)
-> void
{
   initialize<Architecture_t>(fWeights, m);
   initialize<Architecture_t>(fBiases,  EInitialization::kZero);
}


//_____________________________________________________________________________

template<typename Architecture_t>
HiddenLayer<Architecture_t>::Output(Matrix_t &input, Matrix_t &fWeights, Matrix_t fBiases)
-> void
{

}


//______________________________________________________________________________
template<typename Architecture_t>
class TSDAE
{
public:
  size_t fBatchsize;
  size_t fInputUnits;
  size_t fOutputUnits;
  size_t fNumHiddenLayers;

  std::vector<TDAE> fDae;
  std::vector<LogisticRegression>fLogReg;
  std::vector<size_t> fNumHiddenUnitsPerLayer;


  TSDAE(size_t fBatchsize, size_t fInputUnits, size_t fOutputUnits,
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
                             fBatchsize(batchSize),fInputUnits(inputUnits),
                             fOutputUnits(outputUnits),
                             fNumHiddenLayers(numHiddenLayers),
                             fNumHiddenUnitsPerLayer(numHiddenUnitsPerLayer)
{

std::vector<TDAE>fDae(fNumHiddenLayers);
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

  //construct a sigmoid layer and a denoise layer
  fSigmoidLayer[i] = new HiddenLayer(fBatchsize, inputSize,numHiddenUnitsPerLayer[i]);
  fDae[i] = new TDAE(fBatchsize, inputSize, numHiddenUnitsPerLayer[i]);

}
fLogReg = new LogisticRegression(fBatchsize, fNumHiddenUnitsPerLayer[fNumHiddenLayers -1], fOutputUnits);

}

template<typename Architecture_t>
TSDAE<Architecture_t>::TSDAE(const TSDAE &sdae):
                       fBatchsize(sdae.fBatchsize),
                       fInputUnits(sdae.fInputUnits),
                       fOutputUnits(sdae.fOutputUnits),
                       fNumHiddenLayers(sdae.fNumHiddenLayers),
                       fNumHiddenUnitsPerLayer(sdae.fNumHiddenUnitsPerLayer)

{

}

//______________________________________________________________________________
//with this function, we want to train each and every layer with every input and
//for all the epochs. This will train the denoise layers one after other with all inputs.

template<typename Architecture_t>
TSDAE<Architecture_t>::Pretrain(Matrix_t &input, Double_t learningRate, Double_t corruptionLevel, size_t epochs)
-> void
{
  Matrix_t layerInput;
  Matrix_t prevLayerInput;
  size_t prevLayerSize;

  Matrix_t trainX(fInputUnits,1);

  for(size_t i =0; i < fNumHiddenLayers; i++)//for every layer
  {
    for(size_t epoch = 0; epoch<epochs; epoch++)//for every epoch
    {
      for(size_t n=0; n < fBatchsize; n++)//for every example
      {
        for(size_t m =0; m<fInputUnits; m++)//every unit of input
        {
          trainX(m,1) = input((n*fInputUnits+m),1);//getting each training value to pretrain via dae layer, trainX get new input set for every new n
        }

        //input in layer
        for(size_t l=0; l<=i; l++)
        {
          if(l==0)//for first layer
          {
            for(size_t j=0; j<fInputUnits; j++)
            {
              layerInput[j]=trainX[j];
            }
          }
          else
          {
            if(l==1)
            {
              prevLayerSize = fInputUnits;
            }
            else
            {
              prevLayerSize = fNumHiddenUnitsPerLayer[l-2];

            }

            prevLayerInput(prevLayerSize,1);
            for(size_t j = 0; j<prevLayerSize; j++)
            {
              prevLayerInput(j,1)= layerInput(j,1);
            }
            delete layerInput;
            layerInput(fNumHiddenUnitsPerLayer[l-1]);
          }

        }
        fDae[i]->train(layerInput, learningRate, corruptionLevel);

      }
    }
  }
}

//______________________________________________________________________________

template<typename Architecture_t>
TSDAE<Architecture_t>::Finetune(Matrix_t &input, Matrix_t &outputLabel, Double_t learningRate, Int_t epochs)
{
  Matrix_t layerInput;
  Matrix_t prevLayerInput;
  size_t prevLayerSize;

  Matrix_t trainX(fInputUnits,1);
  Matrix_t trainY(fOutputUnits,1);

  for(size_t epoch=0; epoch<epochs; epoch++)
  {
    for(size_t n=0; n<fBatchsize; n++)
    {


      for(size_t m=0; m<fInputUnits; m++)
      {
        trainX(m,1)=input(n*fInputUnits+m);
      }
      for(size_t m=0; m<fOutputUnits; m++)
      {
        trainY(m,1)=outputLabel(n*fOutputUnits+m);
      }



      //input in layer
      for(size_t i=0; i<fNumHiddenLayers; i++)
      {
        if(i==0)
        {
          Matrix_t prevLayerInput(fInputUnits,1);
          for(size_t j=0; j<fInputUnits;j++)
          {
            prevLayerInput(j,1)=trainX(j,1);
          }
        }
        else
        {
          Matrix_t prevLayerInput(fNumHiddenUnitsPerLayer[i-1],1);
          for(size_t i=0; j<fNumHiddenUnitsPerLayer[i-1];j++)
          {
            prevLayerInput(j,1)=layerInput(j,1);
          }
          delete layerInput;
        }
        Matrix_t layerInput(fNumHiddenUnitsPerLayer[i],1);
        //Sigmoid

        delete prevLayerInput;


      }
      fLogReg->train(layerInput, trainY, learningRate);
    }
  }
  delete layerInput;
  delete trainX;
  delete trainY;
}

//______________________________________________________________________________



}// namespace DAE
}// namespace DNN
}// namespace TMVA
