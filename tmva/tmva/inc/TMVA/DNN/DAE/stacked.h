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



}// namespace DAE
}// namespace DNN
}// namespace TMVA
