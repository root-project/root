// @(#)root/test:$Id$
// Author: Alejandro García Montoro 07/2017

#include "TMath.h"
#include "Fit/FitUtil.h"
#include "Fit/BinData.h"
#include "Math/WrappedMultiTF1.h"

#include "TF2.h"
#include "TH2D.h"
#include "HFitInterface.h"
#include "Fit/Fitter.h"
#include "Fit/Chi2FCN.h"

#include "Math/Random.h"
#include "TRandom3.h"
#include "TStopwatch.h"

#include "gtest/gtest.h"

#include <iostream>
#include <random>

#ifdef R__HAS_VECCORE

#include <VecCore/VecCore>

using namespace ROOT::Math;
using namespace ROOT::Fit::FitUtil;

using ROOT::Double_v;
using WrappedMultiTF1Vec = WrappedMultiTF1Templ<Double_v>;
using BaseGradFunc = IMultiGradFunction;

template <class T>
T func(const T *data, const double *params)
{
   return params[0] * exp(-(*data + (-130.)) * (*data + (-130.)) / 2) +
          params[1] * exp(-(params[2] * (*data * (0.01)) - params[3] * ((*data) * (0.01)) * ((*data) * (0.01))));
}

const double parameters[4] = {1, 1000, 7.5, 1.5};

int numTimes = 1000;

TEST(TemplatedGradient, EvalChi2Gradient_ScalarSerial)
{
   TF1 *f = new TF1("fvCore", func<double>, 100, 200, 4);
   f->SetParameters(parameters);

   TH1D *h1f = new TH1D("h1f", "Test random numbers", 12801, 100, 200);
   gRandom->SetSeed(1);
   h1f->FillRandom("fvCore", 1000000);

   ROOT::Fit::BinData data;
   ROOT::Fit::FillData(data, h1f, f);

   WrappedMultiTF1 fitFunction(*f, f->GetNdim());
   double grad[4];
   unsigned nPoints = data.NPoints();
   const unsigned int executionPolicy = ROOT::Fit::kSerial;

   TStopwatch watch;

   watch.Start();
   for (int i = 0; i < numTimes; i++)
      Evaluate<double>::EvalChi2Gradient(fitFunction, data, parameters, grad, nPoints, executionPolicy);
   watch.Stop();

   std::cout << "Scalar serial: " << std::endl;
   for (unsigned i = 0; i < 4; i++)
      std::cout << grad[i] << ", ";
   std::cout << std::endl;

   std::cout << "Scalar serial - Time: " << watch.RealTime() / numTimes << std::endl;
}

TEST(TemplatedGradient, EvalChi2Gradient_ScalarMultithread)
{
   TF1 *f = new TF1("fvCore", func<double>, 100, 200, 4);
   f->SetParameters(parameters);

   TH1D *h1f = new TH1D("h1f", "Test random numbers", 12801, 100, 200);
   gRandom->SetSeed(1);
   h1f->FillRandom("fvCore", 1000000);

   ROOT::Fit::BinData data;
   ROOT::Fit::FillData(data, h1f, f);

   WrappedMultiTF1 fitFunction(*f, f->GetNdim());
   double grad[4];
   unsigned nPoints = data.NPoints();
   const unsigned int executionPolicy = ROOT::Fit::kMultithread;

   TStopwatch watch;

   watch.Start();
   for (int i = 0; i < numTimes; i++)
      Evaluate<double>::EvalChi2Gradient(fitFunction, data, parameters, grad, nPoints, executionPolicy);
   watch.Stop();

   std::cout << "Scalar multithread: " << std::endl;
   for (unsigned i = 0; i < 4; i++)
      std::cout << grad[i] << ", ";
   std::cout << std::endl;

   std::cout << "Scalar multithread - Time: " << watch.RealTime() / numTimes << std::endl;
}

TEST(TemplatedGradient, EvalChi2Gradient_VectorizedSerial)
{
   TF1 *fVec = new TF1("fvCoreVec", func<Double_v>, 100, 200, 4);
   fVec->SetParameters(parameters);

   TH1D *h1fVec = new TH1D("h1fVec", "Test random numbers", 12801, 100, 200);
   gRandom->SetSeed(1);
   h1fVec->FillRandom("fvCoreVec", 1000000);

   WrappedMultiTF1Templ<Double_v> fitFunctionVec(*fVec, fVec->GetNdim());

   ROOT::Fit::BinData data;
   ROOT::Fit::FillData(data, h1fVec, fVec);

   double grad[4];
   unsigned nPoints = data.NPoints();
   const unsigned int executionPolicy = ROOT::Fit::kSerial;

   TStopwatch watch;

   watch.Start();
   for (int i = 0; i < numTimes; i++)
      Evaluate<Double_v>::EvalChi2Gradient(fitFunctionVec, data, parameters, grad, nPoints, executionPolicy);
   watch.Stop();

   std::cout << "Vector serial: " << std::endl;
   for (unsigned i = 0; i < 4; i++)
      std::cout << grad[i] << ", ";
   std::cout << std::endl;

   std::cout << "Vector serial - Time: " << watch.RealTime() / numTimes << std::endl;
}

TEST(TemplatedGradient, EvalChi2Gradient_VectorizedMultithread)
{
   TF1 *fVec = new TF1("fvCoreVec", func<Double_v>, 100, 200, 4);
   fVec->SetParameters(parameters);

   TH1D *h1fVec = new TH1D("h1fVec", "Test random numbers", 12801, 100, 200);
   gRandom->SetSeed(1);
   h1fVec->FillRandom("fvCoreVec", 1000000);

   WrappedMultiTF1Templ<Double_v> fitFunctionVec(*fVec, fVec->GetNdim());

   ROOT::Fit::BinData data;
   ROOT::Fit::FillData(data, h1fVec, fVec);

   double grad[4];
   unsigned nPoints = data.NPoints();
   const unsigned int executionPolicy = ROOT::Fit::kMultithread;

   TStopwatch watch;

   watch.Start();
   for (int i = 0; i < numTimes; i++)
      Evaluate<Double_v>::EvalChi2Gradient(fitFunctionVec, data, parameters, grad, nPoints, executionPolicy);
   watch.Stop();

   std::cout << "Vector multithread: " << std::endl;
   for (unsigned i = 0; i < 4; i++)
      std::cout << grad[i] << ", ";
   std::cout << std::endl;

   std::cout << "Vector multithread - Time: " << watch.RealTime() / numTimes << std::endl;
}

TEST(TemplatedGradient, EvalChi2Gradient_Comparison)
{
   // Scalar serial
   TF1 *f = new TF1("fvCore", func<double>, 100, 200, 4);
   f->SetParameters(parameters);

   TH1D *h1f = new TH1D("h1f", "Test random numbers", 12801, 100, 200);
   gRandom->SetSeed(1);
   h1f->FillRandom("fvCore", 1000000);

   ROOT::Fit::BinData data;
   ROOT::Fit::FillData(data, h1f, f);

   WrappedMultiTF1 fitFunction(*f, f->GetNdim());
   double grad[4];
   unsigned nPoints = data.NPoints();
   const unsigned int executionPolicy = ROOT::Fit::kMultithread;

   Evaluate<double>::EvalChi2Gradient(fitFunction, data, parameters, grad, nPoints, executionPolicy);

   std::cout << "Serial: " << std::endl;
   for (unsigned i = 0; i < 4; i++)
      std::cout << grad[i] << ", ";
   std::cout << std::endl;

   // Vectorized multithreaded
   TF1 *fVec = new TF1("fvCoreVec", func<Double_v>, 100, 200, 4);
   fVec->SetParameters(parameters);

   TH1D *h1fVec = new TH1D("h1fVec", "Test random numbers", 12801, 100, 200);
   gRandom->SetSeed(1);
   h1fVec->FillRandom("fvCoreVec", 1000000);

   WrappedMultiTF1Templ<Double_v> fitFunctionVec(*fVec, fVec->GetNdim());

   ROOT::Fit::BinData dataVec;
   ROOT::Fit::FillData(dataVec, h1fVec, fVec);

   double gradVec[4];
   unsigned nPointsVec = dataVec.NPoints();
   const unsigned int executionPolicyVec = ROOT::Fit::kSerial;

   Evaluate<Double_v>::EvalChi2Gradient(fitFunctionVec, dataVec, parameters, gradVec, nPointsVec, executionPolicyVec);

   std::cout << "Vectorized: " << std::endl;
   for (unsigned i = 0; i < 4; i++)
      std::cout << gradVec[i] << ", ";
   std::cout << std::endl;
}

#endif
