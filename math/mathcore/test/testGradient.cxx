// @(#)root/test:$Id$
// Author: Alejandro García Montoro 07/2017

#include "Fit/FitUtil.h"
#include "Fit/BinData.h"
#include "Math/WrappedMultiTF1.h"

#include "TF1.h"
#include "TH1.h"
#include "HFitInterface.h"
#include "Fit/Fitter.h"
#include "Fit/Chi2FCN.h"

#include "TRandom3.h"

#include "gtest/gtest.h"

#include <iostream>
#include <chrono>

// Class to encapsulate the types that define how the gradient test is performed
//    DataType defines how to instantiate the gradient evaluation: Double_t, Double_v.
//    ExecutionPolicyType defines the execution policy: kSerial, kMultithread...
template <typename U, ROOT::Fit::ExecutionPolicy V>
struct GradientTestTraits {
   using DataType = U;
   static constexpr ROOT::Fit::ExecutionPolicy ExecutionPolicyType = V;
};

using ScalarSerial = GradientTestTraits<Double_t, ROOT::Fit::kSerial>;
using ScalarMultithread = GradientTestTraits<Double_t, ROOT::Fit::kMultithread>;
#ifdef R__HAS_VECCORE
using VectorialSerial = GradientTestTraits<ROOT::Double_v, ROOT::Fit::kSerial>;
using VectorialMultithread = GradientTestTraits<ROOT::Double_v, ROOT::Fit::kMultithread>;
#endif

// Model function to test the gradient evaluation
template <class T>
static T modelFunction(const T *data, const double *params)
{
   return params[0] * exp(-(*data + (-130.)) * (*data + (-130.)) / 2) +
          params[1] * exp(-(params[2] * (*data * (0.01)) - params[3] * ((*data) * (0.01)) * ((*data) * (0.01))));
}

// Helper class used to encapsulate the calls to the gradient interfaces, templated with a GradientTestTraits type
template <class T>
struct GradientTestEvaluation {
   // Types to instantiate Chi2FCN: the gradient function interface handles the parameters, so its base typedef
   // has to be always a double; the parametric function interface is templated to test both serial and vectorial
   // cases.
   using GradFunctionType = ROOT::Math::IGradientFunctionMultiDimTempl<double>;
   using BaseFunctionType = ROOT::Math::IParamMultiFunctionTempl<typename T::DataType>;

   GradientTestEvaluation()
   {
      // Create unique names for TF1 and TH1. This prevents possible memory leaks when multiple tests are instantiated.
      std::stringstream streamTF1, streamTH1;

      streamTF1 << "f" << this;
      streamTH1 << "h" << this;

      std::string nameTF1 = streamTF1.str(), nameTH1 = streamTH1.str();

      // Create TF1 from model function
      TF1 *f = new TF1(nameTF1.c_str(), modelFunction<typename T::DataType>, 100, 200, 4);
      f->SetParameters(fParams);

      // Assure the to-be-created histogram does not replace an old one
      auto oldTH1 = gROOT->FindObject(nameTH1.c_str());
      if (oldTH1)
         delete oldTH1;

      // Create TH1 and fill it with values from model function
      TH1D *h1f = new TH1D(nameTH1.c_str(), "Test random numbers", 12801, 100, 200);
      gRandom->SetSeed(1);
      h1f->FillRandom(nameTF1.c_str(), 1000000);

      ROOT::Fit::FillData(fData, h1f, f);
      fNumPoints = fData.NPoints();

      fFitFunction = new ROOT::Math::WrappedMultiTF1Templ<typename T::DataType>(*f, f->GetNdim());

      // Instantiate the Chi2FCN object, responsible for evaluating the gradient.
      fFitter =
         new ROOT::Fit::Chi2FCN<GradFunctionType, BaseFunctionType>(fData, *fFitFunction, T::ExecutionPolicyType);
   }

   Double_t BenchmarkSolution(Double_t *solution)
   {
      std::chrono::time_point<std::chrono::system_clock> start, end;

      start = std::chrono::system_clock::now();
      for (int i = 0; i < fNumRepetitions; i++)
         fFitter->Gradient(fParams, solution);
      end = std::chrono::system_clock::now();

      std::chrono::duration<Double_t> timeElapsed = end - start;

      return timeElapsed.count() / fNumRepetitions;
   }

   static const int fNumParams = 4;
   static const int fNumRepetitions = 1000;

   const Double_t fParams[fNumParams] = {1, 1000, 7.5, 1.5};
   unsigned int fNumPoints;

   ROOT::Math::WrappedMultiTF1Templ<typename T::DataType> *fFitFunction;
   ROOT::Fit::BinData fData;
   ROOT::Fit::Chi2FCN<GradFunctionType, BaseFunctionType> *fFitter;
};

// Test class: creates a reference solution (computing the gradient with scalar values in a serial scenario), and
// compares its values and its performance against the evaluation of the gradient specified by the GradientTestTraits
// type.
template <class T>
class GradientTest : public ::testing::Test, public GradientTestEvaluation<T> {
protected:
   virtual void SetUp()
   {
      GradientTestEvaluation<ScalarSerial> reference;
      fReferenceSolution.resize(this->fNumParams);
      fReferenceTime = reference.BenchmarkSolution(fReferenceSolution.data());
   }

   Double_t fReferenceTime;
   std::vector<Double_t> fReferenceSolution;
};

// Types used by Google Test to instantiate the tests.
#ifdef R__HAS_VECCORE
typedef ::testing::Types<ScalarMultithread, VectorialSerial, VectorialMultithread> TestTypes;
#else
typedef ::testing::Types<ScalarMultithread> TestTypes;
#endif

TYPED_TEST_CASE(GradientTest, TestTypes);

// Test EvalChi2Gradient and outputs its speedup agains the scalar serial case.
TYPED_TEST(GradientTest, EvalChi2Gradient)
{
   Double_t solution[TestFixture::fNumParams];

   Double_t benchmarkTime = TestFixture::BenchmarkSolution(solution);

   std::cout << std::fixed << std::setprecision(4);
   std::cout << "Speed-up with respect to scalar serial case: " << TestFixture::fReferenceTime / benchmarkTime;
   std::cout << std::endl;

   for (unsigned i = 0; i < TestFixture::fNumParams; i++)
      EXPECT_NEAR(solution[i], TestFixture::fReferenceSolution[i], 1e-6);
}
