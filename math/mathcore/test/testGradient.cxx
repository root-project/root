// @(#)root/test:$Id$
// Author: Alejandro García Montoro 07/2017

#include "Fit/BinData.h"
#include "Fit/FitExecutionPolicy.h"
#include "Fit/FitUtil.h"
#include "Fit/UnBinData.h"
#include "Math/WrappedMultiTF1.h"

#include "Fit/Chi2FCN.h"
#include "Fit/Fitter.h"
#include "Fit/LogLikelihoodFCN.h"
#include "Fit/PoissonLikelihoodFCN.h"
#include "HFitInterface.h"
#include "TF1.h"
#include "TH1.h"

#include "TRandom3.h"

#include "gtest/gtest.h"

#include <chrono>
#include <iostream>
#include <string>

// Class to encapsulate the types that define how the gradient test is
// performed; it also stores information strings about the types.
//    DataType defines how to instantiate the gradient evaluation: Double_t,
//    Double_v.
//    ExecutionPolicyType defines the execution policy: kSerial, kMultithread...
//    DataInfoStr points to a human-readable string describing DataType (e.g.,
//    "Scalar", "Vectorial")
//    PolicyInfoStr points to a human-readable string describing
//    ExecutionPolicyType (e.g., "Serial", "Multithread")
template <typename U, ROOT::Fit::ExecutionPolicy V, const char *dataInfoStr, const char *policyInfoStr>
struct GradientTestTraits {
   using DataType = U;
   static constexpr ROOT::Fit::ExecutionPolicy ExecutionPolicyType() { return V; };

   static void PrintTypeInfo(const std::string &fittingInfo)
   {
      std::cout << "---------------- TEST INFO ----------------" << std::endl;
      std::cout << "- Fitting type:     " << fittingInfo << std::endl;
      std::cout << "- Data type:        " << dataInfoStr << std::endl;
      std::cout << "- Execution policy: " << policyInfoStr << std::endl;
      std::cout << "-------------------------------------------" << std::endl;
   }
};

// Info strings describing data types
char scalarStr[] = "Scalar";
char vectorStr[] = "Vectorial";

// Info strings describing execution policies
char serialStr[] = "Serial";
char mthreadStr[] = "Multithread";

// Typedefs of GradientTestTraits for scalar (serial and multithreaded)
// scenarios
using ScalarSerial = GradientTestTraits<Double_t, ROOT::Fit::ExecutionPolicy::kSerial, scalarStr, serialStr>;
using ScalarMultithread = GradientTestTraits<Double_t, ROOT::Fit::ExecutionPolicy::kMultithread, scalarStr, mthreadStr>;

#ifdef R__HAS_VECCORE

// Typedefs of GradientTestTraits for vectorial (serial and multithreaded)
// scenarios
using VectorialSerial = GradientTestTraits<ROOT::Double_v, ROOT::Fit::ExecutionPolicy::kSerial, vectorStr, serialStr>;
using VectorialMultithread =
   GradientTestTraits<ROOT::Double_v, ROOT::Fit::ExecutionPolicy::kMultithread, vectorStr, mthreadStr>;

#endif

// Model function to test the gradient evaluation
template <class T>
static T modelFunction(const T *data, const double *params)
{
   return params[0] * exp(-(*data + (-130.)) * (*data + (-130.)) / 2) +
          params[1] * exp(-(params[2] * (*data * (0.01)) - params[3] * ((*data) * (0.01)) * ((*data) * (0.01))));
}

// Helper class used to encapsulate the calls to the gradient interfaces,
// templated with a GradientTestTraits type
template <class T, class U>
struct GradientTestEvaluation {
   // Types to instantiate Chi2FCN: the gradient function interface handles the
   // parameters, so its base typedef
   // has to be always a double; the parametric function interface is templated
   // to test both serial and vectorial
   // cases.
   using GradFunctionType = ROOT::Math::IGradientFunctionMultiDimTempl<double>;
   using BaseFunctionType = ROOT::Math::IParamMultiFunctionTempl<typename T::DataType>;

   GradientTestEvaluation()
   {
      // Create TF1 from model function and initialize the fit function
      std::stringstream streamTF1;
      streamTF1 << "f" << this;
      std::string nameTF1 = streamTF1.str();

      fModelFunction = new TF1(nameTF1.c_str(), modelFunction<typename T::DataType>, 100, 200, 4);
      fModelFunction->SetParameters(fParams);

      fFitFunction =
         new ROOT::Math::WrappedMultiTF1Templ<typename T::DataType>(*fModelFunction, fModelFunction->GetNdim());

      // Assure the to-be-created histogram does not replace an old one
      std::stringstream streamTH1;
      streamTH1 << "h" << this;
      std::string nameTH1 = streamTH1.str();

      auto oldTH1 = gROOT->FindObject(nameTH1.c_str());
      if (oldTH1)
         delete oldTH1;

      // Create TH1 and fill it with values from model function
      fNumPoints = 12801;
      fHistogram = new TH1D(nameTH1.c_str(), "Test random numbers", fNumPoints, 100, 200);
      gRandom->SetSeed(1);
      fHistogram->FillRandom(nameTF1.c_str(), 1000000);

      // Fill (binned or unbinned) data
      FillData();
   }

   // Fill binned data
   template <class FitType = U>
   typename std::enable_if<std::is_same<FitType, U>::value && std::is_same<FitType, ROOT::Fit::BinData>::value>::type
   FillData()
   {
      fData = new ROOT::Fit::BinData();
      ROOT::Fit::FillData(*fData, fHistogram, fModelFunction);
   }

   // Fill unbinned data
   template <class FitType = U>
   typename std::enable_if<std::is_same<FitType, U>::value && std::is_same<FitType, ROOT::Fit::UnBinData>::value>::type
   FillData()
   {
      fData = new ROOT::Fit::UnBinData(fNumPoints);

      TAxis *coords = fHistogram->GetXaxis();
      for (unsigned i = 0; i < fNumPoints; i++)
         fData->Add(coords->GetBinCenter(i));
   }

   virtual void SetFitter() = 0;

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
   static const int fNumRepetitions = 100;

   const Double_t fParams[fNumParams] = {1, 1000, 7.5, 1.5};
   unsigned int fNumPoints;

   TF1 *fModelFunction;
   TH1D *fHistogram;
   ROOT::Math::WrappedMultiTF1Templ<typename T::DataType> *fFitFunction;
   U *fData;
   ROOT::Fit::BasicFCN<GradFunctionType, BaseFunctionType, U> *fFitter;
};

template <class T>
struct Chi2GradientTestEvaluation : public GradientTestEvaluation<T, ROOT::Fit::BinData> {
   using typename GradientTestEvaluation<T, ROOT::Fit::BinData>::GradFunctionType;
   using typename GradientTestEvaluation<T, ROOT::Fit::BinData>::BaseFunctionType;

   Chi2GradientTestEvaluation() { SetFitter(); }

   virtual void SetFitter() override
   {
      this->fFitter = new ROOT::Fit::Chi2FCN<GradFunctionType, BaseFunctionType>(*(this->fData), *(this->fFitFunction),
                                                                                 T::ExecutionPolicyType());
   }
};

template <class T>
struct PoissonLikelihoodGradientTestEvaluation : public GradientTestEvaluation<T, ROOT::Fit::BinData> {
   using typename GradientTestEvaluation<T, ROOT::Fit::BinData>::GradFunctionType;
   using typename GradientTestEvaluation<T, ROOT::Fit::BinData>::BaseFunctionType;

   PoissonLikelihoodGradientTestEvaluation() { SetFitter(); }

   virtual void SetFitter() override
   {
      this->fFitter = new ROOT::Fit::PoissonLikelihoodFCN<GradFunctionType, BaseFunctionType>(
         *(this->fData), *(this->fFitFunction), 0, true, T::ExecutionPolicyType());
   }
};

template <class T>
struct LogLikelihoodGradientTestEvaluation : public GradientTestEvaluation<T, ROOT::Fit::UnBinData> {
   using typename GradientTestEvaluation<T, ROOT::Fit::UnBinData>::GradFunctionType;
   using typename GradientTestEvaluation<T, ROOT::Fit::UnBinData>::BaseFunctionType;

   LogLikelihoodGradientTestEvaluation() { SetFitter(); }

   virtual void SetFitter() override
   {
      this->fFitter = new ROOT::Fit::LogLikelihoodFCN<GradFunctionType, BaseFunctionType>(
         *(this->fData), *(this->fFitFunction), 0, false, T::ExecutionPolicyType());
   }
};

// Test class: creates a reference solution (computing the gradient with scalar
// values in a serial scenario), and
// compares its values and its performance against the evaluation of the
// gradient specified by the GradientTestTraits
// type.
template <class T>
class Chi2GradientTest : public ::testing::Test, public Chi2GradientTestEvaluation<T> {
protected:
   virtual void SetUp()
   {
      T::PrintTypeInfo("Chi2FCN");

      // Set up the scalar serial reference fitter, to benchmark times.
      Chi2GradientTestEvaluation<ScalarSerial> reference;

      fReferenceSolution.resize(this->fNumParams);
      fReferenceTime = reference.BenchmarkSolution(fReferenceSolution.data());
   }

   Double_t fReferenceTime;
   std::vector<Double_t> fReferenceSolution;
};

// Test class: creates a reference solution (computing the gradient with scalar
// values in a serial scenario), and
// compares its values and its performance against the evaluation of the
// gradient specified by the GradientTestTraits
// type.
template <class T>
class PoissonLikelihoodGradientTest : public ::testing::Test, public PoissonLikelihoodGradientTestEvaluation<T> {
protected:
   virtual void SetUp()
   {
      T::PrintTypeInfo("PoissonLikelihoodFCN");

      // Set up the scalar serial reference fitter, to benchmark times.
      PoissonLikelihoodGradientTestEvaluation<ScalarSerial> reference;

      fReferenceSolution.resize(this->fNumParams);
      fReferenceTime = reference.BenchmarkSolution(fReferenceSolution.data());
   }

   Double_t fReferenceTime;
   std::vector<Double_t> fReferenceSolution;
};

// Test class: creates a reference solution (computing the gradient with scalar
// values in a serial scenario), and
// compares its values and its performance against the evaluation of the
// gradient specified by the GradientTestTraits
// type.
template <class T>
class LogLikelihoodGradientTest : public ::testing::Test, public LogLikelihoodGradientTestEvaluation<T> {
protected:
   virtual void SetUp()
   {
      T::PrintTypeInfo("LogLikelihoodFCN");

      // Set up the scalar serial reference fitter, to benchmark times.
      LogLikelihoodGradientTestEvaluation<ScalarSerial> reference;

      fReferenceSolution.resize(this->fNumParams);
      fReferenceTime = reference.BenchmarkSolution(fReferenceSolution.data());
   }

   Double_t fReferenceTime;
   std::vector<Double_t> fReferenceSolution;
};

// Types used by Google Test to instantiate the tests.
#ifdef R__HAS_VECCORE
#  ifdef R__USE_IMT
typedef ::testing::Types<ScalarMultithread, VectorialSerial, VectorialMultithread> TestTypes;
#  else
typedef ::testing::Types<ScalarSerial, VectorialSerial, VectorialMultithread> TestTypes;
#  endif
#else
#  ifdef R__USE_IMT
typedef ::testing::Types<ScalarSerial> TestTypes;
#  else
typedef ::testing::Types<ScalarSerial> TestTypes;
#  endif
#endif

TYPED_TEST_CASE(Chi2GradientTest, TestTypes);

// Test EvalChi2Gradient and outputs its speedup against the scalar serial case.
TYPED_TEST(Chi2GradientTest, Chi2Gradient)
{
   Double_t solution[TestFixture::fNumParams];

   Double_t benchmarkTime = TestFixture::BenchmarkSolution(solution);

   std::cout << std::fixed << std::setprecision(4);
   std::cout << "Speed-up with respect to scalar serial case: " << TestFixture::fReferenceTime / benchmarkTime;
   std::cout << std::endl;

   for (unsigned i = 0; i < TestFixture::fNumParams; i++) {
      EXPECT_NEAR(solution[i], TestFixture::fReferenceSolution[i], 1e-6);
   }
}

TYPED_TEST_CASE(PoissonLikelihoodGradientTest, TestTypes);

// Test EvalChi2Gradient and outputs its speedup against the scalar serial case.
TYPED_TEST(PoissonLikelihoodGradientTest, PoissonLikelihoodGradient)
{
   Double_t solution[TestFixture::fNumParams];

   Double_t benchmarkTime = TestFixture::BenchmarkSolution(solution);

   std::cout << std::fixed << std::setprecision(4);
   std::cout << "Speed-up with respect to scalar serial case: " << TestFixture::fReferenceTime / benchmarkTime;
   std::cout << std::endl;

   for (unsigned i = 0; i < TestFixture::fNumParams; i++) {
      EXPECT_NEAR(solution[i], TestFixture::fReferenceSolution[i], 1e-6);
   }
}

TYPED_TEST_CASE(LogLikelihoodGradientTest, TestTypes);

// Test EvalChi2Gradient and outputs its speedup against the scalar serial case.
TYPED_TEST(LogLikelihoodGradientTest, LogLikelihoodGradient)
{
   Double_t solution[TestFixture::fNumParams];

   Double_t benchmarkTime = TestFixture::BenchmarkSolution(solution);

   std::cout << std::fixed << std::setprecision(4);
   std::cout << "Speed-up with respect to scalar serial case: " << TestFixture::fReferenceTime / benchmarkTime;
   std::cout << std::endl;

   for (unsigned i = 0; i < TestFixture::fNumParams; i++) {
      EXPECT_NEAR(solution[i], TestFixture::fReferenceSolution[i], 1e-6);
   }
}
