// @(#)root/test:$Id$
// Author: Alejandro García Montoro 08/2017

#include "RConfigure.h"
#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
#include "Fit/Fitter.h"
#include "Fit/PoissonLikelihoodFCN.h"
#include "HFitInterface.h"
#include "TH1.h"
#include "TH2.h"
#include "TF2.h"
#include "TROOT.h"
#include "TRandom.h"

#include "gtest/gtest.h"

#include <iostream>
#include <string>


// Gradient 2D function
template <class T>
class GradFunc2D : public ROOT::Math::IParamMultiGradFunctionTempl<T> {
public:
   void SetParameters(const double *p) override {
      std::copy(p, p + NPar(), fParameters);
      // compute integral in interval [0,1][0,1]
      fIntegral = Integral(p);
   }

   const double *Parameters() const override { return fParameters; }

   ROOT::Math::IBaseFunctionMultiDimTempl<T> *Clone() const override
   {
      GradFunc2D<T> *f = new GradFunc2D<T>();
      f->SetParameters(fParameters);
      return f;
   }

   unsigned int NDim() const override { return 2; }

   unsigned int NPar() const override { return 5; }

   void ParameterGradient(const T *x, const double * p, T *grad) const override
   {
      if (p == nullptr) {
         ParameterGradient(x, fParameters, grad);
         return;
      }
      T xx = (1. - x[0] );
      T yy = (1. - x[1] );
      T fval = FVal(x,p);
      grad[0] =  fval / fIntegral;
      grad[1] =  p[0] * ( xx / fIntegral - fval / (2. * fIntegral * fIntegral ) );
      grad[2] =  p[0] * ( xx * xx / fIntegral -  fval / (3. * fIntegral * fIntegral ) );
      grad[3] =  p[0] * ( yy / fIntegral - fval / (2. * fIntegral * fIntegral ) );
      grad[4] =  p[0] * ( yy * yy / fIntegral -  fval / (3. * fIntegral * fIntegral ) );
   }

   // return integral in interval {0,1}{0,1}
   double Integral(const double * p)
   {
      return 1. +  (p[1] + p[3] )/ 2. + (p[2] + p[4] )/ 3.;
   }

private:

   T FVal(const T * x, const double *p) const
   {
      // use a function based on Bernstein polynomial which have easy normalization
      T xx = (1. - x[0] );
      T yy = (1. - x[1] );
      T fval =  1. + p[1] * xx + p[2] * xx * xx + p[3] * yy + p[4] * yy * yy;
      return fval;
   }

   T DoEvalPar(const T *x, const double *p) const override
   {
      if (p == nullptr)
         return DoEvalPar(x, fParameters);
      return p[0] * FVal(x,p) / fIntegral;
   }

   T DoParameterDerivative(const T *x, const double *p, unsigned int ipar) const override
   {
      std::vector<T> grad(NPar());
      ParameterGradient(x, p, &grad[0]);
      return grad[ipar];
   }

   double fParameters[5] = {0,0,0,0,0};
   double fIntegral = 1.0;
};

struct LikelihoodFitType {};
struct Chi2FitType {};

template <typename U, typename V, typename F>
struct GradientFittingTestTraits {
   using DataType = U;
   using FittingDataType = V;
   using FitType = F;
};

// Typedefs of GradientTestTraits for scalar (binned and unbinned) data
using ScalarChi2 = GradientFittingTestTraits<Double_t, ROOT::Fit::BinData, Chi2FitType>;
using ScalarBinned = GradientFittingTestTraits<Double_t, ROOT::Fit::BinData, LikelihoodFitType>;
using ScalarUnBinned = GradientFittingTestTraits<Double_t, ROOT::Fit::UnBinData, LikelihoodFitType>;

// Typedefs of GradientTestTraits for vectorial (binned and unbinned) data
#ifdef R__HAS_STD_EXPERIMENTAL_SIMD
using VectorialChi2 = GradientFittingTestTraits<ROOT::Double_v, ROOT::Fit::BinData, Chi2FitType>;
using VectorialBinned = GradientFittingTestTraits<ROOT::Double_v, ROOT::Fit::BinData, LikelihoodFitType>;
using VectorialUnBinned = GradientFittingTestTraits<ROOT::Double_v, ROOT::Fit::UnBinData, LikelihoodFitType>;
#endif

int printLevel = 0;

template <class T>
class GradientFittingTest : public ::testing::Test {
protected:
   void SetUp() override
   {
      // Create TF2 from model function and initialize the fit function
      std::stringstream streamTF2;
      streamTF2 << "f" << this;
      std::string nameTF2 = streamTF2.str();

      GradFunc2D<typename T::DataType> fitFunction;
      fFunction = new TF2(nameTF2.c_str(), fitFunction, 0., 1., 0, 1, 5);
      fFunction->SetNpx(300);
      fFunction->SetNpy(300);
      double p0[5] = {1., 1., 2., 3., 0.5};
      fFunction->SetParameters(p0);
      assert(fFunction->GetNpar() == 5);

      // Assure the to-be-created histogram does not replace an old one
      std::stringstream streamTH1;
      streamTH1 << "h" << this;
      std::string nameTH2 = streamTH1.str();

      auto oldTH2 = gROOT->FindObject(nameTH2.c_str());
      if (oldTH2)
         delete oldTH2;

      fHistogram = new TH2D(nameTH2.c_str(), nameTH2.c_str(), fNumPoints, 0, 1., 99, 0., 1.);

      // Fill the histogram
      gRandom->SetSeed(222);
      for (int i = 0; i < 1000000; ++i) {
         double x, y = 0;
         fFunction->GetRandom2(x, y);
         fHistogram->Fill(x, y);
      }


      // Create the function
      GradFunc2D<typename T::DataType> function;

      double p[5] = {50., 1., 1, 2., 1.};
      function.SetParameters(p);

      // Create the fitter from the function
      fFitter.SetFunction(function);
      //fFitter.SetFunction(function,false);
      fFitter.Config().SetMinimizer("Minuit2");


      // Fill the binned or unbinned data
      FillData();


      if (printLevel>1) fFitter.Config().MinimizerOptions().SetPrintLevel(printLevel);


   }


   // Fill binned data
   template <class U = typename T::FittingDataType>
   typename std::enable_if<std::is_same<U, typename T::FittingDataType>::value &&
                           std::is_same<U, ROOT::Fit::BinData>::value>::type
   FillData()
   {
      // fill fit data
      fData = new ROOT::Fit::BinData(fNumPoints, 2);
      ROOT::Fit::FillData(*fData, fHistogram, fFunction);
   }

   // Fill unbinned data
   template <class U = typename T::FittingDataType>
   typename std::enable_if<std::is_same<U, typename T::FittingDataType>::value &&
                           std::is_same<U, ROOT::Fit::UnBinData>::value>::type
   FillData()
   {
      unsigned int npoints = 100*fNumPoints + 1;
      //npoints = 101;
      fData = new ROOT::Fit::UnBinData(npoints, 2);

      gRandom->SetSeed(111); // to get the same data
      for (unsigned i = 0; i < npoints; i++) {
         double xdata, ydata = 0;
         fFunction->GetRandom2(xdata, ydata);
         fData->Add(xdata, ydata);
      }

      // for unbin data we need to fix the overall normalization parameter
      fFitter.Config().ParamsSettings()[0].SetValue(1);
      fFitter.Config().ParamsSettings()[0].Fix();
   }


   // Perform the Fit
   template <class F = typename T::FitType>
   typename std::enable_if<std::is_same<F, typename T::FitType>::value &&
                           std::is_same<F, LikelihoodFitType>::value>::type
   Fit()
   {
      std::cout << "Doing a binned likelihood Fit " << std::endl;
      // the fit is extended in case of bin data types
      bool extended = std::is_same<ROOT::Fit::BinData,typename T::FittingDataType>::value;
      fFitter.LikelihoodFit(*fData, extended,  fExecutionPolicy);
   }

   template <class F = typename T::FitType>
   typename std::enable_if<std::is_same<F, typename T::FitType>::value &&
                           std::is_same<F, Chi2FitType>::value>::type
   Fit()
   {
      std::cout << "Doing a chi2 Fit " << std::endl;
      fFitter.Fit(*fData, fExecutionPolicy );
   }

   // function actually running the test.
   // We define here the condition to say that the test is valid
   bool RunFit(ROOT::EExecutionPolicy executionPolicy) {
      fExecutionPolicy = executionPolicy;
      if (printLevel>0) {
         std::cout << "**************************************\n";
         if (fExecutionPolicy == ROOT::EExecutionPolicy::kSequential)
            std::cout << "   RUN SEQUENTIAL \n";
         else if (fExecutionPolicy == ROOT::EExecutionPolicy::kMultiThread)
            std::cout << "   RUN MULTI-THREAD \n";
         else if (fExecutionPolicy == ROOT::EExecutionPolicy::kMultiProcess)
            std::cout << "   RUN MULTI-PROCESS \n";

         std::cout << "**************************************\n";
      }
      Fit();
      if (printLevel>0) fFitter.Result().Print(std::cout);
      return (fFitter.Result().IsValid() && fFitter.Result().Edm() < 0.001);
   }


   TF2 *fFunction;
   typename T::FittingDataType *fData;
   TH2D *fHistogram;
   ROOT::Fit::Fitter fFitter;
   ROOT::EExecutionPolicy fExecutionPolicy = ROOT::EExecutionPolicy::kSequential;
   static const unsigned fNumPoints = 401;
};

// Types used by Google Test to instantiate the tests.
#ifdef R__HAS_STD_EXPERIMENTAL_SIMD
typedef ::testing::Types<ScalarChi2, ScalarBinned, ScalarUnBinned, VectorialChi2, VectorialBinned, VectorialUnBinned> TestTypes;

//typedef ::testing::Types<ScalarBinned,VectorialBinned> TestTypes;
#else
typedef ::testing::Types<ScalarChi2, ScalarBinned, ScalarUnBinned> TestTypes;
#endif



// Declare that the GradientFittingTest class should be instantiated with the types defined by TestTypes
TYPED_TEST_SUITE_P(GradientFittingTest);

// Test the fitting using the gradient is successful
TYPED_TEST_P(GradientFittingTest, Sequential)
{
   EXPECT_TRUE(TestFixture::RunFit(ROOT::EExecutionPolicy::kSequential));
}

#ifdef R__HAS_IMT
TYPED_TEST_P(GradientFittingTest, Multithread)
{
   EXPECT_TRUE(TestFixture::RunFit(ROOT::EExecutionPolicy::kMultiThread));
}
REGISTER_TYPED_TEST_SUITE_P(GradientFittingTest,Sequential,Multithread);
#else
REGISTER_TYPED_TEST_SUITE_P(GradientFittingTest,Sequential);
#endif

INSTANTIATE_TYPED_TEST_SUITE_P(GradientFitting, GradientFittingTest, TestTypes);

// Linear 1D function p0 + p1 * (x - 0.5), normalized in [0,1] for p0 = 1, with a zero
// parameter Hessian, or failing to compute it if goodHessian is false
class LinearFunc : public ROOT::Math::IParamMultiGradFunction {
public:
   explicit LinearFunc(bool goodHessian = true) : fGoodHessian(goodHessian) {}
   void SetParameters(const double *p) override { std::copy(p, p + 2, fParameters); }
   const double *Parameters() const override { return fParameters; }
   ROOT::Math::IMultiGenFunction *Clone() const override
   {
      auto f = new LinearFunc(fGoodHessian);
      f->SetParameters(fParameters);
      return f;
   }
   unsigned int NDim() const override { return 1; }
   unsigned int NPar() const override { return 2; }
   bool HasParameterHessian() const override { return true; }
   bool ParameterHessian(const double *, const double *, double *h) const override
   {
      std::fill(h, h + 3, 0.);
      return fGoodHessian;
   }

private:
   double DoEvalPar(const double *x, const double *p) const override
   {
      if (p == nullptr)
         p = fParameters;
      return p[0] + p[1] * (x[0] - 0.5);
   }
   double DoParameterDerivative(const double *x, const double *, unsigned int ipar) const override
   {
      return ipar == 0 ? 1. : x[0] - 0.5;
   }

   double fParameters[2] = {1., 0.};
   bool fGoodHessian = true;
};

// Test that FitMethodFunction::Hessian() reports a failure of the model function Hessian
TEST(GradientFitting, HessianFailure)
{
   TH1D h("hHessianFailure", "", 10, 0., 1.);
   for (int i = 1; i <= 10; ++i)
      h.SetBinContent(i, 10.);
   ROOT::Fit::BinData data;
   ROOT::Fit::FillData(data, &h);
   LinearFunc func(false);
   ROOT::Fit::PoissonLLGradFunction fcn(data, func);
   ASSERT_TRUE(fcn.HasHessian());
   const double params[2] = {10., 0.};
   double hess[3];
   EXPECT_FALSE(fcn.Hessian(params, hess));
}

// Test that the parameter errors of fits with and without gradient agree when the full
// Hessian is not available (bin integral and unbinned fits)
TEST(GradientFitting, ErrorsWithoutFullHessian)
{
   TRandom rng(1);
   ROOT::Fit::DataOptions opt;
   opt.fIntegral = true;
   ROOT::Fit::BinData binData(opt);
   ROOT::Fit::UnBinData unbinData(5000);
   TH1D h("hGradientFitting", "", 20, 0., 1.);
   for (int i = 0; i < 5000; ++i) {
      double x = rng.Uniform();
      // accept-reject sampling of the pdf 1 + 0.8 * (x - 0.5)
      while (rng.Uniform(1.4) > 1. + 0.8 * (x - 0.5))
         x = rng.Uniform();
      h.Fill(x);
      unbinData.Add(x);
   }
   ROOT::Fit::FillData(binData, &h);
   ASSERT_TRUE(binData.HasBinEdges());

   enum class EFit {
      kChi2,
      kPoisson,
      kUnbinned
   };
   auto parErrors = [&](EFit fit, bool useGradient) {
      ROOT::Fit::Fitter fitter;
      fitter.Config().SetMinimizer("Minuit2");
      fitter.SetFunction(LinearFunc(), useGradient);
      bool ok = false;
      if (fit == EFit::kUnbinned) {
         fitter.Config().ParSettings(0).Fix();
         ok = fitter.LikelihoodFit(unbinData);
      } else {
         fitter.Config().ParSettings(0).SetValue(250.);
         ok = fit == EFit::kChi2 ? fitter.Fit(binData) : fitter.LikelihoodFit(binData);
      }
      EXPECT_TRUE(ok);
      return fitter.Result().Errors();
   };

   for (EFit fit : {EFit::kChi2, EFit::kPoisson, EFit::kUnbinned}) {
      const auto expected = parErrors(fit, false);
      const auto errors = parErrors(fit, true);
      for (unsigned int i = 0; i < 2; ++i) {
         EXPECT_NEAR(errors[i], expected[i], 1e-2 * expected[i]) << "fit type " << int(fit) << ", parameter " << i;
      }
   }
}

int main(int argc, char** argv) {

   // Disables elapsed time by default.
   //::testing::GTEST_FLAG(print_time) = false;

   // Parse command line arguments
   for (Int_t i = 1 ;  i < argc ; i++) {
      std::string arg = argv[i] ;
      if (arg == "-v") {
         std::cout << "---running in verbose mode" << std::endl;
         printLevel = 1;
      } else if (arg == "-vv") {
         std::cout << "---running in very verbose mode" << std::endl;
         printLevel = 2;
      } else if (arg == "-vvv") {
         std::cout << "---running in very very verbose mode" << std::endl;
         printLevel = 3;
      }
   }

  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
