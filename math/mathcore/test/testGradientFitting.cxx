// @(#)root/test:$Id$
// Author: Alejandro García Montoro 08/2017

#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
#include "Fit/Fitter.h"
#include "HFitInterface.h"
#include "TH2.h"
#include "TF2.h"

#include "gtest/gtest.h"

#include <iostream>
#include <string>

// Gradient 2D function
template <class T>
class GradFunc2D : public ROOT::Math::IParamMultiGradFunctionTempl<T> {
public:
   void SetParameters(const double *p) { std::copy(p, p + NPar(), fParameters); }

   const double *Parameters() const { return fParameters; }

   ROOT::Math::IBaseFunctionMultiDimTempl<T> *Clone() const
   {
      GradFunc2D<T> *f = new GradFunc2D<T>();
      f->SetParameters(fParameters);
      return f;
   }

   unsigned int NDim() const { return 2; }

   unsigned int NPar() const { return 5; }

   void ParameterGradient(const T *x, const double *, T *grad) const
   {
      grad[0] = x[0] * x[0];
      grad[1] = x[0];
      grad[2] = x[1] * x[1];
      grad[3] = x[1];
      grad[4] = 1;
   }

private:
   T DoEvalPar(const T *x, const double *p) const
   {
      return p[0] * x[0] * x[0] + p[1] * x[0] + p[2] * x[1] * x[1] + p[3] * x[1] + p[4];
   }

   T DoParameterDerivative(const T *x, const double *p, unsigned int ipar) const
   {
      std::vector<T> grad(NPar());
      ParameterGradient(x, p, &grad[0]);
      return grad[ipar];
   }

   double fParameters[5];
};

template <typename U, typename V>
struct GradientFittingTestTraits {
   using DataType = U;
   using FittingDataType = V;
};

// Typedefs of GradientTestTraits for scalar (binned and unbinned) data
using ScalarBinned = GradientFittingTestTraits<Double_t, ROOT::Fit::BinData>;
using ScalarUnBinned = GradientFittingTestTraits<Double_t, ROOT::Fit::UnBinData>;

// Typedefs of GradientTestTraits for vectorial (binned and unbinned) data
#ifdef R__HAS_VECCORE
using VectorialBinned = GradientFittingTestTraits<ROOT::Double_v, ROOT::Fit::BinData>;
using VectorialUnBinned = GradientFittingTestTraits<ROOT::Double_v, ROOT::Fit::UnBinData>;
#endif

template <class T>
class GradientFittingTest : public ::testing::Test {
protected:
   virtual void SetUp()
   {
      // Create TF2 from model function and initialize the fit function
      std::stringstream streamTF2;
      streamTF2 << "f" << this;
      std::string nameTF2 = streamTF2.str();

      GradFunc2D<typename T::DataType> fitFunction;
      fFunction = new TF2(nameTF2.c_str(), fitFunction, 0., 10., 0, 10, 5);
      double p0[5] = {1., 2., 0.5, 1., 3.};
      fFunction->SetParameters(p0);
      assert(fFunction->GetNpar() == 5);

      // Assure the to-be-created histogram does not replace an old one
      std::stringstream streamTH1;
      streamTH1 << "h" << this;
      std::string nameTH2 = streamTH1.str();

      auto oldTH2 = gROOT->FindObject(nameTH2.c_str());
      if (oldTH2)
         delete oldTH2;

      fHistogram = new TH2D(nameTH2.c_str(), nameTH2.c_str(), fNumPoints, 0, 10., 30, 0., 10.);

      // Fill the histogram
      for (int i = 0; i < 10000; ++i) {
         double x, y = 0;
         fFunction->GetRandom2(x, y);
         fHistogram->Fill(x, y);
      }

      // Fill the binned or unbinned data
      FillData();

      // Create the function
      GradFunc2D<typename T::DataType> function;

      double p[5] = {2., 1., 1, 2., 100.};
      function.SetParameters(p);

      // Create the fitter from the function
      fFitter.SetFunction(function);
      fFitter.Config().SetMinimizer("Minuit");
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
      fData = new ROOT::Fit::UnBinData(fNumPoints, 2);

      TAxis *x = fHistogram->GetXaxis();
      TAxis *y = fHistogram->GetYaxis();

      for (unsigned i = 0; i < fNumPoints; i++)
         fData->Add(x->GetBinCenter(i), y->GetBinCenter(i));
   }

   TF2 *fFunction;
   typename T::FittingDataType *fData;
   TH2D *fHistogram;
   ROOT::Fit::Fitter fFitter;

   static const unsigned fNumPoints = 6801;
};

// Types used by Google Test to instantiate the tests.
#ifdef R__HAS_VECCORE
typedef ::testing::Types<ScalarBinned, ScalarUnBinned, VectorialBinned, VectorialUnBinned> TestTypes;
#else
typedef ::testing::Types<ScalarBinned, ScalarUnBinned> TestTypes;
#endif

// Declare that the GradientFittingTest class should be instantiated with the types defined by TestTypes
TYPED_TEST_CASE(GradientFittingTest, TestTypes);

// Test the fitting using the gradient is successful
TYPED_TEST(GradientFittingTest, GradientFitting)
{
   // TestFixture::fFitter.Config().MinimizerOptions().SetPrintLevel(3);
   EXPECT_TRUE(TestFixture::fFitter.Fit(*TestFixture::fData));
   TestFixture::fFitter.Result().Print(std::cout);
}
