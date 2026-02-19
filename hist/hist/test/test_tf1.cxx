#include "TF1.h"
#include "TF1NormSum.h"

#include "TError.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TPluginManager.h"
#include "TROOT.h"

#include "gtest/gtest.h"
#include "ROOT/TestSupport.hxx"

#include <cmath>
#include <iostream>

class MyClass {
public:
   double operator()(double *x, double *p) { return *x + *p; };
};

class MyClassConst {
public:
   double operator()(const double *x, const double *p) { return *x + *p; };
};

double func(const double *x, const double *p)
{
   return *x + *p;
}
double functionConst(const double *x, const double *p)
{
   return *x + *p;
}

constexpr Float_t delta = 1.E-11f;

void coeffNamesGeneric(TString &formula, TObjArray *coeffNames)
{
   TF1 cn0("cn0", formula, 0, 1);
   ASSERT_EQ(cn0.GetNpar(), coeffNames->GetEntries());
   for (int i = 0; i < coeffNames->GetEntries(); i++) {
      TObjString *coeffObj = (TObjString *)coeffNames->At(i);
      TString coeffName = coeffObj->GetString();
      EXPECT_EQ(coeffName, TString(cn0.GetParName(i)));
   }
}

// Test that the NSUM names are copied correctly
TEST(TF1, NsumCoeffNames)
{
   TObjArray *coeffNames = new TObjArray();
   coeffNames->SetOwner(kTRUE);
   coeffNames->Add(new TObjString("sg"));
   coeffNames->Add(new TObjString("bg"));
   coeffNames->Add(new TObjString("Mean"));
   coeffNames->Add(new TObjString("Sigma"));
   coeffNames->Add(new TObjString("Slope"));
   TString formula("NSUM([sg] * gaus, [bg] * expo)");

   coeffNamesGeneric(formula, coeffNames);

   delete coeffNames;
}

// Test that the NSUM is normalized as we'd expect
TEST(TF1, Normalization)
{
   double xmin = -5;
   double xmax = 5;
   TF1 n0("n0", "NSUM(.5 * gaus, .5 * (x+[0])**2)", xmin, xmax);
   EXPECT_NEAR(n0.Integral(xmin, xmax), 1, delta);
   n0.SetParameter(4, 1); // should not affect integral
   EXPECT_NEAR(n0.Integral(xmin, xmax), 1, delta);
   n0.SetParameter(0, 0);
   EXPECT_NEAR(n0.Integral(xmin, xmax), .5, delta);

   TF1 n1("n1", "NSUM([sg] * gaus, [bg] * (x+[0])**2)", xmin, xmax);
   n1.SetParameter(0, .5);
   n1.SetParameter(1, .5);
   EXPECT_NEAR(n1.Integral(xmin, xmax), 1, delta);
   n0.SetParameter(0, 0);
   EXPECT_NEAR(n0.Integral(xmin, xmax), .5, delta);

   TF1 n2("n2", "NSUM([sg] * gaus, -0.5 * (x+[0])**2)", xmin, xmax);
   n2.SetParameter(0, .5);
   EXPECT_NEAR(n2.GetParameter(1), -.5, delta);
   EXPECT_NEAR(n2.Integral(xmin, xmax), 0, delta);
   n2.SetParameter(0, 0);
   EXPECT_NEAR(n2.Integral(xmin, xmax), -.5, delta);
}
// Test analytical exponential integral when p1 == 0
TEST(TF1AnalyticalIntegral, ExponentialP1Zero)
{
   TF1 f("f_expo_zero", "expo", 0.0, 2.0);
   f.SetParameters(1.2, 0.0); // p0 = 1.2, p1 = 0

   const double a = 0.3;
   const double b = 1.7;

   const double result = f.Integral(a, b);

   const double expected = std::exp(1.2) * (b - a);

   EXPECT_NEAR(result, expected, 1e-12);
}

// Test analytical Gaussian integral with invalid sigma
void voigtHelper(double sigma, double lg)
{
   TF1 lor("lor", "breitwigner", -20, 20);
   lor.SetParameters(1, 0, lg);
   TF1 mygausn("mygausn", "gausn", -20, 20);
   mygausn.SetParameters(1, 0, sigma);

   TF1 conv("conv", "CONV(lor, mygausn)", -20, 20);

   // Voigt should just be the convolution of the gaussian and lorentzian
   TF1 myvoigt("myvoigt", "TMath::Voigt(x, [0], [1])", -20, 20);
   myvoigt.SetParameters(sigma, lg);

   for (double x = -19.5; x < 20; x += .5)
      EXPECT_NEAR(conv.Eval(x), myvoigt.Eval(x), .01 * conv.Eval(x));
}

// Test that the voigt can be expressed as a convolution of a gaussian and lorentzian
// Check that the values match to within 1%
TEST(TF1, ConvVoigt)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.optionalDiag(kError, "TVirtualFFT::FFT", "handler not found");
   diags.optionalDiag(kWarning, "TF1Convolution::MakeFFTConv",
                      "Cannot use FFT, probably FFTW package is not available. Switch to numerical convolution");

   {
      // Try to load the FFTW plugin. On failure, skip the test.
      R__LOCKGUARD(gROOTMutex);

      TPluginHandler *h = gROOT->GetPluginManager()->FindHandler("TVirtualFFT", "fftwc2c");
      if (!h || h->LoadPlugin() == -1)
         GTEST_SKIP() << "Didn't find the FFTW plugin";
   }

   voigtHelper(.1, 1);
   voigtHelper(1, .1);
   voigtHelper(1, 1);
}

// Test that we can change the range of TF1NormSum and TF1Convolution
TEST(TF1, SetRange)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.optionalDiag(kError, "TVirtualFFT::FFT", "handler not found");
   diags.optionalDiag(kWarning, "TF1Convolution::MakeFFTConv",
                      "Cannot use FFT, probably FFTW package is not available. Switch to numerical convolution");

   // Define TF1 using NSUM
   TF1 f1("f1", "NSUM([sg] * gaus, [bg] * expo)", 0, 1);
   f1.SetParameters(1, 1, 0, 1, 1);
   EXPECT_NEAR(f1.Integral(0, 1), 2, delta);

   // set range and check value of integral again
   f1.SetRange(-5, 5);
   EXPECT_NEAR(f1.Integral(-5, 5), 2, delta);

#ifdef R__HAS_MATHMORE
   // The following relies on GSL integrators, which are only available with MathMore

   // now same thing with CONV
   TF1 f2("f2", "CONV(gaus, breitwigner)", 0, 1);
   f2.SetParameters(1, 1, 1, 1, 1, 1);
   f2.SetRange(-100, 100); // making our convolution much more accurate
   // Numeric integration of this function suffers from roundoff errors, so the default 1.E-12 accuracy won't be reached.
   // By reducing the tolerance, we get rid of a GSL warning, which was picked up by the log checkers.
   constexpr double tolerance = 1.E-6;
   EXPECT_NEAR(f2.Integral(-20, 20, tolerance), 2.466, .005);
#endif
}

// Test that we can copy and clone TF1 objects based on NSUM and CONV
TEST(TF1, CopyClone)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.optionalDiag(kError, "TVirtualFFT::FFT", "handler not found");
   diags.optionalDiag(kWarning, "TF1Convolution::MakeFFTConv",
                      "Cannot use FFT, probably FFTW package is not available. Switch to numerical convolution");
   diags.optionalDiag(kWarning, "TClass::Init", "no dictionary for class pair<TString,int> is available");

   // Define original TF1 using NSUM
   TF1 f1("f1", "NSUM(gaus, expo)", -5, 5);
   f1.SetParameters(1, 1, 0, 1, 1);

   // Make copy and test
   TF1 f2;
   f1.Copy(f2);
   for (double x = -5; x < 5; x += .5)
      EXPECT_NEAR(f1.Eval(x), f2.Eval(x), delta);

   // Make clone and test
   std::unique_ptr<TF1> f3{dynamic_cast<TF1 *>(f1.Clone())};
   for (double x = -5; x < 5; x += .5)
      EXPECT_NEAR(f1.Eval(x), f3->Eval(x), delta);

   // Same thing for CONV

   // Define original TF1 using NSUM
   TF1 f4("f4", "CONV(gaus, breitwigner)", -15, 15);
   f4.SetParameters(1, 1, 1, 1, 1, 1);

   // Make copy
   TF1 f5;
   f4.Copy(f5);
   for (double x = -5; x < 5; x += .5)
      EXPECT_NEAR(f4.Eval(x), f5.Eval(x), delta);

   // Make clone
   std::unique_ptr<TF1> f6{dynamic_cast<TF1 *>(f4.Clone())};
   for (double x = -5; x < 5; x += .5)
      EXPECT_NEAR(f4.Eval(x), f6->Eval(x), delta);
}

TEST(TF1, Constructors)
{
   MyClass testObj, testObjConst;
   double x = 1;
   double p = 1;

   std::function<double(const double *data, const double *param)> stdFunction = functionConst;
   std::vector<TF1> vtf1;
   vtf1.emplace_back(TF1("Functor", testObj, 0, 1, 0));
   vtf1.emplace_back(TF1("FunctorConst", testObjConst, 0, 1, 0));
   vtf1.emplace_back(TF1("FunctorPtr", &testObj, 0, 1, 0));
   vtf1.emplace_back(TF1("FunctorConstPtr", &testObjConst, 0, 1, 0));
   vtf1.emplace_back(TF1("function", func));
   vtf1.emplace_back(TF1("functionConst", functionConst));
   vtf1.emplace_back(TF1("lambda", [](double *x1, double *x2) { return *x1 + *x2; }));
   vtf1.emplace_back(TF1("lambdaConst", [](double *x1, double *x2) { return *x1 + *x2; }));
   vtf1.emplace_back(TF1("stdFunction", stdFunction));

   for (auto tf1 : vtf1)
      EXPECT_EQ(tf1(&x, &p), 2);
}

TEST(TF1, Save)
{
   TF1 linear("linear", "x", -10., 10.);
   linear.SetNpx(20);

   Double_t args[1];

   // save with explicit range
   linear.Save(-10, 10, 0, 0, 0, 0);

   // test at position of saved bins
   for (Double_t x = -10.; x <= 10.; x += 1.) {
      args[0] = x;
      EXPECT_NEAR(x, linear.GetSave(args), 1e-10);
   }

   // test linear approximation
   for (Double_t x = -10.; x <= 10.; x += 0.77) {
      args[0] = x;
      EXPECT_NEAR(x, linear.GetSave(args), 1e-10);
   }

   // test outside range
   args[0] = -11;
   EXPECT_EQ(0., linear.GetSave(args));

   args[0] = 11;
   EXPECT_EQ(0., linear.GetSave(args));


   // now test saved at middle of bins
   linear.Save(0, 0, 0, 0, 0, 0);

   // test at position of saved bins
   for (Double_t x = -9.5; x <= 9.5; x += 1.) {
      args[0] = x;
      EXPECT_NEAR(x, linear.GetSave(args), 1e-10);
   }

   // test linear approximation
   for (Double_t x = -9.5; x <= 9.5; x += 0.77) {
      args[0] = x;
      EXPECT_NEAR(x, linear.GetSave(args), 1e-10);
   }

   // test outside range
   args[0] = -11;
   EXPECT_EQ(0., linear.GetSave(args));

   args[0] = 11;
   EXPECT_EQ(0., linear.GetSave(args));
}
