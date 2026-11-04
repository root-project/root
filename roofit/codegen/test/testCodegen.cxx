// Tests for the C++ code that the RooFit codegen backend generates.
// Author: Jonas Rembser, CERN 2026

#include <RooFit/CodegenContext.h>
#include <RooFit/Detail/NormalizationHelpers.h>

#include <RooAddPdf.h>
#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooChebychev.h>
#include <RooConstVar.h>
#include <RooFFTConvPdf.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooLandau.h>
#include <RooLognormal.h>
#include <RooRealVar.h>

#include <TInterpreter.h>

#include <gtest/gtest.h>

#include <locale>
#include <memory>
#include <regex>
#include <sstream>
#include <string>

namespace {

/// Formats decimal points as ',' like a German locale does. Such a locale is
/// not installed on every test machine, so it is built from a custom facet
/// instead of requested by name. The resulting locale is unnamed, so making it
/// global does not also switch the C locale that std::strtod() uses.
struct CommaPunct : std::numpunct<char> {
   char do_decimal_point() const override { return ','; }
};

/// Generate the code for `arg` with `loc` as the global locale, restoring the
/// previous global locale even if code generation throws.
std::string codeUnderLocale(RooAbsArg &arg, std::locale const &loc)
{
   const std::locale old = std::locale::global(loc);
   std::string code;
   try {
      RooFit::Experimental::CodegenContext ctx;
      ctx.buildFunction(arg);
      code = ctx.collectedCode();
   } catch (...) {
      std::locale::global(old);
      throw;
   }
   std::locale::global(old);
   return code;
}

/// Erase the global counter from the generated function name, which differs
/// between two code generations of the same model.
std::string normalized(std::string const &code)
{
   return std::regex_replace(code, std::regex{"roo_codegen_[0-9]+"}, "roo_codegen_N");
}

/// Generate the code for `arg` as a standalone function of `params`, compile
/// it with the interpreter like RooFuncWrapper does, and return a pointer to
/// it. The signature is the one that CodegenContext::buildFunction() emits.
using CodegenFunc = double (*)(double *params, double const *obs, double const *xlArr);

CodegenFunc compileFunction(RooAbsArg &arg, RooArgList const &params)
{
   RooFit::Experimental::CodegenContext ctx;
   for (std::size_t i = 0; i < params.size(); ++i) {
      ctx.addResult(&params[i], "params[" + std::to_string(i) + "]");
   }
   const std::string funcName = ctx.buildFunction(arg);
   gInterpreter->Declare("#include <RooFit/CodegenImpl.h>\n");
   if (!gInterpreter->Declare(ctx.collectedCode().c_str())) {
      ADD_FAILURE() << "generated code does not compile:\n" << ctx.collectedCode();
      return nullptr;
   }
   return reinterpret_cast<CodegenFunc>(gInterpreter->ProcessLine((funcName + ";").c_str()));
}

/// How a plain stream formats 0.5 under `loc`, to verify that the facet is
/// actually in effect (otherwise the tests below would pass vacuously).
std::string formatWithStream(double val, std::locale const &loc)
{
   const std::locale old = std::locale::global(loc);
   std::stringstream ss;
   ss << val;
   std::locale::global(old);
   return ss.str();
}

} // namespace

// The generated code is C++ source, so its number formatting must not follow
// the global locale: under a comma-decimal locale the literals came out as
// "0,5", which does not compile, and inside a function call argument list the
// comma even turns one argument into two.
TEST(RooFitCodegen, ValueLiteralsAreLocaleIndependent)
{
   const std::locale comma{std::locale::classic(), new CommaPunct};
   ASSERT_EQ(formatWithStream(0.5, comma), "0,5");

   RooRealVar x{"x", "x", 0.5, -10, 10};
   RooRealVar mean{"mean", "mean", 1.25};
   RooConstVar sigma{"sigma", "sigma", 0.75};
   RooGaussian gauss{"gauss", "gauss", x, mean, sigma};
   x.setConstant(true);
   mean.setConstant(true);

   const std::string code = codeUnderLocale(gauss, comma);

   EXPECT_EQ(normalized(code), normalized(codeUnderLocale(gauss, std::locale::classic())));
   for (const char *literal : {"0.5", "1.25", "0.75"}) {
      EXPECT_NE(code.find(literal), std::string::npos) << literal << " missing from:\n" << code;
   }
   for (const char *corrupted : {"0,5", "1,25", "0,75"}) {
      EXPECT_EQ(code.find(corrupted), std::string::npos) << corrupted << " emitted in:\n" << code;
   }
}

// Same for the doubles that the codegen implementations pass to the generated
// function calls directly (here the observable range of RooChebychev), which
// are formatted by CodegenContext::buildArg() and not by codegen's
// doubleToString().
TEST(RooFitCodegen, CallArgumentsAreLocaleIndependent)
{
   const std::locale comma{std::locale::classic(), new CommaPunct};

   RooRealVar x{"x", "x", 0.125, -0.5, 2.25};
   RooRealVar a1{"a1", "a1", 0.375};
   RooChebychev cheby{"cheby", "cheby", x, a1};

   const std::string code = codeUnderLocale(cheby, comma);

   EXPECT_EQ(normalized(code), normalized(codeUnderLocale(cheby, std::locale::classic())));
   for (const char *literal : {"-0.5", "2.25", "0.375"}) {
      EXPECT_NE(code.find(literal), std::string::npos) << literal << " missing from:\n" << code;
   }
   for (const char *corrupted : {"0,5", "2,25", "0,375"}) {
      EXPECT_EQ(code.find(corrupted), std::string::npos) << corrupted << " emitted in:\n" << code;
   }
}

// The code generated for a RooLognormal with the standard parametrization
// called a RooFit::Detail::MathFuncs function that does not exist, so it did
// not compile. Check both parametrizations against the reference evaluation.
TEST(RooFitCodegen, LognormalParametrizations)
{
   RooRealVar x{"x", "x", 1.5, 0.1, 10.0};
   RooRealVar m0{"m0", "m0", 0.7, 0.1, 2.3};
   RooRealVar k{"k", "k", 0.7, 0.1, 0.95};
   RooArgList params{x, m0, k};

   for (bool useStandardParametrization : {false, true}) {
      RooLognormal pdf{"pdf", "pdf", x, m0, k, useStandardParametrization};
      CodegenFunc func = compileFunction(pdf, params);
      ASSERT_NE(func, nullptr) << "useStandardParametrization = " << useStandardParametrization;

      double paramVals[] = {x.getVal(), m0.getVal(), k.getVal()};
      EXPECT_DOUBLE_EQ(func(paramVals, nullptr, nullptr), pdf.getVal())
         << "useStandardParametrization = " << useStandardParametrization;
   }
}

// Compare the generated code for RooFFTConvPdf, pdf value divided by
// normalization integral, with the reference implementation for all buffer
// strategies and interpolation orders, with and without the default shifts.
TEST(RooFitCodegen, FFTConvPdf)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl{RooFit::WARNING};

   RooRealVar x{"x", "x", -10.0, 30.0};
   x.setBins(1000, "cache");
   RooRealVar ml{"ml", "ml", 5.0, -20.0, 20.0};
   RooRealVar sl{"sl", "sl", 1.0, 0.1, 10.0};
   RooRealVar mg{"mg", "mg", 0.0, -5.0, 5.0};
   RooRealVar sg{"sg", "sg", 2.0, 0.1, 10.0};
   RooLandau landau{"lx", "lx", x, ml, sl};
   RooGaussian gauss{"gauss", "gauss", x, mg, sg};
   RooArgList params{x, ml, sl, mg, sg};
   RooArgSet normSet{x};

   for (auto strategy : {RooFFTConvPdf::Extend, RooFFTConvPdf::Flat, RooFFTConvPdf::Mirror}) {
      for (int intOrder : {0, 1, 2}) {
         for (bool zeroShifts : {false, true}) {
            RooFFTConvPdf conv{"conv", "conv", x, landau, gauss, intOrder};
            conv.setBufferStrategy(strategy);
            if (zeroShifts) {
               conv.setShift(0.0, 0.0);
            }
            std::unique_ptr<RooAbsReal> integral{conv.createIntegral(x)};

            CodegenFunc pdfFunc = compileFunction(conv, params);
            CodegenFunc integralFunc = compileFunction(*integral, params);
            ASSERT_NE(pdfFunc, nullptr);
            ASSERT_NE(integralFunc, nullptr);

            for (double mlVal : {5.0, 7.5}) {
               ml.setVal(mlVal);
               for (double xVal = -9.5; xVal < 30.0; xVal += 1.3) {
                  x.setVal(xVal);
                  double paramVals[] = {xVal, ml.getVal(), sl.getVal(), mg.getVal(), sg.getVal()};
                  const double ref = conv.getVal(normSet);
                  const double val = pdfFunc(paramVals, nullptr, nullptr) / integralFunc(paramVals, nullptr, nullptr);
                  EXPECT_NEAR(val, ref, 1e-10 * ref + 1e-14)
                     << "strategy " << strategy << ", intOrder " << intOrder << ", zeroShifts " << zeroShifts << ", ml "
                     << mlVal << ", x " << xVal;
               }
            }
         }
      }
   }
}

// The shape of a RooAddPdf input depends on its normalization set, which has
// to be the convolution observable. The generated code takes the inputs from
// the compiled computation graph, like in a fit.
TEST(RooFitCodegen, FFTConvPdfAddPdfInput)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl{RooFit::WARNING};

   RooRealVar x{"x", "x", -10.0, 30.0};
   x.setBins(1000, "cache");
   RooRealVar mA{"mA", "mA", 6.0, 0.0, 12.0};
   RooRealVar sA{"sA", "sA", 0.8, 0.1, 5.0};
   RooRealVar mB{"mB", "mB", 14.0, 8.0, 20.0};
   RooRealVar sB{"sB", "sB", 3.0, 0.5, 10.0};
   RooRealVar fAB{"fAB", "fAB", 0.3, 0.0, 1.0};
   RooGaussian gA{"gA", "gA", x, mA, sA};
   RooGaussian gB{"gB", "gB", x, mB, sB};
   RooAddPdf sumAB{"sumAB", "sumAB", {gA, gB}, {fAB}};
   RooRealVar mg{"mg", "mg", 0.0};
   RooRealVar sg{"sg", "sg", 2.0, 0.1, 10.0};
   RooGaussian gauss{"gauss", "gauss", x, mg, sg};
   RooArgList params{x, mA, sA, mB, sB, fAB, sg};
   RooArgSet normSet{x};

   RooFFTConvPdf conv{"conv", "conv", x, sumAB, gauss};
   // The compiled pdf is already normalized over x
   std::unique_ptr<RooAbsReal> compiled = RooFit::Detail::compileForNormSet(conv, normSet);
   CodegenFunc pdfFunc = compileFunction(*compiled, params);
   ASSERT_NE(pdfFunc, nullptr);

   for (double fVal : {0.3, 0.7}) {
      fAB.setVal(fVal);
      for (double xVal = -9.5; xVal < 30.0; xVal += 1.3) {
         x.setVal(xVal);
         double paramVals[] = {xVal, mA.getVal(), sA.getVal(), mB.getVal(), sB.getVal(), fAB.getVal(), sg.getVal()};
         const double ref = conv.getVal(normSet);
         EXPECT_NEAR(pdfFunc(paramVals, nullptr, nullptr), ref, 1e-10 * ref + 1e-14)
            << "fAB " << fVal << ", x " << xVal;
      }
   }
}

// Integrals over a sub-range of the convolution observable are not supported,
// neither as partial sums of the cache (no interpolation) nor via numeric
// integration of the interpolated cache. Both must fail clearly.
TEST(RooFitCodegen, FFTConvPdfSubRangeIntegral)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl{RooFit::WARNING};

   RooRealVar x{"x", "x", -10.0, 30.0};
   x.setBins(1000, "cache");
   x.setRange("sub", -5.0, 20.0);
   RooRealVar ml{"ml", "ml", 5.0, -20.0, 20.0};
   RooRealVar sl{"sl", "sl", 1.0, 0.1, 10.0};
   RooRealVar mg{"mg", "mg", 0.0};
   RooRealVar sg{"sg", "sg", 2.0, 0.1, 10.0};
   RooLandau landau{"lx", "lx", x, ml, sl};
   RooGaussian gauss{"gauss", "gauss", x, mg, sg};
   RooArgList params{x, ml, sl, sg};

   for (int intOrder : {0, 2}) {
      RooFFTConvPdf conv{"conv", "conv", x, landau, gauss, intOrder};
      std::unique_ptr<RooAbsReal> integral{conv.createIntegral(x, "sub")};
      RooFit::Experimental::CodegenContext ctx;
      for (std::size_t i = 0; i < params.size(); ++i) {
         ctx.addResult(&params[i], "params[" + std::to_string(i) + "]");
      }
      EXPECT_THROW(ctx.buildFunction(*integral), std::runtime_error) << "intOrder " << intOrder;
   }
}
