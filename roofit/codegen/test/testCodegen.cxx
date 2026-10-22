// Tests for the C++ code that the RooFit codegen backend generates.
// Author: Jonas Rembser, CERN 2026

#include <RooFit/CodegenContext.h>

#include <RooArgList.h>
#include <RooChebychev.h>
#include <RooConstVar.h>
#include <RooGaussian.h>
#include <RooLognormal.h>
#include <RooRealVar.h>

#include <TInterpreter.h>

#include <gtest/gtest.h>

#include <locale>
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
