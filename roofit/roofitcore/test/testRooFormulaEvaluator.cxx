// Tests for the JIT-free RooFit formula evaluation backend
// (RooFormulaParser + RooExprEvaluator), and its silent-TFormula-fallback
// contract.
// Author: Jonas Rembser, CERN 2026

#include "../src/RooFormulaUtils.h"
#include "../src/RooFormulaParser.h"
#include "../src/RooExprEvaluator.h"

#include <RooAddition.h>
#include <RooDataSet.h>
#include <RooFit/Evaluator.h>
#include <RooFormulaVar.h>
#include <RooGaussian.h>
#include <RooGenericPdf.h>
#include <RooGlobalFunc.h>
#include <RooRealVar.h>

#include <Math/PdfFuncMathCore.h>
#include <RConfigure.h> // for R__HAS_VDT
#include <ROOT/TestSupport.hxx>
#include <TFormula.h>
#include <TInterpreter.h>
#include <TMath.h>
#include <TSystem.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <locale>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef __linux__
#include <fstream>
#endif

namespace {

/// Bitwise comparison; any two NaNs count as equal.
bool sameBits(double a, double b)
{
   if (std::isnan(a) && std::isnan(b))
      return true;
   return std::memcmp(&a, &b, sizeof(double)) == 0;
}

/// Portable substitutes for the standard-library uniform distributions, whose
/// output sequences are implementation-defined: drawn through them, the fixed
/// seeds in this file produce different draws on libstdc++ and libc++, so a
/// failure seen on one platform does not reproduce on another and the coverage
/// silently differs. std::mt19937 itself is fully specified by the standard, so
/// drawing from it directly is reproducible everywhere.
double uniformDouble(std::mt19937 &rng, double lo, double hi)
{
   return lo + (hi - lo) * (static_cast<double>(rng()) * (1.0 / 4294967296.0)); // mt19937 yields [0, 2^32)
}

/// Uniform draw from [0, n).
int uniformInt(std::mt19937 &rng, int n)
{
   return static_cast<int>(rng() % static_cast<std::uint32_t>(n));
}

/// Comparison against a value that JIT-compiled code produced: a TFormula, or
/// C++ emitted from the program and compiled by the interpreter. Bitwise, but
/// tolerating floating-point contraction: cling compiles with clang's default
/// `-ffp-contract=on`, so wherever FMA is part of the target ISA (arm64, or an
/// x86-64 host with FMA) it fuses `a * b + c` into a single rounding step,
/// while the evaluator applies one instruction at a time and cannot contract.
/// That costs the last bits (a few ulps in practice); any real disagreement
/// -- a wrong precedence, a wrong function, a wrong integer rule -- changes
/// the value by far more than this tolerance, or turns it into a special
/// value, which is still required to match exactly.
bool agreesWithJit(double a, double b)
{
   if (sameBits(a, b)) {
      return true;
   }
   if (!std::isfinite(a) || !std::isfinite(b)) {
      return false;
   }
   return std::abs(a - b) <= 5e-14 * std::max(1.0, std::abs(b));
}

/// Compile with the JIT-free parser and evaluate. The expression must parse.
double astVal(std::string const &expr, std::vector<double> const &vars = {})
{
   auto prog = RooFormulaParser::compile(expr, vars.size());
   if (!prog) {
      ADD_FAILURE() << "expression unexpectedly failed to parse: " << expr;
      return std::numeric_limits<double>::quiet_NaN();
   }
   return RooExprEvaluator{prog}.eval(vars.data());
}

bool astParses(std::string const &expr, unsigned int nVars = 1)
{
   return RooFormulaParser::compile(expr, nVars) != nullptr;
}

/// Whether the evaluation engine is the JIT-free expression backend.
bool isAstBackend(RooFormulaEvaluator const &ev)
{
   return dynamic_cast<RooExprEvaluator const *>(&ev) != nullptr;
}

/// Set or unset ROOFIT_FORMULA_BACKEND for the lifetime of the object,
/// resetting the backend's read-once cache on both ends.
class ScopedBackendEnv {
public:
   ScopedBackendEnv(const char *value)
   {
      if (const char *old = std::getenv("ROOFIT_FORMULA_BACKEND"))
         _old = old;
      // gSystem instead of setenv()/unsetenv(), which MSVC does not have
      if (value)
         gSystem->Setenv("ROOFIT_FORMULA_BACKEND", value);
      else
         gSystem->Unsetenv("ROOFIT_FORMULA_BACKEND");
      RooFormulaInternal::resetFormulaBackendForTesting();
   }
   ~ScopedBackendEnv()
   {
      if (_old.empty())
         gSystem->Unsetenv("ROOFIT_FORMULA_BACKEND");
      else
         gSystem->Setenv("ROOFIT_FORMULA_BACKEND", _old.c_str());
      RooFormulaInternal::resetFormulaBackendForTesting();
   }

private:
   std::string _old;
};

} // namespace

// Precedence must match C++ exactly (that is what cling compiled). Each case
// evaluates to a different number under a plausible wrong precedence.
TEST(RooFormulaEvaluator, Precedence)
{
   EXPECT_EQ(astVal("1+2*3"), 7.0);
   EXPECT_EQ(astVal("6-8/2."), 2.0);
   EXPECT_EQ(astVal("2*3+4*5"), 26.0);
   EXPECT_EQ(astVal("1-2-3"), -4.0);    // left-associative
   EXPECT_EQ(astVal("10-2+3"), 11.0);   // left-associative
   EXPECT_EQ(astVal("16/4./2."), 2.0);  // left-associative
   EXPECT_EQ(astVal("1<2==1"), 1.0);    // (1<2)==1, not 1<(2==1)
   EXPECT_EQ(astVal("1||0&&0"), 1.0);   // 1||(0&&0), not (1||0)&&0
   EXPECT_EQ(astVal("1+1<1+3"), 1.0);   // (1+1)<(1+3)
   EXPECT_EQ(astVal("0<1==0<1"), 1.0);  // relational binds tighter than equality
   EXPECT_EQ(astVal("1?2:3+10"), 2.0);  // ternary is loosest: 1?2:(13)
   EXPECT_EQ(astVal("1?2:3?4:5"), 2.0); // right-associative
   EXPECT_EQ(astVal("0?2:3?4:5"), 4.0); // 0?2:(3?4:5)
   EXPECT_EQ(astVal("0?2:0?4:5"), 5.0);
   EXPECT_EQ(astVal("1?0?6:7:8"), 7.0); // nested in the middle operand
   EXPECT_EQ(astVal("2--3"), 5.0);      // binary minus then unary minus
   EXPECT_EQ(astVal("2- -3"), 5.0);
   EXPECT_EQ(astVal("-2*3"), -6.0);
   EXPECT_EQ(astVal("!0"), 1.0);
   EXPECT_EQ(astVal("!3"), 0.0);
   EXPECT_EQ(astVal("!!3"), 1.0);
   EXPECT_EQ(astVal("!1<2"), 1.0); // (!1)<2
}

// `^` (and `**`) follow TFormula::HandleExponentiation: right-associative,
// tighter than `*`, `/` and unary minus, one optional sign in the exponent,
// and the textual exponent `2` becomes TMath::Sq. Reference values were
// checked against TFormula/cling directly.
TEST(RooFormulaEvaluator, PowerOperator)
{
   EXPECT_EQ(astVal("2^3"), 8.0);
   EXPECT_EQ(astVal("2**3"), 8.0);
   EXPECT_EQ(astVal("-2^2"), -4.0);   // -(2^2), not (-2)^2
   EXPECT_EQ(astVal("2^3^2"), 512.0); // right-assoc: 2^(3^2)
   EXPECT_EQ(astVal("5*2^3"), 40.0);  // not (5*2)^3
   EXPECT_EQ(astVal("2^2*3"), 12.0);  // not 2^(2*3)
   EXPECT_EQ(astVal("2^3+1"), 9.0);
   EXPECT_EQ(astVal("x[0]^-2", {2.0}), 0.25); // sign in the exponent
   EXPECT_TRUE(sameBits(astVal("x[0]^-2.5", {2.0}), std::pow(2.0, -2.5)));
   EXPECT_TRUE(sameBits(astVal("x[0]^+2.5", {2.0}), std::pow(2.0, 2.5)));
   EXPECT_TRUE(sameBits(astVal("x[0]^-2^2", {2.0}), std::pow(2.0, -4.0))); // pow(x,-(2^2))
   EXPECT_TRUE(sameBits(astVal("x[0]^2", {1.3}), 1.3 * 1.3));              // TMath::Sq special case
   EXPECT_TRUE(sameBits(astVal("x[0]**2", {1.3}), 1.3 * 1.3));
   EXPECT_TRUE(sameBits(astVal("x[0]^2.0", {1.3}), std::pow(1.3, 2.0)));
   EXPECT_TRUE(sameBits(astVal("sin(x[0])^2", {0.7}), std::sin(0.7) * std::sin(0.7)));
   EXPECT_EQ(astVal("0x64^2"), 10000.0);
   EXPECT_TRUE(sameBits(astVal("x[0]^sin(x[0])", {2.0}), std::pow(2.0, std::sin(2.0))));
   EXPECT_EQ(astVal("2^x[0]*3", {3.0}), 24.0);
}

TEST(RooFormulaEvaluator, ComparisonsYieldExactly0Or1)
{
   EXPECT_TRUE(sameBits(astVal("x[0]>2", {3.0}), 1.0));
   EXPECT_TRUE(sameBits(astVal("x[0]>2", {1.0}), 0.0));
   EXPECT_TRUE(sameBits(astVal("(x[0]>2)*3", {3.0}), 3.0));
   EXPECT_TRUE(sameBits(astVal("x[0]<=0.5", {0.5}), 1.0));
   EXPECT_TRUE(sameBits(astVal("x[0]==0.25", {0.25}), 1.0));
   EXPECT_TRUE(sameBits(astVal("x[0]!=0.25", {0.25}), 0.0));
   // comparisons with NaN are false, like in C++
   const double nan = std::numeric_limits<double>::quiet_NaN();
   EXPECT_TRUE(sameBits(astVal("x[0]<1", {nan}), 0.0));
   EXPECT_TRUE(sameBits(astVal("x[0]>=1", {nan}), 0.0));
   EXPECT_TRUE(sameBits(astVal("x[0]==x[0]", {nan}), 0.0));
   // && and || convert their operands like C++ bool conversion (NaN is true)
   EXPECT_TRUE(sameBits(astVal("x[0]&&1", {nan}), 1.0));
   EXPECT_TRUE(sameBits(astVal("0.5&&0"), 0.0));
   EXPECT_TRUE(sameBits(astVal("0.0||0.25"), 1.0));
}

TEST(RooFormulaEvaluator, TernarySelectsExactValue)
{
   // Both branches are evaluated (no branching, matching a future vectorized
   // path), but the selected value is exact.
   EXPECT_TRUE(sameBits(astVal("x[0]>0 ? log(x[0]) : -1", {-2.0}), -1.0));
   EXPECT_TRUE(sameBits(astVal("x[0]>0 ? log(x[0]) : -1", {2.0}), std::log(2.0)));
   const double nan = std::numeric_limits<double>::quiet_NaN();
   EXPECT_TRUE(sameBits(astVal("x[0] ? 1 : 2", {nan}), 1.0)); // NaN converts to true
}

// Every allow-listed function in every accepted spelling, against the exact
// call the JIT-compiled code would have made.
TEST(RooFormulaEvaluator, Functions)
{
   // The argument must stay opaque to the optimizer: for a compile-time
   // constant, GCC folds std::sinh(v) & co. at compile time with MPFR's
   // correct rounding, which differs by one ulp from the libm call this test
   // file would emit and that the evaluator makes at run time.
   static volatile double vOpaque = 0.7311;
   const double v = vOpaque;
   auto check1 = [&](const char *expr, double expected) {
      EXPECT_TRUE(sameBits(astVal(std::string(expr) + "(x[0])", {v}), expected)) << expr;
   };
   check1("sqrt", std::sqrt(v));
   check1("std::sqrt", std::sqrt(v));
   check1("TMath::Sqrt", TMath::Sqrt(v));
   check1("exp", std::exp(v));
   check1("std::exp", std::exp(v));
   check1("TMath::Exp", TMath::Exp(v));
   check1("log", std::log(v));
   check1("std::log", std::log(v));
   check1("TMath::Log", TMath::Log(v));
   check1("log10", std::log10(v));
   check1("std::log10", std::log10(v));
   check1("TMath::Log10", TMath::Log10(v));
   check1("sin", std::sin(v));
   check1("std::sin", std::sin(v));
   check1("TMath::Sin", TMath::Sin(v));
   check1("cos", std::cos(v));
   check1("std::cos", std::cos(v));
   check1("TMath::Cos", TMath::Cos(v));
   check1("tan", std::tan(v));
   check1("std::tan", std::tan(v));
   check1("TMath::Tan", TMath::Tan(v));
   check1("asin", std::asin(v));
   check1("std::asin", std::asin(v));
   check1("TMath::ASin", TMath::ASin(v));
   check1("acos", std::acos(v));
   check1("std::acos", std::acos(v));
   check1("TMath::ACos", TMath::ACos(v));
   check1("atan", std::atan(v));
   check1("std::atan", std::atan(v));
   check1("TMath::ATan", TMath::ATan(v));
   check1("sinh", std::sinh(v));
   check1("std::sinh", std::sinh(v));
   check1("TMath::SinH", TMath::SinH(v));
   check1("cosh", std::cosh(v));
   check1("std::cosh", std::cosh(v));
   check1("TMath::CosH", TMath::CosH(v));
   check1("tanh", std::tanh(v));
   check1("std::tanh", std::tanh(v));
   check1("TMath::TanH", TMath::TanH(v));
   check1("asinh", std::asinh(v));
   check1("std::asinh", std::asinh(v));
   check1("TMath::ASinH", TMath::ASinH(v));
   check1("atanh", std::atanh(v));
   check1("std::atanh", std::atanh(v));
   check1("TMath::ATanH", TMath::ATanH(v));
   check1("floor", std::floor(v));
   check1("std::floor", std::floor(v));
   check1("TMath::Floor", TMath::Floor(v));
   check1("ceil", std::ceil(v));
   check1("std::ceil", std::ceil(v));
   check1("TMath::Ceil", TMath::Ceil(v));
   check1("erf", std::erf(v));
   check1("std::erf", std::erf(v));
   check1("TMath::Erf", TMath::Erf(v));
   check1("erfc", std::erfc(v));
   check1("std::erfc", std::erfc(v));
   check1("TMath::Erfc", TMath::Erfc(v));
   check1("tgamma", std::tgamma(v));
   check1("std::tgamma", std::tgamma(v));
   check1("lgamma", std::lgamma(v));
   check1("std::lgamma", std::lgamma(v));
   check1("abs", std::fabs(v));
   check1("std::abs", std::fabs(v));
   check1("fabs", std::fabs(v));
   check1("std::fabs", std::fabs(v));
   check1("TMath::Abs", TMath::Abs(v));
   check1("sq", v * v);
   check1("TMath::Sq", TMath::Sq(v));

   // acosh needs an argument >= 1
   static volatile double wOpaque = 1.5;
   const double w = wOpaque;
   EXPECT_TRUE(sameBits(astVal("acosh(x[0])", {w}), std::acosh(w)));
   EXPECT_TRUE(sameBits(astVal("std::acosh(x[0])", {w}), std::acosh(w)));
   EXPECT_TRUE(sameBits(astVal("TMath::ACosH(x[0])", {w}), TMath::ACosH(w)));

   // int() is a C++ functional cast: truncation towards zero
   EXPECT_EQ(astVal("int(2.7)"), 2.0);
   EXPECT_EQ(astVal("int(-2.7)"), -2.0);

   // TMath::SignBit uses std::signbit (note: true for -0.0)
   EXPECT_EQ(astVal("TMath::SignBit(x[0])", {-2.0}), 1.0);
   EXPECT_EQ(astVal("TMath::SignBit(x[0])", {2.0}), 0.0);
   EXPECT_EQ(astVal("TMath::SignBit(x[0])", {-0.0}), 1.0);

   // two-argument functions
   EXPECT_TRUE(sameBits(astVal("pow(x[0],x[1])", {1.7, 2.5}), std::pow(1.7, 2.5)));
   EXPECT_TRUE(sameBits(astVal("std::pow(x[0],x[1])", {1.7, 2.5}), std::pow(1.7, 2.5)));
   EXPECT_TRUE(sameBits(astVal("TMath::Power(x[0],x[1])", {1.7, 2.5}), TMath::Power(1.7, 2.5)));
   EXPECT_TRUE(sameBits(astVal("atan2(x[0],x[1])", {1.0, 2.0}), std::atan2(1.0, 2.0)));
   EXPECT_TRUE(sameBits(astVal("std::atan2(x[0],x[1])", {1.0, 2.0}), std::atan2(1.0, 2.0)));
   EXPECT_TRUE(sameBits(astVal("TMath::ATan2(x[0],x[1])", {1.0, 2.0}), TMath::ATan2(1.0, 2.0)));
   // TMath::ATan2 differs from std::atan2 for x == -0.0; keep each spelling exact
   EXPECT_TRUE(sameBits(astVal("TMath::ATan2(x[0],x[1])", {0.0, -0.0}), TMath::ATan2(0.0, -0.0)));
   EXPECT_TRUE(sameBits(astVal("atan2(x[0],x[1])", {0.0, -0.0}), std::atan2(0.0, -0.0)));
   EXPECT_TRUE(sameBits(astVal("fmod(x[0],x[1])", {7.5, 2.0}), std::fmod(7.5, 2.0)));
   EXPECT_TRUE(sameBits(astVal("std::fmod(x[0],x[1])", {7.5, 2.0}), std::fmod(7.5, 2.0)));
   EXPECT_TRUE(sameBits(astVal("min(x[0],x[1])", {1.0, 2.0}), 1.0));
   EXPECT_TRUE(sameBits(astVal("max(x[0],x[1])", {1.0, 2.0}), 2.0));
   EXPECT_TRUE(sameBits(astVal("min(2,3)"), 2.0));
   EXPECT_TRUE(sameBits(astVal("std::max(x[0],3.0)", {5.0}), 5.0));
   // std::min/max and TMath::Min/Max have opposite NaN behavior; both must be exact
   const double nan = std::numeric_limits<double>::quiet_NaN();
   EXPECT_TRUE(std::isnan(astVal("min(x[0],1.0)", {nan})));    // std::min(NaN, 1) = NaN
   EXPECT_TRUE(sameBits(astVal("min(1.0,x[0])", {nan}), 1.0)); // std::min(1, NaN) = 1
   EXPECT_TRUE(sameBits(astVal("TMath::Min(x[0],1.0)", {nan}), 1.0));
   EXPECT_TRUE(std::isnan(astVal("TMath::Min(1.0,x[0])", {nan})));
   // sign resolves to TMath::Sign, which is std::copysign for doubles
   EXPECT_TRUE(sameBits(astVal("sign(1.5,x[0])", {-3.0}), -1.5));
   EXPECT_TRUE(sameBits(astVal("sign(1.5,x[0])", {3.0}), 1.5));
   EXPECT_TRUE(sameBits(astVal("sign(1.5,x[0])", {-0.0}), -1.5)); // copysign semantics
   EXPECT_TRUE(sameBits(astVal("TMath::Sign(1.5,x[0])", {-0.0}), -1.5));
   EXPECT_TRUE(sameBits(astVal("x[0]*sign(1.,x[0]+2.)", {-3.0}), 3.0));

   // zero-argument constants
   EXPECT_TRUE(sameBits(astVal("TMath::Pi()"), TMath::Pi()));
   EXPECT_TRUE(sameBits(astVal("TMath::TwoPi()"), TMath::TwoPi()));
   EXPECT_TRUE(sameBits(astVal("TMath::PiOver2()"), TMath::PiOver2()));
   EXPECT_TRUE(sameBits(astVal("TMath::E()"), TMath::E()));

   // TMath::Gaus with 1 to 4 arguments (default args mean=0, sigma=1, norm=false)
   EXPECT_TRUE(sameBits(astVal("TMath::Gaus(x[0])", {1.0}), TMath::Gaus(1.0)));
   EXPECT_TRUE(sameBits(astVal("TMath::Gaus(x[0],2)", {1.0}), TMath::Gaus(1.0, 2)));
   EXPECT_TRUE(sameBits(astVal("TMath::Gaus(x[0],2,3)", {1.0}), TMath::Gaus(1.0, 2, 3)));
   EXPECT_TRUE(sameBits(astVal("TMath::Gaus(x[0],2,3,1)", {1.0}), TMath::Gaus(1.0, 2, 3, true)));
}

TEST(RooFormulaEvaluator, Literals)
{
   EXPECT_EQ(astVal("1e2"), 100.0);
   EXPECT_EQ(astVal("2e+3"), 2000.0);
   EXPECT_TRUE(sameBits(astVal("0.2e-6"), 0.2e-6));
   EXPECT_TRUE(sameBits(astVal("-7.94004e+06"), -7.94004e+06));
   EXPECT_EQ(astVal(".5"), 0.5);
   EXPECT_EQ(astVal("1."), 1.0);
   EXPECT_TRUE(sameBits(astVal("3.360779"), 3.360779));
   EXPECT_EQ(astVal("0x64"), 100.0); // hex, like cling
   EXPECT_EQ(astVal("010"), 8.0);    // octal, like cling
   EXPECT_EQ(astVal("0"), 0.0);
}

TEST(RooFormulaEvaluator, Variables)
{
   EXPECT_EQ(astVal("x[0]", {3.0}), 3.0);
   EXPECT_EQ(astVal("x[1]+2*x[0]", {3.0, 4.0}), 10.0);
   EXPECT_EQ(astVal("x[2]", {0.0, 0.0, 7.0}), 7.0);
   EXPECT_EQ(astVal("-x[0]", {3.0}), -3.0);

   // used-variable tracking
   auto prog = RooFormulaParser::compile("x[1]+1", 3);
   ASSERT_TRUE(prog);
   RooExprEvaluator ev{prog};
   EXPECT_FALSE(ev.usesVariable(0));
   EXPECT_TRUE(ev.usesVariable(1));
   EXPECT_FALSE(ev.usesVariable(2));
   EXPECT_EQ(ev.processedFormula(), "x[1]+1");
}

// Everything the JIT-free path does not support must return nullptr from the
// parser (and thus silently fall back to the TFormula backend).
TEST(RooFormulaEvaluator, FallbackTriggers)
{
   // unknown identifiers (undefined variables land here too)
   EXPECT_FALSE(astParses("y+1"));
   EXPECT_FALSE(astParses("unknownFunc(x[0])"));
   EXPECT_FALSE(astParses("pi"));        // TFormula constants are not part of the dialect
   EXPECT_FALSE(astParses("TMath::Pi")); // constants require the call syntax
   EXPECT_FALSE(astParses("ROOT::Math::normal_pdf(x[0],1.,2.)"));
   // wrong arity
   EXPECT_FALSE(astParses("erf(x[0],1.0)"));
   EXPECT_FALSE(astParses("TMath::Gaus(x[0],1,2,3,4)"));
   // `%` does not compile on doubles in cling, so TFormula rejects it today
   EXPECT_FALSE(astParses("x[0]%2"));
   // integer division truncates in cling; not reproduced in double arithmetic
   EXPECT_FALSE(astParses("1/2"));
   EXPECT_FALSE(astParses("7/2+x[0]"));
   EXPECT_FALSE(astParses("(x[0]>1)/2"));
   EXPECT_FALSE(astParses("int(x[0])/2"));
   EXPECT_FALSE(astParses("-1/2"));
   // ... but other all-integer arithmetic is value-identical in doubles
   EXPECT_EQ(astVal("7*3+2"), 23.0);
   EXPECT_EQ(astVal("7./2"), 3.5);
   EXPECT_EQ(astVal("7/2."), 3.5);
   // min/max with mixed int/double arguments does not compile in cling
   EXPECT_FALSE(astParses("min(x[0],3)"));
   EXPECT_FALSE(astParses("max(3,x[0])"));
   EXPECT_FALSE(astParses("TMath::Min(x[0],3)"));
   EXPECT_TRUE(astParses("min(x[0],3.0)"));
   // TFormula's textual `^` rewrite breaks next to `,` or `:`; both are
   // invalid in TFormula today and must stay invalid (fall back)
   EXPECT_FALSE(astParses("pow(x[0],2^3)"));
   EXPECT_FALSE(astParses("x[0]>0?1:x[0]^2"));
   EXPECT_TRUE(astParses("x[0]>0?1:(x[0]^2)"));
   // `^` with an explicit sign on a parenthesized exponent: TFormula's
   // textual rewrite pushes the sign onto only the first term inside the
   // group (`x^-(a+b)` compiles as pow(x,-(a)+b) today), so fall back
   EXPECT_FALSE(astParses("x[0]^-(x[0]+1)"));
   EXPECT_FALSE(astParses("x[0]^-(2)"));
   EXPECT_FALSE(astParses("x[0]^+(x[0]*2)"));
   EXPECT_TRUE(astParses("x[0]^(-x[0]-1)")); // sign inside the group is fine
   EXPECT_TRUE(astParses("x[0]^-2.5"));
   EXPECT_TRUE(astParses("x[0]^-sin(x[0])"));
   // cling resolves TMath::Sign with a bool first argument to the generic
   // template, which returns bool -- not copysign
   EXPECT_FALSE(astParses("sign(x[0]>1, x[0])"));
   EXPECT_FALSE(astParses("TMath::Sign(!x[0], x[0])"));
   EXPECT_FALSE(astParses("sign(TMath::SignBit(x[0]), x[0])"));
   EXPECT_TRUE(astParses("sign(1, x[0])")); // an int first argument is copysign-like
   // bool/int mixes in min/max do not compile in cling either
   EXPECT_FALSE(astParses("min(x[0]>1, 2)"));
   EXPECT_TRUE(astParses("min(x[0]>1, x[0]>2)"));
   EXPECT_TRUE(astParses("abs(x[0]>1)")); // abs(bool) promotes to int and works
   // cling compiles the TFormula code with -Wparentheses promoted to an
   // error, so a bare chained comparison is invalid in TFormula today
   EXPECT_FALSE(astParses("x[0]<x[0]<2"));
   EXPECT_FALSE(astParses("x[0]<=x[0]>1"));
   EXPECT_FALSE(astParses("1+x[0]<2<3"));
   EXPECT_TRUE(astParses("(x[0]<x[0])<2"));
   EXPECT_TRUE(astParses("x[0]<(x[0]<2)"));
   EXPECT_TRUE(astParses("x[0]==x[0]==1")); // ... while equality chains do compile
   EXPECT_TRUE(astParses("x[0]<2==x[0]>1"));
   // a textually adjacent `++` is TFormula's linear-combination separator
   // (one fit parameter per part), not an addition
   EXPECT_FALSE(astParses("2++3"));
   EXPECT_FALSE(astParses("x[0]++x[1]"));
   EXPECT_FALSE(astParses("++x[0]"));
   EXPECT_TRUE(astParses("2+ +3")); // with whitespace it is an ordinary addition
   // runs of three or more `-` (with or without whitespace) survive TFormula's
   // double-negation rewrite as a `--` pre-decrement, which cling rejects
   EXPECT_FALSE(astParses("2---3"));
   EXPECT_FALSE(astParses("---x[0]"));
   EXPECT_FALSE(astParses("x[0]- - -x[1]", 2));
   EXPECT_TRUE(astParses("2--3"));
   EXPECT_TRUE(astParses("x[0]- -x[1]", 2));
   EXPECT_TRUE(astParses("-(-(-x[0]))"));
   // TFormula parameters and parametrized shortcuts
   EXPECT_FALSE(astParses("[0]+x[0]"));
   EXPECT_FALSE(astParses("x[0]+[0]*0xaf"));
   EXPECT_FALSE(astParses("gaus"));
   EXPECT_FALSE(astParses("pol1"));
   EXPECT_FALSE(astParses("gaus(0)+pol1(3)"));
   // string literals, malformed numbers, stray characters
   EXPECT_FALSE(astParses("\"abc\""));
   EXPECT_FALSE(astParses("1e"));
   EXPECT_FALSE(astParses("1.5f"));
   EXPECT_FALSE(astParses("08"));
   EXPECT_FALSE(astParses("x[0] & 1"));
   EXPECT_FALSE(astParses("x[0] | 1"));
   EXPECT_FALSE(astParses("x[0] = 1"));
   EXPECT_FALSE(astParses(""));
   // malformed variable references
   EXPECT_FALSE(astParses("x[a]"));
   EXPECT_FALSE(astParses("x[0"));
   EXPECT_FALSE(astParses("x[0+1]"));
   // referencing more variables than provided
   EXPECT_FALSE(RooFormulaParser::compile("x[1]+1", 1));
   EXPECT_TRUE(RooFormulaParser::compile("x[1]+1", 2));
   // deep nesting falls back instead of overflowing the parser stack
   std::string deep(300, '(');
   deep += "1";
   deep += std::string(300, ')');
   EXPECT_FALSE(astParses(deep));
   std::string shallow(50, '(');
   shallow += "1";
   shallow += std::string(50, ')');
   EXPECT_TRUE(astParses(shallow));
}

// Int-typed constant arithmetic that leaves the int32 range wrapped around in
// cling's int arithmetic ("100000*100000" is 1410065408 there, not 1e10).
// That is not reproduced in double arithmetic: such formulas must fall back
// to the TFormula backend, so their values are unchanged. Integer literals
// too large for int32 have type long (decimal) or unsigned int (hex/octal) in
// C++ -- not the int typing tracked by the parser -- and fall back too.
TEST(RooFormulaEvaluator, IntOverflowFallback)
{
   EXPECT_FALSE(astParses("100000*100000"));
   EXPECT_FALSE(astParses("x[0]*(100000*100000)"));
   EXPECT_FALSE(astParses("2000000000+2000000000"));
   EXPECT_FALSE(astParses("2147483647+1"));
   EXPECT_FALSE(astParses("0-2000000000-2000000000"));
   EXPECT_FALSE(astParses("-(0-2147483647-1)")); // negating INT_MIN overflows
   EXPECT_TRUE(astParses("100000*100000."));     // double arithmetic is fine
   EXPECT_TRUE(astParses("100000.*100000"));
   EXPECT_EQ(astVal("2147483647*1"), 2147483647.0);
   EXPECT_EQ(astVal("0-2147483647-1"), -2147483648.0); // INT_MIN itself is in range
   // literals out of int32 range
   EXPECT_FALSE(astParses("3000000000"));
   EXPECT_FALSE(astParses("0x80000000"));
   EXPECT_FALSE(astParses("min(3000000000,2)"));
   EXPECT_TRUE(astParses("2147483647"));
   EXPECT_TRUE(astParses("0x7fffffff"));

   ScopedBackendEnv env{nullptr};
   RooRealVar x("x", "x", 5.0);

   // The fallback must reproduce cling's wrapped value exactly.
   RooArgList vars{x};
   auto f = RooFormulaUtils::makeFormulaEvaluator("f", "x*(100000*100000)", vars);
   EXPECT_FALSE(isAstBackend(*f));
   TFormula ref("ref", "x*(100000*100000)", /*addToGlobList=*/false);
   ASSERT_TRUE(ref.IsValid());
   double xv = 5.0;
   EXPECT_TRUE(sameBits(RooFormulaUtils::evalFormula(*f, vars), ref.EvalPar(&xv)));

   // A long-typed literal on its own is valid in cling with the same value;
   // the fallback keeps such formulas working.
   auto g = RooFormulaUtils::makeFormulaEvaluator("g", "3000000000+0*x", vars);
   EXPECT_FALSE(isAstBackend(*g));
   EXPECT_EQ(RooFormulaUtils::evalFormula(*g, vars), 3000000000.0);

   // min(long, int) does not compile in cling, so this formula threw at
   // construction before the JIT-free backend existed. It must still throw
   // instead of silently yielding 2.
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TFormula::InputFormulaIntoCling", "Error compiling formula expression in Cling",
                         false);
      diags.requiredDiag(kError, "TFormula::ProcessFormula", " is invalid", false);
      diags.optionalDiag(kError, "prepareMethod", "Can't compile function TFormula", false);
      diags.optionalDiag(kError, "cling", "no matching function", false);
      EXPECT_THROW(RooFormulaUtils::makeFormulaEvaluator("h", "min(3000000000,2)+x", vars), std::runtime_error);
   }
}

// "0x1e+2" is one single (invalid) pp-number in C++, not 0x1e + 2: cling
// refused it, so formula construction threw. The lexer must not split
// it into two tokens; such formulas keep failing via the TFormula fallback.
// TFormula strips whitespace before compiling, so "0x1e + 2" is equally
// invalid.
TEST(RooFormulaEvaluator, HexLiteralWithExponentSign)
{
   EXPECT_FALSE(astParses("0x1e+2"));
   EXPECT_FALSE(astParses("0x1E-2"));
   EXPECT_FALSE(astParses("0x1e + 2"));
   EXPECT_FALSE(astParses("x[0]+0x1e+2"));
   EXPECT_FALSE(astParses("0x1e+x[0]"));
   EXPECT_EQ(astVal("0x1e"), 30.0);
   EXPECT_EQ(astVal("0x1f+2"), 33.0); // final digit not e/E: ordinary addition
   EXPECT_EQ(astVal("2+0x1e"), 32.0); // sign before the literal is fine
   EXPECT_EQ(astVal("x[0]-0x1e", {5.0}), -25.0);
   EXPECT_EQ(astVal("(0x1e)+2"), 32.0); // ')' ends the pp-number

   // On the default backend the formula must still throw at construction,
   // exactly like before (via the TFormula fallback path).
   ScopedBackendEnv env{nullptr};
   RooRealVar x("x", "x", 1.0);
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TFormula::InputFormulaIntoCling", "Error compiling formula expression in Cling",
                         false);
      diags.requiredDiag(kError, "TFormula::ProcessFormula", " is invalid", false);
      diags.optionalDiag(kError, "prepareMethod", "Can't compile function TFormula", false);
      diags.optionalDiag(kError, "cling", "invalid suffix", false);
      EXPECT_THROW(RooFormulaUtils::makeFormulaEvaluator("f", "0x1e+2+x", RooArgList{x}), std::runtime_error);
   }
}

// A long chain of `^` recursed once per operator without a depth guard and
// segfaulted on parser stack overflow. It must fail the parse cleanly (and
// fall back) like deeply nested parentheses do.
TEST(RooFormulaEvaluator, DeepPowerChainFallsBack)
{
   std::string deep = "1";
   for (int i = 0; i < 200000; ++i) {
      deep += "^1";
   }
   EXPECT_FALSE(astParses(deep));
   // a modest chain still parses
   std::string shallow = "2";
   for (int i = 0; i < 50; ++i) {
      shallow += "^1";
   }
   EXPECT_TRUE(astParses(shallow));
   EXPECT_EQ(astVal(shallow), 2.0);
}

// Identical formula strings share one immutable program via the registry.
TEST(RooFormulaEvaluator, ProgramSharing)
{
   auto p1 = RooFormulaParser::compile("x[0]*2+sin(x[0])", 1);
   auto p2 = RooFormulaParser::compile("x[0]*2+sin(x[0])", 5);
   ASSERT_TRUE(p1);
   EXPECT_EQ(p1.get(), p2.get());
}

TEST(RooFormulaEvaluator, RooFormulaIntegration)
{
   // This test is about the default backend; shield it from an ambient
   // ROOFIT_FORMULA_BACKEND setting.
   ScopedBackendEnv env{nullptr};

   RooRealVar x("x", "x", 2.0);
   RooRealVar y("y", "y", 3.0);
   RooArgList vars{x, y};

   auto f = RooFormulaUtils::makeFormulaEvaluator("f", "x*y+sin(x)", vars);
   EXPECT_TRUE(isAstBackend(*f));
   EXPECT_TRUE(sameBits(RooFormulaUtils::evalFormula(*f, vars), 2.0 * 3.0 + std::sin(2.0)));

   // copies share the backend and evaluate identically
   auto fCopy = f->clone();
   EXPECT_TRUE(isAstBackend(*fCopy));
   EXPECT_TRUE(sameBits(RooFormulaUtils::evalFormula(*fCopy, vars), RooFormulaUtils::evalFormula(*f, vars)));

   // used-variable pruning in the owning classes works on the AST path
   RooFormulaVar g("g", "y*2", {x, y});
   EXPECT_EQ(g.dependents().size(), 1u);
   EXPECT_TRUE(g.dependents().find("y"));

   // the stored expression is the processed formula (persistence!)
   RooFormulaVar fVar("fVar", "x*y+sin(x)", {x, y});
   EXPECT_STREQ(fVar.expression(), "x[0]*x[1]+sin(x[0])");
   EXPECT_TRUE(sameBits(fVar.getVal(), 2.0 * 3.0 + std::sin(2.0)));

   // on the AST path, the expression is emitted as C++ for codegen instead of
   // JIT-compiling a TFormula; there is no JIT'd function name
   ASSERT_TRUE(f->canEmitCpp());
   auto varName = [](unsigned int i) { return "v[" + std::to_string(i) + "]"; };
   EXPECT_EQ(f->emitCpp(varName), "(((v[0]) * (v[1])) + std::sin((v[0])))");
   EXPECT_TRUE(f->uniqueFuncName().empty());
   EXPECT_TRUE(fVar.getUniqueFuncName().empty());
}

// Codegen indexes the pruned list of actually-used dependents. Since the
// owning classes prune unused variables and reindex the expression before the
// evaluation engine is created, emitFormulaCpp() must name the variables
// consistently with dependents().
TEST(RooFormulaEvaluator, EmitCppDependentRemap)
{
   ScopedBackendEnv env{nullptr};

   RooRealVar x("x", "x", 2.0);
   RooRealVar y("y", "y", 3.0);

   // x (list index 0) is unused, so y is dependents()[0].
   RooFormulaVar f("f", "y*2", {x, y});
   ASSERT_EQ(f.dependents().size(), 1u);
   std::string expr = f.emitFormulaCpp([](unsigned int i) { return "dep" + std::to_string(i); });
   EXPECT_EQ(expr, "((dep0) * 2.0)");
}

TEST(RooFormulaEvaluator, BackendOverride)
{
   RooRealVar x("x", "x", 2.0);

   RooArgList vars{x};
   {
      ScopedBackendEnv env{nullptr}; // default: AST with silent fallback
      auto f = RooFormulaUtils::makeFormulaEvaluator("f", "x*2", vars);
      EXPECT_TRUE(isAstBackend(*f));
      EXPECT_EQ(RooFormulaUtils::evalFormula(*f, vars), 4.0);
      // unsupported expression: silently falls back and still works
      auto g = RooFormulaUtils::makeFormulaEvaluator("g", "ROOT::Math::normal_pdf(x,1.,0.)", vars);
      EXPECT_FALSE(isAstBackend(*g));
      EXPECT_TRUE(sameBits(RooFormulaUtils::evalFormula(*g, vars), ROOT::Math::normal_pdf(2.0, 1.0, 0.0)));
      // the fallback backend cannot emit C++; codegen instead calls the
      // cling-JIT-compiled TFormula function by its unique name
      EXPECT_FALSE(g->canEmitCpp());
      EXPECT_TRUE(g->emitCpp([](unsigned int) { return std::string{"v"}; }).empty());
      EXPECT_FALSE(g->uniqueFuncName().empty());
   }
   {
      ScopedBackendEnv env{"tformula"};
      auto f = RooFormulaUtils::makeFormulaEvaluator("f", "x*2", vars);
      EXPECT_FALSE(isAstBackend(*f));
      EXPECT_EQ(RooFormulaUtils::evalFormula(*f, vars), 4.0);
   }
   {
      ScopedBackendEnv env{"ast"};
      auto f = RooFormulaUtils::makeFormulaEvaluator("f", "x*2", vars);
      EXPECT_TRUE(isAstBackend(*f));
      EXPECT_EQ(RooFormulaUtils::evalFormula(*f, vars), 4.0);
      // unsupported expression: fail loudly instead of falling back
      EXPECT_THROW(RooFormulaUtils::makeFormulaEvaluator("g", "ROOT::Math::normal_pdf(x,1.,0.)", vars),
                   std::runtime_error);
   }
}

// The public backend query on RooFormulaVar and RooGenericPdf: true when the
// formula is evaluated by the JIT-free AST backend (the default for supported
// expressions), false on the TFormula fallback backend.
TEST(RooFormulaEvaluator, PublicBackendQuery)
{
   RooRealVar a("a", "a", 1.1);
   RooRealVar x("x", "x", 2.0);
   RooRealVar b("b", "b", 0.3);

   {
      ScopedBackendEnv env{nullptr};
      RooFormulaVar f("f", "a*x+b", {a, x, b});
      EXPECT_TRUE(f.formulaUsesAstBackend());
      EXPECT_TRUE(f.getUniqueFuncName().empty());
      RooGenericPdf p("p", "a*x+b", {a, x, b});
      EXPECT_TRUE(p.formulaUsesAstBackend());
      EXPECT_TRUE(p.getUniqueFuncName().empty());
   }
   {
      ScopedBackendEnv env{"tformula"};
      RooFormulaVar f("f", "a*x+b", {a, x, b});
      EXPECT_FALSE(f.formulaUsesAstBackend());
      EXPECT_FALSE(f.getUniqueFuncName().empty());
      RooGenericPdf p("p", "a*x+b", {a, x, b});
      EXPECT_FALSE(p.formulaUsesAstBackend());
      EXPECT_FALSE(p.getUniqueFuncName().empty());
   }
}

namespace {

// The real-world formula corpus (see corpus.txt in the RooFit JIT-free
// formula planning notes): every RooFormulaVar/RooGenericPdf construction
// string in roofit/*/test/, factory strings, the RooFit tutorials, formulas
// RooFit itself generates (HistFactory, RooProdPdf, convolution bases, ...),
// and the RooFit-realistic subset of test/TFormulaParsingTests.h.
// The category-state entry "cat==cat::c1" appears in its processed form.
const char *const kCorpus[] = {
   "floor(x / 2.0) + 1.0",
   "floor(x / 2.0)",
   "floor(x)",
   "floor(x) + floor(y)",
   "floor(x) + 1.0",
   "TMath::Floor(x) + TMath::Ceil(x) + TMath::Abs(x) + TMath::Tan(x) + TMath::ASin(x / 2.) + TMath::ACos(x / 2.) + "
   "TMath::ATan(x) + TMath::Pi() + TMath::E()",
   "TMath::PiOver2() + TMath::TanH(x) + TMath::SinH(x) + TMath::Log10(2. + x)",
   "2 * x[0] * x[1]",
   "@0*0.2e-6 + @1*0.1",
   "x + y",
   "x + 2.0",
   "gauss",
   "std::exp(-0.5 * (x*x))",
   "x + shift",
   "std::pow(x,a)",
   "(x-5)*(x-5)*1.2",
   "x[0]",
   "a1 + x",
   "a1 + x + a2 *x*x",
   "exp(-2.*x)",
   "TMath::Gaus(x, 3, 2)",
   "x*x*x+1",
   "exp(-0.5*x)",
   "TMath::Gaus(x, 5, 0.7)",
   "TMath::Gaus(x, 8, 0.8)",
   "x[0]==1", // processed form of "cat==cat::c1"
   "cat",
   "catIndex > 0.5",
   "(1+0.1*abs(x)+sin(sqrt(abs(x*alpha+0.1))))",
   "sqrt(mean2)",
   "a0-a1*sqrt(10*abs(y))",
   "0.1*x",
   "0.9*x",
   "0.0*y",
   "0.1*y*y",
   "log10(@0)-log10(@1)",
   "(x*x+10)",
   "x*x+10",
   "(1-a)+a*cos((x-c)/b)",
   "((1-ax)+ax*cos((x-cx)/bx))*((1-ay)+ay*cos((y-cy)/by))",
   "0.5*(std::erf((t-1)/0.5)+1)",
   "exp(-@0/ @1)*cosh(@0*@2/2)",
   "exp(-@0/ @1)*sinh(@0*@2/2)",
   "@1/@0",
   "@0*@1*(1-2*@2)",
   "@0",
   "x - x + 1.0",
   "y - y + 1.0",
   "0.1 + x*(a + b*x)",
   "0.1 + x*(a + x*(b + x*(c + d * x)))",
   "log(a*x)",
   "ROOT::Math::breitwigner_pdf(x, b, a)",
   "ROOT::Math::gaussian_pdf(x, s, m)",
   "ROOT::Math::gaussian_pdf(theX, 1, 0)",
   "x*std::sqrt(x) + y*std::sqrt(y) + x*y",
   "x",
   "std::exp(-0.5*(x - mean1)^2/width^2)",
   "std::exp(-0.5*(x - mean2)^2/width^2)",
   "delta/(sigma*std::sqrt(TMath::TwoPi()))*std::exp(-0.5*(gamma+delta*TMath::ASinH((mass-mu)/sigma))*(gamma+delta*"
   "TMath::ASinH((mass-mu)/sigma)))/std::sqrt(1+(mass-mu)*(mass-mu)/(sigma*sigma))",
   "var*(par + 1)",
   "std::exp(-0.5*(x - mean) * (x - mean) / (sigma * sigma))",
   "x * y",
   "r + B + y",
   "2.7*@0",
   "x + 0",
   "b*(y<100)",
   "1.0 + 1.0*pow(@0,1) + 1.0*pow(@0,2)",
   "1.0 + x - x + y - y",
   "nbkg_func + 0*x",
   "-x[0]",
   "x[0] - x[0] + 1",
   "1 - x",
   "1 + x - x",
   "x + y + z",
   "mu+shift",
   "sigma*1.5",
   "sqrt(@0)",
   "2 * @0",
   "mu*S+B",
   "1+1./sqrt(n_off)",
   "1+1./sqrt(y)",
   "sqrt(n_off)",
   "sqrt(y)",
   "sqrt(y0)",
   "sig+bkg1",
   "sig+bkg2",
   "1+0.02*alpha_bkg",
   "1+0.02*alpha_bkg_A",
   "1+0.05*alpha_bkg_B",
   "0.5 * pow(1.2, e1)",
   "5 * pow(1.3, b1)",
   "2*sig*pow(1.2, beta)",
   "eff * sig + bkg",
   "0.07 * x + 2.0",
   "@0*@2+d",
   "a * x + c",
   "x[1] * x[0] + x[2]",
   "@1 * @0 + @2",
   "@0   *  2  *      @1 +   @2",
   "f * 3.0/2000. * x * x + (1 - f) / 20.",
   "x*x+1",
   "1/mean",
   "x^4+5*x^3+2*x^2+x+1",
   "1+sin(2*@0)",
   "acos(cpsi)",
   "abs(mean)<a",
   "0.5*(TMath::Erf((t-1)/0.5)+1)",
   "M/3.360779",
   "M/2",
   "x[0] / x[1]",
   "pow(@0,4) -5 * pow(@0,3) +5 * pow(@0,2) + 5 * pow(@0,1) - 6",
   "std::pow(std::sin(1.27 * x[2] * x[0] / x[1]), 2)",
   "1./sqrt(mu)",
   "1",
   "sqrt(2.)/sigma",
   "100 + mu * 1000",
   "x[0]*(pow(x[1],x[2])-1.)",
   "@0/@1",
   "1./x[0]",
   "pow((@0-@1),2)*@2",
   "pow(@0,2)*@1",
   "std::pow((@0-1.5),2) * @1",
   "exp(-@0/@1)",
   "exp(@0/@1)",
   "(@0/@1)*exp(-@0/@1)",
   "(@0/@1)*(@0/@1)*exp(-@0/@1)",
   "exp(-@0/@1)*sin(@0*@2)",
   "exp(@0/@1)*sin(@0*@2)",
   "exp(-@0/@1)*cos(@0*@2)",
   "exp(@0/@1)*cos(@0*@2)",
   "exp(-@0/@1)*sinh(@0*@2/2)",
   "exp(@0/@1)*sinh(@0*@2/2)",
   "exp(-@0/@1)*cosh(@0*@2/2)",
   "exp(@0/@1)*cosh(@0*@2/2)",
   "exp(-abs(@0)/@1)",
   "exp(-abs(@0)/@1)*sin(@0*@2)",
   "exp(-abs(@0)/@1)*cos(@0*@2)",
   "exp(-abs(@0)/@1)*sinh(@0*@2/2)",
   "exp(-abs(@0)/@1)*cosh(@0*@2/2)",
   "x^-2.5",
   "x^+2.5",
   "std::pow(x,2.5)",
   "TMath::Power(x,2.5)",
   "(x<190)?(-18.7813+(((2.49368+(10.3321/(x^0.881126)))*exp(-((x^-1.66603)/0.074916)))-(-17.5757*exp(-((x^-1464.26)/"
   "-7.94004e+06))))):(1.09984+(0.394544*exp(-(x/562.407))))",
   "x*sign(1.,x+2.)",
   "x*TMath::Sign(1,x+2)",
   "TMath::SignBit(x-2)",
   "sqrt(1.+sq(x))",
   "sq(1.+std::sqrt(x))",
   "ROOT::Math::normal_pdf(x,1,2)",
   "x - y",
   "x + 2*y + 3*z",
   "x[1] + 1",
   "x + 1",
   "0x64^2+x",
   "x^0x000c+1",
};

/// Bring a user-level corpus formula into the processed `x[i]` dialect that
/// RooFormulaUtils::processFormula() would produce, by assigning consecutive
/// indices to named variables (after any explicitly indexed x[i]/@i).
std::string normalizeCorpusEntry(std::string const &in, int &nVars)
{
   const std::size_t n = in.size();
   int maxIndex = -1;

   // First pass: find explicitly indexed references x[i] and @i.
   for (std::size_t i = 0; i < n; ++i) {
      if (in[i] == '@' && i + 1 < n && std::isdigit(static_cast<unsigned char>(in[i + 1]))) {
         maxIndex = std::max(maxIndex, in[i + 1] - '0');
      } else if (in[i] == 'x' && i + 1 < n && in[i + 1] == '[' &&
                 (i == 0 || !std::isalnum(static_cast<unsigned char>(in[i - 1])))) {
         maxIndex = std::max(maxIndex, std::atoi(in.c_str() + i + 2));
      }
   }

   std::map<std::string, int> named;
   std::string out;
   std::size_t i = 0;
   auto isIdentChar = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
   while (i < n) {
      const char c = in[i];
      if (std::isdigit(static_cast<unsigned char>(c)) ||
          (c == '.' && i + 1 < n && std::isdigit(static_cast<unsigned char>(in[i + 1])))) {
         // numeric literal (also hex): copy verbatim
         if (c == '0' && i + 1 < n && (in[i + 1] == 'x' || in[i + 1] == 'X')) {
            out += in[i++];
            out += in[i++];
            while (i < n && std::isalnum(static_cast<unsigned char>(in[i])))
               out += in[i++];
            continue;
         }
         while (i < n && (std::isdigit(static_cast<unsigned char>(in[i])) || in[i] == '.'))
            out += in[i++];
         if (i < n && (in[i] == 'e' || in[i] == 'E')) {
            std::size_t k = i + 1;
            if (k < n && (in[k] == '+' || in[k] == '-'))
               ++k;
            if (k < n && std::isdigit(static_cast<unsigned char>(in[k]))) {
               while (i < k)
                  out += in[i++];
               while (i < n && std::isdigit(static_cast<unsigned char>(in[i])))
                  out += in[i++];
            }
         }
         continue;
      }
      if (c == '@' && i + 1 < n && std::isdigit(static_cast<unsigned char>(in[i + 1]))) {
         out += "x[";
         ++i;
         while (i < n && std::isdigit(static_cast<unsigned char>(in[i])))
            out += in[i++];
         out += "]";
         continue;
      }
      if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
         std::size_t start = i;
         while (i < n && isIdentChar(in[i]))
            ++i;
         while (i + 2 < n && in[i] == ':' && in[i + 1] == ':' &&
                (std::isalpha(static_cast<unsigned char>(in[i + 2])) || in[i + 2] == '_')) {
            i += 2;
            while (i < n && isIdentChar(in[i]))
               ++i;
         }
         const std::string name = in.substr(start, i - start);
         if (name == "x" && i < n && in[i] == '[') {
            out += name;
            while (i < n && in[i] != ']')
               out += in[i++];
            if (i < n)
               out += in[i++]; // ']'
            continue;
         }
         // function call?
         std::size_t j = i;
         while (j < n && std::isspace(static_cast<unsigned char>(in[j])))
            ++j;
         if (j < n && in[j] == '(') {
            out += name;
            continue;
         }
         // named variable
         auto it = named.find(name);
         if (it == named.end())
            it = named.emplace(name, ++maxIndex).first;
         out += "x[" + std::to_string(it->second) + "]";
         continue;
      }
      out += in[i++];
   }
   nVars = maxIndex + 1;
   return out;
}

} // namespace

// First differential smoke test (the full differential campaign is Phase 3):
// every corpus entry that parses on the JIT-free path must evaluate
// identically to a directly constructed TFormula on random inputs (bitwise up
// to floating-point contraction in the JIT, see agreesWithJit()). Also report
// the corpus coverage of the JIT-free path.
TEST(RooFormulaEvaluator, DifferentialCorpus)
{
   int nTotal = 0;
   int nAst = 0;
   std::vector<std::string> fallbacks;

   for (const char *entry : kCorpus) {
      int nVars = 0;
      const std::string processed = normalizeCorpusEntry(entry, nVars);

      TFormula ref("ref", processed.c_str(), /*addToGlobList=*/false);
      ASSERT_TRUE(ref.IsValid()) << "corpus entry no longer valid in TFormula: " << entry
                                 << "\n  processed: " << processed;
      ++nTotal;

      auto prog = RooFormulaParser::compile(processed, nVars);
      if (!prog) {
         fallbacks.push_back(entry);
         continue;
      }
      ++nAst;
      RooExprEvaluator ast{prog};

      std::mt19937 rng{1234u + static_cast<unsigned>(nTotal)};
      for (int trial = 0; trial < 5; ++trial) {
         std::vector<double> pars(std::max(nVars, 1));
         for (double &p : pars) {
            p = trial == 0 ? 0.5 : trial == 1 ? 2.0 : uniformDouble(rng, -3.0, 3.0);
         }
         const double a = ast.eval(pars.data());
         const double t = ref.EvalPar(pars.data());
         EXPECT_TRUE(agreesWithJit(a, t))
            << "AST and TFormula disagree for: " << entry << "\n  processed: " << processed
            << "\n  ast = " << std::hexfloat << a << " tformula = " << t << std::defaultfloat;
      }
   }

   const double coverage = static_cast<double>(nAst) / nTotal;
   std::cout << "JIT-free evaluator corpus coverage: " << nAst << "/" << nTotal << " = " << 100. * coverage << "%\n";
   for (auto const &f : fallbacks) {
      std::cout << "  fallback: " << f << "\n";
   }
   RecordProperty("CorpusSize", nTotal);
   RecordProperty("CorpusOnAstPath", nAst);
   RecordProperty("CorpusCoveragePercent", static_cast<int>(std::round(100. * coverage)));
   // Hard floor only: the detailed number is reported above. A drop below 90%
   // means the allow-list lost something that real-world formulas need.
   EXPECT_GE(coverage, 0.9);
}

namespace {

using EmittedFunc = double (*)(double const *);

/// Declare the emitted expression as a function of the variable array `v` in
/// the interpreter and return a pointer to the compiled function.
EmittedFunc compileEmitted(std::string const &expr)
{
   static bool headersDeclared =
      gInterpreter->Declare("#include \"TMath.h\"\n#include <algorithm>\n#include <cmath>\n#include <limits>\n");
   if (!headersDeclared) {
      return nullptr;
   }
   static int counter = 0;
   const std::string fname = "rooFormulaEmitTestFunc" + std::to_string(counter++);
   const std::string code = "double " + fname + "(double const *v) { return " + expr + "; }";
   if (!gInterpreter->Declare(code.c_str())) {
      return nullptr;
   }
   return reinterpret_cast<EmittedFunc>(gInterpreter->Calc(("(void *) " + fname).c_str()));
}

} // namespace

// Emitted-code agreement (Phase 3 item 5, brought forward minimally): for
// every corpus expression the JIT-free parser accepts, emit the C++
// expression, compile it with the interpreter, and require agreement with AST
// evaluation on random inputs (bitwise up to floating-point contraction in the
// JIT, see agreesWithJit()). The C++ is emitted from the same
// instruction vector that eval() walks, so this directly validates the
// emission itself: the operator spellings, the function-name mapping table,
// and the exact round-trip of numeric literals.
TEST(RooFormulaEvaluator, EmittedCppDifferential)
{
   auto varName = [](unsigned int i) { return "v[" + std::to_string(i) + "]"; };

   int iEntry = 0;
   for (const char *entry : kCorpus) {
      ++iEntry;
      int nVars = 0;
      const std::string processed = normalizeCorpusEntry(entry, nVars);

      auto prog = RooFormulaParser::compile(processed, nVars);
      if (!prog) {
         continue; // fallback expressions have no emission; covered elsewhere
      }
      RooExprEvaluator ast{prog};

      const std::string expr = ast.emitCpp(varName);
      ASSERT_FALSE(expr.empty()) << entry;
      EmittedFunc fn = compileEmitted(expr);
      ASSERT_NE(fn, nullptr) << "emitted C++ failed to compile for: " << entry << "\n  emitted: " << expr;

      std::mt19937 rng{987u + static_cast<unsigned>(iEntry)};
      for (int trial = 0; trial < 5; ++trial) {
         std::vector<double> pars(std::max(nVars, 1));
         for (double &p : pars) {
            p = trial == 0 ? 0.5 : trial == 1 ? 2.0 : uniformDouble(rng, -3.0, 3.0);
         }
         const double a = ast.eval(pars.data());
         const double c = fn(pars.data());
         EXPECT_TRUE(agreesWithJit(a, c))
            << "AST and emitted C++ disagree for: " << entry << "\n  emitted: " << expr << "\n  ast = " << std::hexfloat
            << a << " emitted = " << c << std::defaultfloat;
      }
   }
}

// Numeric literals must survive emit -> compile -> eval bitwise, so they are
// emitted with max_digits10 (17) significant digits. A lossy emission (e.g.
// the default 6-digit %g formatting) would show up here: 0.1 and friends are
// not exactly representable.
TEST(RooFormulaEvaluator, EmittedLiteralRoundTrip)
{
   auto varName = [](unsigned int) { return std::string{"v[0]"}; };

   auto prog = RooFormulaParser::compile("x[0]*0.1+0.2e-6*3.360779", 1);
   ASSERT_TRUE(prog);
   RooExprEvaluator ast{prog};
   const std::string expr = ast.emitCpp(varName);
   // 0.1 must be emitted with enough digits for an exact round-trip
   EXPECT_NE(expr.find("0.10000000000000001"), std::string::npos) << expr;
   EmittedFunc fn = compileEmitted(expr);
   ASSERT_NE(fn, nullptr) << expr;
   for (double v : {0.3, 1.0, 7.7, 1e30, 1e-30, -2.5}) {
      EXPECT_TRUE(sameBits(ast.eval(&v), fn(&v))) << expr << " at v = " << v;
   }

   // integer-spelled literals must be emitted with double type: `2` in the
   // formula dialect is emitted as `2.0`
   auto prog2 = RooFormulaParser::compile("7./2", 0);
   ASSERT_TRUE(prog2);
   EXPECT_EQ(RooExprEvaluator{prog2}.emitCpp(varName), "(7.0 / 2.0)");
}

// The emitted C++ must not depend on the global locale: under a comma-decimal
// locale a default-constructed stream would format 0.5 as "0,5", corrupting
// the generated code. No comma-decimal OS locale is installed on every test
// machine, so the locale is built from a custom numpunct facet instead. The
// global locale is restored before any assertion can bail out of the test.
TEST(RooFormulaEvaluator, EmitCppLocaleIndependent)
{
   struct CommaPunct : std::numpunct<char> {
      char do_decimal_point() const override { return ','; }
   };

   const std::locale old = std::locale::global(std::locale{std::locale::classic(), new CommaPunct});
   std::string expr;
   std::string formatted;
   try {
      std::stringstream ss; // sanity check: the facet is actually in effect
      ss << 0.5;
      formatted = ss.str();
      auto prog = RooFormulaParser::compile("x[0]*0.5+1.25", 1);
      if (prog) {
         expr = RooExprEvaluator{prog}.emitCpp([](unsigned int i) { return "v[" + std::to_string(i) + "]"; });
      }
   } catch (...) {
      std::locale::global(old);
      throw;
   }
   std::locale::global(old);

   EXPECT_EQ(formatted, "0,5");
   EXPECT_NE(expr.find("0.5"), std::string::npos) << expr;
   EXPECT_NE(expr.find("1.25"), std::string::npos) << expr;
   EXPECT_EQ(expr.find(','), std::string::npos) << expr;
}

namespace {

/// A randomly generated expression in the processed `x[i]` dialect, together
/// with the C++ type (double, int, or bool) its cling compilation would have.
struct RandomExpr {
   enum class Type : std::uint8_t {
      Double,
      Int,
      Bool
   };
   std::string text;
   Type type = Type::Double;
   bool isIntegral() const { return type != Type::Double; }
};

/// Generates random well-formed expressions over exactly the grammar the
/// JIT-free parser supports: all binary and unary operators including `^` and
/// `**`, comparisons, logical operators, the ternary operator, and every
/// function spelling in the RooFormulaFunctions allow-list (sampled directly
/// from the table, so new entries are covered automatically, including the
/// zero-argument constants and the multi-arity TMath::Gaus).
///
/// The generator composes strings level by level along the C++ operator
/// precedence, so the string parses back to the generated structure and the
/// tracked double/int/bool typing is that of the actual parse. The typing is
/// used to steer around the deliberately unsupported constructs (truncating
/// integer division, min/max with mixed argument types, sign() with a
/// bool-typed first argument) by inserting a `1.0 *` factor. The textual
/// pitfalls of the dialect are avoided structurally: `^` expressions are
/// always parenthesized (a `^` operand adjacent to `,` or `:` is invalid in
/// TFormula), a signed exponent never starts with `(` (TFormula's rewrite
/// distributes the sign into the group), comparisons are not chained, stacked
/// signs are parenthesized (`++` is TFormula's linear-combination separator
/// and runs of three `-` are invalid), and the else branch of a ternary is
/// parenthesized (TFormula reads the `:` in front of it as the tail of a `::`
/// scope and then leaves a short function alias behind it unrewritten, so
/// `1?2:sq(1.)` does not compile while `1?2:(sq(1.))` does). Everything the
/// generator produces must therefore parse on the AST path *and* be a valid
/// TFormula, so the differential test can compare every expression with no
/// skips.
class RandomExprGenerator {
public:
   RandomExprGenerator(unsigned int seed, unsigned int nVars) : _rng{seed}, _nVars{nVars} {}

   RandomExpr gen(int depth) { return genTernary(depth); }

private:
   double chance() { return uniformDouble(_rng, 0., 1.); }
   int pick(int n) { return uniformInt(_rng, n); }

   /// Turn an int- or bool-typed operand into a double-typed one with the
   /// same value.
   RandomExpr forceDouble(RandomExpr e)
   {
      if (e.isIntegral()) {
         e.text = "(1.0 * (" + e.text + "))";
         e.type = RandomExpr::Type::Double;
      }
      return e;
   }

   RandomExpr genTernary(int depth)
   {
      if (depth > 0 && chance() < 0.10) {
         RandomExpr c = genBinary(1, depth - 1);
         RandomExpr a = genTernary(depth - 1);
         RandomExpr b = genTernary(depth - 1);
         RandomExpr out;
         out.text = c.text + " ? " + a.text + " : (" + b.text + ")";
         if (a.type == RandomExpr::Type::Double || b.type == RandomExpr::Type::Double) {
            out.type = RandomExpr::Type::Double;
         } else if (a.type == RandomExpr::Type::Bool && b.type == RandomExpr::Type::Bool) {
            out.type = RandomExpr::Type::Bool;
         } else {
            out.type = RandomExpr::Type::Int;
         }
         return out;
      }
      return genBinary(1, depth);
   }

   /// Precedence levels: 1 `||`, 2 `&&`, 3 `== !=`, 4 `< <= > >=`, 5 `+ -`,
   /// 6 `* /`; operands of a level come from the next-tighter level, so no
   /// parentheses are needed to reproduce the intended structure.
   RandomExpr genBinary(int level, int depth)
   {
      if (level > 6)
         return genUnary(depth);
      RandomExpr lhs = genBinary(level + 1, depth);
      static constexpr double probs[7] = {0.0, 0.06, 0.06, 0.08, 0.10, 0.35, 0.35};
      while (depth > 0 && chance() < probs[level]) {
         --depth;
         RandomExpr rhs = genBinary(level + 1, depth);
         const char *op = nullptr;
         switch (level) {
         case 1:
            op = "||";
            lhs.type = RandomExpr::Type::Bool;
            break;
         case 2:
            op = "&&";
            lhs.type = RandomExpr::Type::Bool;
            break;
         case 3:
            op = pick(2) ? "==" : "!=";
            lhs.type = RandomExpr::Type::Bool;
            break;
         case 4:
            switch (pick(4)) {
            case 0: op = "<"; break;
            case 1: op = "<="; break;
            case 2: op = ">"; break;
            default: op = ">="; break;
            }
            lhs.type = RandomExpr::Type::Bool;
            break;
         case 5:
            op = pick(2) ? "+" : "-";
            lhs.type = lhs.isIntegral() && rhs.isIntegral() ? RandomExpr::Type::Int : RandomExpr::Type::Double;
            break;
         case 6:
            if (pick(3) == 0) {
               // avoid truncating integer division (unsupported: falls back)
               if (lhs.isIntegral())
                  rhs = forceDouble(rhs);
               op = "/";
               lhs.type = RandomExpr::Type::Double;
            } else {
               op = "*";
               lhs.type = lhs.isIntegral() && rhs.isIntegral() ? RandomExpr::Type::Int : RandomExpr::Type::Double;
            }
            break;
         }
         lhs.text += std::string{" "} + op + " " + rhs.text;
         // no bare chained comparison (`a < b < c`): invalid in TFormula,
         // where cling compiles with -Wparentheses promoted to an error
         if (level == 4)
            break;
      }
      return lhs;
   }

   RandomExpr genUnary(int depth)
   {
      if (depth > 0 && chance() < 0.15) {
         RandomExpr e = genUnary(depth - 1);
         const int which = pick(3); // -, +, !
         if (which == 2) {
            e.text = "!(" + e.text + ")";
            e.type = RandomExpr::Type::Bool;
         } else {
            // parenthesize an operand that starts with a sign or `!`, to
            // avoid the `++` / `---` pitfalls (see FallbackTriggers)
            if (e.text[0] == '-' || e.text[0] == '+' || e.text[0] == '!')
               e.text = "(" + e.text + ")";
            e.text = (which == 0 ? "-" : "+") + e.text;
            if (e.isIntegral())
               e.type = RandomExpr::Type::Int; // bool promotes to int
         }
         return e;
      }
      return genPower(depth);
   }

   /// `^`/`**` exponentiation, always parenthesized as a whole; base and
   /// exponent are primaries (with one optional sign on the exponent),
   /// matching how the operator appears in real formulas and staying within
   /// what TFormula's textual pow() rewrite scans correctly.
   RandomExpr genPower(int depth)
   {
      if (depth > 0 && chance() < 0.12) {
         RandomExpr base = genPrimary(depth - 1);
         std::string sign;
         if (chance() < 0.3)
            sign = pick(2) ? "-" : "+";
         RandomExpr exponent = genPrimary(depth - 1);
         // a signed exponent must not start with `(`: TFormula's textual
         // rewrite distributes the sign into the group (falls back)
         if (!sign.empty() && exponent.text[0] == '(')
            sign.clear();
         const char *op = pick(4) == 0 ? "**" : "^";
         RandomExpr out;
         out.text = "(" + base.text + op + sign + exponent.text + ")";
         out.type = RandomExpr::Type::Double;
         return out;
      }
      return genPrimary(depth);
   }

   RandomExpr genPrimary(int depth)
   {
      const double r = chance();
      if (depth <= 0 || r < 0.4) {
         if (chance() < 0.55) {
            RandomExpr out;
            out.text = "x[" + std::to_string(pick(_nVars)) + "]";
            return out;
         }
         return genLiteral();
      }
      if (r < 0.6) {
         RandomExpr e = genTernary(depth - 1);
         e.text = "(" + e.text + ")";
         return e;
      }
      return genCall(depth - 1);
   }

   RandomExpr genLiteral()
   {
      static const char *const kIntLiterals[] = {"0", "1", "2", "3", "7", "42", "0x1f"};
      static const char *const kDoubleLiterals[] = {"0.5",  "1.5",  "2.5",    "3.360779", "0.25",          ".5",
                                                    "1.",   "1e-3", "2e+2",   "1e300",    "1e-300",        "0.1",
                                                    "1e30", "13.7", "1.5e-8", "2.0",      "6.62607015e-34"};
      RandomExpr out;
      if (chance() < 0.4) {
         out.text = kIntLiterals[pick(std::end(kIntLiterals) - std::begin(kIntLiterals))];
         out.type = RandomExpr::Type::Int;
      } else {
         out.text = kDoubleLiterals[pick(std::end(kDoubleLiterals) - std::begin(kDoubleLiterals))];
      }
      return out;
   }

   /// A call to a random entry of the actual allow-list table.
   RandomExpr genCall(int depth)
   {
      auto const *tab = RooFormulaFunctions::table();
      auto const &entry = tab[pick(static_cast<int>(RooFormulaFunctions::tableSize()))];

      RandomExpr args[4];
      for (unsigned int i = 0; i < entry.arity; ++i) {
         args[i] = genTernary(depth);
      }

      // `int(x)` is a C++ functional cast, and converting a NaN or an
      // out-of-int-range double is undefined behaviour: cling and the compiler
      // that built the evaluator are then free to disagree, and they do. Guard
      // the argument into the safe range so that the generated expression is
      // well-defined. The guard is NaN-safe: a comparison with NaN is false,
      // so a NaN argument selects the constant.
      if (std::strcmp(entry.name, "int") == 0) {
         const std::string arg = "(" + args[0].text + ")";
         args[0].text = "(" + arg + " > -1e9 && " + arg + " < 1e9 ? " + arg + " : (0.0))";
         args[0].type = RandomExpr::Type::Double;
      }

      using RooFormulaFunctions::TypeRule;
      using Type = RandomExpr::Type;
      Type type = Type::Double;
      switch (entry.rule) {
      case TypeRule::Double: break;
      case TypeRule::SameAsFirstArg:
         if (entry.arity >= 1)
            type = args[0].type == Type::Bool ? Type::Int : args[0].type;
         break;
      case TypeRule::Int: type = Type::Int; break;
      case TypeRule::Bool: type = Type::Bool; break;
      case TypeRule::Sign:
         // sign() with a bool first argument is not copysign in cling (falls back)
         if (args[0].type == Type::Bool)
            args[0] = forceDouble(args[0]);
         type = args[0].type;
         break;
      case TypeRule::MinMax:
         // mixed argument types do not compile in cling (falls back)
         if (args[0].type != args[1].type) {
            args[0] = forceDouble(args[0]);
            args[1] = forceDouble(args[1]);
         }
         type = args[0].type;
         break;
      }

      RandomExpr out;
      out.type = type;
      out.text = std::string{entry.name} + "(";
      for (unsigned int i = 0; i < entry.arity; ++i) {
         if (i > 0)
            out.text += ", ";
         out.text += args[i].text;
      }
      out.text += ")";
      return out;
   }

   std::mt19937 _rng;
   unsigned int _nVars = 0;
};

constexpr unsigned int kRandomExprSeed = 20260827u; // fixed: failures must reproduce
constexpr unsigned int kRandomExprVars = 3;

/// Fill the input vector for the given trial: the first trials use edge
/// values (0, +-1, very large, very small, negatives that drive sqrt/log/pow
/// into NaN territory), later ones random values from the given generator.
void fillRandomInputs(int trial, double *pars, std::mt19937 &rng)
{
   switch (trial) {
   case 0:
      for (unsigned int j = 0; j < kRandomExprVars; ++j)
         pars[j] = 0.0;
      break;
   case 1:
      for (unsigned int j = 0; j < kRandomExprVars; ++j)
         pars[j] = j == 1 ? -1.0 : 1.0;
      break;
   case 2:
      pars[0] = 1e300;
      pars[1] = 1e-300;
      pars[2] = -1.0;
      break;
   case 3:
      pars[0] = -2.5;
      pars[1] = -1e300;
      pars[2] = -1e-300;
      break;
   default: {
      const double scale = trial % 2 ? 3.0 : 50.0;
      for (unsigned int j = 0; j < kRandomExprVars; ++j)
         pars[j] = uniformDouble(rng, -scale, scale);
      break;
   }
   }
}

} // namespace

// Random-expression differential campaign: several hundred generated
// expressions, each evaluated through both backends on several input vectors
// including edge cases. Results must agree bitwise up to floating-point
// contraction in the JIT (see agreesWithJit(); NaN counts as equal to NaN).
// The generator and the input draws use only portable arithmetic on the
// engine, so the fixed seeds produce the same expressions and the same inputs
// on every platform and a failure reproduces exactly; to investigate one,
// print `expr.text` and the hexfloat values from the failure message.
TEST(RooFormulaEvaluator, RandomExpressionDifferential)
{
   constexpr int nExprs = 500;
   constexpr int nTrials = 8;

   RandomExprGenerator gen{kRandomExprSeed, kRandomExprVars};
   std::mt19937 inputRng{987654u};

   for (int iExpr = 0; iExpr < nExprs; ++iExpr) {
      const RandomExpr expr = gen.gen(4);

      std::string error;
      auto prog = RooFormulaParser::compile(expr.text, kRandomExprVars, &error);
      ASSERT_TRUE(prog) << "generated expression unexpectedly failed to parse: " << expr.text << "\n  error: " << error;
      RooExprEvaluator ast{prog};

      TFormula ref("ref", expr.text.c_str(), /*addToGlobList=*/false);
      ASSERT_TRUE(ref.IsValid()) << "generated expression is invalid in TFormula (the generator must only produce "
                                    "expressions valid in both dialects): "
                                 << expr.text;

      for (int trial = 0; trial < nTrials; ++trial) {
         double pars[kRandomExprVars];
         fillRandomInputs(trial, pars, inputRng);
         const double a = ast.eval(pars);
         const double t = ref.EvalPar(pars);
         EXPECT_TRUE(agreesWithJit(a, t))
            << "AST and TFormula backends disagree for: " << expr.text << "\n  inputs: " << pars[0] << " " << pars[1]
            << " " << pars[2] << "\n  ast = " << std::hexfloat << a << " tformula = " << t << std::defaultfloat;
      }
   }
}

// Emitted-code agreement on random expressions (the corpus counterpart is
// EmittedCppDifferential): emit the C++ for a sample of the same generated
// expression stream, compile it with the interpreter, and require agreement
// with AST evaluation (bitwise up to floating-point contraction in the JIT,
// see agreesWithJit()). Uses the same seed as
// RandomExpressionDifferential, so this covers a prefix of the same
// expressions.
TEST(RooFormulaEvaluator, EmittedCppRandomExpressions)
{
   constexpr int nExprs = 150;
   constexpr int nTrials = 6;

   auto varName = [](unsigned int i) { return "v[" + std::to_string(i) + "]"; };

   RandomExprGenerator gen{kRandomExprSeed, kRandomExprVars};
   std::mt19937 inputRng{192837u};

   for (int iExpr = 0; iExpr < nExprs; ++iExpr) {
      const RandomExpr expr = gen.gen(4);

      auto prog = RooFormulaParser::compile(expr.text, kRandomExprVars);
      ASSERT_TRUE(prog) << expr.text;
      RooExprEvaluator ast{prog};

      const std::string emitted = ast.emitCpp(varName);
      ASSERT_FALSE(emitted.empty()) << expr.text;
      EmittedFunc fn = compileEmitted(emitted);
      ASSERT_NE(fn, nullptr) << "emitted C++ failed to compile for: " << expr.text << "\n  emitted: " << emitted;

      for (int trial = 0; trial < nTrials; ++trial) {
         double pars[kRandomExprVars];
         fillRandomInputs(trial, pars, inputRng);
         const double a = ast.eval(pars);
         const double c = fn(pars);
         EXPECT_TRUE(agreesWithJit(a, c))
            << "AST and emitted C++ disagree for: " << expr.text << "\n  emitted: " << emitted
            << "\n  inputs: " << pars[0] << " " << pars[1] << " " << pars[2] << "\n  ast = " << std::hexfloat << a
            << " emitted = " << c << std::defaultfloat;
      }
   }
}

namespace {

// The vectorized doEval() path evaluates exp/log/sin/cos with the fast VDT
// implementations when ROOT is built with VDT (like all RooBatchCompute pdf
// kernels), in which case batch and per-event scalar results agree within
// RooBatchCompute's own batch-vs-scalar tolerance (_toleranceCompareBatches
// in roofit/test/vectorisedPDFs/VectorisedPDFTests.h). Without VDT, every
// vectorized operation is the exact same double-precision operation the
// scalar evaluator applies, so agreement must be bitwise.
//
// The VDT functions are only approximations over their normal argument range
// and do not reproduce libm's special values: vdt::fast_log() of zero or of a
// negative number returns a finite garbage value instead of -Inf or NaN, and
// that value then propagates through the rest of the expression. Special
// values are therefore not comparable at all on a VDT build; their bitwise
// propagation is checked on non-VDT builds (and holds unconditionally for the
// scalar path, which always calls the exact libm functions).
bool batchAgrees(double batch, double ref)
{
   if (sameBits(batch, ref)) {
      return true;
   }
#ifdef R__HAS_VDT
   if (!std::isfinite(batch) || !std::isfinite(ref)) {
      return true;
   }
   return std::abs(batch - ref) <= 5e-14 * std::max(1.0, std::abs(ref));
#else
   return false;
#endif
}

/// Batch-evaluate `processedExpr` (in the x[i] dialect) through
/// RooFit::Evaluator, exercising RooFormula::doEval() with mixed span sizes:
/// x[0] gets the vector input `xData` (span of size N), all other x[i] are
/// scalar parameters (spans of size 1) with value scalarVals[i], and one
/// trailing unused dependent exercises the empty-span handling.
std::vector<double> batchEvalFormula(std::string const &processedExpr, std::vector<double> const &xData,
                                     std::vector<double> const &scalarVals)
{
   const std::size_t nVars = scalarVals.size();
   RooArgList vars;
   std::vector<std::unique_ptr<RooRealVar>> owned;
   for (std::size_t i = 0; i <= nVars; ++i) { // one extra, unused dependent
      const std::string name = "v" + std::to_string(i);
      const double val = i == 0 ? (xData.empty() ? 1.0 : xData[0]) : (i < nVars ? scalarVals[i] : 0.5);
      owned.emplace_back(std::make_unique<RooRealVar>(name.c_str(), name.c_str(), val, -1e300, 1e300));
      vars.add(*owned.back());
   }
   RooFormulaVar f("f", processedExpr.c_str(), vars);
   RooFit::Evaluator ev(f);
   ev.setInput("v0", {xData.data(), xData.size()}, false);
   std::span<const double> out = ev.run();
   return {out.begin(), out.end()};
}

/// Per-event scalar reference for the same inputs, through the same compiled
/// program that the vectorized path executes.
std::vector<double> scalarRefFormula(std::string const &processedExpr, std::vector<double> const &xData,
                                     std::vector<double> const &scalarVals)
{
   auto prog = RooFormulaParser::compile(processedExpr, scalarVals.size() + 1);
   if (!prog) {
      ADD_FAILURE() << "expression unexpectedly failed to parse: " << processedExpr;
      return {};
   }
   RooExprEvaluator ev{prog};
   std::vector<double> pars(scalarVals.size() + 1);
   for (std::size_t i = 1; i < scalarVals.size(); ++i) {
      pars[i] = scalarVals[i];
   }
   std::vector<double> out(xData.size());
   for (std::size_t i = 0; i < xData.size(); ++i) {
      pars[0] = xData[i];
      out[i] = ev.eval(pars.data());
   }
   return out;
}

/// Compare a batch output against the per-event reference. An output of size
/// 1 means the evaluator collapsed the case to a single value (x[0] unused or
/// a size-1 input span): compare only the first reference value then.
void expectBatchMatches(std::vector<double> const &out, std::vector<double> const &ref, std::string const &what)
{
   ASSERT_FALSE(ref.empty()) << what;
   if (out.size() == 1) {
      EXPECT_TRUE(batchAgrees(out[0], ref[0]))
         << what << "\n  batch = " << std::hexfloat << out[0] << " scalar = " << ref[0] << std::defaultfloat;
      return;
   }
   ASSERT_EQ(out.size(), ref.size()) << what;
   for (std::size_t i = 0; i < out.size(); ++i) {
      ASSERT_TRUE(batchAgrees(out[i], ref[i])) << what << "\n  event " << i << ": batch = " << std::hexfloat << out[i]
                                               << " scalar = " << ref[i] << std::defaultfloat;
   }
}

} // namespace

// Differential test of the vectorized doEval() against per-event scalar
// evaluation on batch sizes around the bufferSize=64 chunking boundaries,
// with an expression covering ternary, comparisons, logical operators and the
// vectorizable functions, and with input values that produce NaN (log of a
// negative number, sqrt of a negative number) and Inf (division by zero), to
// check that special values propagate identically.
TEST(RooFormulaEvaluator, VectorizedDoEvalEdgeSizes)
{
   ScopedBackendEnv env{nullptr};

   const std::string expr = "x[0]*x[1] + sin(x[0])*cos(x[2]) + (x[0] > 0.5 ? log(x[0] - 1.0) : -x[0])"
                            " + sqrt(x[0] - 2.0) + (x[0] != 0.0 && x[1] > 0.0) + 1.0/x[0] + exp(-x[0])";
   const std::vector<double> scalarVals{0.0, 1.5, 0.7};

   std::mt19937 rng{20260828u};

   for (std::size_t n : {1u, 2u, 63u, 64u, 65u, 127u, 128u, 1000u}) {
      std::vector<double> xData(n);
      for (double &v : xData) {
         v = uniformDouble(rng, -3.0, 8.0);
      }
      // special values: division by zero -> Inf, negatives -> NaN from log/sqrt
      if (n > 2) {
         xData[0] = 0.0;
         xData[1] = -1.0;
         xData[2] = 1.75; // log(0.75) finite, sqrt(-0.25) NaN
      }
      auto out = batchEvalFormula(expr, xData, scalarVals);
      auto ref = scalarRefFormula(expr, xData, scalarVals);
      expectBatchMatches(out, ref, "n = " + std::to_string(n));
   }
}

// Differential doEval() over the real-world corpus: every entry the JIT-free
// parser accepts is evaluated through the batch path (x[0] vectorized, other
// variables scalar, plus an unused dependent) and compared per event against
// scalar evaluation of the same program.
TEST(RooFormulaEvaluator, VectorizedDoEvalCorpus)
{
   ScopedBackendEnv env{nullptr};

   constexpr std::size_t nEvents = 197; // 3 full chunks plus a remainder

   std::mt19937 rng{555u};

   for (const char *entry : kCorpus) {
      int nVars = 0;
      const std::string processed = normalizeCorpusEntry(entry, nVars);
      auto prog = RooFormulaParser::compile(processed, std::max(nVars, 1));
      if (!prog) {
         continue; // TFormula-fallback expressions are not vectorized
      }

      std::vector<double> xData(nEvents);
      for (double &v : xData) {
         v = uniformDouble(rng, 0.1, 3.0);
      }
      xData[0] = 0.0;
      xData[1] = -1.5;

      std::vector<double> scalarVals(std::max(nVars, 1));
      for (std::size_t i = 1; i < scalarVals.size(); ++i) {
         scalarVals[i] = uniformDouble(rng, 0.1, 3.0);
      }

      auto out = batchEvalFormula(processed, xData, scalarVals);
      auto ref = scalarRefFormula(processed, xData, scalarVals);
      expectBatchMatches(out, ref, std::string{"corpus entry: "} + entry + "\n  processed: " + processed);
   }
}

// Differential doEval() on randomly generated expressions (same generator and
// seed as RandomExpressionDifferential), covering ternary/comparison/logical
// operators and the whole function allow-list in random combinations.
TEST(RooFormulaEvaluator, VectorizedDoEvalRandomExpressions)
{
   ScopedBackendEnv env{nullptr};

   constexpr int nExprs = 150;
   constexpr std::size_t nEvents = 130; // two full chunks plus a remainder

   RandomExprGenerator gen{kRandomExprSeed, kRandomExprVars};
   std::mt19937 inputRng{424242u};

   for (int iExpr = 0; iExpr < nExprs; ++iExpr) {
      const RandomExpr expr = gen.gen(4);

      std::vector<double> xData(nEvents);
      for (double &v : xData) {
         v = uniformDouble(inputRng, -5.0, 5.0);
      }
      xData[0] = 0.0;
      xData[1] = 1e300;
      xData[2] = -1e-300;

      std::vector<double> scalarVals(kRandomExprVars);
      for (std::size_t i = 1; i < scalarVals.size(); ++i) {
         scalarVals[i] = uniformDouble(inputRng, -5.0, 5.0);
      }

      auto out = batchEvalFormula(expr.text, xData, scalarVals);
      auto ref = scalarRefFormula(expr.text, xData, scalarVals);
      expectBatchMatches(out, ref, "random expression: " + expr.text);
   }
}

// A large batch through the vectorized path: the ~30-instruction expression
// from the Phase 5 benchmarks over 10^6 events, compared per event.
TEST(RooFormulaEvaluator, VectorizedDoEvalLargeBatch)
{
   ScopedBackendEnv env{nullptr};

   const std::string expr = "0.5*exp(-0.5*(x[0]-x[1])*(x[0]-x[1])/(x[2]*x[2])) + 0.3*sin(0.5*x[0]+x[1])*cos(x[0]*x[2])"
                            " + 0.2/(1.0+x[0]*x[0]) + sqrt(abs(x[0]*x[1])+1.0) + 0.1*log(1.0+exp(-x[0]))";
   const std::vector<double> scalarVals{0.0, 1.0, 2.0};

   constexpr std::size_t nEvents = 1000000;
   std::vector<double> xData(nEvents);
   std::mt19937 rng{31415u};
   for (double &v : xData) {
      v = uniformDouble(rng, 0.0, 10.0);
   }

   auto out = batchEvalFormula(expr, xData, scalarVals);
   auto ref = scalarRefFormula(expr, xData, scalarVals);
   expectBatchMatches(out, ref, "large batch");
}

// The all-scalar case (every input span has size 1, e.g. a formula of
// parameters only -- the HistFactory NormFactor shape of issue #21052) must
// short-circuit to a single scalar evaluation, bitwise identical to eval(),
// with or without VDT.
TEST(RooFormulaEvaluator, VectorizedDoEvalAllScalar)
{
   ScopedBackendEnv env{nullptr};

   RooRealVar a("a", "a", 1.3, 0.0, 10.0);
   RooRealVar b("b", "b", 0.7, 0.0, 10.0);
   RooRealVar unused("unused", "unused", 2.0, 0.0, 10.0);
   RooFormulaVar f("f", "exp(-a*b) + sin(a)/b", {a, b, unused});

   const double refVal = f.getVal();
   RooFit::Evaluator ev(f);
   std::span<const double> out = ev.run();
   ASSERT_EQ(out.size(), 1u);
   EXPECT_TRUE(sameBits(out[0], refVal)) << std::hexfloat << out[0] << " vs " << refVal << std::defaultfloat;
}

// The TFormula fallback backend keeps its scalar per-event loop in doEval();
// its batch results must stay identical to scalar evaluation.
TEST(RooFormulaEvaluator, TFormulaBackendDoEval)
{
   ScopedBackendEnv env{"tformula"};

   RooRealVar x("x", "x", 5.0, 0.0, 10.0);
   RooRealVar p("p", "p", 1.5, 0.1, 3.0);
   RooRealVar q("q", "q", 0.7, 0.1, 3.0); // unused by the formula
   RooFormulaVar f("f", "x*p + sin(x) + (x > 5.0 ? log(x) : -x)", {x, p, q});

   std::vector<double> xData(100);
   std::mt19937 rng{777u};
   for (double &v : xData) {
      v = uniformDouble(rng, 0.0, 10.0);
   }

   RooFit::Evaluator ev(f);
   ev.setInput("x", {xData.data(), xData.size()}, false);
   std::span<const double> out = ev.run();
   // The TFormula backend was used: there is a JIT-compiled function.
   EXPECT_FALSE(f.getUniqueFuncName().empty());
   ASSERT_EQ(out.size(), xData.size());
   // Reference through the JIT-free scalar program, which evaluates
   // identically to the JIT-compiled TFormula (the Phase 3 contract), bitwise
   // up to the contraction the JIT applies and the interpreter loop cannot.
   // A reference expression inlined here would not be reliable either: this
   // test file is compiled with FMA contraction enabled.
   auto ref = scalarRefFormula("x[0]*x[1] + sin(x[0]) + (x[0] > 5.0 ? log(x[0]) : -x[0])", xData, {0.0, 1.5});
   ASSERT_EQ(ref.size(), xData.size());
   for (std::size_t i = 0; i < xData.size(); ++i) {
      EXPECT_TRUE(agreesWithJit(out[i], ref[i])) << "event " << i;
   }
}

namespace {

/// Resident set size of this process in kB, or -1 where unsupported.
long vmRSSkB()
{
#ifdef __linux__
   std::ifstream in("/proc/self/status");
   std::string line;
   while (std::getline(in, line)) {
      if (line.rfind("VmRSS:", 0) == 0)
         return std::atol(line.c_str() + 6);
   }
#endif
   return -1;
}

/// A formula string with numeric literals that are distinct for each `i`,
/// mirroring the TRExFitter-style expression NormFactors reported in
/// https://github.com/root-project/root/issues/21052, e.g.
/// `(1-(SFb_pcbt90_20_40*0.035)-(SFc_pcbt85_20_40*0.138))/0.518`.
std::string distinctLiteralExpr(int i)
{
   char buf[64];
   const double k1 = 0.001 + 1e-6 * i;
   const double k2 = 0.100 + 1e-6 * i;
   std::snprintf(buf, sizeof(buf), "(1-(a*%.6f)-(b*%.6f))/0.518", k1, k2);
   return buf;
}

} // namespace

// Regression test for https://github.com/root-project/root/issues/21052:
// constructing many RooFormulaVars whose formulas differ only in their
// numeric literals must not JIT-compile anything. On the old TFormula
// backend, each distinct-literal formula misses the JIT'd-function cache and
// costs two cling JIT compilations (the formula and its validation clone) at
// ~130 kB and ~6 ms each -- about 8 GB and several minutes for this N, which
// is the reported out-of-memory failure. Note that literals distinct per
// formula are essential: with identical formula bodies the JIT'd-function
// cache absorbs the repetition and the old path passes this test too.
TEST(RooFormulaEvaluator, ManyDistinctLiteralFormulas)
{
   // The default backend must handle this workload; shield the test from an
   // ambient ROOFIT_FORMULA_BACKEND override.
   ScopedBackendEnv env{nullptr};

   constexpr int n = 30000;

   RooRealVar a("a", "a", 1.0, 0.1, 3.0);
   RooRealVar b("b", "b", 1.0, 0.1, 3.0);

   const long rss0 = vmRSSkB();
   const auto t0 = std::chrono::steady_clock::now();

   std::vector<std::unique_ptr<RooFormulaVar>> fvs;
   fvs.reserve(n);
   for (int i = 0; i < n; ++i) {
      const std::string name = "f_" + std::to_string(i);
      fvs.emplace_back(std::make_unique<RooFormulaVar>(name.c_str(), distinctLiteralExpr(i).c_str(), RooArgList(a, b)));
      RooFormulaVar &fv = *fvs.back();
      fv.getVal(); // force evaluation like a real fit setup would
      // The whole batch must run on the JIT-free expression backend: a
      // formula on the TFormula backend always reports the (non-empty) name
      // of its cling-JIT-compiled function, so a non-empty name here means a
      // JIT compilation happened for this formula.
      ASSERT_TRUE(fv.getUniqueFuncName().empty()) << name;
   }

   const auto t1 = std::chrono::steady_clock::now();
   const long rss1 = vmRSSkB();

   // Spot-check values and that the expressions are emittable as inline C++
   // (the positive counterpart of the empty unique function name: exactly the
   // JIT-free expression backend can emit C++).
   auto varName = [](unsigned int i) { return "v[" + std::to_string(i) + "]"; };
   for (int i : {0, 12345, n - 1}) {
      const double k1 = 0.001 + 1e-6 * i;
      const double k2 = 0.100 + 1e-6 * i;
      EXPECT_DOUBLE_EQ(fvs[i]->getVal(), (1. - k1 - k2) / 0.518) << i;
      EXPECT_FALSE(fvs[i]->emitFormulaCpp(varName).empty()) << i;
   }

   // Order-of-magnitude resource bounds, far above what the JIT-free path
   // needs (measured: 1.1 s wall, 82 MB RSS growth) and far below what the
   // per-formula JIT path would cost (minutes, gigabytes).
   const double wallSeconds = std::chrono::duration<double>(t1 - t0).count();
   EXPECT_LT(wallSeconds, 60.);
   if (rss0 >= 0) {
      EXPECT_LT((rss1 - rss0) / 1024., 400.) << "RSS growth in MB";
   }
}

// The codegen counterpart of ManyDistinctLiteralFormulas, also for
// https://github.com/root-project/root/issues/21052: before the JIT-free
// backend, codegen forced one TFormula JIT compilation per RooFormulaVar just
// to obtain a function name for the generated code to call. Now the
// expressions are inlined into the generated code, so building and evaluating
// the likelihood of a model with many distinct-literal formulas involves no
// per-formula JIT. Cling is still invoked once, for the whole squashed
// likelihood function: the claim under test is O(1) compilations per model
// instead of O(N) per formula.
TEST(RooFormulaEvaluator, ManyDistinctLiteralFormulasCodegen)
{
   ScopedBackendEnv env{nullptr};

   constexpr int n = 1000;

   RooRealVar x("x", "x", 0.0, -10.0, 10.0);
   RooRealVar a("a", "a", 1.0, 0.1, 3.0);
   RooRealVar b("b", "b", 1.0, 0.1, 3.0);
   RooRealVar sigma("sigma", "sigma", 2.0, 0.1, 10.0);

   const long rss0 = vmRSSkB();
   const auto t0 = std::chrono::steady_clock::now();

   // A Gaussian whose mean is the sum of n distinct-literal formula terms,
   // each scaled such that the sum stays of order one.
   std::vector<std::unique_ptr<RooFormulaVar>> fvs;
   RooArgList terms;
   fvs.reserve(n);
   for (int i = 0; i < n; ++i) {
      const std::string name = "f_" + std::to_string(i);
      const std::string expr = "(" + distinctLiteralExpr(i) + " - 0.9)/" + std::to_string(n) + ".0";
      fvs.emplace_back(std::make_unique<RooFormulaVar>(name.c_str(), expr.c_str(), RooArgList(a, b)));
      terms.add(*fvs.back());
   }
   RooAddition mean("mean", "mean", terms);
   RooGaussian gauss("gauss", "gauss", x, mean, sigma);

   RooDataSet data("data", "data", x);
   for (int i = 0; i < 20; ++i) {
      x.setVal(-3.0 + 0.3 * i);
      data.add(x);
   }

   // CodegenNoGrad is the codegen backend without the (clad-dependent)
   // gradient generation, which is not needed to test the compilation count.
   std::unique_ptr<RooAbsReal> nll{gauss.createNLL(data, RooFit::EvalBackend::CodegenNoGrad())};
   const double nllVal = nll->getVal();

   const auto t1 = std::chrono::steady_clock::now();
   const long rss1 = vmRSSkB();

   // Codegen must not have created any TFormula behind the scenes (it used
   // to call getVal() and getUniqueFuncName() on each formula, forcing one
   // JIT compilation per formula): every formula still has no JIT'd function
   // name, i.e. it stayed on the JIT-free expression backend throughout.
   for (auto const &fv : fvs) {
      ASSERT_TRUE(fv->getUniqueFuncName().empty()) << fv->GetName();
   }

   // The generated code must agree with the regular CPU evaluation backend.
   std::unique_ptr<RooAbsReal> nllRef{gauss.createNLL(data, RooFit::EvalBackend::Cpu())};
   EXPECT_NEAR(nllVal, nllRef->getVal(), 1e-10 * std::abs(nllRef->getVal()));

   // Order-of-magnitude resource bounds (measured: 0.6 s wall, 40 MB RSS
   // growth, dominated by the one cling compilation of the squashed
   // likelihood function).
   const double wallSeconds = std::chrono::duration<double>(t1 - t0).count();
   EXPECT_LT(wallSeconds, 60.);
   if (rss0 >= 0) {
      EXPECT_LT((rss1 - rss0) / 1024., 400.) << "RSS growth in MB";
   }
}
