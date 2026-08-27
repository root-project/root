// Tests for the JIT-free RooFit formula evaluation backend
// (RooFormulaParser + RooExprEvaluator), and its silent-TFormula-fallback
// contract.
// Author: Jonas Rembser, CERN 2026

#include "../src/RooFormulaUtils.h"
#include "../src/RooFormulaParser.h"
#include "../src/RooExprEvaluator.h"

#include <RooFormulaVar.h>
#include <RooRealVar.h>

#include <Math/PdfFuncMathCore.h>
#include <ROOT/TestSupport.hxx>
#include <TFormula.h>
#include <TMath.h>
#include <TSystem.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

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

   // the codegen accessor lazily provides a JIT-compiled TFormula function
   // even on the AST path
   EXPECT_FALSE(fVar.getUniqueFuncName().empty());
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
   EXPECT_GE(coverage, 0.9);
}
