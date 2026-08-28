/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooExprEvaluator.h"

#include "TMath.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace RooFormulaFunctions {

namespace {

// Entry construction helpers (C++17 has no designated initializers).
Entry F0(const char *name, double (*fn)())
{
   Entry e;
   e.name = name;
   e.arity = 0;
   e.fn0 = fn;
   return e;
}

Entry F1(const char *name, double (*fn)(double), TypeRule rule = TypeRule::Double, const char *cppName = nullptr)
{
   Entry e;
   e.name = name;
   e.cppName = cppName;
   e.arity = 1;
   e.rule = rule;
   e.fn1 = fn;
   return e;
}

Entry F2(const char *name, double (*fn)(double, double), TypeRule rule = TypeRule::Double,
         const char *cppName = nullptr)
{
   Entry e;
   e.name = name;
   e.cppName = cppName;
   e.arity = 2;
   e.rule = rule;
   e.fn2 = fn;
   return e;
}

Entry F3(const char *name, double (*fn)(double, double, double))
{
   Entry e;
   e.name = name;
   e.arity = 3;
   e.fn3 = fn;
   return e;
}

Entry F4(const char *name, double (*fn)(double, double, double, double))
{
   Entry e;
   e.name = name;
   e.arity = 4;
   e.fn4 = fn;
   return e;
}

// The allow-list. For every accepted spelling, the function pointer calls
// exactly what the cling-JIT-compiled TFormula code called for that spelling:
// bare and std:: spellings resolve to the libm/std functions, TMath::
// spellings call the TMath functions (whose implementations are not always
// identical to libm, e.g. TMath::Erf goes through ROOT::Math::erf and
// TMath::ATan2 special-cases x == 0). This makes the two evaluation backends
// agree bitwise.
//
// Bare spellings work in TFormula because cling resolves them against the
// global namespace (libm) and `using namespace std`; the only genuine
// TFormula shortcuts are `sign` -> TMath::Sign and `sq` -> TMath::Sq
// (see TFormula::FillDefaults). All spellings below were cross-checked
// against what TFormula accepts today; do not add a spelling TFormula
// would reject (e.g. there is no std::sq or std::sign).
std::vector<Entry> makeTable()
{
   auto castInt = +[](double x) { return static_cast<double>(static_cast<int>(x)); };
   auto square = +[](double x) { return x * x; };
   // std::min/std::max compare exactly like this; the asymmetric NaN behavior
   // (min(NaN, 1) = NaN but min(1, NaN) = 1) must be reproduced, so no fmin/fmax.
   auto stdMin = +[](double a, double b) { return b < a ? b : a; };
   auto stdMax = +[](double a, double b) { return a < b ? b : a; };
   // TMath::Min/Max compare differently from std::min/max (visible with NaNs).
   auto tmathMin = +[](double a, double b) { return TMath::Min(a, b); };
   auto tmathMax = +[](double a, double b) { return TMath::Max(a, b); };
   // Both the `sign` shortcut and TMath::Sign resolve to TMath::Sign, which is
   // std::copysign for double arguments. (For an int first argument cling picks
   // the generic TMath::Sign template, which also follows the sign bit of the
   // second argument and is numerically identical for int values. A *bool*
   // first argument is rejected by the parser: the template would return bool,
   // turning e.g. Sign(true, -1.) into +1.)
   auto sign = +[](double a, double b) { return TMath::Sign(a, b); };
   auto signBit = +[](double x) { return std::signbit(x) ? 1.0 : 0.0; };

   return {
      // clang-format off
      // one-argument functions, libm/std spellings
      F1("sqrt",         +[](double x) { return std::sqrt(x); }),
      F1("std::sqrt",    +[](double x) { return std::sqrt(x); }),
      F1("exp",          +[](double x) { return std::exp(x); }),
      F1("std::exp",     +[](double x) { return std::exp(x); }),
      F1("log",          +[](double x) { return std::log(x); }),
      F1("std::log",     +[](double x) { return std::log(x); }),
      F1("log10",        +[](double x) { return std::log10(x); }),
      F1("std::log10",   +[](double x) { return std::log10(x); }),
      F1("sin",          +[](double x) { return std::sin(x); }),
      F1("std::sin",     +[](double x) { return std::sin(x); }),
      F1("cos",          +[](double x) { return std::cos(x); }),
      F1("std::cos",     +[](double x) { return std::cos(x); }),
      F1("tan",          +[](double x) { return std::tan(x); }),
      F1("std::tan",     +[](double x) { return std::tan(x); }),
      F1("asin",         +[](double x) { return std::asin(x); }),
      F1("std::asin",    +[](double x) { return std::asin(x); }),
      F1("acos",         +[](double x) { return std::acos(x); }),
      F1("std::acos",    +[](double x) { return std::acos(x); }),
      F1("atan",         +[](double x) { return std::atan(x); }),
      F1("std::atan",    +[](double x) { return std::atan(x); }),
      F1("sinh",         +[](double x) { return std::sinh(x); }),
      F1("std::sinh",    +[](double x) { return std::sinh(x); }),
      F1("cosh",         +[](double x) { return std::cosh(x); }),
      F1("std::cosh",    +[](double x) { return std::cosh(x); }),
      F1("tanh",         +[](double x) { return std::tanh(x); }),
      F1("std::tanh",    +[](double x) { return std::tanh(x); }),
      F1("asinh",        +[](double x) { return std::asinh(x); }),
      F1("std::asinh",   +[](double x) { return std::asinh(x); }),
      F1("acosh",        +[](double x) { return std::acosh(x); }),
      F1("std::acosh",   +[](double x) { return std::acosh(x); }),
      F1("atanh",        +[](double x) { return std::atanh(x); }),
      F1("std::atanh",   +[](double x) { return std::atanh(x); }),
      F1("floor",        +[](double x) { return std::floor(x); }),
      F1("std::floor",   +[](double x) { return std::floor(x); }),
      F1("ceil",         +[](double x) { return std::ceil(x); }),
      F1("std::ceil",    +[](double x) { return std::ceil(x); }),
      F1("erf",          +[](double x) { return std::erf(x); }),
      F1("std::erf",     +[](double x) { return std::erf(x); }),
      F1("erfc",         +[](double x) { return std::erfc(x); }),
      F1("std::erfc",    +[](double x) { return std::erfc(x); }),
      F1("tgamma",       +[](double x) { return std::tgamma(x); }),
      F1("std::tgamma",  +[](double x) { return std::tgamma(x); }),
      F1("lgamma",       +[](double x) { return std::lgamma(x); }),
      F1("std::lgamma",  +[](double x) { return std::lgamma(x); }),
      // abs is integer-preserving in C++ (::abs(int), std::abs(int)); fabs is not
      F1("abs",          +[](double x) { return std::fabs(x); }, TypeRule::SameAsFirstArg),
      F1("std::abs",     +[](double x) { return std::fabs(x); }, TypeRule::SameAsFirstArg),
      F1("fabs",         +[](double x) { return std::fabs(x); }),
      F1("std::fabs",    +[](double x) { return std::fabs(x); }),
      // C++ functional cast: truncation towards zero
      F1("int",          castInt, TypeRule::Int, "int"),
      // TFormula shortcut for TMath::Sq(Double_t)
      F1("sq",           square, TypeRule::Double, "TMath::Sq"),
      // one-argument functions, TMath spellings
      F1("TMath::Sqrt",  +[](double x) { return TMath::Sqrt(x); }),
      F1("TMath::Exp",   +[](double x) { return TMath::Exp(x); }),
      F1("TMath::Log",   +[](double x) { return TMath::Log(x); }),
      F1("TMath::Log10", +[](double x) { return TMath::Log10(x); }),
      F1("TMath::Sin",   +[](double x) { return TMath::Sin(x); }),
      F1("TMath::Cos",   +[](double x) { return TMath::Cos(x); }),
      F1("TMath::Tan",   +[](double x) { return TMath::Tan(x); }),
      F1("TMath::ASin",  +[](double x) { return TMath::ASin(x); }),
      F1("TMath::ACos",  +[](double x) { return TMath::ACos(x); }),
      F1("TMath::ATan",  +[](double x) { return TMath::ATan(x); }),
      F1("TMath::SinH",  +[](double x) { return TMath::SinH(x); }),
      F1("TMath::CosH",  +[](double x) { return TMath::CosH(x); }),
      F1("TMath::TanH",  +[](double x) { return TMath::TanH(x); }),
      F1("TMath::ASinH", +[](double x) { return TMath::ASinH(x); }),
      F1("TMath::ACosH", +[](double x) { return TMath::ACosH(x); }),
      F1("TMath::ATanH", +[](double x) { return TMath::ATanH(x); }),
      F1("TMath::Floor", +[](double x) { return TMath::Floor(x); }),
      F1("TMath::Ceil",  +[](double x) { return TMath::Ceil(x); }),
      F1("TMath::Erf",   +[](double x) { return TMath::Erf(x); }),
      F1("TMath::Erfc",  +[](double x) { return TMath::Erfc(x); }),
      F1("TMath::Abs",   +[](double x) { return TMath::Abs(x); }, TypeRule::SameAsFirstArg),
      F1("TMath::Sq",    square),
      F1("TMath::SignBit", signBit, TypeRule::Bool),
      // two-argument functions
      F2("pow",          +[](double a, double b) { return std::pow(a, b); }),
      F2("std::pow",     +[](double a, double b) { return std::pow(a, b); }),
      F2("TMath::Power", +[](double a, double b) { return TMath::Power(a, b); }),
      F2("atan2",        +[](double a, double b) { return std::atan2(a, b); }),
      F2("std::atan2",   +[](double a, double b) { return std::atan2(a, b); }),
      F2("TMath::ATan2", +[](double a, double b) { return TMath::ATan2(a, b); }),
      F2("fmod",         +[](double a, double b) { return std::fmod(a, b); }),
      F2("std::fmod",    +[](double a, double b) { return std::fmod(a, b); }),
      F2("min",          stdMin, TypeRule::MinMax),
      F2("std::min",     stdMin, TypeRule::MinMax),
      F2("TMath::Min",   tmathMin, TypeRule::MinMax),
      F2("max",          stdMax, TypeRule::MinMax),
      F2("std::max",     stdMax, TypeRule::MinMax),
      F2("TMath::Max",   tmathMax, TypeRule::MinMax),
      // TFormula shortcut for TMath::Sign
      F2("sign",         sign, TypeRule::Sign, "TMath::Sign"),
      F2("TMath::Sign",  sign, TypeRule::Sign),
      // zero-argument constants (folded to Op::Const at parse time)
      F0("TMath::Pi",     +[]() { return TMath::Pi(); }),
      F0("TMath::TwoPi",  +[]() { return TMath::TwoPi(); }),
      F0("TMath::PiOver2",+[]() { return TMath::PiOver2(); }),
      F0("TMath::E",      +[]() { return TMath::E(); }),
      // TMath::Gaus with its default arguments mean=0, sigma=1, norm=false
      F1("TMath::Gaus",  +[](double x) { return TMath::Gaus(x); }),
      F2("TMath::Gaus",  +[](double x, double m) { return TMath::Gaus(x, m); }),
      F3("TMath::Gaus",  +[](double x, double m, double s) { return TMath::Gaus(x, m, s); }),
      F4("TMath::Gaus",  +[](double x, double m, double s, double n) { return TMath::Gaus(x, m, s, n != 0.0); }),
      // clang-format on
   };
}

std::vector<Entry> const &theTable()
{
   static const std::vector<Entry> t = makeTable();
   return t;
}

} // namespace

Entry const *table()
{
   return theTable().data();
}

std::size_t tableSize()
{
   return theTable().size();
}

Entry const *find(std::string const &name, unsigned int nArgs, std::uint32_t &index)
{
   auto const &tab = theTable();
   for (std::size_t i = 0; i < tab.size(); ++i) {
      if (tab[i].arity == nArgs && name == tab[i].name) {
         index = static_cast<std::uint32_t>(i);
         return &tab[i];
      }
   }
   return nullptr;
}

} // namespace RooFormulaFunctions

////////////////////////////////////////////////////////////////////////////////
/// Interpret the instruction sequence with a fixed-size value stack. Stack
/// depth was bounded at parse time and is re-checked once up front, so no
/// per-instruction bounds checks are needed.
double RooExprEvaluator::eval(const double *vars) const
{
   // The depth bound is established at parse time; refuse to evaluate rather
   // than trust it, so a future accounting bug cannot overflow the stack.
   if (_program->stackDepth > kMaxStackDepth) {
      throw std::runtime_error("RooExprEvaluator: expression program exceeds the maximum stack depth");
   }

   double stack[kMaxStackDepth];
   std::size_t sp = 0;
   auto const *funcs = RooFormulaFunctions::table();

   for (Instr const &ins : _program->code) {
      switch (ins.op) {
      case Op::Const: stack[sp++] = ins.konst; break;
      case Op::Var: stack[sp++] = vars[ins.arg]; break;
      case Op::Add:
         --sp;
         stack[sp - 1] = stack[sp - 1] + stack[sp];
         break;
      case Op::Sub:
         --sp;
         stack[sp - 1] = stack[sp - 1] - stack[sp];
         break;
      case Op::Mul:
         --sp;
         stack[sp - 1] = stack[sp - 1] * stack[sp];
         break;
      case Op::Div:
         --sp;
         stack[sp - 1] = stack[sp - 1] / stack[sp];
         break;
      case Op::Neg: stack[sp - 1] = -stack[sp - 1]; break;
      case Op::Not: stack[sp - 1] = stack[sp - 1] == 0.0 ? 1.0 : 0.0; break;
      case Op::LT:
         --sp;
         stack[sp - 1] = stack[sp - 1] < stack[sp] ? 1.0 : 0.0;
         break;
      case Op::LE:
         --sp;
         stack[sp - 1] = stack[sp - 1] <= stack[sp] ? 1.0 : 0.0;
         break;
      case Op::GT:
         --sp;
         stack[sp - 1] = stack[sp - 1] > stack[sp] ? 1.0 : 0.0;
         break;
      case Op::GE:
         --sp;
         stack[sp - 1] = stack[sp - 1] >= stack[sp] ? 1.0 : 0.0;
         break;
      case Op::EQ:
         --sp;
         stack[sp - 1] = stack[sp - 1] == stack[sp] ? 1.0 : 0.0;
         break;
      case Op::NE:
         --sp;
         stack[sp - 1] = stack[sp - 1] != stack[sp] ? 1.0 : 0.0;
         break;
      case Op::And:
         --sp;
         stack[sp - 1] = (stack[sp - 1] != 0.0 && stack[sp] != 0.0) ? 1.0 : 0.0;
         break;
      case Op::Or:
         --sp;
         stack[sp - 1] = (stack[sp - 1] != 0.0 || stack[sp] != 0.0) ? 1.0 : 0.0;
         break;
      case Op::Select:
         sp -= 2;
         stack[sp - 1] = stack[sp - 1] != 0.0 ? stack[sp] : stack[sp + 1];
         break;
      case Op::Pow:
         --sp;
         stack[sp - 1] = std::pow(stack[sp - 1], stack[sp]);
         break;
      case Op::Sq: stack[sp - 1] *= stack[sp - 1]; break;
      case Op::IntNorm: stack[sp - 1] += 0.0; break;
      case Op::Call1: stack[sp - 1] = funcs[ins.arg].fn1(stack[sp - 1]); break;
      case Op::Call2:
         --sp;
         stack[sp - 1] = funcs[ins.arg].fn2(stack[sp - 1], stack[sp]);
         break;
      case Op::Call3:
         sp -= 2;
         stack[sp - 1] = funcs[ins.arg].fn3(stack[sp - 1], stack[sp], stack[sp + 1]);
         break;
      case Op::Call4:
         sp -= 3;
         stack[sp - 1] = funcs[ins.arg].fn4(stack[sp - 1], stack[sp], stack[sp + 1], stack[sp + 2]);
         break;
      }
   }

   return stack[0];
}

namespace {

/// Format a double as a C++ expression of type double that parses back to the
/// exact same value: max_digits10 (17) significant decimal digits guarantee
/// the bitwise round-trip (this is also the formatting convention used
/// elsewhere in RooFit codegen).
std::string emitDouble(double val)
{
   if (std::isnan(val)) {
      return "std::numeric_limits<double>::quiet_NaN()";
   }
   if (std::isinf(val)) {
      return val > 0 ? "std::numeric_limits<double>::infinity()" : "(-std::numeric_limits<double>::infinity())";
   }
   std::stringstream ss;
   // The formatting must not depend on the global locale: a comma decimal
   // separator (e.g. from a German locale) would corrupt the emitted C++.
   ss.imbue(std::locale::classic());
   ss << std::setprecision(std::numeric_limits<double>::max_digits10) << val;
   std::string out = ss.str();
   // The emitted literal must have type double: an integer-looking literal
   // like `2` would be an int in C++, with different division semantics.
   if (out.find_first_of(".eE") == std::string::npos) {
      out += ".0";
   }
   // Parse-time constant folding (e.g. of TMath::Pi()) can in principle
   // produce negative values; keep every stack entry self-contained.
   if (out[0] == '-') {
      out = "(" + out + ")";
   }
   return out;
}

/// The C++ spelling emitted for a function-table entry (see Entry::cppName).
std::string emissionName(RooFormulaFunctions::Entry const &entry)
{
   if (entry.cppName) {
      return entry.cppName;
   }
   const std::string name = entry.name;
   return name.find("::") == std::string::npos ? "std::" + name : name;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Emit the expression as C++ source by walking the same instruction sequence
/// that eval() interprets, so evaluation and code generation cannot diverge
/// structurally. Every intermediate result is kept fully parenthesized, so no
/// operator-precedence reasoning is involved. The emitted operators and
/// function spellings reproduce what the cling-JIT-compiled TFormula code
/// called for the same formula, so all three semantic paths (AST evaluation,
/// emitted C++, TFormula fallback) agree bitwise.
std::string RooExprEvaluator::emitCpp(std::function<std::string(unsigned int)> const &varName) const
{
   std::vector<std::string> stack;
   auto const *funcs = RooFormulaFunctions::table();

   auto pop = [&]() {
      std::string out = std::move(stack.back());
      stack.pop_back();
      return out;
   };
   auto binary = [&](const char *sym) {
      const std::string b = pop();
      stack.back() = "(" + stack.back() + " " + sym + " " + b + ")";
   };

   for (Instr const &ins : _program->code) {
      switch (ins.op) {
      case Op::Const: stack.push_back(emitDouble(ins.konst)); break;
      case Op::Var: stack.push_back("(" + varName(ins.arg) + ")"); break;
      case Op::Add: binary("+"); break;
      case Op::Sub: binary("-"); break;
      case Op::Mul: binary("*"); break;
      case Op::Div: binary("/"); break;
      case Op::Neg: stack.back() = "(-" + stack.back() + ")"; break;
      case Op::Not: stack.back() = "(!" + stack.back() + ")"; break;
      case Op::LT: binary("<"); break;
      case Op::LE: binary("<="); break;
      case Op::GT: binary(">"); break;
      case Op::GE: binary(">="); break;
      case Op::EQ: binary("=="); break;
      case Op::NE: binary("!="); break;
      // In C++, `&&`/`||` short-circuit and `?:` evaluates only the taken
      // branch, while eval() always evaluates both operands. The resulting
      // values are identical; the difference is only observable if
      // floating-point exceptions are trapped.
      case Op::And: binary("&&"); break;
      case Op::Or: binary("||"); break;
      case Op::Select: {
         const std::string b = pop();
         const std::string a = pop();
         stack.back() = "(" + stack.back() + " ? " + a + " : " + b + ")";
         break;
      }
      case Op::Pow: {
         const std::string b = pop();
         stack.back() = "std::pow(" + stack.back() + ", " + b + ")";
         break;
      }
      case Op::Sq: stack.back() = "TMath::Sq(" + stack.back() + ")"; break;
      case Op::IntNorm: stack.back() = "(" + stack.back() + " + 0.0)"; break;
      case Op::Call1: stack.back() = emissionName(funcs[ins.arg]) + "(" + stack.back() + ")"; break;
      case Op::Call2: {
         const std::string b = pop();
         stack.back() = emissionName(funcs[ins.arg]) + "(" + stack.back() + ", " + b + ")";
         break;
      }
      case Op::Call3: {
         const std::string c = pop();
         const std::string b = pop();
         stack.back() = emissionName(funcs[ins.arg]) + "(" + stack.back() + ", " + b + ", " + c + ")";
         break;
      }
      case Op::Call4: {
         const std::string d = pop();
         const std::string c = pop();
         const std::string b = pop();
         stack.back() = emissionName(funcs[ins.arg]) + "(" + stack.back() + ", " + b + ", " + c + ", " + d + ")";
         break;
      }
      }
   }

   return stack.back();
}

/// \endcond
