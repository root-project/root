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

#ifndef ROO_EXPR_EVALUATOR
#define ROO_EXPR_EVALUATOR

#include "RooFormulaEvaluator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/// Functions that can be called from a formula on the JIT-free evaluation
/// path. Each table entry corresponds to one accepted spelling (e.g. `sin`,
/// `std::sin`, and `TMath::Sin` are three entries), and its function pointer
/// reproduces exactly what the cling-JIT-compiled code called for that
/// spelling, so that both backends agree bitwise.
namespace RooFormulaFunctions {

/// How the C++ result type of a call depends on the argument types. The parser
/// tracks int-ness of subexpressions to reproduce cling's expression typing
/// (in particular to detect integer division, which is not supported).
enum class TypeRule : std::uint8_t {
   Double,         ///< result is always double
   SameAsFirstArg, ///< result type equals the type of the first argument (abs, sign)
   Int,            ///< result is an integer type (`int(x)` cast, TMath::SignBit)
   MinMax          ///< int if both args are int; mixed int/double does not compile in cling
};

struct Entry {
   const char *name = nullptr; ///< accepted spelling in the formula
   std::uint8_t arity = 0;
   TypeRule rule = TypeRule::Double;
   double (*fn0)() = nullptr;
   double (*fn1)(double) = nullptr;
   double (*fn2)(double, double) = nullptr;
   double (*fn3)(double, double, double) = nullptr;
   double (*fn4)(double, double, double, double) = nullptr;
};

Entry const *table();
std::size_t tableSize();

/// Find the entry for the given spelling and argument count, or return
/// nullptr if there is none. The index output is the position in table().
Entry const *find(std::string const &name, unsigned int nArgs, std::uint32_t &index);

} // namespace RooFormulaFunctions

/// RooFormulaEvaluator implementation that interprets a compiled postfix
/// instruction sequence, without any use of the interpreter/JIT. Instances are
/// created via RooFormulaParser::compile(), which returns a shared immutable
/// Program: identical formula strings share one instruction vector.
///
/// eval() is const, touches no globals and no mutable state, so it is safe to
/// call concurrently from multiple threads.
class RooExprEvaluator final : public RooFormulaEvaluator {
public:
   enum class Op : std::uint8_t {
      Const,   ///< push konst
      Var,     ///< push vars[arg]
      Add,     ///< a + b
      Sub,     ///< a - b
      Mul,     ///< a * b
      Div,     ///< a / b
      Neg,     ///< -a
      Not,     ///< !a  (exactly 0.0 or 1.0)
      LT,      ///< a < b   (exactly 0.0 or 1.0, likewise below)
      LE,      ///< a <= b
      GT,      ///< a > b
      GE,      ///< a >= b
      EQ,      ///< a == b
      NE,      ///< a != b
      And,     ///< a && b  (no short-circuit: both operands are always evaluated)
      Or,      ///< a || b  (no short-circuit)
      Select,  ///< c ? a : b  (both branches are always evaluated)
      Pow,     ///< std::pow(a, b), from the `^`/`**` operator or pow()
      Sq,      ///< a * a, from TFormula's `expr^2` -> TMath::Sq(expr) rewrite
      IntNorm, ///< a + 0.0: maps -0.0 to +0.0 where cling would have used integer arithmetic
      Call1,   ///< RooFormulaFunctions::table()[arg].fn1
      Call2,   ///< RooFormulaFunctions::table()[arg].fn2
      Call3,   ///< RooFormulaFunctions::table()[arg].fn3
      Call4    ///< RooFormulaFunctions::table()[arg].fn4
   };

   struct Instr {
      Op op = Op::Const;
      std::uint32_t arg = 0;
      double konst = 0.0;
   };

   /// A compiled formula: immutable after construction and shared between all
   /// RooFormula instances with the same processed formula string.
   struct Program {
      std::vector<Instr> code;
      std::vector<bool> usedVars; ///< usedVars[i] is true if `x[i]` appears in the formula
      std::string formula;        ///< the processed formula string this was compiled from
      unsigned int stackDepth = 0;
   };

   /// Maximum evaluation stack depth (checked at compile time in the parser).
   static constexpr unsigned int kMaxStackDepth = 256;

   RooExprEvaluator(std::shared_ptr<const Program> program) : _program{std::move(program)} {}

   double eval(const double *vars) const override;

   std::unique_ptr<RooFormulaEvaluator> clone() const override { return std::make_unique<RooExprEvaluator>(_program); }

   /// Whether `x[i]` appears in the formula, as recorded while parsing.
   bool usesVariable(unsigned int i) const { return i < _program->usedVars.size() && _program->usedVars[i]; }

   /// The processed formula string this program was compiled from.
   std::string processedFormula() const { return _program->formula; }

private:
   std::shared_ptr<const Program> _program;
};

#endif

/// \endcond
