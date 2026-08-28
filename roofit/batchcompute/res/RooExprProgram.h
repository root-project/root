/*
 * Project: RooFit
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOFIT_BATCHCOMPUTE_ROOEXPRPROGRAM_H
#define ROOFIT_BATCHCOMPUTE_ROOEXPRPROGRAM_H

#include <cstddef>
#include <cstdint>

namespace RooBatchCompute {

/// Opcodes of the postfix expression programs compiled by RooFit's JIT-free
/// formula backend (see RooFormulaParser in RooFitCore). The same instruction
/// sequence drives both the scalar per-event evaluation in RooFitCore and the
/// chunked, vectorized batch evaluation in
/// RooBatchComputeInterface::computeExprProgram().
enum class ExprOp : std::uint8_t {
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
   // Unary calls whose semantics are exactly the corresponding std/libm
   // function, split out from Call1 so that batch backends can substitute a
   // fast vectorizable implementation (VDT, hardware sqrt). fn1 carries the
   // exact scalar function, which is what per-event evaluation calls.
   Exp,   ///< std::exp(a)
   Log,   ///< std::log(a)
   Sin,   ///< std::sin(a)
   Cos,   ///< std::cos(a)
   Sqrt,  ///< std::sqrt(a)
   Call1, ///< fn1(a)
   Call2, ///< fn2(a, b)
   Call3, ///< fn3(a, b, c)
   Call4  ///< fn4(a, b, c, d)
};

/// One instruction of a postfix expression program. Call instructions carry
/// the resolved function pointer, so evaluation involves no lookup table;
/// `arg` additionally keeps the index into RooFitCore's function allow-list
/// (RooFormulaFunctions) that the call was resolved from, which C++ emission
/// uses to reproduce the exact spelling.
struct ExprInstr {
   ExprOp op = ExprOp::Const;
   std::uint32_t arg = 0; ///< Var: variable index; calls: function-table index
   union {
      double konst = 0.0;                            ///< Const
      double (*fn1)(double);                         ///< Call1 and Exp...Sqrt
      double (*fn2)(double, double);                 ///< Call2
      double (*fn3)(double, double, double);         ///< Call3
      double (*fn4)(double, double, double, double); ///< Call4
   };
};

/// Maximum expression stack depth accepted by computeExprProgram(), which
/// stack-allocates one bufferSize-sized chunk buffer per stack slot. Deeper
/// programs must be evaluated with the scalar per-event fallback.
constexpr std::uint32_t maxExprProgramStackDepth = 24;

} // End namespace RooBatchCompute

#endif
