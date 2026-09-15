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

/// Device-representable identity of the function a call instruction calls.
///
/// The host interpreters call through the function pointer in the instruction,
/// which reproduces exactly what the cling-JIT-compiled code called for that
/// spelling. A GPU kernel cannot call a host function pointer, so every call
/// instruction additionally carries this identity, which the CUDA backend
/// switches on. There is one value per distinct host implementation, not per
/// accepted spelling: `sin`, `std::sin` and `TMath::Sin` all resolve to the
/// same libm call and share a value, while `TMath::Erf` (Cephes) is separate
/// from `erf` (libm) because the host implementations differ.
///
/// A call instruction whose function has no device implementation keeps
/// ExprFunc::None; RooFitCore refuses to schedule such a program on the GPU
/// (see RooExprEvaluator::Program::cudaCapable). The five spelling families
/// with their own opcode (Exp, Log, Sin, Cos, Sqrt) are identified by the
/// opcode and do not need a value here, but carry one anyway.
enum class ExprFunc : std::uint8_t {
   None = 0, ///< no device implementation
   // Unary. The values shared with a dedicated opcode come first.
   Exp,
   Log,
   Sin,
   Cos,
   Sqrt,
   Log10,
   Tan,
   ASin,
   ACos,
   ATan,
   SinH,
   CosH,
   TanH,
   ASinH,
   ACosH,
   ATanH,
   Floor,
   Ceil,
   Erf,
   Erfc,
   TMathErf,  ///< TMath::Erf, which is ROOT::Math::erf (Cephes) on the host
   TMathErfc, ///< TMath::Erfc, likewise
   TGamma,
   LGamma,
   Abs,     ///< std::abs/std::fabs/TMath::Abs
   CastInt, ///< the `int(x)` functional cast: truncation towards zero
   Square,  ///< `sq`/TMath::Sq
   SignBit, ///< TMath::SignBit
   Gaus1,   ///< TMath::Gaus(x)
   // Binary.
   Pow,
   ATan2,      ///< std::atan2
   TMathATan2, ///< TMath::ATan2, which special-cases x == 0
   Fmod,
   StdMin, ///< std::min: asymmetric in NaN, unlike TMath::Min
   StdMax,
   TMathMin,
   TMathMax,
   CopySign, ///< `sign`/TMath::Sign
   Gaus2,    ///< TMath::Gaus(x, mean)
   // Ternary and quaternary.
   Gaus3, ///< TMath::Gaus(x, mean, sigma)
   Gaus4  ///< TMath::Gaus(x, mean, sigma, norm)
};

/// One instruction of a postfix expression program. Call instructions carry
/// the resolved function pointer, so evaluation involves no lookup table;
/// `arg` additionally keeps the index into RooFitCore's function allow-list
/// (RooFormulaFunctions) that the call was resolved from, which C++ emission
/// uses to reproduce the exact spelling, and `func` identifies the function
/// for backends that cannot use the host function pointer.
struct ExprInstr {
   ExprOp op = ExprOp::Const;
   ExprFunc func = ExprFunc::None; ///< calls: which function (for GPU dispatch)
   std::uint32_t arg = 0;          ///< Var: variable index; calls: function-table index
   union {
      double konst = 0.0;                            ///< Const
      double (*fn1)(double);                         ///< Call1 and Exp...Sqrt
      double (*fn2)(double, double);                 ///< Call2
      double (*fn3)(double, double, double);         ///< Call3
      double (*fn4)(double, double, double, double); ///< Call4
   };
};

/// Maximum expression stack depth accepted by computeExprProgram(), which
/// stack-allocates one bufferSize-sized chunk buffer per stack slot on the CPU
/// and one per-thread stack of this size on the GPU. Deeper programs must be
/// evaluated with the scalar per-event fallback.
constexpr std::uint32_t maxExprProgramStackDepth = 24;

} // End namespace RooBatchCompute

#endif
