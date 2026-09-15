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

#include "RooExprProgram.h"

#include <ROOT/RSpan.hxx>

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
/// tracks the double/int/bool typing of subexpressions to reproduce cling's
/// expression typing (in particular to detect integer division, which is not
/// supported, and the bool-typed constructs that behave differently).
enum class TypeRule : std::uint8_t {
   Double,         ///< result is always double
   SameAsFirstArg, ///< abs: result type follows the first argument (bool promotes to int)
   Int,            ///< result is int (the `int(x)` functional cast)
   Bool,           ///< result is bool (TMath::SignBit)
   Sign,           ///< sign/TMath::Sign: result type follows the first argument; a bool
                   ///< first argument is rejected, because cling resolves it to the
                   ///< generic TMath::Sign template returning bool -- not copysign
   MinMax          ///< result type is the common argument type; mixed argument types
                   ///< (int/double or bool/int) do not compile in cling
};

struct Entry {
   const char *name = nullptr; ///< accepted spelling in the formula
   /// Spelling emitted in generated C++ for this entry. A nullptr means
   /// "derive from name": qualified names (TMath::Erf, std::sin) are emitted
   /// as-is, bare libm/std names get a std:: qualification (sin -> std::sin,
   /// resolving to the same function the JIT-compiled code called). Only
   /// entries whose emission cannot be derived this way set it explicitly.
   const char *cppName = nullptr;
   std::uint8_t arity = 0;
   TypeRule rule = TypeRule::Double;
   /// Opcode emitted for a call to this entry when arity == 1. Entries whose
   /// semantics are exactly a std/libm function with a fast vectorizable
   /// batch implementation (the exp/log/sin/cos/sqrt spelling families) get
   /// the corresponding dedicated opcode instead of the generic Call1, so
   /// that RooBatchCompute::computeExprProgram() can vectorize them. Scalar
   /// evaluation and C++ emission treat those opcodes exactly like Call1.
   RooBatchCompute::ExprOp op1 = RooBatchCompute::ExprOp::Call1;
   /// Device-representable identity of `fn0`..`fn4`, copied into the call
   /// instruction so that the CUDA backend can dispatch without the host
   /// function pointer. Entries that leave this at ExprFunc::None keep their
   /// programs off the GPU.
   RooBatchCompute::ExprFunc func = RooBatchCompute::ExprFunc::None;
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
   /// The instruction set is shared with RooBatchCompute (see
   /// RooExprProgram.h): the same instruction vector drives the scalar
   /// per-event eval() here and the chunked, vectorized batch evaluation in
   /// RooBatchCompute::computeExprProgram().
   using Op = RooBatchCompute::ExprOp;
   using Instr = RooBatchCompute::ExprInstr;

   /// A compiled formula: immutable after construction and shared between all
   /// RooFormula instances with the same processed formula string.
   struct Program {
      std::vector<Instr> code;
      std::vector<bool> usedVars; ///< usedVars[i] is true if `x[i]` appears in the formula
      std::string formula;        ///< the processed formula string this was compiled from
      unsigned int stackDepth = 0;
      /// Whether RooBatchCompute's CUDA backend can evaluate this program:
      /// the stack fits the fixed-size per-thread stack, and every call
      /// resolves to a function with a device implementation. Determined once
      /// at parse time.
      bool cudaCapable = false;
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

   bool canEmitCpp() const override { return true; }

   std::string emitCpp(std::function<std::string(unsigned int)> const &varName) const override;

   /// The compiled instruction sequence, for handing to
   /// RooBatchCompute::computeExprProgram().
   std::span<const Instr> code() const { return {_program->code.data(), _program->code.size()}; }

   /// The program's maximum expression stack depth.
   unsigned int stackDepth() const { return _program->stackDepth; }

   /// Whether this program can be evaluated by RooBatchCompute's CUDA backend.
   bool cudaCapable() const { return _program->cudaCapable; }

private:
   std::shared_ptr<const Program> _program;
};

#endif

/// \endcond
