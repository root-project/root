// Tests for the CUDA batch evaluation of the JIT-free RooFit formula backend
// (RooBatchCompute::computeExprProgram() in RooBatchCompute.cu), differential
// against the CPU backend.
// Author: Jonas Rembser, CERN 2026

#include "../src/RooExprEvaluator.h"
#include "../src/RooFormulaParser.h"

#include <RooBatchCompute.h>
#include <RooDataSet.h>
#include <RooFit/Evaluator.h>
#include <RooFormulaVar.h>
#include <RooGenericPdf.h>
#include <RooRealVar.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef ROOFIT_CUDA

namespace {

/// Bitwise comparison; any two NaNs count as equal.
bool sameBits(double a, double b)
{
   if (std::isnan(a) && std::isnan(b))
      return true;
   return std::memcmp(&a, &b, sizeof(double)) == 0;
}

/// Portable [lo, hi) draw; see the note on reproducibility in
/// testRooFormulaEvaluator.cxx.
double uniformDouble(std::mt19937 &rng, double lo, double hi)
{
   return lo + (hi - lo) * (static_cast<double>(rng()) * (1.0 / 4294967296.0));
}

/// The device math functions are not the host's libm, so GPU results are not
/// bitwise identical to CPU results; they agree within the batch-vs-scalar
/// tolerance RooBatchCompute uses for its pdf kernels
/// (_toleranceCompareBatches in roofit/test/vectorisedPDFs). Special values
/// must still propagate identically: an Inf or NaN appearing on one side only
/// means the two interpreters disagree about the expression, not about the
/// last bits of a libm call.
bool cudaAgrees(double gpu, double cpu)
{
   if (sameBits(gpu, cpu)) {
      return true;
   }
   if (!std::isfinite(gpu) || !std::isfinite(cpu)) {
      return false;
   }
   return std::abs(gpu - cpu) <= 5e-14 * std::max(1.0, std::abs(cpu));
}

/// Whether this machine can run the CUDA backend at all. Building with
/// cuda=ON does not imply that a GPU is present when the tests run.
bool cudaUsable()
{
   RooRealVar x{"x", "x", 1.0, 0.0, 2.0};
   RooFormulaVar f{"f", "x", RooArgList{x}};
   try {
      RooFit::Evaluator ev{f, /*useGPU=*/true};
      ev.run();
   } catch (std::exception const &) {
      return false;
   }
   return true;
}

/// Batch-evaluate the compiled program of `expr` (in the x[i] dialect) over
/// `xData` for x[0], with the other x[i] as scalar parameters, through the CPU
/// or the CUDA implementation of RooBatchCompute::computeExprProgram().
///
/// The CUDA call is made the way RooFit makes it: the vector input already
/// lives on the device, the scalar inputs are spans of size 1 that still live
/// on the host, and the output is a device buffer. Reading the output through
/// hostReadPtr() copies it back on the blocking default stream, which waits
/// for the kernel.
std::vector<double> evalProgram(RooExprEvaluator::Program const &prog, std::vector<double> const &xData,
                                std::vector<double> const &scalarVals, bool useGPU)
{
   const std::size_t n = xData.size();
   std::vector<double> out(n);
   std::span<const RooBatchCompute::ExprInstr> code{prog.code.data(), prog.code.size()};

   if (!useGPU) {
      std::vector<std::span<const double>> spans;
      spans.emplace_back(xData.data(), n);
      for (std::size_t i = 1; i < scalarVals.size(); ++i) {
         spans.emplace_back(&scalarVals[i], 1);
      }
      RooBatchCompute::computeExprProgram({}, code, prog.stackDepth, out, {spans.data(), spans.size()});
      return out;
   }

   auto mgr = RooBatchCompute::dispatchCUDA->createBufferManager();
   auto inBuffer = mgr->makePinnedBuffer(n);
   std::copy(xData.begin(), xData.end(), inBuffer->hostWritePtr());
   auto outBuffer = mgr->makePinnedBuffer(n);

   std::vector<std::span<const double>> spans;
   spans.emplace_back(inBuffer->deviceReadPtr(), n);
   for (std::size_t i = 1; i < scalarVals.size(); ++i) {
      spans.emplace_back(&scalarVals[i], 1);
   }

   RooBatchCompute::CudaInterface::CudaStream *stream = RooBatchCompute::dispatchCUDA->newCudaStream();
   RooBatchCompute::Config cfg;
   cfg.setCudaStream(stream);
   RooBatchCompute::computeExprProgram(cfg, code, prog.stackDepth, {outBuffer->deviceWritePtr(), n},
                                       {spans.data(), spans.size()});
   double const *host = outBuffer->hostReadPtr();
   out.assign(host, host + n);
   RooBatchCompute::dispatchCUDA->deleteCudaStream(stream);
   return out;
}

/// Compare the CPU and the GPU batch evaluation of `expr` event by event.
void expectCudaMatchesCpu(std::string const &expr, std::vector<double> const &xData,
                          std::vector<double> const &scalarVals)
{
   auto prog = RooFormulaParser::compile(expr, scalarVals.size());
   ASSERT_TRUE(prog) << expr;
   ASSERT_TRUE(prog->cudaCapable) << expr;

   auto cpu = evalProgram(*prog, xData, scalarVals, /*useGPU=*/false);
   auto gpu = evalProgram(*prog, xData, scalarVals, /*useGPU=*/true);
   ASSERT_EQ(cpu.size(), xData.size()) << expr;
   ASSERT_EQ(gpu.size(), xData.size()) << expr;
   for (std::size_t i = 0; i < xData.size(); ++i) {
      ASSERT_TRUE(cudaAgrees(gpu[i], cpu[i]))
         << expr << "\n  event " << i << " (x[0] = " << xData[i] << "): gpu = " << std::hexfloat << gpu[i]
         << " cpu = " << cpu[i] << std::defaultfloat;
   }
}

constexpr std::size_t kNEvents = 197; // three full 64-event chunks plus a remainder

std::vector<double> makeInputs()
{
   std::mt19937 rng{20260903u};
   std::vector<double> xData(kNEvents);
   for (double &v : xData) {
      v = uniformDouble(rng, -3.0, 3.0);
   }
   // Values that drive log/sqrt/acosh into NaN and 1/x into Inf, so that the
   // propagation of special values is compared too.
   xData[0] = 0.0;
   xData[1] = -1.5;
   xData[2] = 1.0;
   return xData;
}

class RooFormulaEvaluatorCuda : public testing::Test {
protected:
   void SetUp() override
   {
      static const bool usable = cudaUsable();
      if (!usable) {
         GTEST_SKIP() << "no usable CUDA device";
      }
   }
};

} // namespace

/// Every function spelling in the allow-list, on the GPU against the CPU. A
/// new RooFormulaFunctions entry without a device implementation shows up
/// here: either its programs stop being cudaCapable (caught by the coverage
/// expectation below), or its ExprFunc has no case in the CUDA interpreter and
/// the values disagree.
TEST_F(RooFormulaEvaluatorCuda, Functions)
{
   const std::vector<double> xData = makeInputs();
   const std::vector<double> scalarVals{0.0, 1.7, 0.4};

   auto const *tab = RooFormulaFunctions::table();
   int nChecked = 0;
   for (std::size_t i = 0; i < RooFormulaFunctions::tableSize(); ++i) {
      RooFormulaFunctions::Entry const &entry = tab[i];
      // Build a call with the right number of arguments, mixing the vector
      // input with the two scalar parameters.
      static const char *const kArgs[] = {"x[0]", "x[1]", "x[2]", "x[1]"};
      std::string call = std::string{entry.name} + "(";
      for (unsigned int iArg = 0; iArg < entry.arity; ++iArg) {
         call += (iArg ? ", " : "") + std::string{kArgs[iArg]};
      }
      call += ")";
      // Wrapped so that a bool- or int-typed result still exercises the
      // surrounding double arithmetic.
      const std::string expr = "1.0 * (" + call + ") + 0.5 * x[0]";

      auto prog = RooFormulaParser::compile(expr, scalarVals.size());
      ASSERT_TRUE(prog) << expr;
      EXPECT_TRUE(prog->cudaCapable) << "no device implementation for " << entry.name << "/" << int(entry.arity);
      if (!prog->cudaCapable) {
         continue;
      }
      ++nChecked;
      SCOPED_TRACE(entry.name);
      expectCudaMatchesCpu(expr, xData, scalarVals);
   }
   EXPECT_GT(nChecked, 90) << "the allow-list sweep did not run";
}

/// Operators, precedence, comparisons, logical operators, the ternary, and the
/// propagation of Inf and NaN through them.
TEST_F(RooFormulaEvaluatorCuda, Operators)
{
   const std::vector<double> xData = makeInputs();
   const std::vector<double> scalarVals{0.0, 1.7, 0.4};

   const std::vector<std::string> exprs{
      "x[0]*x[1] + x[2]",
      "x[0] - x[1]*x[2] / (x[0]*x[0] + 0.25)",
      "1.0/x[0]",                // Inf at x[0] == 0
      "log(x[0]) + sqrt(x[0])",  // NaN for negative x[0]
      "-x[0] + +x[1] + !(x[0])", //
      "(x[0] > 0.5) * 3 + (x[0] <= x[1]) + (x[0] == 0.0) + (x[0] != x[2])",
      "(x[0] != 0.0 && x[1] > 0.0) + (x[0] < 0.0 || x[2] > 0.0)",
      "x[0] > 0 ? log(x[0]) : -x[0]", // both branches evaluated
      "x[0]^2 + x[1]**3 + x[0]^x[2]", // Sq and Pow
      "int(x[0] * 10) + sq(x[1])",
      "0.5*exp(-0.5*(x[0]-x[2])*(x[0]-x[2])/(x[1]*x[1])) + 0.3*sin(0.5*x[0]+x[2])*cos(x[0]*x[1])"
      " + 0.2/(1.0+x[0]*x[0]) + sqrt(abs(x[0]*x[2])+1.0) + 0.1*log(1.0+exp(-x[0]))",
   };
   for (std::string const &expr : exprs) {
      expectCudaMatchesCpu(expr, xData, scalarVals);
   }
}

/// Batch sizes around the chunk and grid boundaries of the two interpreters,
/// including the sizes where the grid-stride loop wraps.
TEST_F(RooFormulaEvaluatorCuda, BatchSizes)
{
   const std::string expr = "x[0]*x[1] + sin(x[0])*cos(x[2]) + (x[0] > 0.5 ? log(x[0]) : -x[0]) + 1.0/x[0]";
   const std::vector<double> scalarVals{0.0, 1.5, 0.7};

   // No single-event batch: the CUDA backend tells a per-event device array
   // from a broadcast host value by the span size, which is only unambiguous
   // for more than one event -- and RooFit never schedules a one-event batch
   // on the GPU (see the memory convention on
   // RooBatchComputeInterface::computeExprProgram).
   std::mt19937 rng{424242u};
   for (std::size_t n : {std::size_t(2), std::size_t(63), std::size_t(64), std::size_t(65), std::size_t(512),
                         std::size_t(513), std::size_t(43009), std::size_t(100000)}) {
      std::vector<double> xData(n);
      for (double &v : xData) {
         v = uniformDouble(rng, -3.0, 3.0);
      }
      xData[0] = 0.0;
      SCOPED_TRACE("n = " + std::to_string(n));
      expectCudaMatchesCpu(expr, xData, scalarVals);
   }
}

/// A scalar input (a span of size 1) still lives on the host when the kernel
/// runs, so the CUDA backend has to stage it to the device itself. The trailing
/// dependent the formula does not use gets an empty span, which must never be
/// dereferenced.
TEST_F(RooFormulaEvaluatorCuda, ScalarBroadcastAndUnusedDependent)
{
   const std::vector<double> xData = makeInputs();
   // x[2] is never referenced: its dependent ends up with an empty span.
   expectCudaMatchesCpu("x[0]*x[1] + exp(-x[1])", xData, {0.0, 1.7, 0.4});
}

/// Only formulas on the JIT-free backend go to the GPU: a formula that falls
/// back to TFormula JIT-compiles host code, which cannot run on the device.
TEST_F(RooFormulaEvaluatorCuda, FallbackStaysOnCpu)
{
   RooRealVar x{"x", "x", 1.0, -10.0, 10.0};

   RooFormulaVar ast{"ast", "x*x + sin(x)", RooArgList{x}};
   EXPECT_TRUE(ast.canComputeBatchWithCuda());

   // An integer constant expression that overflows int in cling: not supported
   // by the JIT-free parser, so this falls back to the TFormula backend.
   RooFormulaVar fallback{"fallback", "x*(100000*100000)", RooArgList{x}};
   EXPECT_FALSE(fallback.canComputeBatchWithCuda());

   RooGenericPdf pdf{"pdf", "exp(-0.5*x*x) + 0.1", RooArgList{x}};
   EXPECT_TRUE(pdf.canComputeBatchWithCuda());
}

/// A program deeper than the fixed-size per-thread stack of the GPU
/// interpreter must not be scheduled on the GPU.
TEST_F(RooFormulaEvaluatorCuda, DeepProgramStaysOnCpu)
{
   // A right-nested sum keeps one operand live per open group, so the stack
   // depth grows with the nesting.
   std::string expr = "x[0]";
   for (unsigned int i = 0; i <= RooBatchCompute::maxExprProgramStackDepth; ++i) {
      expr = "x[0] + (" + expr + ")";
   }
   auto prog = RooFormulaParser::compile(expr, 1);
   ASSERT_TRUE(prog);
   ASSERT_GT(prog->stackDepth, RooBatchCompute::maxExprProgramStackDepth);
   EXPECT_FALSE(prog->cudaCapable);

   RooRealVar x{"x", "x", 1.0, -10.0, 10.0};
   RooFormulaVar deep{"deep", expr.c_str(), RooArgList{x}};
   EXPECT_FALSE(deep.canComputeBatchWithCuda());
}

/// End-to-end: an unbinned NLL of a RooGenericPdf must come out the same on
/// both backends.
TEST_F(RooFormulaEvaluatorCuda, UnbinnedNllAgrees)
{
   RooRealVar x{"x", "x", -5, 5};
   RooRealVar a{"a", "a", 1.5, 0.1, 5};
   RooRealVar b{"b", "b", 0.3, -2, 2};
   RooGenericPdf pdf{"pdf", "exp(-0.5*(x-b)*(x-b)/(a*a)) + 0.1*sin(x)*sin(x) + 0.05", {x, a, b}};

   std::unique_ptr<RooDataSet> data{pdf.generate(x, 20000)};

   std::unique_ptr<RooAbsReal> nllCpu{pdf.createNLL(*data, RooFit::EvalBackend::Cpu())};
   std::unique_ptr<RooAbsReal> nllGpu{pdf.createNLL(*data, RooFit::EvalBackend::Cuda())};

   const double cpu = nllCpu->getVal();
   const double gpu = nllGpu->getVal();
   EXPECT_LT(std::abs(gpu - cpu), 1e-8 * std::abs(cpu)) << "cpu = " << cpu << " gpu = " << gpu;
}

#endif // ROOFIT_CUDA
