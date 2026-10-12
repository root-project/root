/*
 * Project: RooFit
 * Authors:
 *   Emmanouil Michalainas, CERN, September 2020
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
\file RooBatchCompute.cxx
\class RbcClass
\ingroup roofit_dev_docs_batchcompute

This file contains the code for cpu computations using the RooBatchCompute library.
**/

#include "RooBatchCompute.h"
#include "RooNaNPacker.h"
#include "RooVDTHeaders.h"
#include "Batches.h"

#include <ROOT/RConfig.hxx>
#include <RConfigure.h>

#ifdef R__USE_IMT
#include <ROOT/TSeq.hxx>
#include <ROOT/TThreadExecutor.hxx>
#endif

#include <Math/Util.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>

#include <vector>

#ifndef RF_ARCH
#error "RF_ARCH should always be defined"
#endif

namespace RooBatchCompute {
namespace RF_ARCH {

namespace {

void fillBatches(Batches &batches, double *output, size_t nEvents, std::size_t nBatches, ArgSpan extraArgs)
{
   batches.extra = extraArgs.data();
   batches.nEvents = nEvents;
   batches.nBatches = nBatches;
   batches.nExtra = extraArgs.size();
   batches.output = output;
}

void fillArrays(std::span<Batch> arrays, VarSpan vars, std::size_t nEvents)
{
   for (std::size_t i = 0; i < vars.size(); i++) {
      arrays[i]._array = vars[i].data();
      arrays[i]._isVector = vars[i].empty() || vars[i].size() >= nEvents;
   }
}

inline void advance(Batches &batches, std::size_t nEvents)
{
   for (std::size_t i = 0; i < batches.nBatches; i++) {
      Batch &arg = batches.args[i];
      arg._array += arg._isVector * nEvents;
   }
   batches.output += nEvents;
}

/// Run one compute function over the event range [begin, begin + count).
/// The `_isVector` flags of the inputs are determined by the total number of
/// events, such that the same inputs are considered per-event arrays no matter
/// how the full range is split into sub-ranges.
void computeRange(void (*computeFn)(Batches &), double *output, std::size_t totalNEvents, VarSpan vars,
                  ArgSpan extraArgs, std::size_t begin, std::size_t count)
{
   std::vector<Batch> arrays(vars.size());
   Batches batches;
   fillBatches(batches, output, count, vars.size(), extraArgs);
   fillArrays(arrays, vars, totalNEvents);
   batches.args = arrays.data();
   advance(batches, begin);

   std::size_t events = count;
   batches.nEvents = bufferSize;
   while (events > bufferSize) {
      computeFn(batches);
      advance(batches, bufferSize);
      events -= bufferSize;
   }
   batches.nEvents = events;
   computeFn(batches);
}

#ifdef R__USE_IMT

// The number of events per task when computing multi-threaded. The chunk
// boundaries must not depend on the number of threads, such that also the
// results of the multi-threaded reductions are bitwise independent of the
// requested number of threads.
constexpr std::size_t parallelChunkSize = 16384;

// Batches with fewer events than this are always evaluated single-threaded,
// because the scheduling overhead would exceed the gain from parallelization.
constexpr std::size_t minParallelSize = 2 * parallelChunkSize;

std::size_t numChunks(std::size_t nEvents)
{
   return (nEvents + parallelChunkSize - 1) / parallelChunkSize;
}

/// Get a cached executor for the given number of threads. All executors share
/// ROOT's global thread pool: the first executor created in the process
/// determines its size, and requests for a different number of threads later
/// on print a warning. Sharing one pool makes sure that parallel evaluations
/// in several fits (or on top of user-level parallelism) running at the same
/// time don't oversubscribe the machine.
ROOT::TThreadExecutor &executorFor(int nThreads)
{
   static std::mutex mutex;
   // The map is deliberately leaked: this library is loaded with dlopen(), so
   // there is no guaranteed destruction order between its statics and the TBB
   // runtime underlying the executor, and destroying the thread pool after
   // TBB tore down its scheduler can crash or hang at process exit.
   static auto &executors = *new std::map<int, std::unique_ptr<ROOT::TThreadExecutor>>;
   std::lock_guard<std::mutex> lock{mutex};
   auto &executor = executors[nThreads];
   if (!executor) {
      executor = std::make_unique<ROOT::TThreadExecutor>(nThreads);
   }
   return *executor;
}

/// Call func(iChunk) in parallel, using up to nThreads threads, for the
/// fixed-size event chunks covering [0, nEvents).
template <class Func>
void parallelForChunks(int nThreads, std::size_t nEvents, Func &&func)
{
   ROOT::TThreadExecutor &executor = executorFor(nThreads);
   // Usually, the chunks are handed out to the threads dynamically one by one
   // for optimal load balancing. But if the shared thread pool has more
   // threads than requested (because something else initialized it with a
   // higher concurrency before), the loop is instead split statically into at
   // most nThreads tasks to still honor the requested number of threads.
   const unsigned int nTasks = executor.GetPoolSize() > static_cast<unsigned int>(nThreads) ? nThreads : 0;
   const ROOT::TSeq<unsigned int> chunkIndices{0u, static_cast<unsigned int>(numChunks(nEvents))};
   executor.Foreach([&](unsigned int iChunk) { func(iChunk); }, chunkIndices, nTasks);
}

/// First event of a given fixed-size chunk.
std::size_t chunkBegin(std::size_t iChunk)
{
   return iChunk * parallelChunkSize;
}

/// Number of events in a given fixed-size chunk (the last one can be shorter).
std::size_t chunkSize(std::size_t iChunk, std::size_t nEvents)
{
   return std::min(parallelChunkSize, nEvents - chunkBegin(iChunk));
}

#endif // R__USE_IMT

bool useParallelEvaluation(RooBatchCompute::Config const &cfg, std::size_t nEvents)
{
#ifdef R__USE_IMT
   return cfg.nThreads() > 1 && nEvents >= minParallelSize;
#else
   (void)cfg;
   (void)nEvents;
   return false;
#endif
}

} // namespace

std::vector<void (*)(Batches &)> getFunctions();

/// This class overrides some RooBatchComputeInterface functions, for the
/// purpose of providing a CPU specific implementation of the library.
class RooBatchComputeClass : public RooBatchComputeInterface {
public:
   RooBatchComputeClass() : _computeFunctions(getFunctions())
   {
      // Set the dispatch pointer to this instance of the library upon loading
      dispatchCPU = this;
   }

   Architecture architecture() const override { return Architecture::RF_ARCH; };
   std::string architectureName() const override
   {
      // transform to lower case to match the original architecture name passed to the compiler
      std::string out = _R_QUOTEVAL_(RF_ARCH);
      std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
      return out;
   };

   void compute(Config const &, Computer computer, std::span<double> output, VarSpan vars, ArgSpan extraArgs) override;
   void computeExprProgram(Config const &, std::span<const ExprInstr> code, unsigned int stackDepth,
                           std::span<double> output, VarSpan vars) override;
   double reduceSum(Config const &, InputArr input, size_t n) override;
   ReduceNLLOutput reduceNLL(Config const &, std::span<const double> probas, std::span<const double> weights,
                             std::span<const double> offsetProbas) override;

   std::unique_ptr<AbsBufferManager> createBufferManager() const override;

   CudaInterface::CudaStream *newCudaStream() const override { throw std::bad_function_call(); }
   void deleteCudaStream(CudaInterface::CudaStream *) const override { throw std::bad_function_call(); }
   void synchronizeCudaStream(CudaInterface::CudaStream *) const override { throw std::bad_function_call(); }

private:
   const std::vector<void (*)(Batches &)> _computeFunctions;
};

/** Compute multiple values using optimized functions.
This method creates a Batches object and passes it to the correct compute function.
If the configuration requests more than one thread and the batch is large
enough, the events are processed in fixed-size chunks by parallel tasks.
\param cfg Configuration, steering among other things the number of threads.
\param computer An enum specifying the compute function to be used.
\param output The array where the computation results are stored.
\param vars A std::span containing pointers to the variables involved in the computation.
\param extraArgs An optional std::span containing extra double values that may participate in the computation. **/
void RooBatchComputeClass::compute(Config const &cfg, Computer computer, std::span<double> output, VarSpan vars,
                                   ArgSpan extraArgs)
{
   const std::size_t nEvents = output.size();
   auto computeFn = _computeFunctions[computer];

#ifdef R__USE_IMT
   if (useParallelEvaluation(cfg, nEvents)) {
      const std::size_t nChunks = numChunks(nEvents);

      // Some compute functions use the extra arguments also as scratch space
      // or as output parameters (e.g. for evaluation error counts), so every
      // task works on its own copy and the differences to the original values
      // are merged back afterwards.
      const std::size_t nExtra = extraArgs.size();
      std::vector<double> extraCopies(nChunks * nExtra);
      for (std::size_t iChunk = 0; iChunk < nChunks; ++iChunk) {
         std::copy(extraArgs.begin(), extraArgs.end(), extraCopies.begin() + iChunk * nExtra);
      }

      parallelForChunks(cfg.nThreads(), nEvents, [&](std::size_t iChunk) {
         computeRange(computeFn, output.data(), nEvents, vars, {extraCopies.data() + iChunk * nExtra, nExtra},
                      chunkBegin(iChunk), chunkSize(iChunk, nEvents));
      });

      for (std::size_t k = 0; k < nExtra; ++k) {
         double delta = 0.0;
         for (std::size_t iChunk = 0; iChunk < nChunks; ++iChunk) {
            delta += extraCopies[iChunk * nExtra + k] - extraArgs[k];
         }
         extraArgs[k] += delta;
      }
      return;
   }
#else
   (void)cfg;
#endif

   computeRange(computeFn, output.data(), nEvents, vars, extraArgs, 0, nEvents);
}

namespace {

/// Apply a unary operation in place on a chunk at the top of the value stack.
template <class F>
inline void exprUnaryOp(double *__restrict a, std::size_t len, F f)
{
   for (std::size_t k = 0; k < len; ++k) {
      a[k] = f(a[k]);
   }
}

/// Apply a binary operation on two chunks, storing the result in the first.
template <class F>
inline void exprBinaryOp(double *__restrict a, const double *__restrict b, std::size_t len, F f)
{
   for (std::size_t k = 0; k < len; ++k) {
      a[k] = f(a[k], b[k]);
   }
}

} // namespace

/** Evaluate a postfix expression program over a batch of events.

The evaluation is chunked over bufferSize events, exactly like compute() and
the stack temporaries in ComputeFunctions.cxx, so that all intermediate value
buffers stay resident in L1 cache. Within a chunk, each instruction is applied
across the whole chunk: the per-instruction loops are trivial elementwise
operations that the compiler auto-vectorizes for the target architecture of
each RooBatchCompute library, and the interpreter dispatch cost is amortized
over bufferSize events. Scalar inputs (spans of size 1) are broadcast once per
chunk, hoisting the broadcast decision out of the per-event loop.

Exp/Log/Sin/Cos use the fast vectorizable VDT implementations when ROOT is
built with VDT, exactly like the pdf compute kernels, in which case batch
results can differ from per-event scalar evaluation within the usual
RooBatchCompute batch-vs-scalar tolerance (relative ~5e-14) over the normal
argument range, and the special values are not reproduced either (fast_log()
of a non-positive number returns a finite garbage value instead of -Inf or
NaN). All other operations apply the exact same double-precision operation per
event that the scalar evaluator applies, so without VDT the results are
bitwise identical to scalar evaluation. **/
void RooBatchComputeClass::computeExprProgram(Config const &, std::span<const ExprInstr> code, unsigned int stackDepth,
                                              std::span<double> output, VarSpan vars)
{
   if (stackDepth > maxExprProgramStackDepth) {
      throw std::runtime_error("expression program exceeds the computeExprProgram() stack-depth limit");
   }

   double stack[maxExprProgramStackDepth][bufferSize];

   const std::size_t nEvents = output.size();
   for (std::size_t begin = 0; begin < nEvents; begin += bufferSize) {
      const std::size_t len = std::min(bufferSize, nEvents - begin);
      std::size_t sp = 0;
      for (ExprInstr const &ins : code) {
         switch (ins.op) {
         case ExprOp::Const: {
            const double val = ins.konst;
            double *__restrict out = stack[sp++];
            for (std::size_t k = 0; k < len; ++k) {
               out[k] = val;
            }
            break;
         }
         case ExprOp::Var: {
            std::span<const double> v = vars[ins.arg];
            double *__restrict out = stack[sp++];
            if (v.size() == 1) {
               const double val = v[0];
               for (std::size_t k = 0; k < len; ++k) {
                  out[k] = val;
               }
            } else {
               const double *__restrict in = v.data() + begin;
               for (std::size_t k = 0; k < len; ++k) {
                  out[k] = in[k];
               }
            }
            break;
         }
         case ExprOp::Add:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return a + b; });
            break;
         case ExprOp::Sub:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return a - b; });
            break;
         case ExprOp::Mul:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return a * b; });
            break;
         case ExprOp::Div:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return a / b; });
            break;
         case ExprOp::Neg: exprUnaryOp(stack[sp - 1], len, [](double a) { return -a; }); break;
         case ExprOp::Not: exprUnaryOp(stack[sp - 1], len, [](double a) { return a == 0.0 ? 1.0 : 0.0; }); break;
         case ExprOp::LT:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return a < b ? 1.0 : 0.0; });
            break;
         case ExprOp::LE:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return a <= b ? 1.0 : 0.0; });
            break;
         case ExprOp::GT:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return a > b ? 1.0 : 0.0; });
            break;
         case ExprOp::GE:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return a >= b ? 1.0 : 0.0; });
            break;
         case ExprOp::EQ:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return a == b ? 1.0 : 0.0; });
            break;
         case ExprOp::NE:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return a != b ? 1.0 : 0.0; });
            break;
         case ExprOp::And:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len,
                         [](double a, double b) { return (a != 0.0 && b != 0.0) ? 1.0 : 0.0; });
            break;
         case ExprOp::Or:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len,
                         [](double a, double b) { return (a != 0.0 || b != 0.0) ? 1.0 : 0.0; });
            break;
         case ExprOp::Select: {
            sp -= 2;
            double *__restrict c = stack[sp - 1];
            const double *__restrict a = stack[sp];
            const double *__restrict b = stack[sp + 1];
            for (std::size_t k = 0; k < len; ++k) {
               c[k] = c[k] != 0.0 ? a[k] : b[k];
            }
            break;
         }
         case ExprOp::Pow:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, [](double a, double b) { return std::pow(a, b); });
            break;
         case ExprOp::Sq: exprUnaryOp(stack[sp - 1], len, [](double a) { return a * a; }); break;
         case ExprOp::IntNorm: exprUnaryOp(stack[sp - 1], len, [](double a) { return a + 0.0; }); break;
         case ExprOp::Exp: exprUnaryOp(stack[sp - 1], len, [](double a) { return fast_exp(a); }); break;
         case ExprOp::Log: exprUnaryOp(stack[sp - 1], len, [](double a) { return fast_log(a); }); break;
         case ExprOp::Sin: exprUnaryOp(stack[sp - 1], len, [](double a) { return fast_sin(a); }); break;
         case ExprOp::Cos: exprUnaryOp(stack[sp - 1], len, [](double a) { return fast_cos(a); }); break;
         case ExprOp::Sqrt: exprUnaryOp(stack[sp - 1], len, [](double a) { return std::sqrt(a); }); break;
         case ExprOp::Call1: exprUnaryOp(stack[sp - 1], len, ins.fn1); break;
         case ExprOp::Call2:
            --sp;
            exprBinaryOp(stack[sp - 1], stack[sp], len, ins.fn2);
            break;
         case ExprOp::Call3: {
            sp -= 2;
            double *__restrict a = stack[sp - 1];
            const double *__restrict b = stack[sp];
            const double *__restrict c = stack[sp + 1];
            for (std::size_t k = 0; k < len; ++k) {
               a[k] = ins.fn3(a[k], b[k], c[k]);
            }
            break;
         }
         case ExprOp::Call4: {
            sp -= 3;
            double *__restrict a = stack[sp - 1];
            const double *__restrict b = stack[sp];
            const double *__restrict c = stack[sp + 1];
            const double *__restrict d = stack[sp + 2];
            for (std::size_t k = 0; k < len; ++k) {
               a[k] = ins.fn4(a[k], b[k], c[k], d[k]);
            }
            break;
         }
         }
      }
      const double *__restrict res = stack[0];
      double *__restrict out = output.data() + begin;
      for (std::size_t k = 0; k < len; ++k) {
         out[k] = res[k];
      }
   }
}

namespace {

inline std::pair<double, double> getLog(double prob, ReduceNLLOutput &out)
{
   if (prob <= 0.0) {
      out.nNonPositiveValues++;
      return {std::log(prob), -prob};
   }

   if (std::isinf(prob)) {
      out.nInfiniteValues++;
   }

   if (std::isnan(prob)) {
      out.nNaNValues++;
      return {prob, RooNaNPacker::unpackNaN(prob)};
   }

   return {std::log(prob), 0.0};
}

} // namespace

double RooBatchComputeClass::reduceSum(Config const &cfg, InputArr input, size_t n)
{
#ifdef R__USE_IMT
   if (useParallelEvaluation(cfg, n)) {
      const std::size_t nChunks = numChunks(n);
      std::vector<ROOT::Math::KahanSum<double, 4u>> partials(nChunks);
      parallelForChunks(cfg.nThreads(), n, [&](std::size_t iChunk) {
         const std::size_t begin = chunkBegin(iChunk);
         partials[iChunk] =
            ROOT::Math::KahanSum<double, 4u>::Accumulate(input + begin, input + begin + chunkSize(iChunk, n));
      });
      // Combine the partial sums in fixed chunk order, so that the result
      // doesn't depend on the number of threads.
      ROOT::Math::KahanSum<double, 4u> total;
      for (auto const &partial : partials) {
         total += partial;
      }
      return total.Sum();
   }
#else
   (void)cfg;
#endif
   return ROOT::Math::KahanSum<double, 4u>::Accumulate(input, input + n).Sum();
}

namespace {

/// Accumulator for the negative log-likelihood reduction over one event range.
struct NLLPartialResult {
   ROOT::Math::KahanSum<double> nllSum;
   double badness = 0.0;
   ReduceNLLOutput counters;
};

void reduceNLLRange(std::span<const double> probas, std::span<const double> weights,
                    std::span<const double> offsetProbas, std::size_t begin, std::size_t end, NLLPartialResult &result)
{
   for (std::size_t i = begin; i < end; ++i) {

      if (0. == weights[i])
         continue;

      std::pair<double, double> logOut = getLog(probas.size() == 1 ? probas[0] : probas[i], result.counters);
      double term = logOut.first;
      result.badness += logOut.second;

      if (!offsetProbas.empty()) {
         term -= std::log(offsetProbas[i]);
      }

      term *= -weights[i];

      result.nllSum.Add(term);
   }
}

} // namespace

ReduceNLLOutput RooBatchComputeClass::reduceNLL(Config const &cfg, std::span<const double> probas,
                                                std::span<const double> weights, std::span<const double> offsetProbas)
{
   const std::size_t n = weights.size();
   NLLPartialResult result;

#ifdef R__USE_IMT
   if (useParallelEvaluation(cfg, n)) {
      const std::size_t nChunks = numChunks(n);
      std::vector<NLLPartialResult> partials(nChunks);
      parallelForChunks(cfg.nThreads(), n, [&](std::size_t iChunk) {
         const std::size_t begin = chunkBegin(iChunk);
         reduceNLLRange(probas, weights, offsetProbas, begin, begin + chunkSize(iChunk, n), partials[iChunk]);
      });
      // Combine the partial results in fixed chunk order, so that the result
      // doesn't depend on the number of threads.
      for (auto const &partial : partials) {
         result.nllSum += partial.nllSum;
         result.badness += partial.badness;
         result.counters.nInfiniteValues += partial.counters.nInfiniteValues;
         result.counters.nNonPositiveValues += partial.counters.nNonPositiveValues;
         result.counters.nNaNValues += partial.counters.nNaNValues;
      }
   } else {
      reduceNLLRange(probas, weights, offsetProbas, 0, n, result);
   }
#else
   (void)cfg;
   reduceNLLRange(probas, weights, offsetProbas, 0, n, result);
#endif

   ReduceNLLOutput out = result.counters;
   out.nllSum = result.nllSum.Sum();
   out.nllSumCarry = result.nllSum.Carry();

   if (result.badness != 0.) {
      // Some events with evaluation errors. Return "badness" of errors.
      out.nllSum = RooNaNPacker::packFloatIntoNaN(result.badness);
      out.nllSumCarry = 0.0;
   }

   return out;
}

namespace {

class ScalarBufferContainer {
public:
   ScalarBufferContainer() {}
   ScalarBufferContainer(std::size_t size)
   {
      if (size != 1)
         throw std::runtime_error("ScalarBufferContainer can only be of size 1");
   }

   double const *hostReadPtr() const { return &_val; }
   double const *deviceReadPtr() const { return &_val; }

   double *hostWritePtr() { return &_val; }
   double *deviceWritePtr() { return &_val; }

   void assignFromHost(std::span<const double> input) { _val = input[0]; }
   void assignFromDevice(std::span<const double>) { throw std::bad_function_call(); }

private:
   double _val;
};

class CPUBufferContainer {
public:
   CPUBufferContainer(std::size_t size) : _vec(size) {}

   double const *hostReadPtr() const { return _vec.data(); }
   double const *deviceReadPtr() const
   {
      throw std::bad_function_call();
      return nullptr;
   }

   double *hostWritePtr() { return _vec.data(); }
   double *deviceWritePtr()
   {
      throw std::bad_function_call();
      return nullptr;
   }

   void assignFromHost(std::span<const double> input) { _vec.assign(input.begin(), input.end()); }
   void assignFromDevice(std::span<const double>) { throw std::bad_function_call(); }

private:
   std::vector<double> _vec;
};

template <class Container>
class BufferImpl : public AbsBuffer {
public:
   using Queue = std::queue<std::unique_ptr<Container>>;

   BufferImpl(std::size_t size, Queue &queue) : _queue{queue}
   {
      if (_queue.empty()) {
         _vec = std::make_unique<Container>(size);
      } else {
         _vec = std::move(_queue.front());
         _queue.pop();
      }
   }

   ~BufferImpl() override { _queue.emplace(std::move(_vec)); }

   double const *hostReadPtr() const override { return _vec->hostReadPtr(); }
   double const *deviceReadPtr() const override { return _vec->deviceReadPtr(); }

   double *hostWritePtr() override { return _vec->hostWritePtr(); }
   double *deviceWritePtr() override { return _vec->deviceWritePtr(); }

   void assignFromHost(std::span<const double> input) override { _vec->assignFromHost(input); }
   void assignFromDevice(std::span<const double> input) override { _vec->assignFromDevice(input); }

   Container &vec() { return *_vec; }

private:
   std::unique_ptr<Container> _vec;
   Queue &_queue;
};

using ScalarBuffer = BufferImpl<ScalarBufferContainer>;
using CPUBuffer = BufferImpl<CPUBufferContainer>;

struct BufferQueuesMaps {
   std::map<std::size_t, ScalarBuffer::Queue> scalarBufferQueuesMap;
   std::map<std::size_t, CPUBuffer::Queue> cpuBufferQueuesMap;
};

class BufferManager : public AbsBufferManager {

public:
   BufferManager() : _queuesMaps{std::make_unique<BufferQueuesMaps>()} {}

   std::unique_ptr<AbsBuffer> makeScalarBuffer() override
   {
      return std::make_unique<ScalarBuffer>(1, _queuesMaps->scalarBufferQueuesMap[1]);
   }
   std::unique_ptr<AbsBuffer> makeCpuBuffer(std::size_t size) override
   {
      return std::make_unique<CPUBuffer>(size, _queuesMaps->cpuBufferQueuesMap[size]);
   }
   std::unique_ptr<AbsBuffer> makeGpuBuffer(std::size_t) override { throw std::bad_function_call(); }
   std::unique_ptr<AbsBuffer> makePinnedBuffer(std::size_t, CudaInterface::CudaStream * = nullptr) override
   {
      throw std::bad_function_call();
   }

private:
   std::unique_ptr<BufferQueuesMaps> _queuesMaps;
};

} // namespace

std::unique_ptr<AbsBufferManager> RooBatchComputeClass::createBufferManager() const
{
   return std::make_unique<BufferManager>();
}

/// Static object to trigger the constructor which overwrites the dispatch pointer.
static RooBatchComputeClass computeObj;

} // End namespace RF_ARCH
} // End namespace RooBatchCompute
