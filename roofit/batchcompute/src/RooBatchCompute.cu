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
\file RooBatchCompute.cu
\class RbcClass
\ingroup roofit_dev_docs_batchcompute

This file contains the code for cuda computations using the RooBatchCompute library.
**/

#include "RooBatchCompute.h"
#include "Batches.h"
#include "CudaInterface.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <map>
#include <queue>
#include <stdexcept>
#include <vector>

namespace RooBatchCompute {
namespace CUDA {

constexpr int blockSize = 512;

namespace {

void fillBatches(Batches &batches, double *output, size_t nEvents, std::size_t nBatches, std::size_t nExtraArgs)
{
   batches.nEvents = nEvents;
   batches.nBatches = nBatches;
   batches.nExtra = nExtraArgs;
   batches.output = output;
}

void fillArrays(Batch *arrays, VarSpan vars, double *buffer, double *bufferDevice, std::size_t nEvents)
{
   for (int i = 0; i < vars.size(); i++) {
      const std::span<const double> &span = vars[i];
      arrays[i]._isVector = span.empty() || span.size() >= nEvents;
      if (!arrays[i]._isVector) {
         // In the scalar case, the value is not on the GPU yet, so we have to
         // copy the value to the GPU buffer.
         buffer[i] = span[0];
         arrays[i]._array = bufferDevice + i;
      } else {
         // In the vector input cases, they are already on the GPU, so we can
         // fill be buffer with some dummy value and set the input span
         // directly.
         buffer[i] = 0.0;
         arrays[i]._array = span.data();
      }
   }
}

int getGridSize(std::size_t n)
{
   // The grid size should be not larger than the order of number of streaming
   // multiprocessors (SMs) in an Nvidia GPU. The number 84 was chosen because
   // the developers were using an Nvidia RTX A4500, which has 46 SMs. This was
   // multiplied by a factor of 1.5, as recommended by stackoverflow.
   //
   // But when there are not enough elements to load the GPU, the number should
   // be lower: that's why there is the std::ceil().
   //
   // Note: for grid sizes larger than 512, the Kahan summation kernels give
   // wrong results. This problem is not understood, but also not really worth
   // investigating further, as that number is unreasonably large anyway.
   constexpr int maxGridSize = 84;
   return std::min(int(std::ceil(double(n) / blockSize)), maxGridSize);
}

} // namespace

std::vector<void (*)(Batches &)> getFunctions();

/// This class overrides some RooBatchComputeInterface functions, for the
/// purpose of providing a cuda specific implementation of the library.
class RooBatchComputeClass : public RooBatchComputeInterface {

public:
   RooBatchComputeClass() : _computeFunctions(getFunctions())
   {
      dispatchCUDA = this; // Set the dispatch pointer to this instance of the library upon loading
   }

   Architecture architecture() const override { return Architecture::CUDA; }
   std::string architectureName() const override { return "cuda"; }

   /** Compute multiple values using cuda kernels.
   This method creates a Batches object and passes it to the correct compute function.
   The compute function is launched as a cuda kernel.
   \param computer An enum specifying the compute function to be used.
   \param output The array where the computation results are stored.
   \param vars A std::span containing pointers to the variables involved in the computation.
   \param extraArgs An optional std::span containing extra double values that may participate in the computation. **/
   void compute(RooBatchCompute::Config const &cfg, Computer computer, std::span<double> output, VarSpan vars,
                ArgSpan extraArgs) override
   {
      using namespace CudaInterface;

      std::size_t nEvents = output.size();

      const std::size_t memSize = sizeof(Batches) + vars.size() * sizeof(Batch) + vars.size() * sizeof(double) +
                                  extraArgs.size() * sizeof(double);

      std::vector<char> hostMem(memSize);
      auto batches = reinterpret_cast<Batches *>(hostMem.data());
      auto arrays = reinterpret_cast<Batch *>(batches + 1);
      auto scalarBuffer = reinterpret_cast<double *>(arrays + vars.size());
      auto extraArgsHost = reinterpret_cast<double *>(scalarBuffer + vars.size());

      DeviceArray<char> deviceMem(memSize);
      auto batchesDevice = reinterpret_cast<Batches *>(deviceMem.data());
      auto arraysDevice = reinterpret_cast<Batch *>(batchesDevice + 1);
      auto scalarBufferDevice = reinterpret_cast<double *>(arraysDevice + vars.size());
      auto extraArgsDevice = reinterpret_cast<double *>(scalarBufferDevice + vars.size());

      fillBatches(*batches, output.data(), nEvents, vars.size(), extraArgs.size());
      fillArrays(arrays, vars, scalarBuffer, scalarBufferDevice, nEvents);
      batches->args = arraysDevice;

      if (!extraArgs.empty()) {
         std::copy(std::cbegin(extraArgs), std::cend(extraArgs), extraArgsHost);
         batches->extra = extraArgsDevice;
      }

      copyHostToDevice(hostMem.data(), deviceMem.data(), hostMem.size(), cfg.cudaStream());

      const int gridSize = getGridSize(nEvents);
      _computeFunctions[computer]<<<gridSize, blockSize, 0, *cfg.cudaStream()>>>(*batchesDevice);

      // The compute might have modified the mutable extra args, so we need to
      // copy them back. This can be optimized if necessary in the future by
      // flagging if the extra args were actually changed.
      if (!extraArgs.empty()) {
         copyDeviceToHost(extraArgsDevice, extraArgs.data(), extraArgs.size(), cfg.cudaStream());
      }
   }
   void computeExprProgram(Config const &cfg, std::span<const ExprInstr> code, unsigned int stackDepth,
                           std::span<double> output, VarSpan vars) override;

   /// Return the sum of an input array
   double reduceSum(RooBatchCompute::Config const &cfg, InputArr input, size_t n) override;
   ReduceNLLOutput reduceNLL(RooBatchCompute::Config const &cfg, std::span<const double> probas,
                             std::span<const double> weights, std::span<const double> offsetProbas) override;

   std::unique_ptr<AbsBufferManager> createBufferManager() const override;

   CudaInterface::CudaEvent *newCudaEvent(bool forTiming) const override
   {
      return new CudaInterface::CudaEvent{forTiming};
   }
   CudaInterface::CudaStream *newCudaStream() const override { return new CudaInterface::CudaStream{}; }
   void deleteCudaEvent(CudaInterface::CudaEvent *event) const override { delete event; }
   void deleteCudaStream(CudaInterface::CudaStream *stream) const override { delete stream; }

   void cudaEventRecord(CudaInterface::CudaEvent *event, CudaInterface::CudaStream *stream) const override
   {
      CudaInterface::cudaEventRecord(*event, *stream);
   }
   void cudaStreamWaitForEvent(CudaInterface::CudaStream *stream, CudaInterface::CudaEvent *event) const override
   {
      stream->waitForEvent(*event);
   }
   bool cudaStreamIsActive(CudaInterface::CudaStream *stream) const override { return stream->isActive(); }

private:
   const std::vector<void (*)(Batches &)> _computeFunctions;

}; // End class RooBatchComputeClass

namespace {

/// TMath::Gaus, ported for the device (the host implementation lives in
/// TMath.cxx, which is not device code).
inline __device__ double gaus(double x, double mean, double sigma, bool norm)
{
   if (sigma == 0.0)
      return 1.e30;
   const double arg = (x - mean) / sigma;
   // for |arg| > 39 the result is zero in double precision
   if (arg < -39.0 || arg > 39.0)
      return 0.0;
   const double res = ::exp(-0.5 * arg * arg);
   return norm ? res / (2.50662827463100024 * sigma) : res; // sqrt(2*Pi)
}

/// Device counterpart of the unary functions in RooFitCore's formula
/// allow-list. There is one case per ExprFunc value, so the compiler warns
/// (-Wswitch) when the enum grows without a device implementation. Values with
/// a different arity, and ExprFunc::None, cannot reach here: the parser only
/// marks a program cudaCapable when every call has a device implementation,
/// and the arity follows from the opcode.
inline __device__ double applyFunc1(ExprFunc f, double a)
{
   switch (f) {
   case ExprFunc::Exp: return ::exp(a);
   case ExprFunc::Log: return ::log(a);
   case ExprFunc::Sin: return ::sin(a);
   case ExprFunc::Cos: return ::cos(a);
   case ExprFunc::Sqrt: return ::sqrt(a);
   case ExprFunc::Log10: return ::log10(a);
   case ExprFunc::Tan: return ::tan(a);
   case ExprFunc::ASin: return ::asin(a);
   case ExprFunc::ACos: return ::acos(a);
   case ExprFunc::ATan: return ::atan(a);
   case ExprFunc::SinH: return ::sinh(a);
   case ExprFunc::CosH: return ::cosh(a);
   case ExprFunc::TanH: return ::tanh(a);
   case ExprFunc::ASinH: return ::asinh(a);
   case ExprFunc::ACosH: return ::acosh(a);
   case ExprFunc::ATanH: return ::atanh(a);
   case ExprFunc::Floor: return ::floor(a);
   case ExprFunc::Ceil: return ::ceil(a);
   // TMath::Erf and TMath::Erfc go through Cephes on the host; the device has
   // only its own erf/erfc. Like every other function here, they agree with
   // the host only to within the batch-vs-scalar tolerance.
   case ExprFunc::Erf:
   case ExprFunc::TMathErf: return ::erf(a);
   case ExprFunc::Erfc:
   case ExprFunc::TMathErfc: return ::erfc(a);
   case ExprFunc::TGamma: return ::tgamma(a);
   case ExprFunc::LGamma: return ::lgamma(a);
   case ExprFunc::Abs: return ::fabs(a);
   case ExprFunc::CastInt: return static_cast<double>(static_cast<int>(a));
   case ExprFunc::Square: return a * a;
   case ExprFunc::SignBit: return ::signbit(a) ? 1.0 : 0.0;
   case ExprFunc::Gaus1: return gaus(a, 0.0, 1.0, false);
   // Values of another arity, and None, cannot reach here. They are listed so
   // that -Wswitch flags a new ExprFunc that has no device implementation.
   case ExprFunc::None:
   case ExprFunc::Pow:
   case ExprFunc::ATan2:
   case ExprFunc::TMathATan2:
   case ExprFunc::Fmod:
   case ExprFunc::StdMin:
   case ExprFunc::StdMax:
   case ExprFunc::TMathMin:
   case ExprFunc::TMathMax:
   case ExprFunc::CopySign:
   case ExprFunc::Gaus2:
   case ExprFunc::Gaus3:
   case ExprFunc::Gaus4: break;
   }
   return ::nan("");
}

/// Device counterpart of the binary functions in the allow-list.
inline __device__ double applyFunc2(ExprFunc f, double a, double b)
{
   switch (f) {
   case ExprFunc::Pow: return ::pow(a, b);
   case ExprFunc::ATan2: return ::atan2(a, b);
   case ExprFunc::TMathATan2:
      // TMath::ATan2 special-cases x == 0 instead of leaving it to atan2().
      if (b != 0.0)
         return ::atan2(a, b);
      return a == 0.0 ? 0.0 : (a > 0.0 ? 1.5707963267948966 : -1.5707963267948966);
   case ExprFunc::Fmod: return ::fmod(a, b);
   // std::min/max and TMath::Min/Max differ in how they order NaN; both
   // orderings are reproduced exactly as the host comparisons are written.
   case ExprFunc::StdMin: return b < a ? b : a;
   case ExprFunc::StdMax: return a < b ? b : a;
   case ExprFunc::TMathMin: return a <= b ? a : b;
   case ExprFunc::TMathMax: return a >= b ? a : b;
   case ExprFunc::CopySign: return ::copysign(a, b);
   case ExprFunc::Gaus2: return gaus(a, b, 1.0, false);
   // Values of another arity, and None, cannot reach here. They are listed so
   // that -Wswitch flags a new ExprFunc that has no device implementation.
   case ExprFunc::None:
   case ExprFunc::Exp:
   case ExprFunc::Log:
   case ExprFunc::Sin:
   case ExprFunc::Cos:
   case ExprFunc::Sqrt:
   case ExprFunc::Log10:
   case ExprFunc::Tan:
   case ExprFunc::ASin:
   case ExprFunc::ACos:
   case ExprFunc::ATan:
   case ExprFunc::SinH:
   case ExprFunc::CosH:
   case ExprFunc::TanH:
   case ExprFunc::ASinH:
   case ExprFunc::ACosH:
   case ExprFunc::ATanH:
   case ExprFunc::Floor:
   case ExprFunc::Ceil:
   case ExprFunc::Erf:
   case ExprFunc::Erfc:
   case ExprFunc::TMathErf:
   case ExprFunc::TMathErfc:
   case ExprFunc::TGamma:
   case ExprFunc::LGamma:
   case ExprFunc::Abs:
   case ExprFunc::CastInt:
   case ExprFunc::Square:
   case ExprFunc::SignBit:
   case ExprFunc::Gaus1:
   case ExprFunc::Gaus3:
   case ExprFunc::Gaus4: break;
   }
   return ::nan("");
}

/// Evaluate one expression program per thread over a batch of events.
///
/// The loops are interchanged with respect to the CPU interpreter: there, one
/// instruction is applied across a chunk of events; here, each thread walks
/// the whole program for its own events, keeping the value stack in per-thread
/// local memory. Input reads are then coalesced across the warp, and no
/// intermediate value ever reaches global memory. All threads execute the same
/// instruction at the same time, so the program itself is read uniformly and
/// stays in cache.
__global__ void exprProgramKernel(const ExprInstr *__restrict code, unsigned int nInstr, const Batch *__restrict vars,
                                  double *__restrict output, std::size_t nEvents)
{
   const std::size_t nThreadsTotal = static_cast<std::size_t>(blockDim.x) * gridDim.x;
   for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nEvents; i += nThreadsTotal) {
      double stack[maxExprProgramStackDepth];
      unsigned int sp = 0;
      for (unsigned int k = 0; k < nInstr; ++k) {
         const ExprInstr ins = code[k];
         switch (ins.op) {
         case ExprOp::Const: stack[sp++] = ins.konst; break;
         case ExprOp::Var: stack[sp++] = vars[ins.arg][i]; break;
         case ExprOp::Add:
            --sp;
            stack[sp - 1] += stack[sp];
            break;
         case ExprOp::Sub:
            --sp;
            stack[sp - 1] -= stack[sp];
            break;
         case ExprOp::Mul:
            --sp;
            stack[sp - 1] *= stack[sp];
            break;
         case ExprOp::Div:
            --sp;
            stack[sp - 1] /= stack[sp];
            break;
         case ExprOp::Neg: stack[sp - 1] = -stack[sp - 1]; break;
         case ExprOp::Not: stack[sp - 1] = stack[sp - 1] == 0.0 ? 1.0 : 0.0; break;
         case ExprOp::LT:
            --sp;
            stack[sp - 1] = stack[sp - 1] < stack[sp] ? 1.0 : 0.0;
            break;
         case ExprOp::LE:
            --sp;
            stack[sp - 1] = stack[sp - 1] <= stack[sp] ? 1.0 : 0.0;
            break;
         case ExprOp::GT:
            --sp;
            stack[sp - 1] = stack[sp - 1] > stack[sp] ? 1.0 : 0.0;
            break;
         case ExprOp::GE:
            --sp;
            stack[sp - 1] = stack[sp - 1] >= stack[sp] ? 1.0 : 0.0;
            break;
         case ExprOp::EQ:
            --sp;
            stack[sp - 1] = stack[sp - 1] == stack[sp] ? 1.0 : 0.0;
            break;
         case ExprOp::NE:
            --sp;
            stack[sp - 1] = stack[sp - 1] != stack[sp] ? 1.0 : 0.0;
            break;
         case ExprOp::And:
            --sp;
            stack[sp - 1] = (stack[sp - 1] != 0.0 && stack[sp] != 0.0) ? 1.0 : 0.0;
            break;
         case ExprOp::Or:
            --sp;
            stack[sp - 1] = (stack[sp - 1] != 0.0 || stack[sp] != 0.0) ? 1.0 : 0.0;
            break;
         case ExprOp::Select:
            // Both branches were evaluated, exactly as on the CPU.
            sp -= 2;
            stack[sp - 1] = stack[sp - 1] != 0.0 ? stack[sp] : stack[sp + 1];
            break;
         case ExprOp::Pow:
            --sp;
            stack[sp - 1] = ::pow(stack[sp - 1], stack[sp]);
            break;
         case ExprOp::Sq: stack[sp - 1] *= stack[sp - 1]; break;
         case ExprOp::IntNorm: stack[sp - 1] += 0.0; break;
         case ExprOp::Exp: stack[sp - 1] = ::exp(stack[sp - 1]); break;
         case ExprOp::Log: stack[sp - 1] = ::log(stack[sp - 1]); break;
         case ExprOp::Sin: stack[sp - 1] = ::sin(stack[sp - 1]); break;
         case ExprOp::Cos: stack[sp - 1] = ::cos(stack[sp - 1]); break;
         case ExprOp::Sqrt: stack[sp - 1] = ::sqrt(stack[sp - 1]); break;
         case ExprOp::Call1: stack[sp - 1] = applyFunc1(ins.func, stack[sp - 1]); break;
         case ExprOp::Call2:
            --sp;
            stack[sp - 1] = applyFunc2(ins.func, stack[sp - 1], stack[sp]);
            break;
         case ExprOp::Call3:
            sp -= 2;
            // TMath::Gaus(x, mean, sigma) is the only ternary entry.
            stack[sp - 1] = gaus(stack[sp - 1], stack[sp], stack[sp + 1], false);
            break;
         case ExprOp::Call4:
            sp -= 3;
            // TMath::Gaus(x, mean, sigma, norm) is the only quaternary entry.
            stack[sp - 1] = gaus(stack[sp - 1], stack[sp], stack[sp + 1], stack[sp + 2] != 0.0);
            break;
         }
      }
      output[i] = stack[0];
   }
}

} // namespace

/** Evaluate a postfix expression program over a batch of events on the GPU.

One thread evaluates the whole program for one event (with a grid-stride loop
over the batch), so the per-event value stack lives in per-thread local memory
and no intermediate result is written to global memory.

The device math functions are not the host's libm, so, unlike the CPU
backends, the results are not bitwise identical to per-event scalar evaluation
on the host; they agree within the usual RooBatchCompute batch-vs-scalar
tolerance. Only programs that RooFitCore marked as cudaCapable get here: their
stack fits maxExprProgramStackDepth and every call has a device
implementation. **/
void RooBatchComputeClass::computeExprProgram(Config const &cfg, std::span<const ExprInstr> code,
                                              unsigned int stackDepth, std::span<double> output, VarSpan vars)
{
   using namespace CudaInterface;

   if (stackDepth > maxExprProgramStackDepth) {
      throw std::runtime_error("expression program exceeds the computeExprProgram() stack-depth limit");
   }

   const std::size_t nEvents = output.size();
   if (nEvents == 0) {
      return;
   }

   // One host-side staging block that mirrors the device block: the program,
   // the per-variable input descriptors, and the values of the scalar inputs
   // (which, unlike the vector inputs, are not on the device yet).
   const std::size_t codeBytes = code.size() * sizeof(ExprInstr);
   const std::size_t varsBytes = vars.size() * sizeof(Batch);
   const std::size_t scalarBytes = vars.size() * sizeof(double);
   const std::size_t memSize = codeBytes + varsBytes + scalarBytes;

   std::vector<char> hostMem(memSize);
   auto codeHost = reinterpret_cast<ExprInstr *>(hostMem.data());
   auto varsHost = reinterpret_cast<Batch *>(hostMem.data() + codeBytes);
   auto scalarsHost = reinterpret_cast<double *>(hostMem.data() + codeBytes + varsBytes);

   DeviceArray<char> deviceMem(memSize);
   auto codeDevice = reinterpret_cast<ExprInstr *>(deviceMem.data());
   auto varsDevice = reinterpret_cast<Batch *>(deviceMem.data() + codeBytes);
   auto scalarsDevice = reinterpret_cast<double *>(deviceMem.data() + codeBytes + varsBytes);

   std::copy(code.begin(), code.end(), codeHost);
   for (std::size_t i = 0; i < vars.size(); ++i) {
      std::span<const double> span = vars[i];
      // Exactly the rule the CPU backend applies: a span of more than one
      // value is per-event, anything else is broadcast. The distinction is not
      // just about broadcasting here -- only the per-event inputs are on the
      // device. A scalar input is the host-side value buffer of a node the
      // Evaluator computed on the CPU, and an empty span belongs to a
      // dependent the formula does not use (no Var instruction reads it, but
      // it must still not leave a host pointer for the kernel to follow), so
      // both are staged into the device scalar buffer.
      const bool isVector = span.size() > 1;
      varsHost[i]._isVector = isVector;
      varsHost[i]._array = isVector ? span.data() : scalarsDevice + i;
      // Only a scalar span may be dereferenced here: a per-event span points
      // into device memory.
      scalarsHost[i] = isVector || span.empty() ? 0.0 : span[0];
   }

   copyHostToDevice(hostMem.data(), deviceMem.data(), hostMem.size(), cfg.cudaStream());

   const int gridSize = getGridSize(nEvents);
   exprProgramKernel<<<gridSize, blockSize, 0, *cfg.cudaStream()>>>(codeDevice, static_cast<unsigned int>(code.size()),
                                                                    varsDevice, output.data(), nEvents);
}

inline __device__ void kahanSumUpdate(double &sum, double &carry, double a, double otherCarry)
{
   // c is zero the first time around. Then is done a summation as the c variable is NEGATIVE
   const double y = a - (carry + otherCarry);
   const double t = sum + y; // Alas, sum is big, y small, so low-order digits of y are lost.

   // (t - sum) cancels the high-order part of y; subtracting y recovers NEGATIVE (low part of y)
   carry = (t - sum) - y;

   // Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
   sum = t;
}

// This is the same implementation of the ROOT::Math::KahanSum::operator+=(KahanSum) but in GPU
inline __device__ void kahanSumReduction(double *shared, size_t n, double *__restrict__ result, int carry_index)
{
   // Stride in first iteration = half of the block dim. Then the half of the half...
   for (int i = blockDim.x / 2; i > 0; i >>= 1) {
      if (threadIdx.x < i && (threadIdx.x + i) < n) {
         kahanSumUpdate(shared[threadIdx.x], shared[carry_index], shared[threadIdx.x + i], shared[carry_index + i]);
      }
      __syncthreads();
   } // Next time around, the lost low part will be added to y in a fresh attempt.
     // Wait until all threads of the block have finished its work

   if (threadIdx.x == 0) {
      result[blockIdx.x] = shared[0];
      result[blockIdx.x + gridDim.x] = shared[carry_index];
   }
}

__global__ void kahanSum(const double *__restrict__ input, const double *__restrict__ carries, size_t n,
                         double *__restrict__ result, bool nll)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   int carry_index = threadIdx.x + blockDim.x;
   const int nThreadsTotal = blockSize * gridDim.x;

   // The first half of the shared memory is for storing the summation and the second half for the carry or compensation
   extern __shared__ double shared[];

   double sum = 0.0;
   double carry = 0.0;

   for (int i = gthIdx; i < n; i += nThreadsTotal) {
      // Note: it does not make sense to use the nll option and provide at the
      // same time external carries.
      double val = nll == 1 ? -std::log(input[i]) : input[i];
      kahanSumUpdate(sum, carry, val, carries ? carries[i] : 0.0);
   }

   shared[thIdx] = sum;
   shared[carry_index] = carry;

   // Wait until all threads in each block have loaded their elements
   __syncthreads();

   kahanSumReduction(shared, n, result, carry_index);
}

__global__ void nllSumKernel(const double *__restrict__ probas, const double *__restrict__ weights,
                             const double *__restrict__ offsetProbas, size_t nProbas, double scalarProba,
                             size_t nWeights, double *__restrict__ result)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   int carry_index = threadIdx.x + blockDim.x;
   const int nThreadsTotal = blockSize * gridDim.x;

   // The first half of the shared memory is for storing the summation and the second half for the carry or compensation
   extern __shared__ double shared[];

   double sum = 0.0;
   double carry = 0.0;

   for (int i = gthIdx; i < nWeights; i += nThreadsTotal) {
      // Note: it does not make sense to use the nll option and provide at the
      // same time external carries.
      double val = -std::log(nProbas == 1 ? scalarProba : probas[i]);
      if (offsetProbas)
         val += std::log(offsetProbas[i]);
      val = weights[i] * val;
      kahanSumUpdate(sum, carry, val, 0.0);
   }

   shared[thIdx] = sum;
   shared[carry_index] = carry;

   // Wait until all threads in each block have loaded their elements
   __syncthreads();

   kahanSumReduction(shared, nWeights, result, carry_index);
}

double RooBatchComputeClass::reduceSum(RooBatchCompute::Config const &cfg, InputArr input, size_t n)
{
   if (n == 0)
      return 0.0;
   const int gridSize = getGridSize(n);
   cudaStream_t stream = *cfg.cudaStream();
   CudaInterface::DeviceArray<double> devOut(2 * gridSize);
   constexpr int shMemSize = 2 * blockSize * sizeof(double);
   kahanSum<<<gridSize, blockSize, shMemSize, stream>>>(input, nullptr, n, devOut.data(), 0);
   kahanSum<<<1, blockSize, shMemSize, stream>>>(devOut.data(), devOut.data() + gridSize, gridSize, devOut.data(), 0);
   double tmp = 0.0;
   CudaInterface::copyDeviceToHost(devOut.data(), &tmp, 1, cfg.cudaStream());
   return tmp;
}

ReduceNLLOutput RooBatchComputeClass::reduceNLL(RooBatchCompute::Config const &cfg, std::span<const double> probas,
                                                std::span<const double> weights, std::span<const double> offsetProbas)
{
   ReduceNLLOutput out;
   if (probas.empty()) {
      return out;
   }
   const int gridSize = getGridSize(weights.size());
   CudaInterface::DeviceArray<double> devOut(2 * gridSize);
   cudaStream_t stream = *cfg.cudaStream();
   constexpr int shMemSize = 2 * blockSize * sizeof(double);

#ifndef NDEBUG
   for (auto span : {probas, weights, offsetProbas}) {
      cudaPointerAttributes attr;
      assert(span.size() == 0 || span.data() == nullptr ||
             (cudaPointerGetAttributes(&attr, span.data()) == cudaSuccess && attr.type == cudaMemoryTypeDevice));
   }
#endif

   nllSumKernel<<<gridSize, blockSize, shMemSize, stream>>>(
      probas.data(), weights.data(), offsetProbas.empty() ? nullptr : offsetProbas.data(), probas.size(),
      probas.size() == 1 ? probas[0] : 0.0, weights.size(), devOut.data());

   kahanSum<<<1, blockSize, shMemSize, stream>>>(devOut.data(), devOut.data() + gridSize, gridSize, devOut.data(), 0);

   double tmpSum = 0.0;
   double tmpCarry = 0.0;
   CudaInterface::copyDeviceToHost(devOut.data(), &tmpSum, 1, cfg.cudaStream());
   CudaInterface::copyDeviceToHost(devOut.data() + 1, &tmpCarry, 1, cfg.cudaStream());

   out.nllSum = tmpSum;
   out.nllSumCarry = tmpCarry;
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
   void assignFromDevice(std::span<const double> input)
   {
      CudaInterface::copyDeviceToHost(input.data(), &_val, input.size(), nullptr);
   }

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
   void assignFromDevice(std::span<const double> input)
   {
      CudaInterface::copyDeviceToHost(input.data(), _vec.data(), input.size(), nullptr);
   }

private:
   std::vector<double> _vec;
};

class GPUBufferContainer {
public:
   GPUBufferContainer(std::size_t size) : _arr(size) {}

   double const *hostReadPtr() const
   {
      throw std::bad_function_call();
      return nullptr;
   }
   double const *deviceReadPtr() const { return _arr.data(); }

   double *hostWritePtr() const
   {
      throw std::bad_function_call();
      return nullptr;
   }
   double *deviceWritePtr() const { return const_cast<double *>(_arr.data()); }

   void assignFromHost(std::span<const double> input)
   {
      CudaInterface::copyHostToDevice(input.data(), deviceWritePtr(), input.size(), nullptr);
   }
   void assignFromDevice(std::span<const double> input)
   {
      CudaInterface::copyDeviceToDevice(input.data(), deviceWritePtr(), input.size(), nullptr);
   }

private:
   CudaInterface::DeviceArray<double> _arr;
};

class PinnedBufferContainer {
public:
   PinnedBufferContainer(std::size_t size) : _arr{size}, _gpuBuffer{size} {}
   std::size_t size() const { return _arr.size(); }

   void setCudaStream(CudaInterface::CudaStream *stream) { _cudaStream = stream; }

   double const *hostReadPtr() const
   {

      if (_lastAccess == LastAccessType::GPU_WRITE) {
         CudaInterface::copyDeviceToHost(_gpuBuffer.deviceReadPtr(), const_cast<double *>(_arr.data()), size(),
                                         _cudaStream);
      }

      _lastAccess = LastAccessType::CPU_READ;
      return const_cast<double *>(_arr.data());
   }
   double const *deviceReadPtr() const
   {

      if (_lastAccess == LastAccessType::CPU_WRITE) {
         CudaInterface::copyHostToDevice(_arr.data(), _gpuBuffer.deviceWritePtr(), size(), _cudaStream);
      }

      _lastAccess = LastAccessType::GPU_READ;
      return _gpuBuffer.deviceReadPtr();
   }

   double *hostWritePtr()
   {
      _lastAccess = LastAccessType::CPU_WRITE;
      return _arr.data();
   }
   double *deviceWritePtr()
   {
      _lastAccess = LastAccessType::GPU_WRITE;
      return _gpuBuffer.deviceWritePtr();
   }

   void assignFromHost(std::span<const double> input) { std::copy(input.begin(), input.end(), hostWritePtr()); }
   void assignFromDevice(std::span<const double> input)
   {
      CudaInterface::copyDeviceToDevice(input.data(), deviceWritePtr(), input.size(), _cudaStream);
   }

private:
   enum class LastAccessType {
      CPU_READ,
      GPU_READ,
      CPU_WRITE,
      GPU_WRITE
   };

   CudaInterface::PinnedHostArray<double> _arr;
   GPUBufferContainer _gpuBuffer;
   CudaInterface::CudaStream *_cudaStream = nullptr;
   mutable LastAccessType _lastAccess = LastAccessType::CPU_READ;
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
using GPUBuffer = BufferImpl<GPUBufferContainer>;
using PinnedBuffer = BufferImpl<PinnedBufferContainer>;

struct BufferQueuesMaps {
   std::map<std::size_t, ScalarBuffer::Queue> scalarBufferQueuesMap;
   std::map<std::size_t, CPUBuffer::Queue> cpuBufferQueuesMap;
   std::map<std::size_t, GPUBuffer::Queue> gpuBufferQueuesMap;
   std::map<std::size_t, PinnedBuffer::Queue> pinnedBufferQueuesMap;
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
   std::unique_ptr<AbsBuffer> makeGpuBuffer(std::size_t size) override
   {
      return std::make_unique<GPUBuffer>(size, _queuesMaps->gpuBufferQueuesMap[size]);
   }
   std::unique_ptr<AbsBuffer> makePinnedBuffer(std::size_t size, CudaInterface::CudaStream *stream = nullptr) override
   {
      auto out = std::make_unique<PinnedBuffer>(size, _queuesMaps->pinnedBufferQueuesMap[size]);
      out->vec().setCudaStream(stream);
      return out;
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

} // End namespace CUDA
} // End namespace RooBatchCompute
