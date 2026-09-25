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
#include "RooNaNPacker.h"
#include "Batches.h"
#include "CudaInterface.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <map>
#include <queue>
#include <stdexcept>
#include <unordered_map>
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

/// Scratch memory attached to a CUDA stream, used for staging small
/// per-kernel-launch data like the Batches descriptor and reduction results.
///
/// The slots form a ring: acquire() returns the next slot, waiting for the
/// completion of the work that was previously enqueued from that slot if it
/// is still in flight (which is rare, given the depth of the ring). Each slot
/// pairs a pinned host buffer with a device buffer of the same capacity, so
/// staging copies are truly asynchronous and no cudaMalloc()/cudaFree() calls
/// happen in the evaluation hot loop.
///
/// Like the rest of the RooBatchCompute library, this class is not
/// thread-safe: RooFit evaluates on a single thread per process.
class StreamScratch {
public:
   struct Slot {
      char *host = nullptr; // pinned host memory
      char *device = nullptr;
      std::size_t capacity = 0;
      cudaEvent_t event = nullptr; // recorded after the last enqueued use
      bool inFlight = false;
   };

   StreamScratch() = default;
   StreamScratch(StreamScratch const &) = delete;
   StreamScratch &operator=(StreamScratch const &) = delete;

   Slot &acquire(std::size_t n)
   {
      Slot &slot = _slots[_next];
      _next = (_next + 1) % _slots.size();
      if (slot.inFlight) {
         ERRCHECK(cudaEventSynchronize(slot.event));
         slot.inFlight = false;
      }
      if (slot.capacity < n) {
         // Reset the slot state before reallocating, so that a throwing
         // allocation can't leave dangling pointers with a stale capacity
         // behind (which would lead to a double free later).
         if (slot.host) {
            ERRCHECK(cudaFreeHost(slot.host));
            slot.host = nullptr;
         }
         if (slot.device) {
            ERRCHECK(cudaFree(slot.device));
            slot.device = nullptr;
         }
         slot.capacity = 0;
         const std::size_t newCapacity = std::max<std::size_t>(n, 1024);
         ERRCHECK(cudaMallocHost(reinterpret_cast<void **>(&slot.host), newCapacity));
         ERRCHECK(cudaMalloc(reinterpret_cast<void **>(&slot.device), newCapacity));
         slot.capacity = newCapacity;
      }
      if (slot.event == nullptr) {
         ERRCHECK(cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming));
      }
      return slot;
   }

   /// Mark the last enqueued use of the slot on the stream. The slot will not
   /// be handed out again before that work has completed.
   void release(Slot &slot, cudaStream_t stream)
   {
      ERRCHECK(cudaEventRecord(slot.event, stream));
      slot.inFlight = true;
   }

   /// A persistent slot for a deferred device-to-host readback: an
   /// asynchronous copy delivers device results (e.g. evaluation error
   /// counters) into the pinned host buffer, and flushDeferred() forwards
   /// them to the destination in the caller's memory once the stream was
   /// synchronized. Slots stay valid from acquireDeferred() until the flush.
   struct DeferredSlot {
      char *host = nullptr; // pinned host memory
      std::size_t capacity = 0;
      double *dst = nullptr;
      std::size_t nPending = 0;
   };

   DeferredSlot &acquireDeferred(std::size_t n)
   {
      if (_deferredCursor == _deferredSlots.size()) {
         _deferredSlots.emplace_back();
      }
      DeferredSlot &slot = _deferredSlots[_deferredCursor++];
      if (slot.capacity < n) {
         // The slot is idle here: its previous use ended with the flush after
         // a stream synchronization. Reset the state before reallocating for
         // exception safety, like in acquire().
         if (slot.host) {
            ERRCHECK(cudaFreeHost(slot.host));
            slot.host = nullptr;
         }
         slot.capacity = 0;
         ERRCHECK(cudaMallocHost(reinterpret_cast<void **>(&slot.host), n));
         slot.capacity = n;
      }
      return slot;
   }

   /// Copy the completed readbacks to their destinations. Must only be
   /// called after the stream was synchronized.
   void flushDeferred()
   {
      for (std::size_t i = 0; i < _deferredCursor; ++i) {
         DeferredSlot &slot = _deferredSlots[i];
         if (slot.dst) {
            std::memcpy(slot.dst, slot.host, slot.nPending * sizeof(double));
            slot.dst = nullptr;
            slot.nPending = 0;
         }
      }
      _deferredCursor = 0;
   }

   ~StreamScratch()
   {
      // Don't use ERRCHECK here: throwing from a destructor would terminate.
      for (Slot &slot : _slots) {
         if (slot.inFlight)
            cudaEventSynchronize(slot.event);
         if (slot.event)
            cudaEventDestroy(slot.event);
         if (slot.host)
            cudaFreeHost(slot.host);
         if (slot.device)
            cudaFree(slot.device);
      }
      for (DeferredSlot &slot : _deferredSlots) {
         if (slot.host)
            cudaFreeHost(slot.host);
      }
   }

private:
   std::array<Slot, 64> _slots;
   std::size_t _next = 0;
   std::vector<DeferredSlot> _deferredSlots;
   std::size_t _deferredCursor = 0;
};

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

      cudaStream_t stream = *cfg.cudaStream();
      StreamScratch &streamScratch = scratch(cfg.cudaStream());
      StreamScratch::Slot &slot = streamScratch.acquire(memSize);

      // The staging area has the same layout in the pinned host buffer and in
      // the device buffer, so it can be uploaded with a single copy.
      auto batches = reinterpret_cast<Batches *>(slot.host);
      auto arrays = reinterpret_cast<Batch *>(batches + 1);
      auto scalarBuffer = reinterpret_cast<double *>(arrays + vars.size());
      auto extraArgsHost = reinterpret_cast<double *>(scalarBuffer + vars.size());

      auto batchesDevice = reinterpret_cast<Batches *>(slot.device);
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

      copyHostToDevice(slot.host, slot.device, memSize, cfg.cudaStream());

      const int gridSize = getGridSize(nEvents);
      _computeFunctions[computer]<<<gridSize, blockSize, 0, stream>>>(*batchesDevice);

      // Only the NormalizedPdf computer mutates its extra args: it uses them
      // as output parameters for the evaluation error counts. Instead of
      // synchronizing the stream to read the counters back immediately, the
      // readback is deferred to avoid stalling the pipeline: an asynchronous
      // copy delivers them into a persistent pinned buffer, and the next
      // synchronizeCudaStream() call forwards them to the caller's span. The
      // caller's memory therefore has to stay valid until then.
      if (computer == NormalizedPdf && !extraArgs.empty()) {
         const std::size_t nBytes = extraArgs.size() * sizeof(double);
         StreamScratch::DeferredSlot &deferredSlot = streamScratch.acquireDeferred(nBytes);
         ERRCHECK(cudaMemcpyAsync(deferredSlot.host, extraArgsDevice, nBytes, cudaMemcpyDeviceToHost, stream));
         deferredSlot.dst = extraArgs.data();
         deferredSlot.nPending = extraArgs.size();
      }

      streamScratch.release(slot, stream);
   }
   void computeExprProgram(Config const &cfg, std::span<const ExprInstr> code, unsigned int stackDepth,
                           std::span<double> output, VarSpan vars) override;

   /// Return the sum of an input array
   double reduceSum(RooBatchCompute::Config const &cfg, InputArr input, size_t n) override;
   ReduceNLLOutput reduceNLL(RooBatchCompute::Config const &cfg, std::span<const double> probas,
                             std::span<const double> weights, std::span<const double> offsetProbas) override;

   std::unique_ptr<AbsBufferManager> createBufferManager() const override;

   CudaInterface::CudaStream *newCudaStream() const override { return new CudaInterface::CudaStream{}; }
   void deleteCudaStream(CudaInterface::CudaStream *stream) const override
   {
      _scratchMap.erase(stream);
      delete stream;
   }
   void synchronizeCudaStream(CudaInterface::CudaStream *stream) const override
   {
      ERRCHECK(::cudaStreamSynchronize(*stream));
      // Deliver deferred readbacks (e.g. the evaluation error counters from
      // compute()) that have completed with the synchronization.
      auto found = _scratchMap.find(stream);
      if (found != _scratchMap.end()) {
         found->second.flushDeferred();
      }
   }

private:
   StreamScratch &scratch(CudaInterface::CudaStream *stream) { return _scratchMap[stream]; }

   const std::vector<void (*)(Batches &)> _computeFunctions;
   mutable std::unordered_map<CudaInterface::CudaStream *, StreamScratch> _scratchMap;

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
   // (which, unlike the vector inputs, are not on the device yet). Both halves
   // come from the stream's scratch pool, so the staging copy below is truly
   // asynchronous: the pinned host buffer outlives this call and is only
   // handed out again once the event recorded by release() has completed.
   const std::size_t codeBytes = code.size() * sizeof(ExprInstr);
   const std::size_t varsBytes = vars.size() * sizeof(Batch);
   const std::size_t scalarBytes = vars.size() * sizeof(double);
   const std::size_t memSize = codeBytes + varsBytes + scalarBytes;

   cudaStream_t stream = *cfg.cudaStream();
   StreamScratch &streamScratch = scratch(cfg.cudaStream());
   StreamScratch::Slot &slot = streamScratch.acquire(memSize);

   auto codeHost = reinterpret_cast<ExprInstr *>(slot.host);
   auto varsHost = reinterpret_cast<Batch *>(slot.host + codeBytes);
   auto scalarsHost = reinterpret_cast<double *>(slot.host + codeBytes + varsBytes);

   auto codeDevice = reinterpret_cast<ExprInstr *>(slot.device);
   auto varsDevice = reinterpret_cast<Batch *>(slot.device + codeBytes);
   auto scalarsDevice = reinterpret_cast<double *>(slot.device + codeBytes + varsBytes);

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

   copyHostToDevice(slot.host, slot.device, memSize, cfg.cudaStream());

   const int gridSize = getGridSize(nEvents);
   exprProgramKernel<<<gridSize, blockSize, 0, stream>>>(codeDevice, static_cast<unsigned int>(code.size()), varsDevice,
                                                         output.data(), nEvents);

   streamScratch.release(slot, stream);
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

/// Computes the negative log likelihood sum with the same semantics as the
/// CPU implementation of RooBatchComputeInterface::reduceNLL(): zero-weight
/// events are skipped, and evaluation problems are counted and accumulated
/// into a "badness" value that the host can pack into a NaN for the error
/// recovery in the minimizer. The `stats` output has the layout
/// [badness, nNonPositive, nNaN, nInfinite] and must be zero-initialized.
__global__ void nllSumKernel(const double *__restrict__ probas, const double *__restrict__ weights,
                             const double *__restrict__ offsetProbas, size_t nProbas, double scalarProba,
                             size_t nWeights, double *__restrict__ result, double *__restrict__ stats)
{
   int thIdx = threadIdx.x;
   int gthIdx = thIdx + blockIdx.x * blockSize;
   int carry_index = threadIdx.x + blockDim.x;
   const int nThreadsTotal = blockSize * gridDim.x;

   // The first half of the shared memory is for storing the summation and the second half for the carry or compensation
   extern __shared__ double shared[];

   double sum = 0.0;
   double carry = 0.0;
   double badness = 0.0;
   unsigned int nNonPositive = 0;
   unsigned int nNaN = 0;
   unsigned int nInfinite = 0;

   for (int i = gthIdx; i < nWeights; i += nThreadsTotal) {
      const double weight = weights[i];
      // Zero-weight events don't contribute to the likelihood. Skipping them
      // also avoids 0 * inf = NaN for zero probabilities.
      if (weight == 0.0) {
         continue;
      }
      const double proba = nProbas == 1 ? scalarProba : probas[i];
      double term;
      if (proba <= 0.0) {
         ++nNonPositive;
         badness += -proba;
         term = std::log(proba);
      } else if (std::isnan(proba)) {
         ++nNaN;
         badness += RooNaNPacker::unpackNaN(proba);
         term = proba;
      } else {
         if (std::isinf(proba)) {
            ++nInfinite;
         }
         term = std::log(proba);
      }
      if (offsetProbas)
         term -= std::log(offsetProbas[i]);
      term *= -weight;
      kahanSumUpdate(sum, carry, term, 0.0);
   }

   // Accumulate the evaluation error statistics over the whole grid. These
   // atomics are on the rare path: they are only executed by threads that
   // actually encountered problematic values.
   if (badness != 0.0)
      atomicAdd(&stats[0], badness);
   if (nNonPositive != 0)
      atomicAdd(&stats[1], double(nNonPositive));
   if (nNaN != 0)
      atomicAdd(&stats[2], double(nNaN));
   if (nInfinite != 0)
      atomicAdd(&stats[3], double(nInfinite));

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
   StreamScratch &streamScratch = scratch(cfg.cudaStream());
   StreamScratch::Slot &slot = streamScratch.acquire(2 * gridSize * sizeof(double));
   auto devOut = reinterpret_cast<double *>(slot.device);
   auto hostOut = reinterpret_cast<double *>(slot.host);
   constexpr int shMemSize = 2 * blockSize * sizeof(double);
   kahanSum<<<gridSize, blockSize, shMemSize, stream>>>(input, nullptr, n, devOut, 0);
   kahanSum<<<1, blockSize, shMemSize, stream>>>(devOut, devOut + gridSize, gridSize, devOut, 0);
   CudaInterface::copyDeviceToHost(devOut, hostOut, 1, cfg.cudaStream());
   // Release right after the last enqueued use of the slot, so that the slot
   // is protected by its event even if the synchronization below throws.
   streamScratch.release(slot, stream);
   ERRCHECK(cudaStreamSynchronize(stream));
   return hostOut[0];
}

ReduceNLLOutput RooBatchComputeClass::reduceNLL(RooBatchCompute::Config const &cfg, std::span<const double> probas,
                                                std::span<const double> weights, std::span<const double> offsetProbas)
{
   ReduceNLLOutput out;
   if (probas.empty()) {
      return out;
   }
   const int gridSize = getGridSize(weights.size());
   cudaStream_t stream = *cfg.cudaStream();
   // Layout of the scratch buffer: [sum, carry, badness, nNonPositive, nNaN,
   // nInfinite, partial sums (gridSize), partial carries (gridSize)].
   StreamScratch &streamScratch = scratch(cfg.cudaStream());
   StreamScratch::Slot &slot = streamScratch.acquire((6 + 2 * gridSize) * sizeof(double));
   auto devOut = reinterpret_cast<double *>(slot.device);
   auto hostOut = reinterpret_cast<double *>(slot.host);
   constexpr int shMemSize = 2 * blockSize * sizeof(double);

#ifndef NDEBUG
   for (auto span : {probas, weights, offsetProbas}) {
      // Scalar spans can point to host memory (e.g. the scalar buffer of an
      // observable-independent pdf), so only spans with more than one element
      // are required to be on the device.
      cudaPointerAttributes attr;
      assert(span.size() <= 1 || span.data() == nullptr ||
             (cudaPointerGetAttributes(&attr, span.data()) == cudaSuccess && attr.type == cudaMemoryTypeDevice));
   }
#endif

   // Zero-initialize the evaluation error statistics for the atomic updates.
   ERRCHECK(cudaMemsetAsync(devOut + 2, 0, 4 * sizeof(double), stream));

   nllSumKernel<<<gridSize, blockSize, shMemSize, stream>>>(
      probas.data(), weights.data(), offsetProbas.empty() ? nullptr : offsetProbas.data(), probas.size(),
      probas.size() == 1 ? probas[0] : 0.0, weights.size(), devOut + 6, devOut + 2);

   kahanSum<<<1, blockSize, shMemSize, stream>>>(devOut + 6, devOut + 6 + gridSize, gridSize, devOut, 0);

   // The sum, its Kahan carry, and the evaluation error statistics are
   // adjacent in the output buffer, so they can be read back in a single copy.
   CudaInterface::copyDeviceToHost(devOut, hostOut, 6, cfg.cudaStream());
   // Release right after the last enqueued use of the slot, so that the slot
   // is protected by its event even if the synchronization below throws.
   streamScratch.release(slot, stream);
   ERRCHECK(cudaStreamSynchronize(stream));

   out.nllSum = hostOut[0];
   out.nllSumCarry = hostOut[1];
   out.nNonPositiveValues = hostOut[3];
   out.nNaNValues = hostOut[4];
   out.nInfiniteValues = hostOut[5];

   if (hostOut[2] != 0.0) {
      // Some events had evaluation errors: return the accumulated "badness"
      // of the errors packed into a NaN, like the CPU implementation, so the
      // minimizer can use it to recover.
      out.nllSum = RooNaNPacker::packFloatIntoNaN(hostOut[2]);
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
         // The copy is asynchronous, and the caller reads the host memory
         // right away, so the stream needs to be synchronized here.
         if (_cudaStream) {
            ERRCHECK(cudaStreamSynchronize(*_cudaStream));
         }
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
